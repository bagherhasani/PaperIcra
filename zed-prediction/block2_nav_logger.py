#!/usr/bin/env python3
"""
Block 2 research logger — starts when the robot moves, logs person + Nav2.

Run in a third terminal while Mode A or B is active:
  source /opt/ros/humble/setup.bash
  source ~/ros2_ws/install/setup.bash
  cd ~/ros2_ws/src/zed-detection/zed-prediction
  python3 block2_nav_logger.py

Outputs (under results/):
  block2_nav_{TEST_NAME}_{stamp}.csv     — 10 Hz samples while moving
  block2_nav_{TEST_NAME}_{stamp}_summary.txt — one segment summary when robot stops

What it catches (for "feels wrong"):
  - person_detected / person_lost
  - cloud_extent_m (enlarged blob)
  - pred_vs_now_dist, pred_bearing_err_deg (prediction off to the side / wrong way)
  - plan_length_m growth when person appears
  - progress_fail_count (Nav2 "Failed to make progress")
  - time_to_stop after person, min clearance
"""

from __future__ import annotations

import csv
import math
import struct
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from rcl_interfaces.msg import Log

try:
    from test_config import TEST_NAME, NAV_CLOUD_MODE
except ImportError:
    TEST_NAME = "block2_unknown"
    NAV_CLOUD_MODE = "?"

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Start logging when |v| or |w| exceeds this; stop segment after idle this long
MOVE_LIN = 0.02  # m/s
MOVE_ANG = 0.05  # rad/s
IDLE_END_S = 2.5
SAMPLE_HZ = 10.0


def yaw_from_quat(z, w):
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def ang_diff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def path_length(path: Path) -> float:
    pts = path.poses
    if len(pts) < 2:
        return 0.0
    s = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1].pose.position.x, pts[i - 1].pose.position.y
        x1, y1 = pts[i].pose.position.x, pts[i].pose.position.y
        s += math.hypot(x1 - x0, y1 - y0)
    return s


def cloud_stats(msg: PointCloud2):
    """Return (n, cx, cy, extent_m) in the cloud's frame (base_link)."""
    if msg is None or msg.width == 0:
        return 0, float("nan"), float("nan"), 0.0
    # Assume xyz float32 contiguous (danger_zone_ros layout)
    n = msg.width * msg.height
    xs, ys = [], []
    step = msg.point_step
    data = msg.data
    for i in range(n):
        off = i * step
        if off + 12 > len(data):
            break
        x, y, _z = struct.unpack_from("<fff", data, off)
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    if not xs:
        return 0, float("nan"), float("nan"), 0.0
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    extent = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys))
    return len(xs), cx, cy, extent


class Block2NavLogger(Node):
    def __init__(self):
        super().__init__("block2_nav_logger")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tag = f"{TEST_NAME}_{stamp}"
        self.csv_path = RESULTS / f"block2_nav_{self.tag}.csv"
        self.sum_path = RESULTS / f"block2_nav_{self.tag}_summary.txt"

        self.mode = str(NAV_CLOUD_MODE)
        self.robot_x = self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_v = self.robot_w = 0.0
        self.cmd_vx = self.cmd_wz = 0.0

        self.person_now_x = self.person_now_y = float("nan")
        self.person_pred_x = self.person_pred_y = float("nan")  # +2s arrow tip
        self.person_seen = False
        self.person_age = 0.0

        self.cloud_n = 0
        self.cloud_cx = self.cloud_cy = float("nan")
        self.cloud_extent = 0.0
        self.cloud_age = 0.0

        self.plan_len = 0.0
        self.local_plan_len = 0.0
        self.progress_fails = 0
        self.last_rosout = ""

        self.moving = False
        self.idle_t0 = None
        self.seg_t0 = None
        self.seg_rows = []
        self.wall0 = time.time()

        # segment accumulators
        self.seg_dist = 0.0
        self.prev_xy = None
        self.min_clear = float("inf")
        self.max_cloud_extent = 0.0
        self.max_plan_len = 0.0
        self.person_frames = 0
        self.wrong_side_frames = 0
        self.enlarged_frames = 0

        fields = [
            "t_wall",
            "t_seg",
            "mode",
            "test_name",
            "moving",
            "robot_x",
            "robot_y",
            "robot_yaw",
            "odom_v",
            "odom_w",
            "cmd_vx",
            "cmd_wz",
            "person_detected",
            "person_now_x",
            "person_now_y",
            "person_pred2s_x",
            "person_pred2s_y",
            "pred_vs_now_dist",
            "pred_bearing_err_deg",
            "cloud_n",
            "cloud_cx",
            "cloud_cy",
            "cloud_extent_m",
            "robot_to_person_m",
            "robot_to_cloud_m",
            "plan_length_m",
            "local_plan_length_m",
            "progress_fail_count",
            "flag_cloud_enlarged",
            "flag_pred_wrong_side",
            "last_nav_warn",
        ]
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fields)
        self.writer.writeheader()
        self.csv_file.flush()

        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 20)
        self.create_subscription(Path, "/plan", self.on_plan, 10)
        self.create_subscription(Path, "/local_plan", self.on_local_plan, 10)
        self.create_subscription(
            MarkerArray, "/person/danger_zone", self.on_markers, 10
        )
        self.create_subscription(
            PointCloud2,
            "/person/predicted_cloud",
            self.on_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(Log, "/rosout", self.on_rosout, 50)

        self.timer = self.create_timer(1.0 / SAMPLE_HZ, self.tick)

        self.get_logger().info(
            f"Block2 logger ready | mode={self.mode} | file={self.csv_path.name}"
        )
        self.get_logger().info(
            "Logging starts automatically when the robot moves."
        )

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_x, self.robot_y = p.x, p.y
        self.robot_yaw = yaw_from_quat(q.z, q.w)
        self.robot_v = msg.twist.twist.linear.x
        self.robot_w = msg.twist.twist.angular.z
        if self.prev_xy is not None and self.moving:
            self.seg_dist += math.hypot(
                self.robot_x - self.prev_xy[0], self.robot_y - self.prev_xy[1]
            )
        self.prev_xy = (self.robot_x, self.robot_y)

    def on_cmd(self, msg: Twist):
        self.cmd_vx = msg.linear.x
        self.cmd_wz = msg.angular.z

    def on_plan(self, msg: Path):
        self.plan_len = path_length(msg)

    def on_local_plan(self, msg: Path):
        self.local_plan_len = path_length(msg)

    def on_markers(self, msg: MarkerArray):
        now = pred = None
        for m in msg.markers:
            if m.action != 0:  # ADD
                continue
            if m.ns == "now_dot" and m.type == 2:  # SPHERE
                now = (m.pose.position.x, m.pose.position.y)
            elif m.ns == "pred_2s" and m.type == 0 and len(m.points) >= 2:  # ARROW
                pred = (m.points[1].x, m.points[1].y)
                if now is None:
                    now = (m.points[0].x, m.points[0].y)
        if now is not None:
            self.person_now_x, self.person_now_y = now
            self.person_seen = True
            self.person_age = 0.0
        if pred is not None:
            self.person_pred_x, self.person_pred_y = pred

    def on_cloud(self, msg: PointCloud2):
        n, cx, cy, extent = cloud_stats(msg)
        self.cloud_n, self.cloud_cx, self.cloud_cy, self.cloud_extent = n, cx, cy, extent
        self.cloud_age = 0.0
        if n > 0:
            self.person_seen = True

    def on_rosout(self, msg: Log):
        s = msg.msg or ""
        if "Failed to make progress" in s:
            self.progress_fails += 1
            self.last_rosout = s[:120]
        elif any(
            k in s
            for k in (
                "NoValidControl",
                "Goal failed",
                "Collision",
                "Aborting handle",
            )
        ):
            self.last_rosout = s[:120]

    def is_commanded_or_moving(self) -> bool:
        return (
            abs(self.robot_v) > MOVE_LIN
            or abs(self.robot_w) > MOVE_ANG
            or abs(self.cmd_vx) > MOVE_LIN
            or abs(self.cmd_wz) > MOVE_ANG
        )

    def metrics(self):
        pred_dist = float("nan")
        bearing_err = float("nan")
        wrong_side = 0
        if (
            math.isfinite(self.person_now_x)
            and math.isfinite(self.person_pred_x)
        ):
            dx = self.person_pred_x - self.person_now_x
            dy = self.person_pred_y - self.person_now_y
            pred_dist = math.hypot(dx, dy)
            # In base_link, person "forward" of themselves along prediction
            # Wrong-side heuristic: cloud/pred mostly behind robot→person ray
            # or prediction distance huge while person nearly still (pred_dist
            # large with small implied speed handled in summary).
            to_person = math.atan2(
                self.person_now_y, self.person_now_x
            ) if (abs(self.person_now_x) + abs(self.person_now_y)) > 1e-3 else 0.0
            pred_bear = math.atan2(dy, dx) if pred_dist > 1e-3 else to_person
            bearing_err = math.degrees(ang_diff(pred_bear, to_person))
            # If prediction points opposite to robot→person (blob behind person
            # from robot view), flag wrong side when |err| > 90 and mode=pred
            if self.mode == "pred" and abs(bearing_err) > 90.0 and pred_dist > 0.15:
                wrong_side = 1

        r_person = float("nan")
        if math.isfinite(self.person_now_x):
            r_person = math.hypot(self.person_now_x, self.person_now_y)

        r_cloud = float("nan")
        if math.isfinite(self.cloud_cx):
            r_cloud = math.hypot(self.cloud_cx, self.cloud_cy)

        # Enlarged: disc radius was 0.22; extent >> that + corridor means fat
        enlarged = 1 if self.cloud_extent > 0.55 else 0

        return pred_dist, bearing_err, wrong_side, r_person, r_cloud, enlarged

    def tick(self):
        self.person_age += 1.0 / SAMPLE_HZ
        self.cloud_age += 1.0 / SAMPLE_HZ
        if self.person_age > 1.0 and self.cloud_age > 1.0:
            self.person_seen = False

        moving_now = self.is_commanded_or_moving()
        now = time.time()

        if moving_now:
            if not self.moving:
                self.moving = True
                self.seg_t0 = now
                self.idle_t0 = None
                self.seg_dist = 0.0
                self.min_clear = float("inf")
                self.max_cloud_extent = 0.0
                self.max_plan_len = 0.0
                self.person_frames = 0
                self.wrong_side_frames = 0
                self.enlarged_frames = 0
                self.progress_fails = 0
                self.get_logger().info(">>> MOTION — logging started")
            self.idle_t0 = None
        else:
            if self.moving:
                if self.idle_t0 is None:
                    self.idle_t0 = now
                elif now - self.idle_t0 >= IDLE_END_S:
                    self.finish_segment()
                    self.moving = False
                    self.idle_t0 = None

        if not self.moving:
            return

        pred_dist, bearing_err, wrong_side, r_person, r_cloud, enlarged = self.metrics()
        if math.isfinite(r_person):
            self.min_clear = min(self.min_clear, r_person)
        if self.person_seen:
            self.person_frames += 1
        self.wrong_side_frames += wrong_side
        self.enlarged_frames += enlarged
        self.max_cloud_extent = max(self.max_cloud_extent, self.cloud_extent)
        self.max_plan_len = max(self.max_plan_len, self.plan_len)

        t_seg = now - self.seg_t0 if self.seg_t0 else 0.0
        row = {
            "t_wall": f"{now - self.wall0:.3f}",
            "t_seg": f"{t_seg:.3f}",
            "mode": self.mode,
            "test_name": TEST_NAME,
            "moving": 1,
            "robot_x": f"{self.robot_x:.4f}",
            "robot_y": f"{self.robot_y:.4f}",
            "robot_yaw": f"{self.robot_yaw:.4f}",
            "odom_v": f"{self.robot_v:.4f}",
            "odom_w": f"{self.robot_w:.4f}",
            "cmd_vx": f"{self.cmd_vx:.4f}",
            "cmd_wz": f"{self.cmd_wz:.4f}",
            "person_detected": int(self.person_seen),
            "person_now_x": f"{self.person_now_x:.4f}" if math.isfinite(self.person_now_x) else "",
            "person_now_y": f"{self.person_now_y:.4f}" if math.isfinite(self.person_now_y) else "",
            "person_pred2s_x": f"{self.person_pred_x:.4f}" if math.isfinite(self.person_pred_x) else "",
            "person_pred2s_y": f"{self.person_pred_y:.4f}" if math.isfinite(self.person_pred_y) else "",
            "pred_vs_now_dist": f"{pred_dist:.4f}" if math.isfinite(pred_dist) else "",
            "pred_bearing_err_deg": f"{bearing_err:.1f}" if math.isfinite(bearing_err) else "",
            "cloud_n": self.cloud_n,
            "cloud_cx": f"{self.cloud_cx:.4f}" if math.isfinite(self.cloud_cx) else "",
            "cloud_cy": f"{self.cloud_cy:.4f}" if math.isfinite(self.cloud_cy) else "",
            "cloud_extent_m": f"{self.cloud_extent:.4f}",
            "robot_to_person_m": f"{r_person:.4f}" if math.isfinite(r_person) else "",
            "robot_to_cloud_m": f"{r_cloud:.4f}" if math.isfinite(r_cloud) else "",
            "plan_length_m": f"{self.plan_len:.4f}",
            "local_plan_length_m": f"{self.local_plan_len:.4f}",
            "progress_fail_count": self.progress_fails,
            "flag_cloud_enlarged": enlarged,
            "flag_pred_wrong_side": wrong_side,
            "last_nav_warn": self.last_rosout.replace(",", ";"),
        }
        self.writer.writerow(row)
        self.csv_file.flush()

    def finish_segment(self):
        dur = (time.time() - self.seg_t0) if self.seg_t0 else 0.0
        pf = self.person_frames
        wrong_pct = 100.0 * self.wrong_side_frames / max(pf, 1)
        enl_pct = 100.0 * self.enlarged_frames / max(1, int(dur * SAMPLE_HZ))

        # Rough verdict for this segment (not a paper metric — triage only)
        issues = []
        if self.progress_fails > 0:
            issues.append(f"progress_fails={self.progress_fails}")
        if self.max_cloud_extent > 0.7:
            issues.append(f"cloud_very_large={self.max_cloud_extent:.2f}m")
        if wrong_pct > 30 and self.mode == "pred":
            issues.append(f"pred_wrong_side={wrong_pct:.0f}%")
        if pf == 0:
            issues.append("no_person_while_moving")
        if not issues:
            verdict = "OK_or_unclear — compare A vs B on min_clear + progress_fails + duration"
        else:
            verdict = "SUSPECT: " + "; ".join(issues)

        text = "\n".join(
            [
                f"TEST={TEST_NAME}  MODE={self.mode}",
                f"duration_s={dur:.2f}",
                f"distance_m={self.seg_dist:.2f}",
                f"person_frames={pf}",
                f"min_robot_to_person_m={self.min_clear if self.min_clear < 1e9 else float('nan'):.3f}",
                f"max_cloud_extent_m={self.max_cloud_extent:.3f}",
                f"max_plan_length_m={self.max_plan_len:.3f}",
                f"progress_fail_count={self.progress_fails}",
                f"pred_wrong_side_pct≈{wrong_pct:.1f}",
                f"cloud_enlarged_pct≈{enl_pct:.1f}",
                f"verdict={verdict}",
                f"csv={self.csv_path}",
            ]
        )
        with open(self.sum_path, "a") as f:
            f.write("\n===== SEGMENT =====\n")
            f.write(text + "\n")
        self.get_logger().info("<<< STOP — segment summary:\n" + text)

    def destroy_node(self):
        if self.moving:
            self.finish_segment()
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = Block2NavLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
