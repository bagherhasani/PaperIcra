#!/usr/bin/env python3
"""
Block 2 path guard: if predicted person cloud intersects the current global plan,
clear local costmap and resend the active Nav2 goal so the path bends quickly.

Run alongside zed2 (Mode B / pred):
  source /opt/ros/humble/setup.bash
  source ~/ros2_ws/install/setup.bash
  cd ~/ros2_ws/src/zed-detection/zed-prediction
  python3 block2_path_guard.py

Only acts when NAV_CLOUD_MODE=pred (reads test_config).
"""

from __future__ import annotations

import math
import struct
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap

try:
    from test_config import NAV_CLOUD_MODE
except ImportError:
    NAV_CLOUD_MODE = "pred"

# How close cloud points must be to the plan (m) to count as conflict
HIT_DIST_M = 0.55
# Only look this far along the plan from the robot (m)
LOOKAHEAD_M = 4.0
# Min seconds between forced replans
COOLDOWN_S = 1.25


def cloud_xy(msg: PointCloud2):
    pts = []
    n = msg.width * msg.height
    step = msg.point_step
    data = msg.data
    for i in range(n):
        off = i * step
        if off + 8 > len(data):
            break
        x, y = struct.unpack_from("<ff", data, off)
        if math.isfinite(x) and math.isfinite(y):
            pts.append((x, y))
    return pts


def min_dist_point_to_path(px, py, path: Path, lookahead_m: float) -> float:
    poses = path.poses
    if len(poses) < 2:
        return float("inf")
    best = float("inf")
    traveled = 0.0
    for i in range(1, len(poses)):
        x0 = poses[i - 1].pose.position.x
        y0 = poses[i - 1].pose.position.y
        x1 = poses[i].pose.position.x
        y1 = poses[i].pose.position.y
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-6:
            continue
        # project point onto segment
        t = ((px - x0) * (x1 - x0) + (py - y0) * (y1 - y0)) / (seg * seg)
        t = max(0.0, min(1.0, t))
        qx = x0 + t * (x1 - x0)
        qy = y0 + t * (y1 - y0)
        best = min(best, math.hypot(px - qx, py - qy))
        traveled += seg
        if traveled > lookahead_m:
            break
    return best


class Block2PathGuard(Node):
    def __init__(self):
        super().__init__("block2_path_guard")
        self.mode = str(NAV_CLOUD_MODE).lower().strip()
        self.plan = None
        self.cloud_pts = []
        self.goal = None
        self.last_replan_t = 0.0
        self.plan_frame = "map"

        self.create_subscription(Path, "/plan", self.on_plan, 10)
        self.create_subscription(
            PointCloud2,
            "/person/predicted_cloud",
            self.on_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 10)
        self.event_pub = self.create_publisher(String, "/block2/replan_event", 10)

        self.clear_cli = self.create_client(
            ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap"
        )
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.timer = self.create_timer(0.15, self.tick)
        self.get_logger().info(
            f"Path guard ready | mode={self.mode} | hit<{HIT_DIST_M}m | "
            f"lookahead={LOOKAHEAD_M}m | cooldown={COOLDOWN_S}s"
        )
        if self.mode != "pred":
            self.get_logger().warn(
                "NAV_CLOUD_MODE is not 'pred' — guard will idle (Mode A)."
            )

    def on_plan(self, msg: Path):
        self.plan = msg
        if msg.header.frame_id:
            self.plan_frame = msg.header.frame_id

    def on_cloud(self, msg: PointCloud2):
        # Cloud is in base_link; plan is in map — transform via TF
        self.cloud_pts = cloud_xy(msg)
        self.cloud_frame = msg.header.frame_id or "base_link"
        self.cloud_stamp = msg.header.stamp

    def on_goal(self, msg: PoseStamped):
        self.goal = msg

    def _cloud_in_plan_frame(self):
        """Transform base_link cloud points into plan frame (map/odom)."""
        if not self.cloud_pts:
            return []
        try:
            from tf2_ros import Buffer, TransformListener
        except ImportError:
            return []
        if not hasattr(self, "_tf_buf"):
            self._tf_buf = Buffer()
            self._tf_listener = TransformListener(self._tf_buf, self)
        try:
            tf = self._tf_buf.lookup_transform(
                self.plan_frame,
                getattr(self, "cloud_frame", "base_link"),
                rclpy.time.Time(),
            )
        except Exception:
            return []
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for x, y in self.cloud_pts:
            out.append((tx + c * x - s * y, ty + s * x + c * y))
        return out

    def tick(self):
        if self.mode != "pred":
            return
        if self.plan is None or len(self.plan.poses) < 2:
            return
        pts = self._cloud_in_plan_frame()
        if not pts:
            return

        hit = False
        min_d = float("inf")
        for x, y in pts:
            d = min_dist_point_to_path(x, y, self.plan, LOOKAHEAD_M)
            min_d = min(min_d, d)
            if d <= HIT_DIST_M:
                hit = True
                break
        if not hit:
            return

        now = time.time()
        if now - self.last_replan_t < COOLDOWN_S:
            return
        self.last_replan_t = now

        self.get_logger().warn(
            f"PRED∩PLAN hit (min_d={min_d:.2f}m) — clear local + replan"
        )
        self.event_pub.publish(
            String(data=f"replan min_d={min_d:.3f} t={now:.3f}")
        )

        # 1) Clear local costmap so new marks / freespace update
        if self.clear_cli.wait_for_service(timeout_sec=0.05):
            req = ClearEntireCostmap.Request()
            self.clear_cli.call_async(req)

        # 2) Resend active goal → BT GoalUpdated / fresh ComputePathToPose
        goal_pose = self.goal
        if goal_pose is None and self.plan is not None and self.plan.poses:
            goal_pose = PoseStamped()
            goal_pose.header = self.plan.header
            goal_pose.pose = self.plan.poses[-1].pose

        if goal_pose is not None and self.nav_client.wait_for_server(timeout_sec=0.05):
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = goal_pose
            self.nav_client.send_goal_async(goal_msg)
        else:
            self.get_logger().info(
                "No goal + no action server — relying on 3 Hz BT replan after clear"
            )


def main():
    rclpy.init()
    node = Block2PathGuard()
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
