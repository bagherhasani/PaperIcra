import time
import math

import cv2
import numpy as np
import pyzed.sl as sl

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ZedFollower(Node):
    def __init__(self):
        super().__init__("zed_color_follower")

        # Publisher to Tracer base velocity command
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Actuation
        # If False, the robot will never move (status-only demo).
        self.actuation_enabled = True

        # Control parameters
        self.target_distance = 1.0   # desired distance to target [m]
        self.min_distance = 0.6      # stop if closer than this [m]
        self.max_lin_speed = 0.2     # maximum forward speed [m/s]
        self.max_ang_speed = 0.5     # maximum turn speed [rad/s]
        self.k_lin = 0.7             # linear gain
        self.k_ang = 1.0             # angular gain

        # Velocity smoothing (exponential low-pass filter)
        self.smooth_alpha = 0.35     # blend factor: 0=frozen, 1=no smoothing
        self.prev_lin = 0.0
        self.prev_ang = 0.0

        # Angular dead-zone: suppress corrections smaller than ~3 degrees
        self.ang_deadzone_rad = 0.052

        # For on-screen feedback
        self.last_lin = 0.0
        self.last_ang = 0.0
        self.last_status = "Idle"

        # Target memory (for search behavior)
        self.last_angle_error = 0.3      # sign = last known direction (+=left, -=right)
        self.last_seen_time = 0.0
        self.search_enabled = True
        self.search_start_delay_sec = 1.0
        self.search_turn_speed = 0.25    # rad/s
        self.search_timeout_sec = 10.0
        self.search_full_turn = True     # complete a full 360° scan when target is lost
        self.search_angle_rad = 0.0      # integrated angle during current search
        self._search_prev_t = None

        # Target locking — trust ZED SDK body.id for stable re-ID
        self.target_id = None
        self.lock_lost_time = None
        self.lock_lost_timeout_sec = 12.0

        # Frame rate cap: grab at camera FPS, process/control at ≤15 Hz
        self.control_hz = 15.0
        self._last_control_t = 0.0

        # ZED camera init
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        # Phase 2: PERFORMANCE instead of NEURAL — sufficient for following, much faster on Jetson
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED: {status}")
            raise RuntimeError("Cannot open ZED camera")

        self.person_mode = True
        self.color_ratio_threshold = 0.06   # 6% of shirt ROI must match target color
        self.enable_object_fallback = False
        self.objdet_enabled = False
        self.objects = sl.Objects()
        self.obj_runtime = sl.ObjectDetectionRuntimeParameters()
        self.obj_runtime.detection_confidence_threshold = 50

        # Positional tracking (helps body tracking stability)
        try:
            tracking_params = sl.PositionalTrackingParameters()
            self.zed.enable_positional_tracking(tracking_params)
        except Exception as e:
            self.get_logger().warn(f"Positional tracking disabled: {e}. Body tracking stability may degrade.")

        # Body Tracking — BODY_18 (Phase 1: was BODY_38; we only use 4 keypoints)
        self.body_enabled = False
        self.bodies = sl.Bodies()
        self.body_runtime = sl.BodyTrackingRuntimeParameters()
        self.body_runtime.detection_confidence_threshold = 40
        try:
            body_param = sl.BodyTrackingParameters()
            body_param.enable_tracking = True
            body_param.enable_body_fitting = True
            body_param.enable_segmentation = True
            body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
            body_param.body_format = sl.BODY_FORMAT.BODY_18  # Phase 1: was BODY_38
            res_bt = self.zed.enable_body_tracking(body_param)
            self.body_enabled = (res_bt == sl.ERROR_CODE.SUCCESS)
            if not self.body_enabled:
                self.get_logger().warn(f"Body tracking failed to enable: {res_bt}. Robot will not follow anyone.")
        except Exception as e:
            self.get_logger().warn(f"Body tracking exception: {e}. Robot will not follow anyone.")
            self.body_enabled = False

        try:
            obj_params = sl.ObjectDetectionParameters()
            obj_params.enable_tracking = True
            if hasattr(sl, "OBJECT_DETECTION_MODEL") and hasattr(sl.OBJECT_DETECTION_MODEL, "MULTI_CLASS_BOX_FAST"):
                obj_params.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
            elif hasattr(sl, "OBJECT_DETECTION_MODEL") and hasattr(sl.OBJECT_DETECTION_MODEL, "MULTI_CLASS_BOX_MEDIUM"):
                obj_params.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
            res = self.zed.enable_object_detection(obj_params)
            self.objdet_enabled = (res == sl.ERROR_CODE.SUCCESS)
        except Exception:
            self.objdet_enabled = False

        # Target color — university blue (Phase 8: was red)
        # OpenCV HSV: H 0..179, S 0..255, V 0..255
        # Blue: H ~100–130. Single range (no hue wrap unlike red).
        # Tune on demo day:
        #   - Lower S to 70 if shirt looks desaturated under indoor lighting
        #   - Narrow H to 105–125 to avoid cyan/indigo crossover
        #   - Watch for blue jeans in background (raise S threshold if needed)
        self.target_hsv_ranges = [
            (np.array([100, 100, 50]), np.array([130, 255, 255])),
        ]

        self.runtime_params = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.point_cloud = sl.Mat()

        self.kernel = np.ones((5, 5), np.uint8)

        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        self.get_logger().info(
            "ZED color follower started. Press 'q' or ESC in the window to quit."
        )

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.last_lin = 0.0
        self.last_ang = 0.0
        self.prev_lin = 0.0   # reset smoothing filter — prevents post-stop creep
        self.prev_ang = 0.0
        self.last_status = "STOPPED"

    def search_for_target(self):
        """Rotate in place to reacquire target (linear.x always 0 for safety)."""
        twist = Twist()
        twist.linear.x = 0.0

        if not self.actuation_enabled:
            self.last_lin = 0.0
            self.last_ang = 0.0
            self.last_status = "SEARCHING (disabled)"
            self.cmd_pub.publish(Twist())   # ensure zero velocity is published
            return

        direction = 1.0 if self.last_angle_error >= 0.0 else -1.0
        w = direction * min(self.search_turn_speed, self.max_ang_speed)

        now = time.time()
        if self._search_prev_t is None:
            self._search_prev_t = now
        dt = max(0.0, now - self._search_prev_t)
        self._search_prev_t = now
        self.search_angle_rad += abs(w) * dt

        if self.search_full_turn and self.search_angle_rad >= (2.0 * math.pi):
            twist.angular.z = 0.0
            self.last_status = "SEARCH DONE (360°) - STOPPED"
            self.last_lin = 0.0
            self.last_ang = 0.0
            self.cmd_pub.publish(twist)
            return

        twist.angular.z = w
        self.last_lin = 0.0
        self.last_ang = float(twist.angular.z)
        self.last_status = "SEARCHING (rotate in place)"
        self.cmd_pub.publish(twist)

    def control_robot(self, distance, angle_rad):
        """Follow controller with velocity smoothing and angular dead-zone."""
        if not self.actuation_enabled:
            self.last_lin = 0.0
            self.last_ang = 0.0
            self.last_status = "LOCKED (no actuation)"
            self.cmd_pub.publish(Twist())   # ensure zero velocity is published
            return

        twist = Twist()

        # Angular dead-zone: ignore small alignment errors to prevent micro-twitching
        if abs(angle_rad) < self.ang_deadzone_rad:
            raw_ang = 0.0
        else:
            raw_ang = self.k_ang * angle_rad
        raw_ang = max(min(raw_ang, self.max_ang_speed), -self.max_ang_speed)

        # Linear control — forward only (no reversing in a crowded demo environment)
        if distance is None or distance < self.min_distance:
            raw_lin = 0.0
            status = "TOO CLOSE - HOLDING"
        else:
            error_d = distance - self.target_distance
            raw_lin = self.k_lin * error_d
            raw_lin = max(0.0, min(raw_lin, self.max_lin_speed))  # clamp: never reverse
            if abs(raw_lin) < 1e-3 and abs(raw_ang) < 1e-3:
                status = "ALIGNED - HOLDING"
            else:
                status = "FOLLOWING TARGET"

        # Exponential low-pass filter (smooths abrupt velocity jumps)
        a = self.smooth_alpha
        smooth_lin = a * raw_lin + (1.0 - a) * self.prev_lin
        smooth_ang = a * raw_ang + (1.0 - a) * self.prev_ang
        self.prev_lin = smooth_lin
        self.prev_ang = smooth_ang

        twist.linear.x = smooth_lin
        twist.angular.z = smooth_ang
        self.last_lin = float(smooth_lin)
        self.last_ang = float(smooth_ang)
        self.last_status = status
        self.cmd_pub.publish(twist)

    def _shirt_color_ratio(self, mask_color_frame, body, img_h, img_w):
        """Check target color ratio in upper 55% of body bounding box.

        Uses bounding_box_2d (always available) rather than skeleton keypoints
        (which fail on partial occlusion). body.mask gates pixels to only
        count those belonging to this specific person when available.

        Returns float in [0, 1].
        """
        try:
            bb2d = body.bounding_box_2d
            pts = np.array([[p[0], p[1]] for p in bb2d], dtype=np.float32)
            x1 = int(np.clip(np.min(pts[:, 0]), 0, img_w - 1))
            y1 = int(np.clip(np.min(pts[:, 1]), 0, img_h - 1))
            x2 = int(np.clip(np.max(pts[:, 0]), 0, img_w - 1))
            y2_full = int(np.clip(np.max(pts[:, 1]), 0, img_h - 1))
            y2 = int(y1 + 0.55 * (y2_full - y1))   # upper 55% = shirt region
            if x2 <= x1 or y2 <= y1:
                return 0.0

            color_roi = mask_color_frame[y1:y2, x1:x2]

            # Gate by per-person segmentation mask when available
            try:
                m = body.mask
                if m is not None:
                    md = m.get_data()
                    if md is not None:
                        if md.ndim == 3:
                            md = md[:, :, 0]
                        person_mask = (md > 0).astype(np.uint8) * 255
                        if person_mask.shape[:2] != (img_h, img_w):
                            person_mask = cv2.resize(
                                person_mask, (img_w, img_h),
                                interpolation=cv2.INTER_NEAREST
                            )
                        seg_roi = person_mask[y1:y2, x1:x2]
                        denom = int(np.count_nonzero(seg_roi))
                        if denom > 0:
                            both = cv2.bitwise_and(color_roi, seg_roi)
                            return float(np.count_nonzero(both)) / float(denom)
            except Exception as e:
                self.get_logger().debug(f"Segmentation mask unavailable, using raw bbox: {e}")

            # Fallback: raw bbox ratio (no segmentation available)
            area = color_roi.size
            return 0.0 if area == 0 else float(np.count_nonzero(color_roi)) / float(area)

        except Exception:
            return 0.0


def main():
    rclpy.init()
    try:
        node = ZedFollower()
    except RuntimeError:
        rclpy.shutdown()
        return

    zed = node.zed
    runtime_params = node.runtime_params
    image = node.image

    prev_time = node.prev_time
    frame_count = node.frame_count
    fps = node.fps
    last_frame_bgr = None

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            last_frame_bgr = frame_bgr

            img_h, img_w, _ = frame_bgr.shape

            # Phase 6: frame rate cap — grab at full camera FPS, process at ≤15 Hz
            now_t = time.time()
            if (now_t - node._last_control_t) < (1.0 / node.control_hz):
                cv2.imshow("ZED Person Follow (Color)", frame_bgr)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            node._last_control_t = now_t

            detected = False
            dist_text = ""

            # Precompute HSV + color mask once per frame
            hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            mask_color_frame = None
            for lower, upper in node.target_hsv_ranges:
                m = cv2.inRange(hsv_frame, lower, upper)
                mask_color_frame = m if mask_color_frame is None else cv2.bitwise_or(mask_color_frame, m)

            # --- Human-first: Body Tracking + shirt color check ---
            if node.person_mode and node.body_enabled:
                zed.retrieve_bodies(node.bodies, node.body_runtime)

                best_body = None
                best_ratio = 0.0

                for b in node.bodies.body_list:
                    if b.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                        continue
                    # If locked, only accept the same SDK id (SDK handles re-ID natively)
                    if node.target_id is not None and b.id != node.target_id:
                        continue
                    ratio = node._shirt_color_ratio(mask_color_frame, b, img_h, img_w)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_body = b

                if best_body is not None and best_ratio >= node.color_ratio_threshold:
                    node.target_id = best_body.id
                    node.lock_lost_time = None
                    node.search_angle_rad = 0.0
                    node._search_prev_t = None
                    node.last_seen_time = time.time()

                    # Centroid from bbox for display crosshair
                    bb2d = best_body.bounding_box_2d
                    pts = np.array([[p[0], p[1]] for p in bb2d], dtype=np.float32)
                    bx1 = int(np.min(pts[:, 0]))
                    by1 = int(np.min(pts[:, 1]))
                    bx2 = int(np.max(pts[:, 0]))
                    by2 = int(np.max(pts[:, 1]))

                    # Phase 4: horizontal ground-plane distance only (was sqrt(X²+Y²+Z²))
                    try:
                        X = float(best_body.position[0])
                        Z = float(best_body.position[2])
                        if not math.isfinite(X) or not math.isfinite(Z) or Z <= 0.05:
                            raise ValueError("invalid body position")

                        distance = float(math.sqrt(X * X + Z * Z))
                        angle_raw = math.atan2(X, Z)
                        angle_error = -angle_raw   # positive => target is to the RIGHT of center → positive angular.z = left turn to face it

                        angle_err_deg = math.degrees(angle_error)
                        dir_text = "center"
                        if angle_err_deg > 3:
                            dir_text = "left"
                        elif angle_err_deg < -3:
                            dir_text = "right"

                        dist_text = f"{distance:.2f} m, {abs(angle_err_deg):.1f} deg {dir_text}"
                        node.last_angle_error = float(angle_error)
                        node.control_robot(distance, angle_error)

                    except Exception:
                        # Fallback: pixel-based steering if 3D position unavailable
                        target_cx = (bx1 + bx2) // 2
                        pixel_error = (target_cx - (img_w // 2)) / float(img_w)
                        angle_error = -pixel_error
                        node.last_angle_error = float(angle_error)
                        node.control_robot(None, angle_error)

                    # Draw bounding box and label
                    cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                    cv2.putText(
                        frame_bgr,
                        f"Target id={node.target_id} ({best_ratio * 100:.1f}%)",
                        (bx1, max(0, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                    )
                    if dist_text:
                        cv2.putText(frame_bgr, dist_text, (20, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    node.last_status = f"LOCKED id={node.target_id} | FOLLOW"
                    detected = True

                else:
                    # Target missing or below threshold — unlock after timeout
                    if node.target_id is not None:
                        if node.lock_lost_time is None:
                            node.lock_lost_time = time.time()
                        elif (time.time() - node.lock_lost_time) >= node.lock_lost_timeout_sec:
                            node.target_id = None
                            node.lock_lost_time = None

            if not detected:
                now = time.time()
                since_seen = (now - node.last_seen_time) if node.last_seen_time > 0 else 1e9

                do_search = (
                    node.search_enabled
                    and node.last_seen_time > 0.0
                    and since_seen >= node.search_start_delay_sec
                )

                if do_search:
                    if node.search_full_turn:
                        if node.search_angle_rad < (2.0 * math.pi):
                            node.search_for_target()
                        else:
                            node.stop_robot()
                            node.last_status = "NO TARGET (360° scan done)"
                    else:
                        if since_seen <= node.search_timeout_sec:
                            node.search_for_target()
                        else:
                            node.stop_robot()
                            node.last_status = "NO TARGET (search timeout)"
                else:
                    node.stop_robot()
                    if node.last_seen_time > 0.0 and since_seen < node.search_start_delay_sec:
                        node.last_status = "NO TARGET (grace - stopped)"
                    else:
                        node.last_status = "NO TARGET (stopped)"

                cv2.putText(frame_bgr, "No TARGET PERSON", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # FPS estimate
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0

            # Overlays
            cv2.putText(frame_bgr, f"ZED 2i - ~{fps:.1f} FPS", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame_bgr, "Press Q or ESC to STOP", (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame_bgr, f"State: {node.last_status}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"v={node.last_lin:.2f} m/s, w={node.last_ang:.2f} rad/s",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Phase 7: drain ROS2 callback queue (enables subscribers without spin() refactor)
            rclpy.spin_once(node, timeout_sec=0)

            cv2.imshow("ZED Person Follow (Color)", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        else:
            # Grab failed — publish stop immediately so the robot does not coast
            node.stop_robot()
            time.sleep(0.02)   # prevent CPU spin on sustained failure
            if last_frame_bgr is not None:
                frame_bgr = last_frame_bgr.copy()
                cv2.putText(frame_bgr, "ZED grab failed (skipping frame)", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("ZED Person Follow (Color)", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    node.stop_robot()
    zed.close()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
