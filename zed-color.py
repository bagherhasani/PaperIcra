#!/usr/bin/env python3
"""
ZED person follower — Roborregos-style register / gallery / track / reacquire.

Demo flow:
  1. Stand in front of the camera (alone if possible).
  2. Press SPACE — robot enrolls a multi-view OSNet gallery (~2 s).
  3. Robot follows using ZED body 3D + short-term ZED track ID.
  4. If the ZED ID is lost, gallery Re-ID reacquires the same person.
  5. If ambiguous (two similar people), STOP — never guess.
  6. R = clear registration.  M = mark eval event.  Q / ESC = quit + summary.
  Logs: eval_logs/follow_eval_*.csv (+ *_summary.txt). Analyze with analyze_eval.py.

No shirt-color identity. Shirt logic removed.
"""

from __future__ import annotations

import csv
import math
import os
import time
from datetime import datetime
from enum import Enum, auto

import cv2
import numpy as np
import pyzed.sl as sl
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

try:
    import tensorrt as trt
    import pycuda.driver as cuda

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENGINE_PATH = os.path.join(_SCRIPT_DIR, "osnet_x1_reid.engine")
_GALLERY_PATH = os.path.join(_SCRIPT_DIR, "target_gallery.npz")
_EVAL_DIR = os.path.join(_SCRIPT_DIR, "eval_logs")


class EvalLogger:
    """Per-frame CSV logger for solo testing / offline analysis."""

    HEADERS = [
        "t",
        "frame",
        "state",
        "status",
        "n_bodies",
        "locked_id",
        "gallery_n",
        "reid_score",
        "reid_best",
        "reid_second",
        "margin",
        "detected",
        "dist_m",
        "angle_deg",
        "x",
        "y",
        "z",
        "lin",
        "ang",
        "fps",
        "event",
    ]

    def __init__(self, out_dir: str = _EVAL_DIR):
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(out_dir, f"follow_eval_{stamp}.csv")
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.HEADERS)
        self._w.writeheader()
        self._f.flush()
        self.rows = 0
        self.t0 = time.time()
        self.events = []

    def log(self, row: dict, event: str = ""):
        row = {k: row.get(k, "") for k in self.HEADERS}
        row["t"] = f"{time.time() - self.t0:.3f}"
        row["event"] = event
        self._w.writerow(row)
        self.rows += 1
        if event:
            self.events.append((row["t"], event))
        if self.rows % 30 == 0:
            self._f.flush()

    def mark(self, event: str):
        # Marker row; log() already records into self.events
        self.log({}, event=event)

    def close_and_summarize(self):
        self._f.flush()
        self._f.close()
        summary_path = self.path.replace(".csv", "_summary.txt")
        # Lightweight read-back analysis
        states = {}
        reid_follow = []
        reid_reacq = []
        detected_follow = 0
        follow_frames = 0
        try:
            with open(self.path, newline="") as f:
                for r in csv.DictReader(f):
                    st = r.get("state", "")
                    states[st] = states.get(st, 0) + 1
                    if st == "FOLLOWING":
                        follow_frames += 1
                        if r.get("detected") in ("1", "True", "true"):
                            detected_follow += 1
                        try:
                            reid_follow.append(float(r["reid_score"]))
                        except Exception:
                            pass
                    if st in ("REACQUIRING", "LOST"):
                        try:
                            v = float(r["reid_best"])
                            if v > 0:
                                reid_reacq.append(v)
                        except Exception:
                            pass
        except Exception:
            pass

        lines = [
            f"log: {self.path}",
            f"rows: {self.rows}",
            f"events: {self.events}",
            f"state_counts: {states}",
        ]
        if follow_frames > 0:
            lines.append(
                f"follow_detect_rate: {detected_follow / follow_frames:.3f} "
                f"({detected_follow}/{follow_frames})"
            )
        if reid_follow:
            lines.append(
                f"reid_while_following: mean={np.mean(reid_follow):.3f} "
                f"min={np.min(reid_follow):.3f} p10={np.percentile(reid_follow, 10):.3f}"
            )
        if reid_reacq:
            lines.append(
                f"reid_best_while_lost: mean={np.mean(reid_reacq):.3f} "
                f"max={np.max(reid_reacq):.3f}"
            )
        lines.append("")
        lines.append("Solo pass criteria (rough):")
        lines.append("  - REGISTER ends with gallery_n >= 12")
        lines.append("  - follow_detect_rate > 0.90 while walking in view")
        lines.append("  - after leave/return: event REACQUIRED and FOLLOWING resumes")
        lines.append("  - reid_while_following mean usually > 0.65 for same clothes/view")
        text = "\n".join(lines) + "\n"
        with open(summary_path, "w") as f:
            f.write(text)
        print("\n===== EVAL SUMMARY =====")
        print(text)
        print(f"CSV: {self.path}")
        print(f"Summary: {summary_path}")
        return self.path, summary_path

_REID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_REID_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  ReID (TensorRT OSNet)
# ═══════════════════════════════════════════════════════════════════════════


class OsNetReID:
    """TensorRT OSNet-x1.0 → 512-d L2-normalized embedding."""

    def __init__(self, engine_path: str):
        if not _TRT_AVAILABLE:
            raise RuntimeError("tensorrt / pycuda not installed")
        self._closed = False
        self._failed = False
        cuda.init()
        self._cuda_ctx = cuda.Device(0).make_context()
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            self._context = self._engine.create_execution_context()
            self._input_shape = (1, 3, 256, 128)
            self._output_shape = (1, 512)
            nbytes_in = int(np.prod(self._input_shape)) * 4
            nbytes_out = int(np.prod(self._output_shape)) * 4
            self._h_in = cuda.pagelocked_empty(self._input_shape, dtype=np.float32)
            self._h_out = cuda.pagelocked_empty(self._output_shape, dtype=np.float32)
            self._d_in = cuda.mem_alloc(nbytes_in)
            self._d_out = cuda.mem_alloc(nbytes_out)
            self._stream = cuda.Stream()
            self._cuda_ctx.pop()
        except Exception:
            try:
                self._cuda_ctx.pop()
                self._cuda_ctx.detach()
            except Exception:
                pass
            raise

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._cuda_ctx.push()
            try:
                if getattr(self, "_stream", None) is not None:
                    self._stream.synchronize()
            except Exception:
                pass
            for name in ("_context", "_engine", "_stream", "_d_in", "_d_out", "_h_in", "_h_out"):
                try:
                    setattr(self, name, None)
                except Exception:
                    pass
            try:
                self._cuda_ctx.pop()
            except Exception:
                pass
            try:
                self._cuda_ctx.detach()
            except Exception:
                pass
        except Exception:
            pass

    def __del__(self):
        self.close()

    def _preprocess(self, frame_bgr: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(w, x2))
        y2 = int(min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - _REID_MEAN) / _REID_STD
        return crop.transpose(2, 0, 1)[np.newaxis, ...]

    def extract(self, frame_bgr: np.ndarray, bbox) -> np.ndarray | None:
        if self._closed or self._failed:
            return None
        inp = self._preprocess(frame_bgr, bbox)
        if inp is None:
            return None
        pushed = False
        try:
            self._cuda_ctx.push()
            pushed = True
            np.copyto(self._h_in, inp)
            cuda.memcpy_htod_async(self._d_in, self._h_in, self._stream)
            self._context.execute_async_v2(
                bindings=[int(self._d_in), int(self._d_out)],
                stream_handle=self._stream.handle,
            )
            cuda.memcpy_dtoh_async(self._h_out, self._d_out, self._stream)
            self._stream.synchronize()
            emb = self._h_out[0].copy()
        except Exception:
            self._failed = True
            return None
        finally:
            if pushed:
                try:
                    self._cuda_ctx.pop()
                except Exception:
                    pass
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb /= norm
        return emb

    @staticmethod
    def similarity(e1: np.ndarray | None, e2: np.ndarray | None) -> float:
        if e1 is None or e2 is None:
            return 0.0
        return float(np.dot(e1, e2))


# ═══════════════════════════════════════════════════════════════════════════
#  Persistent multi-view gallery (Roborregos-style)
# ═══════════════════════════════════════════════════════════════════════════


class TargetGallery:
    """Multi-embedding identity memory. Survives node restart via .npz."""

    def __init__(self, max_size: int = 48, path: str = _GALLERY_PATH):
        self.max_size = max_size
        self.path = path
        self.embeddings: list[np.ndarray] = []
        self.person_id = "target_001"
        self.height_m: float | None = None

    def clear(self):
        self.embeddings = []
        self.height_m = None

    def __len__(self):
        return len(self.embeddings)

    def add(self, emb: np.ndarray, min_novelty: float = 0.02):
        """Append if not a near-duplicate of an existing sample."""
        if emb is None:
            return False
        e = emb.astype(np.float32).reshape(-1)
        n = np.linalg.norm(e)
        if n < 1e-6:
            return False
        e = e / n
        if self.embeddings:
            sims = [float(np.dot(e, g)) for g in self.embeddings]
            if max(sims) > (1.0 - min_novelty):
                return False  # near-duplicate
        self.embeddings.append(e)
        if len(self.embeddings) > self.max_size:
            self.embeddings.pop(0)
        return True

    def score(self, emb: np.ndarray | None, top_k: int = 3) -> float:
        """Mean of top-k cosine similarities (Roborregos / Antobot pattern)."""
        if emb is None or not self.embeddings:
            return 0.0
        e = emb.astype(np.float32).reshape(-1)
        n = np.linalg.norm(e)
        if n < 1e-6:
            return 0.0
        e = e / n
        sims = np.array([float(np.dot(e, g)) for g in self.embeddings], dtype=np.float32)
        k = min(top_k, len(sims))
        return float(np.mean(np.sort(sims)[-k:]))

    def save(self):
        if not self.embeddings:
            return
        np.savez_compressed(
            self.path,
            embeddings=np.stack(self.embeddings, axis=0),
            person_id=np.array(self.person_id),
            height_m=np.array(-1.0 if self.height_m is None else self.height_m),
        )

    def load(self) -> bool:
        if not os.path.isfile(self.path):
            return False
        try:
            data = np.load(self.path, allow_pickle=True)
            embs = data["embeddings"]
            self.embeddings = [embs[i].astype(np.float32) for i in range(len(embs))]
            self.person_id = str(data["person_id"])
            h = float(data["height_m"])
            self.height_m = None if h < 0 else h
            return len(self.embeddings) > 0
        except Exception:
            return False


class FollowState(Enum):
    IDLE = auto()
    REGISTERING = auto()
    FOLLOWING = auto()
    LOST = auto()
    REACQUIRING = auto()


# ═══════════════════════════════════════════════════════════════════════════
#  ROS node
# ═══════════════════════════════════════════════════════════════════════════


class ZedFollower(Node):
    def __init__(self):
        super().__init__("zed_person_follower")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.actuation_enabled = True

        # Follow control
        self.target_distance = 1.2
        self.min_distance = 0.5
        self.max_lin_speed = 0.8
        self.max_ang_speed = 0.9
        self.k_lin = 0.7
        self.k_ang = 1.0

        self.last_lin = 0.0
        self.last_ang = 0.0
        self.last_status = "PRESS SPACE TO REGISTER"
        self.last_angle_error = 0.3
        self.last_seen_time = 0.0

        # Search
        self.search_enabled = True
        self.search_start_delay_sec = 0.8
        self.search_turn_speed = 0.5
        self.search_full_turn = True
        self.search_angle_rad = 0.0
        self._search_prev_t = None

        # Identity / tracking (Roborregos-inspired)
        self.state = FollowState.IDLE
        self.locked_id: int | None = None
        self.logical_target_id = 1
        self.last_target_position = None  # (x,y,z)
        self.enrolled_height_m: float | None = None

        self.gallery = TargetGallery(max_size=48, path=_GALLERY_PATH)
        # ~5s gives time for a slow full turn (views > duration)
        self.register_duration_sec = 5.0
        self.register_start_t = 0.0
        self.register_min_embeds = 16

        # ReID gates
        self.reid_min_follow = 0.55       # same ZED id: soft check
        self.reid_min_reacquire = 0.68    # must clear to relock
        self.reid_strong = 0.78
        self.reid_margin = 0.10           # top1 - top2
        self.confirm_frames = 3
        self._confirm_id = None
        self._confirm_streak = 0
        self.height_tol_m = 0.14
        self.same_id_max_jump_m = 1.2
        self.lock_lost_timeout_sec = 45.0  # keep gallery; only unlock track after this
        self.lock_lost_time = None
        self.gallery_update_interval = 10
        self.gallery_update_min_sim = 0.80
        self._frame_i = 0

        # ReID engine
        try:
            self.reid = OsNetReID(_ENGINE_PATH)
            self.get_logger().info(f"OSNet ReID loaded: {_ENGINE_PATH}")
        except Exception as e:
            self.reid = None
            self.get_logger().error(f"OSNet ReID unavailable — register/reacquire disabled: {e}")

        if self.gallery.load():
            self.get_logger().info(
                f"Loaded saved gallery ({len(self.gallery)} embeddings). "
                "Press SPACE to re-register, or wait for auto-reacquire."
            )
            self.state = FollowState.LOST
            self.last_status = f"GALLERY LOADED ({len(self.gallery)}) — searching"
            self.enrolled_height_m = self.gallery.height_m

        # ZED
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Cannot open ZED: {status}")

        try:
            self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        except Exception:
            pass

        self.body_enabled = False
        self.bodies = sl.Bodies()
        self.body_runtime = sl.BodyTrackingRuntimeParameters()
        self.body_runtime.detection_confidence_threshold = 40
        try:
            body_param = sl.BodyTrackingParameters()
            body_param.enable_tracking = True
            body_param.enable_body_fitting = True
            body_param.enable_segmentation = False
            body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
            body_param.body_format = sl.BODY_FORMAT.BODY_18
            res_bt = self.zed.enable_body_tracking(body_param)
            self.body_enabled = res_bt == sl.ERROR_CODE.SUCCESS
        except Exception:
            self.body_enabled = False

        if not self.body_enabled:
            raise RuntimeError("ZED Body Tracking failed to enable")

        self.runtime_params = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.get_logger().info(
            "Person follower ready. SPACE=register  R=reset  Q/ESC=quit"
        )

    # ── robot control ────────────────────────────────────────────────────

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)
        self.last_lin = 0.0
        self.last_ang = 0.0

    def search_for_target(self):
        twist = Twist()
        if not self.actuation_enabled:
            self.last_status = "SEARCHING (actuation off)"
            return
        direction = 1.0 if self.last_angle_error >= 0.0 else -1.0
        w = direction * min(self.search_turn_speed, self.max_ang_speed)
        now = time.time()
        if self._search_prev_t is None:
            self._search_prev_t = now
        dt = max(0.0, now - self._search_prev_t)
        self._search_prev_t = now
        self.search_angle_rad += abs(w) * dt
        if self.search_full_turn and self.search_angle_rad >= 2.0 * math.pi:
            self.stop_robot()
            self.last_status = "SEARCH DONE (360°) — STOPPED"
            return
        twist.angular.z = w
        self.last_lin = 0.0
        self.last_ang = float(w)
        self.last_status = "SEARCHING"
        self.cmd_pub.publish(twist)

    def control_robot(self, distance, angle_rad):
        if not self.actuation_enabled:
            self.last_lin = 0.0
            self.last_ang = 0.0
            self.last_status = "LOCKED (no actuation)"
            return
        twist = Twist()
        twist.angular.z = max(min(self.k_ang * angle_rad, self.max_ang_speed), -self.max_ang_speed)
        if distance is None or distance < self.min_distance:
            twist.linear.x = 0.0
            status = "TOO CLOSE — HOLDING"
        else:
            err = distance - self.target_distance
            twist.linear.x = max(min(self.k_lin * err, self.max_lin_speed), -self.max_lin_speed)
            status = "FOLLOWING" if abs(twist.linear.x) > 1e-3 or abs(twist.angular.z) > 1e-3 else "ALIGNED"
        self.last_lin = float(twist.linear.x)
        self.last_ang = float(twist.angular.z)
        self.last_status = status
        self.cmd_pub.publish(twist)

    # ── helpers ──────────────────────────────────────────────────────────

    def _bbox_from_body(self, body, img_w, img_h):
        try:
            pts = np.array([[p[0], p[1]] for p in body.bounding_box_2d], dtype=np.float32)
            x1 = int(np.clip(np.min(pts[:, 0]), 0, img_w - 1))
            y1 = int(np.clip(np.min(pts[:, 1]), 0, img_h - 1))
            x2 = int(np.clip(np.max(pts[:, 0]), 0, img_w - 1))
            y2 = int(np.clip(np.max(pts[:, 1]), 0, img_h - 1))
            if x2 <= x1 or y2 <= y1:
                return None
            return x1, y1, x2, y2
        except Exception:
            return None

    def _body_xyz(self, body):
        try:
            x = float(body.position[0])
            y = float(body.position[1])
            z = float(body.position[2])
            if not all(math.isfinite(v) for v in (x, y, z)) or z <= 0.05:
                return None
            return x, y, z
        except Exception:
            return None

    def _estimate_height(self, body) -> float | None:
        """Rough standing height from 3D keypoints if available."""
        try:
            kps = body.keypoint
            if kps is None or len(kps) == 0:
                return None
            # BODY_18: nose=0, mid-hip approx from L/R hip
            ys = []
            for idx in range(min(len(kps), 18)):
                p = kps[idx]
                if p[2] > 0.05 and math.isfinite(p[1]):
                    ys.append(float(p[1]))
            if len(ys) < 4:
                return None
            # Camera Y is up/down; height ≈ max_y - min_y in camera frame is unreliable.
            # Use head-to-ankle span along Y if ankles exist (BODY_18 ankles 10,13? COCO18:
            # 0 nose, 11/14 ankles for BODY_18 in ZED — use extreme keypoints).
            return float(max(ys) - min(ys))
        except Exception:
            return None

    def _valid_bodies(self):
        out = []
        for b in self.bodies.body_list:
            if b.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                continue
            if len(b.keypoint_2d) == 0:
                continue
            out.append(b)
        return out

    def begin_register(self):
        self.gallery.clear()
        self.locked_id = None
        self.lock_lost_time = None
        self.last_target_position = None
        self.enrolled_height_m = None
        self._confirm_id = None
        self._confirm_streak = 0
        self.search_angle_rad = 0.0
        self._search_prev_t = None
        self.register_start_t = time.time()
        self.state = FollowState.REGISTERING
        self.last_status = "REGISTERING — TURN SLOWLY (full 360°)"
        self.stop_robot()
        self.get_logger().info(
            "Registration started — stand ~1.5–2m centered, alone, turn slowly"
        )

    def reset_identity(self):
        self.gallery.clear()
        if os.path.isfile(_GALLERY_PATH):
            try:
                os.remove(_GALLERY_PATH)
            except Exception:
                pass
        self.locked_id = None
        self.lock_lost_time = None
        self.last_target_position = None
        self.enrolled_height_m = None
        self.state = FollowState.IDLE
        self.last_status = "PRESS SPACE TO REGISTER"
        self.stop_robot()
        self.get_logger().info("Identity cleared")


# ═══════════════════════════════════════════════════════════════════════════
#  Drawing
# ═══════════════════════════════════════════════════════════════════════════


def _draw_hud(img, fps, state, status, lin, ang, n_gallery, reid_score, dist_text):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, f"FPS {fps:.1f}  |  {state.name}  |  gallery={n_gallery}",
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, status, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 2)
    cv2.putText(
        img,
        f"v={lin:.2f}  w={ang:.2f}  reid={reid_score:.2f}  {dist_text}",
        (16, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        img,
        "SPACE=register   R=reset   M=mark   Q=quit+summary",
        (16, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
    )


def _draw_body(img, body, bbox, color, label):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ═══════════════════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════════════════


def main():
    rclpy.init()
    try:
        node = ZedFollower()
    except RuntimeError as e:
        print(f"FATAL: {e}")
        rclpy.shutdown()
        return

    zed = node.zed
    window = "ZED Person Follow (Gallery)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_t = time.time()
    frame_count = 0
    fps = 0.0
    hud_reid = 0.0
    hud_dist = ""
    last_frame = None
    eval_log = EvalLogger()
    node.get_logger().info(
        f"Eval logging → {eval_log.path}  |  M=mark event  Q=quit+summary"
    )
    prev_state = node.state

    while True:
        if zed.grab(node.runtime_params) != sl.ERROR_CODE.SUCCESS:
            if last_frame is not None:
                cv2.imshow(window, last_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            continue

        zed.retrieve_image(node.image, sl.VIEW.LEFT)
        frame = node.image.get_data()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        last_frame = frame_bgr
        img_h, img_w = frame_bgr.shape[:2]
        node._frame_i += 1

        if node.reid is not None and (node.reid._closed or node.reid._failed):
            node.get_logger().warn("ReID disabled after CUDA/TRT failure")
            node.reid.close()
            node.reid = None

        zed.retrieve_bodies(node.bodies, node.body_runtime)
        bodies = node._valid_bodies()

        # Score all bodies for UI / selection
        scored = []  # (gallery_score, body, bbox, xyz, height)
        for b in bodies:
            bbox = node._bbox_from_body(b, img_w, img_h)
            xyz = node._body_xyz(b)
            if bbox is None or xyz is None:
                continue
            emb = None
            gscore = 0.0
            if node.reid is not None and len(node.gallery) > 0:
                emb = node.reid.extract(frame_bgr, bbox)
                gscore = node.gallery.score(emb)
            h_est = node._estimate_height(b)
            scored.append((gscore, b, bbox, xyz, h_est, emb))

        target_body = None
        target_bbox = None
        target_xyz = None
        target_score = 0.0
        detected = False

        # ── IDLE ─────────────────────────────────────────────────────────
        if node.state == FollowState.IDLE:
            node.stop_robot()
            node.last_status = "PRESS SPACE TO REGISTER"
            for _, b, bbox, _, _, _ in scored:
                _draw_body(frame_bgr, b, bbox, (180, 180, 180), f"id={b.id}")

        # ── REGISTERING ──────────────────────────────────────────────────
        elif node.state == FollowState.REGISTERING:
            node.stop_robot()
            # Prefer centered + closest person
            best = None
            best_key = -1e9
            for gscore, b, bbox, xyz, h_est, emb in scored:
                x, y, z = xyz
                cx = 0.5 * (bbox[0] + bbox[2])
                center = 1.0 - abs(cx - img_w * 0.5) / (img_w * 0.5)
                key = 2.0 * center - 0.15 * z
                if key > best_key:
                    best_key = key
                    best = (b, bbox, xyz, h_est)

            if best is not None and node.reid is not None:
                b, bbox, xyz, h_est = best
                emb = node.reid.extract(frame_bgr, bbox)
                if emb is not None:
                    node.gallery.add(emb, min_novelty=0.015)
                if h_est is not None:
                    node.enrolled_height_m = h_est
                    node.gallery.height_m = h_est
                node.locked_id = b.id
                node.last_target_position = xyz
                node.last_seen_time = time.time()
                target_body, target_bbox, target_xyz = b, bbox, xyz
                detected = True
                _draw_body(frame_bgr, b, bbox, (0, 200, 255), f"ENROLL id={b.id}")

            elapsed = time.time() - node.register_start_t
            # Coach by phase so you know to keep turning
            frac = elapsed / max(node.register_duration_sec, 1e-3)
            if frac < 0.25:
                pose_hint = "face camera"
            elif frac < 0.5:
                pose_hint = "turn LEFT"
            elif frac < 0.75:
                pose_hint = "show BACK"
            else:
                pose_hint = "turn RIGHT → face camera"
            n_emb = len(node.gallery)
            node.last_status = (
                f"REGISTERING {elapsed:.1f}/{node.register_duration_sec:.0f}s  "
                f"embeds={n_emb}  → {pose_hint}"
            )
            done = (
                elapsed >= node.register_duration_sec
                and n_emb >= node.register_min_embeds
            ) or (
                elapsed >= node.register_duration_sec * 1.4 and n_emb >= 12
            )
            if done and n_emb >= 8:
                node.gallery.save()
                node.state = FollowState.FOLLOWING
                node.lock_lost_time = None
                node.search_angle_rad = 0.0
                node._search_prev_t = None
                # Quality band for the operator (views matter more than raw count)
                if n_emb >= 24:
                    quality = "EXCELLENT"
                elif n_emb >= 16:
                    quality = "GOOD"
                else:
                    quality = "WEAK — press R and re-register while turning"
                node.last_status = (
                    f"REGISTERED {quality} ({n_emb} embeds) — FOLLOW"
                )
                node.get_logger().info(
                    f"Gallery saved ({n_emb} embeddings, {quality}) → {_GALLERY_PATH}"
                )
            elif elapsed >= node.register_duration_sec * 1.8:
                node.last_status = (
                    f"REGISTER FAILED (embeds={n_emb}) — alone? turn? press SPACE"
                )
                node.state = FollowState.IDLE
                node.gallery.clear()

        # ── FOLLOWING ────────────────────────────────────────────────────
        elif node.state == FollowState.FOLLOWING:
            locked = None
            for gscore, b, bbox, xyz, h_est, emb in scored:
                if b.id == node.locked_id:
                    # Reject impossible jumps (ID transfer)
                    if node.last_target_position is not None:
                        tx, ty, tz = node.last_target_position
                        jump = math.sqrt(
                            (xyz[0] - tx) ** 2 + (xyz[1] - ty) ** 2 + (xyz[2] - tz) ** 2
                        )
                        if jump > node.same_id_max_jump_m:
                            node.get_logger().warn(
                                f"ZED id jump {jump:.2f}m — treating as lost"
                            )
                            break
                    # Periodic appearance check
                    if (
                        node.reid is not None
                        and len(node.gallery) > 0
                        and emb is not None
                        and gscore < node.reid_min_follow
                        and node._frame_i % 5 == 0
                    ):
                        node.get_logger().warn(
                            f"Appearance mismatch on locked id (reid={gscore:.2f})"
                        )
                        break
                    locked = (gscore, b, bbox, xyz, h_est, emb)
                    break

            if locked is not None:
                gscore, b, bbox, xyz, h_est, emb = locked
                target_body, target_bbox, target_xyz = b, bbox, xyz
                target_score = gscore
                detected = True
                node.lock_lost_time = None
                node.last_target_position = xyz
                node.last_seen_time = time.time()
                node.search_angle_rad = 0.0
                node._search_prev_t = None

                # Slow trusted gallery growth (high confidence only)
                if (
                    node.reid is not None
                    and emb is not None
                    and gscore >= node.gallery_update_min_sim
                    and node._frame_i % node.gallery_update_interval == 0
                ):
                    if node.gallery.add(emb, min_novelty=0.03):
                        node.gallery.save()

                _draw_body(frame_bgr, b, bbox, (0, 255, 0), f"TARGET id={b.id} {gscore:.2f}")
            else:
                # Lost short-term track → reacquire
                node.state = FollowState.REACQUIRING
                if node.lock_lost_time is None:
                    node.lock_lost_time = time.time()
                node.locked_id = None
                node._confirm_id = None
                node._confirm_streak = 0
                node.stop_robot()
                node.last_status = "LOST TRACK — REACQUIRING"

        # ── REACQUIRING / LOST ────────────────────────────────────────────
        if node.state in (FollowState.REACQUIRING, FollowState.LOST):
            # Rank by gallery score, then height consistency, then position
            candidates = []
            for gscore, b, bbox, xyz, h_est, emb in scored:
                if len(node.gallery) == 0:
                    continue
                if gscore < node.reid_min_reacquire:
                    continue
                if (
                    node.enrolled_height_m is not None
                    and h_est is not None
                    and abs(h_est - node.enrolled_height_m) > node.height_tol_m
                ):
                    continue
                pos_bonus = 0.0
                if node.last_target_position is not None:
                    tx, ty, tz = node.last_target_position
                    d = math.sqrt(
                        (xyz[0] - tx) ** 2 + (xyz[1] - ty) ** 2 + (xyz[2] - tz) ** 2
                    )
                    pos_bonus = max(0.0, 1.0 - d / 3.0) * 0.05
                candidates.append((gscore + pos_bonus, gscore, b, bbox, xyz))

            candidates.sort(key=lambda t: t[0], reverse=True)

            for gscore, b, bbox, xyz, _, _ in scored:
                col = (80, 80, 255) if gscore < node.reid_min_reacquire else (0, 165, 255)
                _draw_body(frame_bgr, b, bbox, col, f"id={b.id} {gscore:.2f}")

            chosen = None
            if len(candidates) >= 1:
                if len(candidates) >= 2:
                    margin = candidates[0][1] - candidates[1][1]
                    if margin < node.reid_margin:
                        node.last_status = f"AMBIGUOUS (Δ={margin:.2f}) — STOPPED"
                        node.stop_robot()
                        chosen = None
                    else:
                        chosen = candidates[0]
                else:
                    chosen = candidates[0]

            if chosen is not None:
                _, gscore, b, bbox, xyz = chosen
                if node._confirm_id == b.id:
                    node._confirm_streak += 1
                else:
                    node._confirm_id = b.id
                    node._confirm_streak = 1
                node.last_status = (
                    f"REACQUIRING {node._confirm_streak}/{node.confirm_frames} "
                    f"reid={gscore:.2f}"
                )
                _draw_body(frame_bgr, b, bbox, (0, 255, 255), f"CAND id={b.id}")
                if node._confirm_streak >= node.confirm_frames:
                    node.locked_id = b.id
                    node.last_target_position = xyz
                    node.lock_lost_time = None
                    node.state = FollowState.FOLLOWING
                    node._confirm_id = None
                    node._confirm_streak = 0
                    node.search_angle_rad = 0.0
                    node._search_prev_t = None
                    target_body, target_bbox, target_xyz = b, bbox, xyz
                    target_score = gscore
                    detected = True
                    node.last_seen_time = time.time()
                    node.last_status = f"REACQUIRED id={b.id}"
                    node.get_logger().info(f"Reacquired person as ZED id={b.id}")
            else:
                node._confirm_id = None
                node._confirm_streak = 0
                if len(node.gallery) == 0:
                    node.state = FollowState.IDLE
                    node.last_status = "PRESS SPACE TO REGISTER"
                    node.stop_robot()
                else:
                    node.state = FollowState.LOST
                    # search behavior below

            if (
                node.lock_lost_time is not None
                and (time.time() - node.lock_lost_time) > node.lock_lost_timeout_sec
            ):
                # Keep gallery; just stop active chase until SPACE or reacquire
                node.last_status = "LOST TIMEOUT — gallery kept, press SPACE to re-enroll"

        # ── Drive / search ───────────────────────────────────────────────
        if detected and target_xyz is not None and node.state == FollowState.FOLLOWING:
            x, y, z = target_xyz
            distance = math.sqrt(x * x + y * y + z * z)
            angle_error = -math.atan2(x, z)
            node.last_angle_error = float(angle_error)
            deg = math.degrees(angle_error)
            side = "center"
            if deg > 3:
                side = "left"
            elif deg < -3:
                side = "right"
            hud_dist = f"{distance:.2f}m  {abs(deg):.0f}° {side}"
            hud_reid = target_score
            node.control_robot(distance, angle_error)
            if target_bbox is not None and target_body is not None:
                _draw_body(
                    frame_bgr,
                    target_body,
                    target_bbox,
                    (0, 255, 0),
                    f"TARGET {target_score:.2f}",
                )
        elif node.state in (FollowState.LOST, FollowState.REACQUIRING):
            now = time.time()
            since = (now - node.last_seen_time) if node.last_seen_time > 0 else 1e9
            if (
                node.search_enabled
                and node.last_seen_time > 0
                and since >= node.search_start_delay_sec
                and len(node.gallery) > 0
            ):
                if node.search_full_turn and node.search_angle_rad < 2.0 * math.pi:
                    node.search_for_target()
                elif node.search_angle_rad >= 2.0 * math.pi:
                    node.stop_robot()
                    if "AMBIGUOUS" not in node.last_status:
                        node.last_status = "LOST — scan done, waiting"
                else:
                    node.stop_robot()
            else:
                node.stop_robot()
        elif node.state != FollowState.REGISTERING:
            # idle etc.
            pass

        # FPS
        frame_count += 1
        now = time.time()
        if now - prev_t >= 1.0:
            fps = frame_count / (now - prev_t)
            prev_t = now
            frame_count = 0

        # Eval metrics this frame
        reid_scores = sorted(
            [float(s[0]) for s in scored if s[0] > 0], reverse=True
        )
        reid_best = reid_scores[0] if reid_scores else 0.0
        reid_second = reid_scores[1] if len(reid_scores) > 1 else 0.0
        margin = reid_best - reid_second if len(reid_scores) > 1 else reid_best
        dist_m = ""
        angle_deg = ""
        xyz_x = xyz_y = xyz_z = ""
        if detected and target_xyz is not None:
            x, y, z = target_xyz
            dist_m = f"{math.sqrt(x * x + y * y + z * z):.3f}"
            angle_deg = f"{math.degrees(-math.atan2(x, z)):.2f}"
            xyz_x, xyz_y, xyz_z = f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"

        event = ""
        if prev_state != node.state:
            event = f"STATE:{prev_state.name}->{node.state.name}"
            prev_state = node.state
            if node.state == FollowState.FOLLOWING and "REACQUIRED" in node.last_status:
                event = "REACQUIRED"
            elif node.state == FollowState.FOLLOWING and "REGISTERED" in node.last_status:
                event = "REGISTERED"

        eval_log.log(
            {
                "frame": node._frame_i,
                "state": node.state.name,
                "status": node.last_status,
                "n_bodies": len(scored),
                "locked_id": "" if node.locked_id is None else node.locked_id,
                "gallery_n": len(node.gallery),
                "reid_score": f"{(hud_reid if detected else 0.0):.4f}",
                "reid_best": f"{reid_best:.4f}",
                "reid_second": f"{reid_second:.4f}",
                "margin": f"{margin:.4f}",
                "detected": int(detected and node.state == FollowState.FOLLOWING),
                "dist_m": dist_m,
                "angle_deg": angle_deg,
                "x": xyz_x,
                "y": xyz_y,
                "z": xyz_z,
                "lin": f"{node.last_lin:.3f}",
                "ang": f"{node.last_ang:.3f}",
                "fps": f"{fps:.2f}",
            },
            event=event,
        )

        _draw_hud(
            frame_bgr,
            fps,
            node.state,
            node.last_status,
            node.last_lin,
            node.last_ang,
            len(node.gallery),
            hud_reid if detected else 0.0,
            hud_dist if detected else "",
        )
        cv2.imshow(window, frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            if node.reid is None:
                node.get_logger().error("Cannot register — ReID engine missing")
                node.last_status = "NO REID ENGINE — cannot register"
            else:
                eval_log.mark("SPACE_REGISTER")
                node.begin_register()
        elif key in (ord("r"), ord("R")):
            eval_log.mark("RESET")
            node.reset_identity()
        elif key in (ord("m"), ord("M")):
            # Manual marker for your solo protocol (e.g. "I left frame now")
            eval_log.mark("MARK")
            node.last_status = f"{node.last_status} | MARK"
            node.get_logger().info("Eval MARK recorded")

    node.stop_robot()
    eval_log.close_and_summarize()
    if node.reid is not None:
        node.reid.close()
    zed.close()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
