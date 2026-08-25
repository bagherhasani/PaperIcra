#!/usr/bin/env python3
"""
ZED person follower — register / gallery / track / reacquire.

Demo flow:
  1. Stand in front of the camera (alone if possible).
  2. Press SPACE — type the person-of-interest name, Enter to confirm — then
     strict InsightFace enroll (min 16 faces, FULL=32, det>=0.75) + body views.
  3. Robot follows using ZED body 3D + short-term ZED track ID.
  4. If ZED ID is lost: InsightFace face reacquire first (official 0.4 threshold);
     body OSNet only as alone/backup. If unsure → WAIT (never guess).
  5. R = clear registration.  M = mark eval event.  Q / ESC = quit + summary.
  6. V = start/stop demo MP4 (saved under eval_logs/) — use this for professor videos
     (windowed by default; F toggles fullscreen). Avoid OS screen recorders on Jetson.
  Logs: eval_logs/follow_eval_*.csv (+ *_summary.txt). Analyze with analyze_eval.py.

  Identity (InsightFace Evaluation Studio patterns):
  - FaceAnalysis.get → Face.normed_embedding
  - gui.core.recognition: normalize / cosine / compare / search_gallery
  - DEFAULT_THRESHOLD = 0.4 (Same Person / Uncertain / Different)

  Lock policy (from trackernodefiles/tracker_node.py):
  - WHILE FOLLOWING: trust ZED body.id — no per-frame body OSNet veto
  - Appearance (face first, body backup) only when the ZED id is lost
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
except Exception:
    # ImportError OR NumPy ABI mismatch (SystemError/AttributeError) must not kill startup
    trt = None  # type: ignore
    cuda = None  # type: ignore
    _TRT_AVAILABLE = False

# InsightFace official APIs (not custom similarity math)
try:
    from insightface.app import FaceAnalysis
    from insightface.gui.core.constants import DEFAULT_THRESHOLD as _IF_FACE_THRESHOLD
    from insightface.gui.core.recognition import (
        aggregate_person_embeddings,
        compare_embeddings,
        cosine_similarity,
        identify_face,
        normalize_embedding,
        search_gallery,
    )

    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None  # type: ignore
    _IF_FACE_THRESHOLD = 0.4

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENGINE_PATH = os.path.join(_SCRIPT_DIR, "osnet_x1_reid.engine")
_GALLERY_PATH = os.path.join(_SCRIPT_DIR, "target_gallery.npz")
_FACE_GALLERY_PATH = os.path.join(_SCRIPT_DIR, "target_face_gallery.npz")
_EVAL_DIR = os.path.join(_SCRIPT_DIR, "eval_logs")
_FACE_PERSON_ID = 1  # single enrolled target (InsightFace person_id)

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
        "face_n",
        "reid_score",
        "reid_best",
        "reid_second",
        "margin",
        "face_score",
        "face_best",
        "face_second",
        "face_margin",
        "lock_source",
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

    @staticmethod
    def _cell_float(row, key):
        """Parse a CSV cell as float. Named to avoid clashing with self._f file handle."""
        try:
            v = row.get(key, "")
            if v == "" or v is None:
                return None
            return float(v)
        except Exception:
            return None

    def close_and_summarize(self):
        self._f.flush()
        self._f.close()
        summary_path = self.path.replace(".csv", "_summary.txt")
        states = {}
        reid_follow = []
        reid_lost = []
        face_lost = []
        detected_follow = 0
        follow_frames = 0
        lost_frames = 0
        n_reacq_face = 0
        n_reacq_body = 0
        n_register = 0
        face_on_reacq = []
        body_on_reacq = []
        lost_enter_t = None
        reacq_latencies = []
        max_face_n = 0
        max_body_n = 0
        fps_vals = []
        try:
            with open(self.path, newline="") as f:
                for r in csv.DictReader(f):
                    st = r.get("state", "")
                    states[st] = states.get(st, 0) + 1
                    ev = r.get("event", "") or ""
                    src = (r.get("lock_source") or "").lower()
                    t = self._cell_float(r, "t")
                    fn = self._cell_float(r, "face_n")
                    bn = self._cell_float(r, "gallery_n")
                    if fn is not None:
                        max_face_n = max(max_face_n, int(fn))
                    if bn is not None:
                        max_body_n = max(max_body_n, int(bn))
                    fp = self._cell_float(r, "fps")
                    if fp is not None and fp > 0:
                        fps_vals.append(fp)

                    if st == "FOLLOWING":
                        follow_frames += 1
                        if r.get("detected") in ("1", "True", "true"):
                            detected_follow += 1
                        rs = self._cell_float(r, "reid_score")
                        if rs is not None:
                            reid_follow.append(rs)
                        if lost_enter_t is not None and t is not None:
                            reacq_latencies.append(t - lost_enter_t)
                            lost_enter_t = None
                    if st in ("REACQUIRING", "LOST"):
                        lost_frames += 1
                        rb = self._cell_float(r, "reid_best")
                        if rb is not None and rb > 0:
                            reid_lost.append(rb)
                        fb = self._cell_float(r, "face_best")
                        if fb is not None and fb > 0:
                            face_lost.append(fb)

                    if "FOLLOWING->LOST" in ev or "FOLLOWING->REACQUIRING" in ev:
                        lost_enter_t = t

                    if ev == "REGISTERED" or (
                        ev == "STATE:REGISTERING->FOLLOWING" and src == "register"
                    ):
                        n_register += 1

                    # Prefer explicit tags; also accept LOST->FOLLOWING + lock_source
                    is_face_reacq = ev == "REACQUIRED_FACE" or (
                        ev == "STATE:LOST->FOLLOWING" and src == "face"
                    ) or (ev == "REACQUIRED" and src == "face")
                    is_body_reacq = ev == "REACQUIRED_BODY" or (
                        ev == "STATE:LOST->FOLLOWING" and src == "body"
                    ) or (ev == "REACQUIRED" and src == "body")

                    if is_face_reacq:
                        n_reacq_face += 1
                        fs = self._cell_float(r, "face_score")
                        if fs is not None and fs > 0:
                            face_on_reacq.append(fs)
                    if is_body_reacq:
                        n_reacq_body += 1
                        bs = self._cell_float(r, "reid_score")
                        if bs is None or bs <= 0:
                            bs = self._cell_float(r, "reid_best")
                        if bs is not None and bs > 0:
                            body_on_reacq.append(bs)
        except Exception as e:
            print(f"[EvalLogger] summary parse error: {e}")

        n_reacq = n_reacq_face + n_reacq_body
        lines = [
            f"log: {self.path}",
            f"rows: {self.rows}",
            f"events: {self.events}",
            f"state_counts: {states}",
            "",
            "=== IDENTITY / REACQUIRE (solo one-target) ===",
            f"register_events: {n_register}",
            f"gallery_max: body={max_body_n} face={max_face_n}",
            f"reacquire_total: {n_reacq}",
            f"reacquire_via_face: {n_reacq_face}",
            f"reacquire_via_body: {n_reacq_body}",
            (
                f"reacquire_face_pct: {100.0 * n_reacq_face / n_reacq:.1f}%"
                if n_reacq
                else "reacquire_face_pct: n/a"
            ),
        ]
        if face_on_reacq:
            lines.append(
                f"face_score_at_reacq: mean={np.mean(face_on_reacq):.3f} "
                f"min={np.min(face_on_reacq):.3f} max={np.max(face_on_reacq):.3f} "
                f"n={len(face_on_reacq)}"
            )
        if body_on_reacq:
            lines.append(
                f"body_score_at_reacq: mean={np.mean(body_on_reacq):.3f} "
                f"min={np.min(body_on_reacq):.3f} max={np.max(body_on_reacq):.3f} "
                f"n={len(body_on_reacq)}"
            )
        if reacq_latencies:
            lines.append(
                f"lost_to_follow_sec: mean={np.mean(reacq_latencies):.2f} "
                f"min={np.min(reacq_latencies):.2f} max={np.max(reacq_latencies):.2f} "
                f"n={len(reacq_latencies)}"
            )
        if face_lost:
            lines.append(
                f"face_best_while_lost: mean={np.mean(face_lost):.3f} "
                f"max={np.max(face_lost):.3f}"
            )
        if reid_lost:
            lines.append(
                f"body_best_while_lost: mean={np.mean(reid_lost):.3f} "
                f"max={np.max(reid_lost):.3f}"
            )
        lines.append("")
        lines.append("=== FOLLOW / RUNTIME ===")
        if follow_frames > 0:
            lines.append(
                f"follow_detect_rate: {detected_follow / follow_frames:.3f} "
                f"({detected_follow}/{follow_frames})"
            )
        lines.append(f"follow_frames: {follow_frames}  lost_frames: {lost_frames}")
        if fps_vals:
            lines.append(
                f"fps: mean={np.mean(fps_vals):.2f} min={np.min(fps_vals):.2f}"
            )
        if reid_follow:
            nonzero = [v for v in reid_follow if v > 1e-6]
            lines.append(
                f"body_reid_while_following: mean={np.mean(reid_follow):.3f} "
                f"(nonzero_n={len(nonzero)}; 0 expected if body ReID off in FOLLOW)"
            )
        lines.append("")
        lines.append("Solo pass criteria (one person leave/return):")
        lines.append("  - REGISTERED once; face_n >= 16 (FULL=32) and gallery_n >= 12")
        lines.append("  - leave then return facing camera → REACQUIRED_FACE")
        lines.append("  - reacquire_via_face > 0; face_score_at_reacq mean >= 0.40")
        lines.append("  - lost_to_follow_sec usually < 8s (CPU face)")
        lines.append("  - follow_detect_rate > 0.90 while in view")
        lines.append("  - press M when you leave, M when you return (optional markers)")
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
        self.person_name = "target"
        self.height_m: float | None = None

    def clear(self):
        self.embeddings = []
        self.height_m = None
        self.person_name = "target"

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
            person_name=np.array(self.person_name),
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
            if "person_name" in data:
                self.person_name = str(data["person_name"])
            h = float(data["height_m"])
            self.height_m = None if h < 0 else h
            return len(self.embeddings) > 0
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════════
#  Face gallery — InsightFace Evaluation Studio enroll / search patterns
#  (insightface.gui.core.recognition + Face.normed_embedding)
# ═══════════════════════════════════════════════════════════════════════════


class FaceGallery:
    """Stores L2-normalized face embeddings the same way InsightFace's library does.

    Enroll: Face.normed_embedding (see app/common.py)
    Match:  search_gallery / identify_face / compare_embeddings
            with DEFAULT_THRESHOLD (0.4)
    Multi-sample: aggregate_person_embeddings(method='mean')
    """

    def __init__(self, max_size: int = 32, path: str = _FACE_GALLERY_PATH):
        self.max_size = max_size
        self.path = path
        self.embeddings: list[np.ndarray] = []
        self.person_id = _FACE_PERSON_ID
        self.person_name = "target"
        self.threshold = float(_IF_FACE_THRESHOLD)  # InsightFace default 0.4

    def clear(self):
        self.embeddings = []
        self.person_name = "target"

    def __len__(self):
        return len(self.embeddings)

    def _gallery_items(self) -> list[dict]:
        # Format expected by insightface.gui.core.recognition.search_gallery
        return [
            {
                "person_id": self.person_id,
                "person_name": self.person_name or "target",
                "sample_id": i,
                "embedding": emb,
            }
            for i, emb in enumerate(self.embeddings)
        ]

    def add(self, embedding, min_novelty: float = 0.02) -> bool:
        """Append a Face.normed_embedding (or raw embedding — normalized here)."""
        if not _INSIGHTFACE_AVAILABLE:
            return False
        e = normalize_embedding(embedding)
        if e is None:
            return False
        if self.embeddings:
            # Skip near-duplicates (same idea as body gallery)
            best = max(float(cosine_similarity(e, g)) for g in self.embeddings)
            if best > (1.0 - min_novelty):
                return False
        self.embeddings.append(e.astype(np.float32))
        if len(self.embeddings) > self.max_size:
            self.embeddings.pop(0)
        return True

    def prototype(self) -> np.ndarray | None:
        """Mean embedding per InsightFace aggregate_person_embeddings."""
        if not self.embeddings or not _INSIGHTFACE_AVAILABLE:
            return None
        samples = [
            {"person_id": self.person_id, "embedding": e} for e in self.embeddings
        ]
        agg = aggregate_person_embeddings(samples, method="mean")
        return agg.get(self.person_id)

    def score(self, embedding) -> float:
        """Best cosine vs enrolled samples (search_gallery best-of-person)."""
        if embedding is None or not self.embeddings or not _INSIGHTFACE_AVAILABLE:
            return 0.0
        results = search_gallery(
            embedding,
            self._gallery_items(),
            top_k=1,
            threshold=self.threshold,
        )
        return float(results[0].similarity) if results else 0.0

    def identify(self, embedding) -> dict:
        """Official identify_face + compare_embeddings decision.

        Returns dict: similarity, decision ('Same Person'|'Uncertain'|'Different Person'),
        status, matched (bool).
        """
        out = {
            "similarity": 0.0,
            "decision": "Different Person",
            "status": "unknown",
            "matched": False,
        }
        if embedding is None or not self.embeddings or not _INSIGHTFACE_AVAILABLE:
            return out
        results = identify_face(
            embedding,
            self._gallery_items(),
            threshold=self.threshold,
            top_k=1,
        )
        if not results:
            return out
        r0 = results[0]
        proto = self.prototype()
        if proto is not None:
            cmp_ = compare_embeddings(embedding, proto, threshold=self.threshold)
            decision = str(cmp_["decision"])
            sim = float(cmp_["similarity"])
        else:
            sim = float(r0.similarity)
            decision = (
                "Same Person"
                if sim >= self.threshold
                else (
                    "Uncertain"
                    if sim >= self.threshold - 0.05
                    else "Different Person"
                )
            )
        out["similarity"] = sim
        out["decision"] = decision
        out["status"] = r0.status
        out["matched"] = decision == "Same Person"
        return out

    def save(self):
        if not self.embeddings:
            return
        np.savez_compressed(
            self.path,
            embeddings=np.stack(self.embeddings, axis=0),
            person_id=np.array(self.person_id),
            person_name=np.array(self.person_name),
            threshold=np.array(self.threshold),
        )

    def load(self) -> bool:
        if not os.path.isfile(self.path):
            return False
        try:
            data = np.load(self.path, allow_pickle=True)
            embs = data["embeddings"]
            self.embeddings = [embs[i].astype(np.float32) for i in range(len(embs))]
            self.person_id = int(data["person_id"])
            if "person_name" in data:
                self.person_name = str(data["person_name"])
            if "threshold" in data:
                self.threshold = float(data["threshold"])
            return len(self.embeddings) > 0
        except Exception:
            return False


def _face_center_in_body(face_bbox, body_bbox) -> bool:
    """Associate InsightFace face bbox with a ZED body bbox (upper torso)."""
    fx1, fy1, fx2, fy2 = [float(v) for v in face_bbox]
    bx1, by1, bx2, by2 = [float(v) for v in body_bbox]
    fcx = 0.5 * (fx1 + fx2)
    fcy = 0.5 * (fy1 + fy2)
    # Face should sit in the upper 60% of the body box and within x-range
    if not (bx1 <= fcx <= bx2):
        return False
    upper = by1 + 0.60 * (by2 - by1)
    return by1 <= fcy <= upper


def _best_face_for_body(faces, body_bbox):
    """Pick largest face whose center lies on this body (InsightFace detect_best_face idea)."""
    best = None
    best_area = -1.0
    for face in faces:
        bbox = getattr(face, "bbox", None)
        if bbox is None:
            continue
        if not _face_center_in_body(bbox, body_bbox):
            continue
        area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        det = float(getattr(face, "det_score", 0.0) or 0.0)
        key = area * (0.5 + 0.5 * det)
        if key > best_area:
            best_area = key
            best = face
    return best


class FollowState(Enum):
    IDLE = auto()
    NAMING = auto()  # type person-of-interest name before enroll
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

        # Follow control — keep ~1.2 m: farther → forward, closer → reverse
        self.target_distance = 1.2
        self.distance_deadband_m = 0.08  # |d - 1.2| within this → stop (no chatter)
        self.min_distance = 0.35  # below this: force full reverse (safety)
        self.max_lin_speed = 0.5
        self.max_ang_speed = 0.5
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
        self.target_name = ""
        self._name_buffer = ""
        self.last_target_position = None  # (x,y,z)
        self.enrolled_height_m: float | None = None

        self.gallery = TargetGallery(max_size=48, path=_GALLERY_PATH)
        self.face_gallery = FaceGallery(max_size=32, path=_FACE_GALLERY_PATH)
        # Strict face-first enroll: fill InsightFace gallery before FOLLOW is allowed.
        # face_n max=32 = FULL LOAD. Cosine "score" is for reacquire, not enroll count.
        self.register_duration_sec = 8.0
        self.register_start_t = 0.0
        self.register_min_embeds = 16
        self.register_min_faces = 16  # hard gate — below this = register FAILED
        self.register_ideal_faces = 32  # FULL LOAD (FaceGallery.max_size)
        self.register_min_det_score = 0.75  # InsightFace Face.det_score; reject blurry/side
        self._register_det_scores: list[float] = []
        self._register_last_det = 0.0
        self._register_last_self_sim = 0.0

        # Body OSNet gates (backup when face unavailable)
        self.reid_min_follow = 0.62
        self.reid_min_reacquire = 0.78
        self.reid_min_alone = 0.80
        self.reid_strong = 0.85
        self.reid_margin = 0.10
        # Face gates from InsightFace DEFAULT_THRESHOLD (+ small margin for crowds)
        self.face_threshold = float(_IF_FACE_THRESHOLD)  # 0.4
        self.face_margin = 0.08
        self.confirm_frames = 5
        self._confirm_id = None
        self._confirm_streak = 0
        self.height_tol_m = 0.14
        self.same_id_max_jump_m = 1.2
        self.lock_lost_timeout_sec = 45.0
        self.lock_lost_time = None
        # NEVER grow gallery after enroll — pollution caused false locks
        self.gallery_updates_enabled = False
        self.gallery_update_interval = 10
        self.gallery_update_min_sim = 0.92
        self.search_max_bodies = 0  # 0 = never spin-search
        self._frame_i = 0
        self._follow_mismatch_streak = 0
        self._follow_mismatch_limit = 5
        self._last_faces = []
        # tracker_node policy: appearance is for reacquire only, not to veto a live lock
        self._lock_source = None  # "face" | "body" | "register"
        self.follow_use_body_veto = False  # True = old thrashy body-OSNet gate
        self._pending_eval_event = None  # set on register/reacquire; logged before control overwrites status

        # Body ReID engine (TensorRT OSNet)
        try:
            self.reid = OsNetReID(_ENGINE_PATH)
            self.get_logger().info(f"OSNet ReID loaded: {_ENGINE_PATH}")
        except Exception as e:
            self.reid = None
            self.get_logger().error(f"OSNet ReID unavailable — body backup disabled: {e}")

        # InsightFace face engine (official FaceAnalysis)
        self.face_app = None
        if _INSIGHTFACE_AVAILABLE:
            try:
                self.face_app = FaceAnalysis(
                    name="buffalo_s",
                    providers=["CPUExecutionProvider"],
                    allowed_modules=["detection", "recognition"],
                )
                # ctx_id=-1 → CPU (ArcFaceONNX.prepare)
                self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
                self.face_gallery.threshold = self.face_threshold
                self.get_logger().info(
                    f"InsightFace loaded (buffalo_s, threshold={self.face_threshold})"
                )
            except Exception as e:
                self.face_app = None
                self.get_logger().error(f"InsightFace unavailable: {e}")
        else:
            self.get_logger().error(
                "insightface not installed — face enroll/reacquire disabled"
            )

        body_ok = self.gallery.load()
        face_ok = self.face_gallery.load()
        if body_ok or face_ok:
            name = (
                self.face_gallery.person_name
                if face_ok and self.face_gallery.person_name
                else (self.gallery.person_name if body_ok else "target")
            )
            if name and name != "target":
                self.target_name = name
            self.get_logger().info(
                f"Loaded galleries body={len(self.gallery)} face={len(self.face_gallery)} "
                f"name={self.target_name or 'target'}. "
                "Press SPACE to re-register, or wait for auto-reacquire."
            )
            self.state = FollowState.LOST
            who = self.target_name or "target"
            self.last_status = (
                f"GALLERY LOADED — {who} "
                f"(body={len(self.gallery)} face={len(self.face_gallery)})"
            )
            self.enrolled_height_m = self.gallery.height_m
            self.last_seen_time = time.time()
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
        """Keep standoff at target_distance (1.2 m).

        distance > 1.2 → move forward
        distance < 1.2 → move backward
        near 1.2      → stop (deadband)
        """
        if not self.actuation_enabled:
            self.last_lin = 0.0
            self.last_ang = 0.0
            self.last_status = "LOCKED (no actuation)"
            return
        twist = Twist()
        twist.angular.z = max(
            min(self.k_ang * angle_rad, self.max_ang_speed), -self.max_ang_speed
        )

        if distance is None:
            twist.linear.x = 0.0
            status = "NO DISTANCE — STOP"
        else:
            # Positive err = too far → forward; negative = too close → reverse
            err = float(distance) - float(self.target_distance)
            db = float(self.distance_deadband_m)

            if distance < self.min_distance:
                # Person very close — back up at full reverse, still turn to face
                twist.linear.x = -abs(self.max_lin_speed)
                status = f"TOO CLOSE {distance:.2f}m — BACK UP"
            elif abs(err) <= db:
                twist.linear.x = 0.0
                status = f"HOLD {self.target_distance:.1f}m"
            else:
                cmd = self.k_lin * err
                twist.linear.x = max(
                    min(cmd, self.max_lin_speed), -self.max_lin_speed
                )
                if err > 0:
                    status = f"APPROACH {distance:.2f}m → {self.target_distance:.1f}m"
                else:
                    status = f"BACK UP {distance:.2f}m → {self.target_distance:.1f}m"

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

    def begin_naming(self):
        """Ask for person-of-interest name (OpenCV keys), then enroll."""
        self._name_buffer = ""
        self.state = FollowState.NAMING
        self.last_status = "TYPE NAME then ENTER (ESC=cancel)"
        self.stop_robot()
        self.get_logger().info("Enter target name — type in the video window, then Enter")

    def begin_register(self, person_name: str | None = None):
        name = (person_name or self.target_name or "target").strip()
        if not name:
            name = "target"
        self.target_name = name
        self.gallery.clear()
        self.face_gallery.clear()
        self.gallery.person_name = name
        self.face_gallery.person_name = name
        self.locked_id = None
        self._lock_source = None
        self.lock_lost_time = None
        self.last_target_position = None
        self.enrolled_height_m = None
        self._confirm_id = None
        self._confirm_streak = 0
        self._follow_mismatch_streak = 0
        self.search_angle_rad = 0.0
        self._search_prev_t = None
        self.register_start_t = time.time()
        self._register_det_scores = []
        self._register_last_det = 0.0
        self._register_last_self_sim = 0.0
        self.state = FollowState.REGISTERING
        self.last_status = (
            f"REGISTERING {name} — FACE CAMERA until face>={self.register_min_faces} "
            f"(FULL={self.register_ideal_faces})"
        )
        self.stop_robot()
        self.get_logger().info(
            f"Registration started for '{name}' — strict InsightFace enroll "
            f"(min_faces={self.register_min_faces}, full={self.register_ideal_faces}, "
            f"min_det={self.register_min_det_score})"
        )

    def reset_identity(self):
        self.gallery.clear()
        self.face_gallery.clear()
        for p in (_GALLERY_PATH, _FACE_GALLERY_PATH):
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        self.locked_id = None
        self._lock_source = None
        self.lock_lost_time = None
        self.last_target_position = None
        self.enrolled_height_m = None
        self.target_name = ""
        self._name_buffer = ""
        self._confirm_id = None
        self._confirm_streak = 0
        self.state = FollowState.IDLE
        self.last_status = "PRESS SPACE TO REGISTER"
        self.stop_robot()
        self.get_logger().info("Identity cleared (body + face galleries)")

    def detect_faces(self, frame_bgr):
        """InsightFace FaceAnalysis.get — returns list of Face with embeddings."""
        if self.face_app is None:
            return []
        try:
            return self.face_app.get(frame_bgr)
        except Exception as e:
            self.get_logger().warn(f"FaceAnalysis.get failed: {e}")
            return []

# ═══════════════════════════════════════════════════════════════════════════
#  Drawing
# ═══════════════════════════════════════════════════════════════════════════


def _outlined_text(img, text, org, scale, color, thickness=2, outline=(0, 0, 0)):
    """Readable text without heavy blending."""
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, outline, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_hud(
    img,
    fps,
    state,
    status,
    lin,
    ang,
    n_gallery,
    reid_score,
    dist_text,
    n_face=0,
    face_score=0.0,
    recording: bool = False,
    target_name: str = "",
):
    h, w = img.shape[:2]
    # Solid bar (no addWeighted — much cheaper on Jetson)
    cv2.rectangle(img, (0, 0), (w, 100), (20, 20, 20), -1)
    who = (target_name or "").strip()
    _outlined_text(
        img,
        f"FPS {fps:.1f}  |  {state.name}  |  face={n_face} body={n_gallery}",
        (14, 32),
        0.85,
        (255, 255, 255),
        2,
    )
    if who:
        _outlined_text(img, f"TARGET: {who}", (14, 68), 1.05, (0, 255, 120), 2)
    else:
        _outlined_text(img, status[:75], (14, 68), 0.75, (80, 255, 80), 2)
    if recording:
        cv2.circle(img, (w - 36, 30), 12, (0, 0, 255), -1)
        _outlined_text(img, "REC", (w - 100, 38), 0.9, (0, 0, 255), 2)
    _outlined_text(
        img,
        "SPACE=register  R=reset  V=record  Q=quit",
        (14, h - 18),
        0.65,
        (190, 190, 190),
        2,
    )


def _draw_name_plate(img, bbox, name: str, color=(0, 255, 0), subtitle: str = ""):
    """Big name above person — light draw path (no frame copies)."""
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    name = (name or "TARGET").strip() or "TARGET"
    scale, thickness = 1.35, 3
    (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad_x, pad_y = 12, 10
    box_w, box_h = tw + 2 * pad_x, th + 2 * pad_y
    if subtitle:
        (sw, sh), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        box_w = max(box_w, sw + 2 * pad_x)
        box_h += sh + 8
    bx1 = max(0, min(x1, img.shape[1] - box_w - 2))
    by2 = max(box_h + 4, y1 - 10)
    by1 = by2 - box_h
    cv2.rectangle(img, (bx1, by1), (bx1 + box_w, by2), color, -1)
    cv2.rectangle(img, (bx1, by1), (bx1 + box_w, by2), (255, 255, 255), 2)
    cv2.putText(
        img,
        name,
        (bx1 + pad_x, by1 + pad_y + th - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            img,
            subtitle,
            (bx1 + pad_x, by2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)


def _draw_body(img, body, bbox, color, label):
    """Fallback simple box+label (non-target overlays)."""
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    _outlined_text(img, label, (x1, max(32, y1 - 10)), 0.75, color, 2)


def _face_roi_from_body(bbox):
    """Approximate face region from upper body when no face bbox yet."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    fx1 = x1 + int(0.22 * bw)
    fx2 = x2 - int(0.22 * bw)
    fy1 = y1 + int(0.02 * bh)
    fy2 = y1 + int(0.32 * bh)
    return fx1, fy1, fx2, fy2


def _draw_face_enroll_overlay(img, face_bbox, name: str, progress: float, det: float = 0.0):
    """Simple face oval + progress (no pulsing blends — keeps FPS up)."""
    if face_bbox is None:
        return
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    if x2 <= x1 or y2 <= y1:
        return
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax, ay = max(12, (x2 - x1) // 2), max(16, (y2 - y1) // 2)
    # Outline only (cheap)
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (0, 255, 255), 3)
    prog = float(max(0.0, min(1.0, progress)))
    cv2.ellipse(img, (cx, cy), (ax + 12, ay + 12), -90, 0, int(360 * prog), (0, 255, 120), 4)
    # One scan line (no fill blend)
    scan_y = y1 + int((0.2 + 0.6 * ((math.sin(time.time() * 3.0) + 1.0) * 0.5)) * (y2 - y1))
    cv2.line(img, (x1 + 4, scan_y), (x2 - 4, scan_y), (0, 255, 255), 2)
    who = (name or "TARGET").strip() or "TARGET"
    _outlined_text(
        img,
        f"FACE SCAN: {who}  {int(100 * prog)}%",
        (max(8, cx - 200), max(36, y1 - 24)),
        1.0,
        (0, 255, 255),
        2,
    )


# Fixed container FPS so playback ≈ wall-clock (live HUD fps can spike and speed up video).
_DEMO_FPS = 12.0


def _open_demo_writer(frame_bgr, fps_hint: float = _DEMO_FPS):
    """Open an MP4/AVI writer for professor demos (Jetson-safe codecs)."""
    os.makedirs(_EVAL_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h, w = frame_bgr.shape[:2]
    # Always use fixed _DEMO_FPS — do NOT use fluctuating live fps (causes speed-up).
    fps_w = float(_DEMO_FPS)
    candidates = [
        (os.path.join(_EVAL_DIR, f"demo_{stamp}.mp4"), "mp4v"),
        (os.path.join(_EVAL_DIR, f"demo_{stamp}.avi"), "XVID"),
        (os.path.join(_EVAL_DIR, f"demo_{stamp}.avi"), "MJPG"),
    ]
    for path, fourcc_name in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(path, fourcc, fps_w, (w, h))
        if writer is not None and writer.isOpened():
            return writer, path
        if writer is not None:
            writer.release()
    return None, None


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
    # Windowed by default — Jetson OS screen recorders go black on fullscreen OpenCV.
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    fullscreen = False

    prev_t = time.time()
    frame_count = 0
    fps = 0.0
    hud_reid = 0.0
    hud_dist = ""
    last_frame = None
    eval_log = EvalLogger()
    demo_writer = None
    demo_path = None
    demo_t0 = None
    demo_n = 0
    node.get_logger().info(
        f"Eval logging → {eval_log.path}  |  M=mark  V=record MP4  F=fullscreen  Q=quit"
    )
    node.get_logger().info(
        f"Demo tip: press V to record @ {_DEMO_FPS:.0f} FPS (eval_logs/). "
        "Do not use OS screen recorder."
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

        # InsightFace face pass only when enrolling or reacquiring (CPU cost)
        need_faces = node.state in (
            FollowState.REGISTERING,
            FollowState.REACQUIRING,
            FollowState.LOST,
        )
        faces = node.detect_faces(frame_bgr) if need_faces else []
        node._last_faces = faces
        for face in faces:
            fb = getattr(face, "bbox", None)
            if fb is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in fb]
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 180, 0), 1)

        # Score bodies. Body OSNet only when enrolling / reacquiring
        # (tracker_node: no per-frame appearance ReID while FOLLOWING).
        need_body_reid = node.state in (
            FollowState.REGISTERING,
            FollowState.REACQUIRING,
            FollowState.LOST,
        )
        # tuple: (body_gscore, body, bbox, xyz, height, body_emb, face_sim, face_decision)
        scored = []
        for b in bodies:
            bbox = node._bbox_from_body(b, img_w, img_h)
            xyz = node._body_xyz(b)
            if bbox is None or xyz is None:
                continue
            emb = None
            gscore = 0.0
            if (
                need_body_reid
                and node.reid is not None
                and len(node.gallery) > 0
            ):
                emb = node.reid.extract(frame_bgr, bbox)
                gscore = node.gallery.score(emb)
            face_sim = 0.0
            face_decision = "Different Person"
            if faces and len(node.face_gallery) > 0:
                face = _best_face_for_body(faces, bbox)
                if face is not None and getattr(face, "normed_embedding", None) is not None:
                    id_res = node.face_gallery.identify(face.normed_embedding)
                    face_sim = float(id_res["similarity"])
                    face_decision = str(id_res["decision"])
            h_est = node._estimate_height(b)
            scored.append((gscore, b, bbox, xyz, h_est, emb, face_sim, face_decision))

        target_body = None
        target_bbox = None
        target_xyz = None
        target_score = 0.0
        hud_face = 0.0
        detected = False

        # ── IDLE ─────────────────────────────────────────────────────────
        if node.state == FollowState.IDLE:
            node.stop_robot()
            node.last_status = "PRESS SPACE TO REGISTER"
            for _, b, bbox, *_rest in scored:
                _draw_body(frame_bgr, b, bbox, (180, 180, 180), f"id={b.id}")

        # ── NAMING (type person-of-interest name) ─────────────────────────
        elif node.state == FollowState.NAMING:
            node.stop_robot()
            buf = node._name_buffer or ""
            node.last_status = f"NAME: {buf}_   (ENTER=start  ESC=cancel)"
            for _, b, bbox, *_rest in scored:
                _draw_body(frame_bgr, b, bbox, (180, 180, 180), f"id={b.id}")
            # Big name prompt in the middle of the frame
            h, w = frame_bgr.shape[:2]
            msg = f"Person of interest: {buf}_"
            _outlined_text(
                frame_bgr,
                msg,
                (max(20, w // 2 - 320), h // 2),
                1.4,
                (0, 255, 255),
                3,
            )
            _outlined_text(
                frame_bgr,
                "Type name  |  ENTER confirm  |  ESC cancel",
                (max(20, w // 2 - 300), h // 2 + 48),
                0.95,
                (220, 220, 220),
                2,
            )

        # ── REGISTERING ──────────────────────────────────────────────────
        elif node.state == FollowState.REGISTERING:
            node.stop_robot()
            # Prefer centered + closest person
            best = None
            best_key = -1e9
            for gscore, b, bbox, xyz, h_est, emb, face_sim, face_decision in scored:
                x, y, z = xyz
                cx = 0.5 * (bbox[0] + bbox[2])
                center = 1.0 - abs(cx - img_w * 0.5) / (img_w * 0.5)
                key = 2.0 * center - 0.15 * z
                if key > best_key:
                    best_key = key
                    best = (b, bbox, xyz, h_est)

            if best is not None:
                b, bbox, xyz, h_est = best
                # Body OSNet enroll (backup)
                if node.reid is not None:
                    emb = node.reid.extract(frame_bgr, bbox)
                    if emb is not None:
                        node.gallery.add(emb, min_novelty=0.015)
                # InsightFace enroll: only high-det faces (strict first step)
                face = _best_face_for_body(faces, bbox)
                face_box = None
                if face is not None and getattr(face, "bbox", None) is not None:
                    face_box = [int(v) for v in face.bbox]
                if face is not None and getattr(face, "normed_embedding", None) is not None:
                    det = float(getattr(face, "det_score", 0.0) or 0.0)
                    node._register_last_det = det
                    if det >= node.register_min_det_score:
                        emb = face.normed_embedding
                        # Self-sim vs gallery (informational; after 1st sample)
                        if len(node.face_gallery) > 0:
                            node._register_last_self_sim = float(
                                node.face_gallery.score(emb)
                            )
                        if node.face_gallery.add(emb, min_novelty=0.02):
                            node._register_det_scores.append(det)
                            hud_face = det
                if h_est is not None:
                    node.enrolled_height_m = h_est
                    node.gallery.height_m = h_est
                node.locked_id = b.id
                node.last_target_position = xyz
                node.last_seen_time = time.time()
                target_body, target_bbox, target_xyz = b, bbox, xyz
                detected = True
                who = node.target_name or "target"
                n_face_now = len(node.face_gallery)
                prog = n_face_now / float(max(1, node.register_ideal_faces))
                # Demo visual: animated face mask + big name (recognition logic unchanged)
                if face_box is None:
                    face_box = _face_roi_from_body(bbox)
                _draw_face_enroll_overlay(
                    frame_bgr,
                    face_box,
                    who,
                    progress=prog,
                    det=node._register_last_det,
                )
                _draw_name_plate(
                    frame_bgr,
                    bbox,
                    who,
                    color=(0, 200, 255),
                    subtitle=f"Registering face  {n_face_now}/{node.register_ideal_faces}",
                )

            elapsed = time.time() - node.register_start_t
            n_emb = len(node.gallery)
            n_face = len(node.face_gallery)
            face_full = n_face >= node.register_ideal_faces
            face_ok = n_face >= node.register_min_faces
            # Stay on FACE CAMERA until strict face gate met; then body turn.
            if not face_ok:
                pose_hint = (
                    f"FACE CAMERA — need face {n_face}/{node.register_min_faces} "
                    f"(FULL={node.register_ideal_faces})"
                )
            elif not face_full:
                frac = elapsed / max(node.register_duration_sec, 1e-3)
                if frac < 0.55:
                    pose_hint = "FACE CAMERA — fill to FULL 32 (small head turns OK)"
                elif frac < 0.70:
                    pose_hint = "turn LEFT (body)"
                elif frac < 0.85:
                    pose_hint = "show BACK (body)"
                else:
                    pose_hint = "turn RIGHT → face camera"
            else:
                pose_hint = "FULL FACE — turn for body views"
            mean_det = (
                float(sum(node._register_det_scores) / len(node._register_det_scores))
                if node._register_det_scores
                else 0.0
            )
            node.last_status = (
                f"REGISTERING {elapsed:.1f}/{node.register_duration_sec:.0f}s  "
                f"face={n_face}/{node.register_ideal_faces} det={mean_det:.2f} "
                f"self={node._register_last_self_sim:.2f} body={n_emb}  → {pose_hint}"
            )
            body_ok = n_emb >= node.register_min_embeds or (
                elapsed >= node.register_duration_sec * 1.4 and n_emb >= 12
            )
            # Strict: NEVER leave REGISTERING without face_ok (min 16 samples)
            done = elapsed >= node.register_duration_sec and body_ok and face_ok
            # Early finish when FULL face load + body enough
            if face_full and body_ok and elapsed >= 3.0:
                done = True
            if done:
                node.gallery.save()
                node.face_gallery.save()
                node.state = FollowState.FOLLOWING
                node._lock_source = "register"
                node._follow_mismatch_streak = 0
                node.lock_lost_time = None
                node.search_angle_rad = 0.0
                node._search_prev_t = None
                if face_full and mean_det >= node.register_min_det_score:
                    quality = "FULL"
                elif n_face >= node.register_min_faces and mean_det >= node.register_min_det_score:
                    quality = "GOOD"
                else:
                    quality = "OK"
                who = node.target_name or "target"
                node.last_status = (
                    f"REGISTERED {who} {quality} "
                    f"(face={n_face}/{node.register_ideal_faces} "
                    f"det={mean_det:.2f} body={n_emb}) — FOLLOW"
                )
                node._pending_eval_event = "REGISTERED"
                node.get_logger().info(
                    f"Galleries saved for '{who}' body={n_emb} → {_GALLERY_PATH} | "
                    f"face={n_face} det_mean={mean_det:.3f} → {_FACE_GALLERY_PATH}"
                )
            elif elapsed >= node.register_duration_sec * 1.8:
                node.last_status = (
                    f"REGISTER FAILED (face={n_face}/{node.register_min_faces} "
                    f"need>={node.register_min_faces}, det>={node.register_min_det_score:.2f}, "
                    f"body={n_emb}) — face camera, alone, well lit"
                )
                node.state = FollowState.IDLE
                node.gallery.clear()
                node.face_gallery.clear()
                node._register_det_scores = []
                node.target_name = ""

        # ── FOLLOWING (tracker_node style: trust ZED id; no body ReID veto) ─
        elif node.state == FollowState.FOLLOWING:
            locked = None
            for gscore, b, bbox, xyz, h_est, emb, face_sim, face_decision in scored:
                if b.id != node.locked_id:
                    continue
                # Only impossible spatial jump can break a live lock
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
                # Optional legacy body veto (OFF by default — caused face↔lost thrash)
                if (
                    node.follow_use_body_veto
                    and node.reid is not None
                    and len(node.gallery) > 0
                    and emb is not None
                    and gscore < node.reid_min_follow
                ):
                    node._follow_mismatch_streak += 1
                    if node._follow_mismatch_streak >= node._follow_mismatch_limit:
                        node.get_logger().warn(
                            f"Appearance mismatch on locked id (reid={gscore:.2f})"
                        )
                        break
                else:
                    node._follow_mismatch_streak = 0
                locked = (gscore, b, bbox, xyz, h_est, emb, face_sim)
                break

            if locked is not None:
                gscore, b, bbox, xyz, h_est, emb, face_sim = locked
                target_body, target_bbox, target_xyz = b, bbox, xyz
                target_score = gscore
                hud_face = face_sim
                detected = True
                node.lock_lost_time = None
                node.last_target_position = xyz
                node.last_seen_time = time.time()
                node.search_angle_rad = 0.0
                node._search_prev_t = None
                src = node._lock_source or "?"
                who = node.target_name or "target"
                _draw_name_plate(
                    frame_bgr,
                    bbox,
                    who,
                    color=(0, 255, 0),
                    subtitle=f"Tracking  via {src}",
                )
            else:
                # Locked ZED id gone → appearance reacquire (face first)
                node.state = FollowState.REACQUIRING
                if node.lock_lost_time is None:
                    node.lock_lost_time = time.time()
                node.locked_id = None
                node._lock_source = None
                node._confirm_id = None
                node._confirm_streak = 0
                node._follow_mismatch_streak = 0
                node.stop_robot()
                node.last_status = "LOST TRACK — REACQUIRING"

        # ── REACQUIRING / LOST (InsightFace face-first, body backup) ───────
        if node.state in (FollowState.REACQUIRING, FollowState.LOST):
            for gscore, b, bbox, xyz, _h, _e, face_sim, face_decision in scored:
                col = (80, 80, 255)
                if face_decision == "Same Person":
                    col = (0, 255, 255)
                elif gscore >= node.reid_min_reacquire:
                    col = (0, 165, 255)
                _draw_body(
                    frame_bgr,
                    b,
                    bbox,
                    col,
                    f"F={face_sim:.2f} {face_decision[:4]}",
                )

            chosen = None
            reject_reason = None
            has_face_gal = len(node.face_gallery) > 0
            has_body_gal = len(node.gallery) > 0

            if not has_face_gal and not has_body_gal:
                reject_reason = "no gallery"
            elif not scored:
                reject_reason = "no bodies"
            else:
                # 1) Face path — InsightFace Same Person (threshold 0.4)
                face_ranked = sorted(
                    [
                        (face_sim, gscore, b, bbox, xyz, face_decision)
                        for gscore, b, bbox, xyz, _h, _e, face_sim, face_decision in scored
                        if face_decision == "Same Person"
                    ],
                    key=lambda t: t[0],
                    reverse=True,
                )
                if face_ranked:
                    best_f, best_bscore, best_b, best_bbox, best_xyz, _dec = face_ranked[0]
                    second_f = face_ranked[1][0] if len(face_ranked) >= 2 else None
                    if second_f is not None and (best_f - second_f) < node.face_margin:
                        reject_reason = f"FACE AMBIGUOUS (Δ={best_f - second_f:.2f})"
                    else:
                        chosen = (best_f, best_b, best_bbox, best_xyz, "face")
                        hud_face = best_f
                elif has_face_gal and faces:
                    # Faces seen but none matched — do NOT fall back to body in crowd
                    if len(scored) >= 2:
                        reject_reason = "FACE REJECT — waiting (crowd)"
                    else:
                        # Alone + face visible but below threshold → refuse
                        reject_reason = "FACE REJECT — not Same Person"
                elif has_face_gal and not faces:
                    # No face visible: body backup only if alone
                    body_ranked = sorted(
                        [
                            (gscore, b, bbox, xyz, h_est)
                            for gscore, b, bbox, xyz, h_est, _e, _fs, _fd in scored
                        ],
                        key=lambda t: t[0],
                        reverse=True,
                    )
                    if len(body_ranked) == 1 and has_body_gal:
                        best_score, best_b, best_bbox, best_xyz, best_h = body_ranked[0]
                        if best_score >= node.reid_min_alone:
                            if (
                                node.enrolled_height_m is not None
                                and best_h is not None
                                and abs(best_h - node.enrolled_height_m) > node.height_tol_m
                            ):
                                reject_reason = "REJECT height"
                            else:
                                chosen = (best_score, best_b, best_bbox, best_xyz, "body")
                        else:
                            reject_reason = (
                                f"BODY REJECT {best_score:.2f}<{node.reid_min_alone:.2f} (no face)"
                            )
                    elif len(body_ranked) >= 2:
                        reject_reason = "NO FACE — crowd, waiting"
                    else:
                        reject_reason = "no face / weak body"
                else:
                    # No face gallery enrolled — legacy body-only path
                    body_ranked = sorted(
                        [
                            (gscore, b, bbox, xyz, h_est)
                            for gscore, b, bbox, xyz, h_est, _e, _fs, _fd in scored
                        ],
                        key=lambda t: t[0],
                        reverse=True,
                    )
                    best_score, best_b, best_bbox, best_xyz, best_h = body_ranked[0]
                    second_score = body_ranked[1][0] if len(body_ranked) >= 2 else None
                    n_vis = len(body_ranked)
                    if second_score is not None and (best_score - second_score) < node.reid_margin:
                        reject_reason = f"AMBIGUOUS (Δ={best_score - second_score:.2f})"
                    else:
                        need = node.reid_min_alone if n_vis == 1 else node.reid_min_reacquire
                        if best_score < need:
                            reject_reason = f"REJECT reid={best_score:.2f}<{need:.2f}"
                        elif (
                            node.enrolled_height_m is not None
                            and best_h is not None
                            and abs(best_h - node.enrolled_height_m) > node.height_tol_m
                        ):
                            reject_reason = "REJECT height"
                        else:
                            chosen = (best_score, best_b, best_bbox, best_xyz, "body")

            if chosen is not None:
                gscore, b, bbox, xyz, how = chosen
                if how == "face":
                    hud_face = gscore
                if node._confirm_id == b.id:
                    node._confirm_streak += 1
                else:
                    node._confirm_id = b.id
                    node._confirm_streak = 1
                node.last_status = (
                    f"REACQUIRING {node._confirm_streak}/{node.confirm_frames} "
                    f"{how}={gscore:.2f}"
                )
                who = node.target_name or "target"
                _draw_body(frame_bgr, b, bbox, (0, 255, 255), f"Match? {who}")
                # Face matches are stronger — confirm faster (tracker_node locks ASAP)
                need_confirm = 3 if how == "face" else node.confirm_frames
                if node._confirm_streak >= need_confirm:
                    who = node.target_name or "target"
                    node.locked_id = b.id
                    node._lock_source = how
                    node.last_target_position = xyz
                    node.lock_lost_time = None
                    node.state = FollowState.FOLLOWING
                    node._confirm_id = None
                    node._confirm_streak = 0
                    node._follow_mismatch_streak = 0
                    node.search_angle_rad = 0.0
                    node._search_prev_t = None
                    target_body, target_bbox, target_xyz = b, bbox, xyz
                    target_score = gscore
                    detected = True
                    node.last_seen_time = time.time()
                    node.last_status = f"REACQUIRED {who} id={b.id} via {how}={gscore:.2f}"
                    node._pending_eval_event = (
                        "REACQUIRED_FACE" if how == "face" else "REACQUIRED_BODY"
                    )
                    node.get_logger().info(
                        f"Reacquired '{who}' ZED id={b.id} via {how} score={gscore:.2f} "
                        f"(lock held until ZED id lost — no body veto)"
                    )
                    _draw_name_plate(
                        frame_bgr,
                        bbox,
                        who,
                        color=(0, 255, 0),
                        subtitle=f"Reacquired via {how}",
                    )
            else:
                node._confirm_id = None
                node._confirm_streak = 0
                if not has_face_gal and not has_body_gal:
                    node.state = FollowState.IDLE
                    node.last_status = "PRESS SPACE TO REGISTER"
                    node.stop_robot()
                else:
                    node.state = FollowState.LOST
                    if reject_reason:
                        node.last_status = f"{reject_reason} — WAITING"

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
            # Name plate already drawn in FOLLOWING branch — keep it dominant (no overwrite)
        elif node.state in (FollowState.LOST, FollowState.REACQUIRING):
            now = time.time()
            since = (now - node.last_seen_time) if node.last_seen_time > 0 else 1e9
            if (
                node.search_enabled
                and node.last_seen_time > 0
                and since >= node.search_start_delay_sec
                and (len(node.gallery) > 0 or len(node.face_gallery) > 0)
                and len(scored) <= node.search_max_bodies
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

        # Eval metrics this frame (body + face numbers for CSV/summary)
        reid_scores = sorted(
            [float(s[0]) for s in scored if s[0] > 0], reverse=True
        )
        reid_best = reid_scores[0] if reid_scores else 0.0
        reid_second = reid_scores[1] if len(reid_scores) > 1 else 0.0
        margin = reid_best - reid_second if len(reid_scores) > 1 else reid_best
        face_scores = sorted(
            [float(s[6]) for s in scored if float(s[6]) > 0], reverse=True
        )
        face_best = face_scores[0] if face_scores else 0.0
        face_second = face_scores[1] if len(face_scores) > 1 else 0.0
        face_margin = (
            face_best - face_second if len(face_scores) > 1 else face_best
        )
        if hud_face <= 0 and face_best > 0:
            hud_face = face_best
        dist_m = ""
        angle_deg = ""
        xyz_x = xyz_y = xyz_z = ""
        if detected and target_xyz is not None:
            x, y, z = target_xyz
            dist_m = f"{math.sqrt(x * x + y * y + z * z):.3f}"
            angle_deg = f"{math.degrees(-math.atan2(x, z)):.2f}"
            xyz_x, xyz_y, xyz_z = f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"

        event = ""
        if node._pending_eval_event:
            event = node._pending_eval_event
            node._pending_eval_event = None
            node.get_logger().info(
                f"EVAL {event} face={hud_face:.3f} body={target_score:.3f} "
                f"id={node.locked_id} body_n={len(node.gallery)} "
                f"face_n={len(node.face_gallery)}"
            )
        elif prev_state != node.state:
            event = f"STATE:{prev_state.name}->{node.state.name}"
        if prev_state != node.state:
            prev_state = node.state

        eval_log.log(
            {
                "frame": node._frame_i,
                "state": node.state.name,
                "status": node.last_status,
                "n_bodies": len(scored),
                "locked_id": "" if node.locked_id is None else node.locked_id,
                "gallery_n": len(node.gallery),
                "face_n": len(node.face_gallery),
                "reid_score": f"{(hud_reid if detected else 0.0):.4f}",
                "reid_best": f"{reid_best:.4f}",
                "reid_second": f"{reid_second:.4f}",
                "margin": f"{margin:.4f}",
                "face_score": f"{hud_face:.4f}",
                "face_best": f"{face_best:.4f}",
                "face_second": f"{face_second:.4f}",
                "face_margin": f"{face_margin:.4f}",
                "lock_source": node._lock_source or "",
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
            n_face=len(node.face_gallery),
            face_score=hud_face,
            recording=(demo_writer is not None),
            target_name=node.target_name or "",
        )
        if demo_writer is not None:
            try:
                # Pace writes to _DEMO_FPS so file duration ≈ wall-clock (not sped up).
                elapsed = time.time() - demo_t0
                target_n = int(elapsed * _DEMO_FPS) + 1
                while demo_n < target_n:
                    demo_writer.write(frame_bgr)
                    demo_n += 1
            except Exception as e:
                node.get_logger().error(f"Demo write failed: {e}")
                demo_writer.release()
                demo_writer = None
                demo_path = None
                demo_t0 = None
                demo_n = 0
        cv2.imshow(window, frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        # ── Naming: type person-of-interest name in the video window ──
        if node.state == FollowState.NAMING:
            if key in (ord("q"),):
                break
            if key in (27,):  # ESC cancel
                node._name_buffer = ""
                node.state = FollowState.IDLE
                node.last_status = "PRESS SPACE TO REGISTER"
            elif key in (13, 10):  # Enter confirm
                name = (node._name_buffer or "").strip()
                if not name:
                    node.last_status = "NAME empty — type a name, then ENTER"
                else:
                    eval_log.mark("SPACE_REGISTER")
                    node.begin_register(name)
            elif key in (8, 127):  # Backspace
                node._name_buffer = node._name_buffer[:-1]
            elif 32 <= key <= 126:
                # Printable ASCII (letters, digits, space, punctuation)
                if len(node._name_buffer) < 32:
                    node._name_buffer += chr(key)
            continue

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            if node.reid is None and node.face_app is None:
                node.get_logger().error("Cannot register — no face/body ReID")
                node.last_status = "NO FACE/BODY ENGINE — cannot register"
            else:
                node.begin_naming()
        elif key in (ord("r"), ord("R")):
            eval_log.mark("RESET")
            node.reset_identity()
        elif key in (ord("m"), ord("M")):
            # Solo protocol: M when you leave, M again when you return
            tag = "MARK_LEAVE" if node.state == FollowState.FOLLOWING else "MARK_RETURN"
            eval_log.mark(tag)
            node.last_status = f"{node.last_status} | {tag}"
            node.get_logger().info(f"EVAL {tag} recorded")
        elif key in (ord("f"), ord("F")):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                window,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
            if not fullscreen:
                cv2.resizeWindow(window, 1280, 720)
            node.get_logger().info(
                f"Display → {'FULLSCREEN' if fullscreen else 'WINDOWED (better for demos)'}"
            )
        elif key in (ord("v"), ord("V")):
            if demo_writer is not None:
                wall = time.time() - demo_t0 if demo_t0 else 0.0
                vid = demo_n / _DEMO_FPS if _DEMO_FPS else 0.0
                demo_writer.release()
                demo_writer = None
                node.last_status = f"DEMO SAVED → {demo_path}"
                node.get_logger().info(
                    f"Demo recording stopped: {demo_path}  "
                    f"wall={wall:.1f}s video={vid:.1f}s frames={demo_n} @ {_DEMO_FPS:.0f}fps"
                )
                eval_log.mark("DEMO_STOP")
                demo_path = None
                demo_t0 = None
                demo_n = 0
            else:
                demo_writer, demo_path = _open_demo_writer(frame_bgr)
                if demo_writer is None:
                    node.last_status = "DEMO RECORD FAILED — codec/open error"
                    node.get_logger().error("Could not open demo VideoWriter")
                else:
                    demo_t0 = time.time()
                    demo_n = 0
                    node.last_status = f"DEMO RECORDING → {os.path.basename(demo_path)}"
                    node.get_logger().info(
                        f"Demo recording started: {demo_path} @ {_DEMO_FPS:.0f} FPS (real-time)"
                    )
                    eval_log.mark("DEMO_START")

    node.stop_robot()
    if demo_writer is not None:
        wall = time.time() - demo_t0 if demo_t0 else 0.0
        demo_writer.release()
        node.get_logger().info(
            f"Demo recording saved: {demo_path}  wall={wall:.1f}s frames={demo_n}"
        )
    eval_log.close_and_summarize()
    if node.reid is not None:
        node.reid.close()
    zed.close()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
