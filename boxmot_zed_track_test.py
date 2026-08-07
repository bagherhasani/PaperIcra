#!/usr/bin/env python3
"""
ZED/webcam → YOLO + BoxMOT → track IDs + CSV eval log.

Why it felt awful (~3 FPS, mushy/laggy):
  - BotSort + OSNet ReID on CPU every frame
  - VGA upscaled (soft)
  - camera buffer lag / extra threads stealing CPU from YOLO

Defaults now (Jetson, pip torch has no CUDA):
  HD720 @ 15, imgsz=256, ByteTrack (no ReID) → ~7–9 detect FPS, sharper image.

CRITICAL: open ZED BEFORE torch/BoxMOT (cu130 torch breaks ZED if loaded first).

  source boxmot-venv/bin/activate
  python boxmot_zed_track_test.py
  python boxmot_zed_track_test.py --tracker botsort   # ReID, much slower

Keys: Q/ESC=quit   M=mark
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
cv2.setNumThreads(1)

_USER_SITE = Path.home() / ".local/lib/python3.10/site-packages"
if _USER_SITE.is_dir() and str(_USER_SITE) not in sys.path:
    sys.path.append(str(_USER_SITE))

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_YOLO = _SCRIPT_DIR / "yolov8n.pt"
_DEFAULT_REID = _SCRIPT_DIR / "osnet_x0_25_msmt17.pt"
_EVAL_DIR = _SCRIPT_DIR / "eval_logs"


class TrackEvalLogger:
    HEADERS = [
        "t", "frame", "n_tracks", "ids", "fps", "infer_ms", "event", "notes",
    ]

    def __init__(self, out_dir: Path = _EVAL_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = out_dir / f"boxmot_track_{stamp}.csv"
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.HEADERS)
        self._w.writeheader()
        self.t0 = time.time()
        self.rows = 0
        self.events = []
        self._prev_ids: set[int] = set()
        self.id_appear = 0
        self.id_disappear = 0
        self.max_tracks = 0

    def log(self, frame_i, ids, fps, infer_ms, event="", notes=""):
        id_set = set(ids)
        if self._prev_ids:
            self.id_appear += len(id_set - self._prev_ids)
            self.id_disappear += len(self._prev_ids - id_set)
        self._prev_ids = id_set
        self.max_tracks = max(self.max_tracks, len(ids))
        row = {
            "t": f"{time.time() - self.t0:.3f}",
            "frame": frame_i,
            "n_tracks": len(ids),
            "ids": " ".join(str(i) for i in ids),
            "fps": f"{fps:.2f}",
            "infer_ms": f"{infer_ms:.1f}",
            "event": event,
            "notes": notes,
        }
        self._w.writerow(row)
        self.rows += 1
        if event:
            self.events.append((row["t"], event))
        if self.rows % 30 == 0:
            self._f.flush()

    def mark(self, name: str):
        self.log(0, list(self._prev_ids), 0.0, 0.0, event=name)

    def close_and_summarize(self):
        if self._f.closed:
            return self.path, None
        self._f.flush()
        self._f.close()
        summary = self.path.with_name(self.path.stem + "_summary.txt")
        duration = time.time() - self.t0
        lines = [
            f"log: {self.path}",
            f"rows: {self.rows}",
            f"duration_s: {duration:.1f}",
            f"events: {self.events}",
            f"max_tracks_seen: {self.max_tracks}",
            f"id_appear_events: {self.id_appear}",
            f"id_disappear_events: {self.id_disappear}",
            "",
            "CPU YOLO ceiling is ~7–10 FPS on this Jetson until TensorRT/CUDA torch.",
            "Default ByteTrack = fast in-view IDs. Leave/return may change id.",
        ]
        try:
            fps_vals = []
            with open(self.path, newline="") as f:
                for r in csv.DictReader(f):
                    try:
                        v = float(r["fps"])
                        if v > 0:
                            fps_vals.append(v)
                    except Exception:
                        pass
            if fps_vals:
                lines.insert(
                    4,
                    f"fps_mean: {float(np.mean(fps_vals)):.2f}  "
                    f"min: {float(np.min(fps_vals)):.2f}  max: {float(np.max(fps_vals)):.2f}",
                )
        except Exception:
            pass
        text = "\n".join(lines) + "\n"
        summary.write_text(text)
        print("\n===== BOXMOT TRACK SUMMARY =====")
        print(text)
        print(f"CSV: {self.path}")
        print(f"Summary: {summary}")
        return self.path, summary


def parse_args():
    p = argparse.ArgumentParser(description="BoxMOT track test (fast Jetson defaults)")
    p.add_argument("--webcam", type=int, default=None)
    p.add_argument("--yolo", type=Path, default=_DEFAULT_YOLO)
    p.add_argument("--reid", type=Path, default=_DEFAULT_REID)
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument(
        "--tracker",
        choices=("bytetrack", "ocsort", "botsort"),
        default="bytetrack",
        help="bytetrack/ocsort=fast. botsort=ReID (slow on CPU)",
    )
    p.add_argument("--imgsz", type=int, default=256, help="YOLO size (256 fastest usable)")
    p.add_argument("--zed-res", choices=("vga", "hd720", "hd1080"), default="hd720")
    p.add_argument("--zed-fps", type=int, default=15)
    p.add_argument(
        "--display-width",
        type=int,
        default=960,
        help="Preview width; 0=native",
    )
    return p.parse_args()


def open_zed(res_name: str, fps: int):
    import pyzed.sl as sl

    zed = sl.Camera()
    init = sl.InitParameters()
    res_map = {
        "vga": sl.RESOLUTION.VGA,
        "hd720": sl.RESOLUTION.HD720,
        "hd1080": sl.RESOLUTION.HD1080,
    }
    init.camera_resolution = res_map[res_name]
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.camera_fps = int(fps)
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        try:
            zed.close()
        except Exception:
            pass
        raise RuntimeError(f"Cannot open ZED: {status}")
    mat = sl.Mat()
    runtime = sl.RuntimeParameters()

    def read():
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        zed.retrieve_image(mat, sl.VIEW.LEFT)
        return cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)

    def close():
        try:
            zed.close()
        except Exception:
            pass

    return read, close, f"ZED-{res_name}@{fps}"


def open_webcam(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {index}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read():
        # drop one buffered frame if present
        cap.grab()
        ok, frame = cap.retrieve()
        if not ok:
            ok, frame = cap.read()
        return frame if ok else None

    def close():
        cap.release()

    return read, close, f"webcam:{index}"


def dets_from_yolo(result, conf_min: float, scale_x: float, scale_y: float) -> np.ndarray:
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = result.boxes.xyxy.cpu().numpy().copy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    keep = (cls == 0) & (conf >= conf_min)
    if not np.any(keep):
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = xyxy[keep]
    xyxy[:, [0, 2]] *= scale_x
    xyxy[:, [1, 3]] *= scale_y
    return np.concatenate(
        [xyxy, conf[keep, None], cls[keep, None]], axis=1
    ).astype(np.float32)


def draw_tracks(frame, tracks: np.ndarray):
    if tracks is None or len(tracks) == 0:
        return
    for t in tracks:
        x1, y1, x2, y2 = map(int, t[:4])
        tid = int(t[4])
        conf = float(t[5]) if len(t) > 5 else 0.0
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            f"id={tid} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 0),
            2,
        )


def _silence_loguru():
    try:
        from loguru import logger as _loguru

        _loguru.remove()
        _loguru.add(sys.stderr, level="ERROR")
    except Exception:
        pass


def load_models(yolo_src: str, reid: Path, device_name: str, tracker_name: str):
    import torch
    from ultralytics import YOLO
    from boxmot import BotSort, ByteTrack, OcSort

    torch.set_num_threads(4)
    _silence_loguru()
    device = torch.device(device_name)
    yolo = YOLO(yolo_src)

    if tracker_name == "bytetrack":
        tracker = ByteTrack(frame_rate=15)
        uses_reid = False
    elif tracker_name == "ocsort":
        tracker = OcSort()
        uses_reid = False
    else:
        if not reid.is_file():
            raise FileNotFoundError(f"ReID weights not found: {reid}")
        tracker = BotSort(
            reid_weights=reid,
            device=device,
            half=False,
            with_reid=True,
        )
        uses_reid = True

        class _NoCMC:
            def apply(self, img, dets=None):
                return np.eye(2, 3, dtype=np.float32)

        tracker.cmc = _NoCMC()

    return yolo, tracker, device, uses_reid


def main():
    args = parse_args()
    yolo_src = str(args.yolo) if args.yolo.is_file() else "yolov8n.pt"

    if args.webcam is not None:
        read, close, name = open_webcam(args.webcam)
    else:
        try:
            print("Opening ZED before loading Torch/BoxMOT...")
            read, close, name = open_zed(args.zed_res, args.zed_fps)
            print(f"ZED open OK ({name})")
        except Exception as e:
            print(f"ZED unavailable ({e}); falling back to webcam 0")
            read, close, name = open_webcam(0)

    print(
        f"Loading YOLO={Path(yolo_src).name}  tracker={args.tracker}  "
        f"device={args.device}  imgsz={args.imgsz}"
    )
    if args.tracker == "botsort":
        print("NOTE: botsort+ReID on CPU is slow (~3 FPS). Prefer default bytetrack.")
    else:
        print("Fast mode: no appearance ReID. Expect ~7–9 FPS on CPU.")

    yolo, tracker, _device, uses_reid = load_models(
        yolo_src, args.reid, args.device, args.tracker
    )

    eval_log = TrackEvalLogger()
    print(f"Logging → {eval_log.path}")
    print("Stand in front of ZED. Q=quit  M=mark")

    window = f"BoxMOT {args.tracker} ({name})"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    prev = time.time()
    fps = 0.0
    n_fps = 0
    frame_i = 0

    try:
        while True:
            frame = read()
            if frame is None:
                continue
            frame_i += 1
            h0, w0 = frame.shape[:2]
            tw = args.imgsz
            th = max(32, int(round(args.imgsz * h0 / max(w0, 1) / 32) * 32))
            infer = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

            t0 = time.time()
            result = yolo(
                infer,
                verbose=False,
                classes=[0],
                conf=args.conf,
                imgsz=args.imgsz,
            )[0]
            dets = dets_from_yolo(result, args.conf, w0 / float(tw), h0 / float(th))
            tracks = tracker.update(dets, frame)
            infer_ms = (time.time() - t0) * 1000.0

            draw_tracks(frame, tracks)

            n_fps += 1
            now = time.time()
            if now - prev >= 1.0:
                fps = n_fps / (now - prev)
                prev = now
                n_fps = 0

            ids = [int(t[4]) for t in tracks] if tracks is not None and len(tracks) else []
            eval_log.log(frame_i, ids, fps, infer_ms)

            mode = "ReID" if uses_reid else "fast"
            cv2.putText(
                frame,
                f"FPS {fps:.1f}  {infer_ms:.0f}ms  ids={ids}  [{mode}]",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if args.display_width and w0 != args.display_width:
                scale = args.display_width / float(w0)
                disp = cv2.resize(
                    frame,
                    (args.display_width, int(h0 * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                disp = frame
            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("m"), ord("M")):
                eval_log.mark("MARK")
                print(f"[MARK] t={time.time() - eval_log.t0:.1f}s ids={ids}")
    except KeyboardInterrupt:
        print("\nInterrupted — writing summary...")
    finally:
        close()
        cv2.destroyAllWindows()
        eval_log.close_and_summarize()


if __name__ == "__main__":
    main()
