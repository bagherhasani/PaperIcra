#!/usr/bin/env python3
"""Analyze a follow_eval_*.csv from zed-color.py solo tests.

Usage:
  python3 analyze_eval.py eval_logs/follow_eval_YYYYMMDD_HHMMSS.csv
  python3 analyze_eval.py   # latest file in eval_logs/
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from collections import Counter

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.join(_SCRIPT_DIR, "eval_logs")


def latest_csv() -> str | None:
    files = sorted(glob.glob(os.path.join(_EVAL_DIR, "follow_eval_*.csv")))
    return files[-1] if files else None


def fnum(row, key, default=None):
    try:
        v = row.get(key, "")
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default


def analyze(path: str) -> None:
    rows = []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"Empty: {path}")
        return

    states = Counter(r["state"] for r in rows if r.get("state"))
    events = [(r["t"], r["event"]) for r in rows if r.get("event")]
    follow = [r for r in rows if r.get("state") == "FOLLOWING"]
    detected = [r for r in follow if r.get("detected") in ("1", "True", "true")]
    reid_f = [fnum(r, "reid_score") for r in follow if fnum(r, "reid_score") is not None]
    reid_f = [v for v in reid_f if v > 0]
    reid_lost = [
        fnum(r, "reid_best")
        for r in rows
        if r.get("state") in ("REACQUIRING", "LOST") and fnum(r, "reid_best", 0) > 0
    ]
    margins = [
        fnum(r, "margin")
        for r in rows
        if r.get("state") in ("REACQUIRING", "LOST") and fnum(r, "margin") is not None
    ]
    dists = [fnum(r, "dist_m") for r in detected if fnum(r, "dist_m") is not None]
    fps = [fnum(r, "fps") for r in rows if fnum(r, "fps", 0) and fnum(r, "fps") > 0]
    gallery_max = max((int(float(r["gallery_n"])) for r in rows if r.get("gallery_n")), default=0)

    reacquired = sum(1 for _, e in events if e == "REACQUIRED")
    registered = sum(1 for _, e in events if e == "REGISTERED")
    marks = [(t, e) for t, e in events if e == "MARK"]

    print(f"\n===== {os.path.basename(path)} =====")
    print(f"frames: {len(rows)}  duration_s≈{rows[-1].get('t', '?')}")
    print(f"states: {dict(states)}")
    print(f"gallery_max: {gallery_max}")
    print(f"REGISTERED events: {registered}  REACQUIRED events: {reacquired}")
    print(f"manual MARKs: {marks}")

    if follow:
        rate = len(detected) / len(follow)
        print(f"follow_detect_rate: {rate:.3f} ({len(detected)}/{len(follow)})")
    if reid_f:
        print(
            f"reid_following: mean={np.mean(reid_f):.3f} "
            f"p10={np.percentile(reid_f, 10):.3f} min={np.min(reid_f):.3f}"
        )
    if reid_lost:
        print(
            f"reid_best_lost: mean={np.mean(reid_lost):.3f} "
            f"max={np.max(reid_lost):.3f} n={len(reid_lost)}"
        )
    if margins:
        print(f"margin_lost: mean={np.mean(margins):.3f} min={np.min(margins):.3f}")
    if dists:
        print(
            f"dist_m (tracked): mean={np.mean(dists):.2f} "
            f"min={np.min(dists):.2f} max={np.max(dists):.2f}"
        )
    if fps:
        print(f"fps: mean={np.mean(fps):.1f} min={np.min(fps):.1f}")

    print("\nPass / fail (solo, one person):")
    checks = [
        ("gallery >= 12 after register", gallery_max >= 12),
        ("follow_detect_rate > 0.90", bool(follow) and len(detected) / len(follow) > 0.90),
        ("at least one REACQUIRED if you left frame", reacquired >= 1 or not marks),
        ("reid_following mean > 0.65", bool(reid_f) and np.mean(reid_f) > 0.65),
        ("fps mean > 8", bool(fps) and np.mean(fps) > 8),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("\nHow to interpret:")
    print("  - Press M when you leave the camera, M again when you return.")
    print("  - Expect STATE→REACQUIRING near first MARK, REACQUIRED near second.")
    print("  - Low margin / AMBIGUOUS with one person = ReID noise or bad gallery.")
    print(f"\nFull path: {path}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest_csv()
    if not path or not os.path.isfile(path):
        print("No CSV found. Run zed-color.py once, then:")
        print("  python3 analyze_eval.py eval_logs/follow_eval_....csv")
        sys.exit(1)
    analyze(path)


if __name__ == "__main__":
    main()
