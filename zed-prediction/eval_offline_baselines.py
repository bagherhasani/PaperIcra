#!/usr/bin/env python3
"""Offline Static and CV scores from any prediction CSV.

Works on today's paper_1s logs and on whatever you record tomorrow.
Does not need ZED or ROS.

Static: future = current actual position.
CV: future = current actual + last measured step, held for HORIZON seconds.
Logged-model: the error_1s column already in the file.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


HORIZON_S = 1.0
PAIR_TOL_S = 0.08


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "t": float(r["time"]),
                    "ax": float(r["actual_px"]),
                    "ay": float(r["actual_py"]),
                    "e": float(r["error_1s"]),
                }
            )
    return rows


def pair_future(rows: list[dict], i: int) -> dict | None:
    target = rows[i]["t"] + HORIZON_S
    best = None
    best_dt = PAIR_TOL_S
    for j in range(i + 1, len(rows)):
        dt = abs(rows[j]["t"] - target)
        if dt <= best_dt:
            best = rows[j]
            best_dt = dt
        if rows[j]["t"] > target + PAIR_TOL_S:
            break
    return best


def summarize(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0, "ade": float("nan"), "fde": float("nan"),
                "max": float("nan"), "median": float("nan")}
    s = sorted(errors)
    mid = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    return {
        "n": len(errors),
        "ade": sum(errors) / len(errors),
        "fde": errors[-1],
        "max": max(errors),
        "median": mid,
    }


def score_logged(rows: list[dict]) -> dict:
    return summarize([r["e"] for r in rows])


def score_static(rows: list[dict]) -> dict:
    errs = []
    for i, r in enumerate(rows):
        fut = pair_future(rows, i)
        if fut is None:
            continue
        errs.append(math.hypot(fut["ax"] - r["ax"], fut["ay"] - r["ay"]))
    return summarize(errs)


def score_cv(rows: list[dict]) -> dict:
    errs = []
    for i in range(1, len(rows)):
        dt = rows[i]["t"] - rows[i - 1]["t"]
        if dt <= 1e-4 or dt > 0.5:
            continue
        fut = pair_future(rows, i)
        if fut is None:
            continue
        vx = (rows[i]["ax"] - rows[i - 1]["ax"]) / dt
        vy = (rows[i]["ay"] - rows[i - 1]["ay"]) / dt
        px = rows[i]["ax"] + vx * HORIZON_S
        py = rows[i]["ay"] + vy * HORIZON_S
        errs.append(math.hypot(fut["ax"] - px, fut["ay"] - py))
    return summarize(errs)


def fmt(name: str, m: dict) -> str:
    if m["n"] == 0:
        return f"{name:14s}  n=0"
    return (
        f"{name:14s}  ADE={m['ade']:.3f}  FDE={m['fde']:.3f}  "
        f"max={m['max']:.3f}  med={m['median']:.3f}  n={m['n']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csvs",
        nargs="*",
        type=Path,
        help="Prediction CSVs. Default: results/paper_1s/*.csv",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    files = args.csvs or sorted((root / "results" / "paper_1s").glob("ekf_prediction_log_*_1s.csv"))
    if not files:
        raise SystemExit("No CSVs found.")

    print(f"Horizon {HORIZON_S:.1f}s  pair tol {PAIR_TOL_S:.2f}s")
    for path in files:
        rows = load_rows(path)
        print(f"\n{path.name}  ({len(rows)} logged frames)")
        print("  " + fmt("logged-model", score_logged(rows)))
        print("  " + fmt("static", score_static(rows)))
        print("  " + fmt("cv", score_cv(rows)))


if __name__ == "__main__":
    main()
