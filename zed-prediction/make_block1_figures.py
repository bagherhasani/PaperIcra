#!/usr/bin/env python3
"""Build Block-1 paper figures from locked full-clip CSVs. No video."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from eval_offline_baselines import score_static, score_cv, load_rows

RESULTS = Path(__file__).resolve().parent / "results"
FIGS = RESULTS / "block1_figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Locked paper runs (full clip, horizon 1 s)
RUNS = {
    "D-walk": {
        "hip": "ekf_prediction_log_hip_Dwalk_1s.csv",
        "ctrv": "ekf_prediction_log_ctrv_Dwalk_1s.csv",
    },
    "Straight": {
        "hip": "ekf_prediction_log_hip_straight_1s.csv",
        "ctrv": "ekf_prediction_log_ctrv_straight_1s.csv",
    },
}


def ade(path: Path) -> float:
    return float(pd.read_csv(path)["error_1s"].mean())


def build_table() -> pd.DataFrame:
    rows = []
    for scene, files in RUNS.items():
        hip_path = RESULTS / files["hip"]
        ctrv_path = RESULTS / files["ctrv"]
        hip_rows = load_rows(hip_path)
        st = score_static(hip_rows)
        cv = score_cv(hip_rows)
        rows.append(
            {
                "scene": scene,
                "Static": st["ade"],
                "CV": cv["ade"],
                "CTRV": ade(ctrv_path),
                "Hip-steering": ade(hip_path),
                "n": st["n"],
            }
        )
    df = pd.DataFrame(rows)
    out_csv = FIGS / "block1_ade_table.csv"
    df.to_csv(out_csv, index=False)
    out_txt = FIGS / "block1_ade_table.txt"
    with out_txt.open("w") as f:
        f.write("Block 1 — 1 s ADE (full clip, locked SVOs)\n")
        f.write("=" * 56 + "\n")
        for _, r in df.iterrows():
            f.write(
                f"{r['scene']:10s}  Static={r['Static']:.3f}  CV={r['CV']:.3f}  "
                f"CTRV={r['CTRV']:.3f}  Hip={r['Hip-steering']:.3f}  (n≈{int(r['n'])})\n"
            )
        f.write("=" * 56 + "\n")
        f.write("Claim: Hip-steering beats Static / CV / CTRV on these ZED clips.\n")
    print(out_txt.read_text())
    return df


def fig_bar(df: pd.DataFrame) -> None:
    models = ["Static", "CV", "CTRV", "Hip-steering"]
    colors = ["#888888", "#4C78A8", "#F58518", "#54A24B"]
    x = np.arange(len(df))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (m, c) in enumerate(zip(models, colors)):
        vals = df[m].to_numpy()
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=c)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(df["scene"])
    ax.set_ylabel("1 s ADE (m)")
    ax.set_title("Block 1 — Pedestrian prediction ADE (lower is better)")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    ax.set_ylim(0, max(df[models].to_numpy().max() * 1.25, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = FIGS / "fig_ade_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_trajectories() -> None:
    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)
    panels = [
        ("D-walk", "hip", 0, 0, "Hip-steering — D-walk"),
        ("D-walk", "ctrv", 0, 1, "CTRV — D-walk"),
        ("Straight", "hip", 1, 0, "Hip-steering — Straight"),
        ("Straight", "ctrv", 1, 1, "CTRV — Straight"),
    ]
    for scene, model, r, c, title in panels:
        ax = fig.add_subplot(gs[r, c])
        df = pd.read_csv(RESULTS / RUNS[scene][model])
        ax.plot(df["actual_py"], df["actual_px"], "r-", lw=2, label="Actual")
        ax.plot(
            df["pred_py_1s"],
            df["pred_px_1s"],
            "g--",
            lw=1.6,
            label="Pred +1 s",
        )
        ax.scatter(
            df["actual_py"].iloc[0],
            df["actual_px"].iloc[0],
            c="red",
            s=40,
            zorder=5,
        )
        ade_v = df["error_1s"].mean()
        ax.set_title(f"{title}\nADE={ade_v:.3f} m", fontsize=11)
        ax.set_xlabel("py (m)")
        ax.set_ylabel("px (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        if r == 0 and c == 0:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Block 1 — Actual vs 1 s prediction paths", fontsize=13, fontweight="bold")
    path = FIGS / "fig_trajectories.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_error_time() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, scene in zip(axes, ["D-walk", "Straight"]):
        for model, color, label in [
            ("hip", "#54A24B", "Hip-steering"),
            ("ctrv", "#F58518", "CTRV"),
        ]:
            df = pd.read_csv(RESULTS / RUNS[scene][model])
            t = df["time"] - df["time"].iloc[0]
            roll = df["error_1s"].rolling(10, min_periods=1).mean()
            ax.plot(t, roll, color=color, lw=2, label=label)
        ax.set_title(scene)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("1 s error (m)" if scene == "D-walk" else "")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Block 1 — Rolling 1 s prediction error", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIGS / "fig_error_over_time.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_overlay_dwalk() -> None:
    """Single strong paper figure: D-walk actual + hip/CTRV predictions."""
    hip = pd.read_csv(RESULTS / RUNS["D-walk"]["hip"])
    ctrv = pd.read_csv(RESULTS / RUNS["D-walk"]["ctrv"])
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.plot(hip["actual_py"], hip["actual_px"], "k-", lw=2.4, label="Actual path")
    ax.plot(
        hip["pred_py_1s"],
        hip["pred_px_1s"],
        color="#54A24B",
        ls="--",
        lw=2,
        label=f"Hip-steering (ADE={hip['error_1s'].mean():.3f} m)",
    )
    ax.plot(
        ctrv["pred_py_1s"],
        ctrv["pred_px_1s"],
        color="#F58518",
        ls=":",
        lw=2,
        label=f"CTRV (ADE={ctrv['error_1s'].mean():.3f} m)",
    )
    ax.scatter(hip["actual_py"].iloc[0], hip["actual_px"].iloc[0], c="k", s=50, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Lateral py (m)")
    ax.set_ylabel("Forward px (m)")
    ax.set_title("D-walk — 1 s prediction overlay")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGS / "fig_dwalk_overlay.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    df = build_table()
    fig_bar(df)
    fig_trajectories()
    fig_error_time()
    fig_overlay_dwalk()
    print(f"\nAll Block 1 figures → {FIGS}")
