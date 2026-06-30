import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CSV_PATH = "ekf_prediction_log.csv"

if not os.path.exists(CSV_PATH):
    print(f"CSV not found: {CSV_PATH}  — run zed2.py first.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

if len(df) == 0:
    print("CSV is empty. Run zed2.py longer first.")
    sys.exit(1)

# Time relative to start
df["time_zero"] = df["time"] - df["time"].iloc[0]

# Key metrics
ade   = df["error_1s"].mean()
fde   = df["error_1s"].iloc[-1]
max_e = df["error_1s"].max()
med_e = df["error_1s"].median()

# Rolling mean error (window = 10 samples)
df["rolling_error"] = df["error_1s"].rolling(window=10, min_periods=1).mean()

print("=" * 45)
print(f"  ADE  (avg displacement error) : {ade:.3f} m")
print(f"  FDE  (final displacement error): {fde:.3f} m")
print(f"  Max error                      : {max_e:.3f} m")
print(f"  Median error                   : {med_e:.3f} m")
print(f"  Samples evaluated              : {len(df)}")
print("=" * 45)

# ── Layout: 3 rows × 2 cols ────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    f"EKF Pedestrian Prediction Evaluation\n"
    f"ADE={ade:.3f} m  |  FDE={fde:.3f} m  |  Max={max_e:.3f} m  |  Median={med_e:.3f} m",
    fontsize=14,
    fontweight="bold"
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

# ── Panel 1: 2D top-down trajectory ───────────────────────
ax1 = fig.add_subplot(gs[0, :])   # spans both columns

ax1.plot(
    df["actual_py"], df["actual_px"],
    color="red", linewidth=2, label="Actual path"
)
ax1.plot(
    df["pred_py_1s"], df["pred_px_1s"],
    color="green", linewidth=2, linestyle="--", label="Predicted path"
)

# Connect actual→predicted at each sample with a thin gray line
for i in range(len(df)):
    ax1.plot(
        [df["actual_py"].iloc[i], df["pred_py_1s"].iloc[i]],
        [df["actual_px"].iloc[i], df["pred_px_1s"].iloc[i]],
        color="gray", alpha=0.3, linewidth=0.8
    )

# Mark start and end
ax1.scatter(df["actual_py"].iloc[0],  df["actual_px"].iloc[0],
            color="red",   s=80, zorder=5, label="Start (actual)")
ax1.scatter(df["actual_py"].iloc[-1], df["actual_px"].iloc[-1],
            color="darkred", s=80, marker="*", zorder=5, label="End (actual)")

ax1.set_xlabel("Lateral / ZED x (m)")
ax1.set_ylabel("Forward / ZED z (m)")
ax1.set_title("2D Top-Down: Actual vs Predicted Trajectory")
ax1.legend(loc="best", fontsize=9)
ax1.grid(True)
ax1.set_aspect("equal", adjustable="datalim")

# ── Panel 2: 2D error over time ───────────────────────────
ax2 = fig.add_subplot(gs[1, 0])

ax2.fill_between(
    df["time_zero"], df["error_1s"],
    alpha=0.25, color="orange"
)
ax2.plot(
    df["time_zero"], df["error_1s"],
    color="orange", linewidth=1.5, label="Error per sample"
)
ax2.plot(
    df["time_zero"], df["rolling_error"],
    color="red", linewidth=2.5, linestyle="--", label="Rolling mean (10 samples)"
)
ax2.axhline(ade, color="purple", linewidth=1.5, linestyle=":", label=f"ADE = {ade:.3f} m")

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("2D Euclidean error (m)")
ax2.set_title("Prediction Error Over Time")
ax2.legend(fontsize=9)
ax2.grid(True)

# ── Panel 3: Forward position ──────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

ax3.plot(df["time_zero"], df["actual_px"],
         color="red",   linewidth=2, label="Actual forward")
ax3.plot(df["time_zero"], df["pred_px_1s"],
         color="green", linewidth=2, linestyle="--", label="Predicted forward")

for i in range(len(df)):
    ax3.plot(
        [df["time_zero"].iloc[i], df["time_zero"].iloc[i]],
        [df["actual_px"].iloc[i],  df["pred_px_1s"].iloc[i]],
        color="gray", alpha=0.3, linewidth=0.8
    )

ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Forward / ZED z (m)")
ax3.set_title("Forward Position: Actual vs Predicted")
ax3.legend(fontsize=9)
ax3.grid(True)

# ── Panel 4: Lateral position ──────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])

ax4.plot(df["time_zero"], df["actual_py"],
         color="blue",   linewidth=2, label="Actual lateral")
ax4.plot(df["time_zero"], df["pred_py_1s"],
         color="cyan",   linewidth=2, linestyle="--", label="Predicted lateral")

for i in range(len(df)):
    ax4.plot(
        [df["time_zero"].iloc[i], df["time_zero"].iloc[i]],
        [df["actual_py"].iloc[i],  df["pred_py_1s"].iloc[i]],
        color="gray", alpha=0.3, linewidth=0.8
    )

ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Lateral / ZED x (m)")
ax4.set_title("Lateral Position: Actual vs Predicted")
ax4.legend(fontsize=9)
ax4.grid(True)

# ── Panel 5: Speed ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])

ax5.plot(df["time_zero"], df["speed"],
         color="green", linewidth=2, label="EKF speed")

ax5_twin = ax5.twinx()
ax5_twin.plot(df["time_zero"], np.degrees(df["heading_rate"]),
              color="orange", linewidth=1.5, linestyle="--", label="Heading rate (°/s)")
ax5_twin.set_ylabel("Heading rate (°/s)", color="orange")
ax5_twin.tick_params(axis="y", labelcolor="orange")

ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Speed (m/s)", color="green")
ax5.tick_params(axis="y", labelcolor="green")
ax5.set_title("EKF Speed and Heading Rate")

lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_twin.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax5.grid(True)

plt.savefig("ekf_eval_dashboard.png", dpi=150, bbox_inches="tight")
print("Saved: ekf_eval_dashboard.png")
plt.show()
