import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("ekf_prediction_log.csv")

if len(df) == 0:
    print("CSV is empty. Run zed2.py longer first.")
    exit()

# make time start from 0
df["time_zero"] = df["time"] - df["time"].iloc[0]

ade = df["error_1s"].mean()
max_error = df["error_1s"].max()

plt.figure(figsize=(10, 6))

# actual forward position in RED
plt.plot(
    df["time_zero"],
    df["actual_px"],
    color="red",
    linewidth=2.5,
    label="Actual forward position"
)

# predicted forward position in GREEN
plt.plot(
    df["time_zero"],
    df["pred_px_1s"],
    color="green",
    linewidth=2.5,
    linestyle="--",
    label="Predicted forward position"
)

# gray vertical lines show where prediction is wrong
for i in range(len(df)):
    plt.plot(
        [df["time_zero"].iloc[i], df["time_zero"].iloc[i]],
        [df["actual_px"].iloc[i], df["pred_px_1s"].iloc[i]],
        color="gray",
        alpha=0.35,
        linewidth=1
    )

plt.xlabel("Time (seconds)")
plt.ylabel("Forward position / ZED z (meters)")

plt.title(
    f"Actual vs Predicted Forward Position Over Time\n"
    f"ADE = {ade:.3f} m | Max Error = {max_error:.3f} m"
)

plt.legend()
plt.grid(True)
plt.show()