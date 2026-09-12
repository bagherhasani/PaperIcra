# ICRA Paper Assets — Pedestrian Motion Prediction

## One-sentence claim

Nav2 treats people as static. We use a ZED camera + EKF to predict where a person will be in 1 second, so the robot can plan around where they will walk, not only where they are now.

---

## What this folder contains

| File | What it is |
|------|------------|
| `ekf_prediction_log_*_1s.csv` | Per-frame prediction errors |
| `ekf_eval_dashboard_*_1s.png` | Plots for each run |

- Horizon: always **1.0 second** ahead
- Sensor: ZED body tracking (position + hip heading)
- Code: `ekf_zed.py` + `zed2.py`

---

## Models used in these results

Only **two** models:

1. **CTRV** — Constant Turn Rate and Velocity (baseline)
2. **Hip-steering** — our / professor model (novelty)

Same EKF, same measurements, same SVO videos. Only the motion model changes.

---

## Extended Kalman Filter (shared by both)

### State vector

```
x = [px, py, v, theta, omega]
```

| Symbol | Meaning |
|--------|---------|
| px, py | Position (meters) |
| v | Speed (m/s) |
| theta | Velocity heading (radians) |
| omega | Heading rate (rad/s) |

### Measurement from ZED

```
z = [px_zed, py_zed, theta_hip]
```

`theta_hip` = hip orientation from the body skeleton.

### Each frame (EKF loop)

```
1. Predict:  x = f(x)
             P = F P F^T + Q

2. Update:   correct x with measurement z
             (using measurement Jacobian H)

3. Forecast: integrate f(x) open-loop for 1.0 second
             → predicted (px, py) for scoring
```

### Measurement model (same for both)

```
h(x) = [px, py, theta]

H = [[1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0]]
```

---

## Model A — CTRV (baseline)

Heading turns using omega from the EKF state:

```
theta_new = theta + omega * dt
```

**If turning** (|omega| big enough):

```
px_new = px + (v / omega) * (sin(theta_new) - sin(theta))
py_new = py + (v / omega) * (-cos(theta_new) + cos(theta))
```

**If straight** (omega ≈ 0):

```
px_new = px + v * cos(theta) * dt
py_new = py + v * sin(theta) * dt
```

Speed v and rate omega stay constant in the predict step.

In words: keep going at current speed and current turn rate.

---

## Model B — Hip-steering (our model)

Heading is pulled toward the hip orientation:

```
d_theta   = b * sin(theta_hip - theta) * dt
theta_new = theta + d_theta

px_new = px + v * cos(theta_new) * dt
py_new = py + v * sin(theta_new) * dt
```

Gain used in these runs: **b = 3.0**

In words: steer velocity heading toward where the hips point.

- `sin(theta_hip - theta) > 0` → hips left of velocity → turn left
- `sin(theta_hip - theta) < 0` → hips right of velocity → turn right

---

## The only difference

| | CTRV | Hip-steering |
|---|---|---|
| What changes heading | EKF state `omega` | Sensor `theta_hip` |
| Novelty | No (standard) | Yes (hip intent) |

Position step is still speed along the (new) heading. Same filter, same measurement z.

---

## How we score

For each frame:

1. Predict position 1.0 s ahead
2. Later compare to where the person actually was
3. Error = Euclidean distance in meters

| Metric | Meaning |
|--------|---------|
| ADE | Average error over the run |
| FDE | Error on the last sample |
| Max / Median | Worst / middle error |

---

## Results (horizon = 1.0 s)

| Scenario | Model | ADE (m) | FDE (m) | Max | Median | Samples |
|----------|--------|--------:|--------:|----:|-------:|--------:|
| D-walk | CTRV | 0.203 | 0.502 | 0.843 | 0.157 | 537 |
| D-walk | **Hip-steering** | **0.192** | 0.528 | 0.844 | 0.175 | 537 |
| Straight | CTRV | 0.206 | 0.111 | 0.655 | 0.153 | 232 |
| Straight | **Hip-steering** | **0.198** | **0.093** | 0.649 | 0.153 | 232 |

Takeaway: Hip-steering has slightly better ADE on both walks at 1 s.

---

## Not in this paper table

Code also has `profidea2` and `profidea3`. They are **not** used in these assets.

---

## Pipeline

```
ZED body tracking
  → (px, py, theta_hip)
    → EKF predict + update
      → open-loop predict 1 s ahead
        → ADE vs future ground truth
```

---

## Next steps (do in this order)

### Done
- [x] Lock horizon = 1.0 s
- [x] Run CTRV vs Hip on D-walk + straight
- [x] Save results in this folder

### Step 1 — Static baseline (do next)
**What:** Add a third “model”: predict future = current position (person never moves).

**Why:** That is what Nav2 costmap assumes. Hip/CTRV should beat it by a lot.

**How:**
1. Add `MOTION_MODEL = "static"` (future px,py = current px,py)
2. Re-run D-walk + straight at 1 s
3. Add one row to the results table: static / CTRV / hip

### Step 2 — Nav2 experiment (only after Step 1)
**What:** Same robot stack, two modes:
- inflate costmap at person **now** (static)
- inflate costmap at person **+1 s** (hip predict)

**How it works:**
```
ZED → EKF hip → predicted pose at +1 s
                    ↓
         publish to Nav2 costmap
                    ↓
         robot plans around future position
```

**Measure:** time-to-goal, time stuck, success / near-miss — predicted vs static.

### Do not do yet
- More motion models (idea2/idea3)
- 4.8 s horizon
- Full building navigation / crowds

