# ICRA Paper Assets — Pedestrian Motion Prediction

## One-sentence claim

Nav2 treats people as **static**. We use a ZED camera + EKF to predict where a person will be in **1 second**, so the robot can plan around **where they will walk**, not only where they are now.

---

## What this folder contains

| File | What it is |
|------|------------|
| `ekf_prediction_log_*_1s.csv` | Per-frame prediction errors |
| `ekf_eval_dashboard_*_1s.png` | Plots for each run |

**Horizon:** always **1.0 second** ahead.  
**Sensor:** ZED body tracking (position + hip heading).  
**Code:** `ekf_zed.py` + `zed2.py`.

---

## Models used in these results (paper table)

Only **two** models:

1. **CTRV** — standard Constant Turn Rate and Velocity (baseline)
2. **Hip-steering** — our / professor model (novelty)

Same EKF, same measurements, same SVO videos. Only the **motion model** changes.

---

## Extended Kalman Filter (shared by both)

### State

\[
\mathbf{x} = [p_x,\ p_y,\ v,\ \theta,\ \omega]^\top
\]

| Symbol | Meaning |
|--------|---------|
| \(p_x, p_y\) | Position (m) |
| \(v\) | Speed (m/s) |
| \(\theta\) | Velocity heading (rad) |
| \(\omega\) | Heading rate (rad/s) |

### Measurement (from ZED)

\[
\mathbf{z} = [p_x^{\text{zed}},\ p_y^{\text{zed}},\ \theta_{\text{hip}}]^\top
\]

Hip heading \(\theta_{\text{hip}}\) comes from the body skeleton (hips).

### Each frame

1. **Predict:** \(\mathbf{x} \leftarrow f(\mathbf{x})\), \(\mathbf{P} \leftarrow \mathbf{F}\mathbf{P}\mathbf{F}^\top + \mathbf{Q}\)
2. **Update:** correct with \(\mathbf{z}\) using measurement Jacobian \(\mathbf{H}\)
3. **Forecast:** open-loop integrate \(f(\cdot)\) for **1.0 s** (no new measurements) → predicted position

### Measurement model (same for both)

\[
h(\mathbf{x}) = [p_x,\ p_y,\ \theta]^\top
\]

\[
\mathbf{H} =
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0
\end{bmatrix}
\]

---

## Model A — CTRV (baseline)

Heading turns at rate \(\omega\) from the **EKF state**:

\[
\theta^{+} = \theta + \omega\,\Delta t
\]

If \(|\omega|\) is large enough (turning):

\[
\begin{aligned}
p_x^{+} &= p_x + \frac{v}{\omega}\bigl(\sin\theta^{+} - \sin\theta\bigr) \\
p_y^{+} &= p_y + \frac{v}{\omega}\bigl(-\cos\theta^{+} + \cos\theta\bigr)
\end{aligned}
\]

If \(\omega \approx 0\) (straight):

\[
\begin{aligned}
p_x^{+} &= p_x + v\cos\theta\,\Delta t \\
p_y^{+} &= p_y + v\sin\theta\,\Delta t
\end{aligned}
\]

Speed \(v\) and rate \(\omega\) stay constant in the predict step.

**In words:** “Keep going at current speed and current turn rate.” Blind to hips after the last update for the open-loop 1 s forecast (uses estimated \(\omega\)).

---

## Model B — Hip-steering (our model)

Heading is pulled toward the **hip orientation** \(\theta_{\text{hip}}\):

\[
\Delta\theta = b \cdot \sin(\theta_{\text{hip}} - \theta) \cdot \Delta t
\]

\[
\theta^{+} = \theta + \Delta\theta
\]

\[
\begin{aligned}
p_x^{+} &= p_x + v\cos\theta^{+}\,\Delta t \\
p_y^{+} &= p_y + v\sin\theta^{+}\,\Delta t
\end{aligned}
\]

Gain used in these runs: \(b = 3.0\).

**In words:** “Steer velocity heading toward where the hips point.”  
\(\sin(\theta_{\text{hip}}-\theta) > 0\) → hips left of velocity → turn left (and vice versa).

---

## The only difference

| | CTRV | Hip-steering |
|---|---|---|
| What changes heading | EKF state \(\omega\) | Sensor \(\theta_{\text{hip}}\) |
| Novelty | No (standard) | Yes (hip intent) |

Position step is still \(v\) along the (new) heading. Same filter, same \(\mathbf{z}\).

---

## How we score

For each frame with a person:

- Predict position **1.0 s ahead**
- Later compare to where the person **actually** was
- Error = Euclidean distance (m)

| Metric | Meaning |
|--------|---------|
| **ADE** | Average error over the run |
| **FDE** | Error on the last sample |
| **Max / Median** | Worst / middle error |

---

## Results in this folder (horizon = 1.0 s)

| Scenario | Model | ADE (m) | FDE (m) | Max | Median | Samples |
|----------|--------|--------:|--------:|----:|-------:|--------:|
| D-walk | CTRV | 0.203 | 0.502 | 0.843 | 0.157 | 537 |
| D-walk | **Hip-steering** | **0.192** | 0.528 | 0.844 | 0.175 | 537 |
| Straight | CTRV | 0.206 | 0.111 | 0.655 | 0.153 | 232 |
| Straight | **Hip-steering** | **0.198** | **0.093** | 0.649 | 0.153 | 232 |

**Takeaway:** Hip-steering is slightly better ADE on both walks at 1 s. Next paper piece: compare against a **static** baseline (predict = stay put), then Nav2 costmap experiment.

---

## Not in this paper table (code exists, not used here)

- `profidea2` — position correction with \(\alpha, \beta\)
- `profidea3` — hip-steering + slow-down on turns

Keep them out of the main ICRA comparison unless needed later.

---

## Pipeline (simple)

```
ZED body tracking
    → (px, py, θ_hip)
        → EKF predict + update
            → open-loop predict 1 s ahead
                → ADE vs future ground truth
```

Later (not these files yet): publish that 1 s pose into Nav2 as predicted occupancy.
