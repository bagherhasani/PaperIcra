# Motion Models: CTRV vs Hip-Steering

## CTRV (Constant Turn Rate and Velocity)

A standard kinematic model from robotics and autonomous driving literature.
The heading rotates at a constant rate `ω` that the EKF estimates from position observations.

**Reference:** Schubert et al. (2008), IEEE Intelligent Vehicles Symposium.
[MathWorks CTRV docs](https://www.mathworks.com/help/fusion/ug/motion-model-ctrvz.html)

```
θ_new = θ + ω·Δt          # ω comes from EKF internal state
x_new = x + v·cos(θ_new)·Δt
y_new = y + v·sin(θ_new)·Δt
```

---

## Prof's Hip-Steering Model

A biomechanics-informed model where the velocity heading is steered toward the hip orientation measured by the ZED sensor.
The idea: where the hips point is where the person is about to go.

```
θ_new = θ + b·sin(θ_hip − θ)·Δt   # θ_hip comes from ZED sensor each frame
x_new = x + v·cos(θ_new)·Δt
y_new = y + v·sin(θ_new)·Δt
```

`b` is the steering gain — how aggressively the heading corrects toward the hips.
`sin(θ_hip − θ)` is the cross product between hip and velocity unit vectors:
positive when hips point left of velocity, negative when right.

---

## The Only Difference

| | CTRV | Hip-Steering |
|---|---|---|
| **Heading driven by** | EKF-estimated `ω` (internal state) | Live `θ_hip` from ZED sensor |
| **Sensor dependency** | Position only | Position + hip orientation |
| **After prediction horizon** | Keeps spinning at same `ω` | Keeps steering toward last known hip angle |
| **Tuning knob** | None | Gain `b` |
| **Works without body tracking** | Yes | No |
