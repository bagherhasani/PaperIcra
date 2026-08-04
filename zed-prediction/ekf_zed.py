from filterpy.kalman import ExtendedKalmanFilter
import numpy as np


class Ekf:
    def __init__(self, initial_px, initial_py, initial_speed,
                 initial_heading, initial_heading_rate, dt=0.033,
                 motion_model="ctrv", steering_gain_b=3.0,
                 profidea2_alpha=0.05, profidea2_beta=1.5,
                 profidea3_k=0.35):

        self.initial_px = initial_px
        self.initial_py = initial_py
        self.initial_speed = initial_speed
        self.initial_heading = initial_heading
        self.initial_heading_rate = initial_heading_rate
        self.dt = dt
        self.motion_model_type = motion_model
        self.steering_gain_b = steering_gain_b
        self.hip_heading = initial_heading

        # Prof idea 2 params
        self.profidea2_alpha = profidea2_alpha
        self.profidea2_beta = profidea2_beta

        # Prof idea 3 params:
        #   k — deceleration gain. k=0 = profidea1, k=0.35 = 35% speed reduction at 90° turn
        self.profidea3_k = profidea3_k

        # Initialize EKF
        # state = [px, py, speed, heading, heading_rate]
        # measurement = [px_zed, py_zed, hip_heading]
        self.ekf = ExtendedKalmanFilter(dim_x=5, dim_z=3)

        # Define X state vector
        self.ekf.x = np.array([
            self.initial_px,
            self.initial_py,
            self.initial_speed,
            self.initial_heading,
            self.initial_heading_rate
        ], dtype=float)

        # Define P: initial uncertainty
        self.ekf.P = np.array([
            [1,   0,   0,   0,   0],
            [0,   1,   0,   0,   0],
            [0,   0, 100,   0,   0],
            [0,   0,   0,  10,   0],
            [0,   0,   0,   0,   5]
        ], dtype=float)

        # Define R: measurement noise
        # measurement = [px_zed, py_zed, hip_heading]
        self.ekf.R = np.array([
            [0.1, 0,   0],
            [0,   0.1, 0],
            [0,   0,   1.0]
        ], dtype=float)

        # Define Q: process noise
        # model uncertainty for [px, py, speed, heading, heading_rate]
        self.ekf.Q = np.array([
            [0.01, 0,    0,   0,    0],
            [0,    0.01, 0,   0,    0],
            [0,    0,    0.1, 0,    0],
            [0,    0,    0,   0.05, 0],
            [0,    0,    0,   0,    0.01]
        ], dtype=float)

    def normalize_angle(self, angle):
        """
        Keep angle between -pi and +pi.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def update_dt(self, dt):
        """
        Update dt for each ZED frame.
        """
        self.dt = dt

    def hip_steering_dot(self, heading, hip_heading):
        """
        v_hat_perp · h_hat with v_hat_perp = left normal to velocity heading.

        Equals sin(hip_heading - heading): positive when hip is left of velocity.
        """
        return np.sin(hip_heading - heading)

    def motion_model_hip_steering(self, x):
        """
        Prof motion model:
            theta_v^{k+1} = theta_v^k + b * (v_hat_perp · h_hat)

        Position uses straight-line step with the updated heading.
        State[4] stores the implied turn rate for logging/plotting.
        """
        px = x[0]
        py = x[1]
        speed = x[2]
        heading = x[3]

        hip = self.hip_heading
        dt = self.dt
        b = self.steering_gain_b

        steer = b * self.hip_steering_dot(heading, hip) * dt
        heading_new = self.normalize_angle(heading + steer)
        heading_rate_new = steer / dt if dt > 0 else 0.0

        px_new = px + speed * np.cos(heading_new) * dt
        py_new = py + speed * np.sin(heading_new) * dt

        return np.array([
            px_new,
            py_new,
            speed,
            heading_new,
            heading_rate_new
        ], dtype=float)

    def F_jacobian_hip_steering(self, x):
        """
        Jacobian of the hip-steering motion model.
        Hip heading is treated as constant during one predict step.
        """
        speed = x[2]
        heading = x[3]
        hip = self.hip_heading
        dt = self.dt
        b = self.steering_gain_b

        d_heading_new_d_heading = 1.0 - b * dt * np.cos(hip - heading)
        heading_new = self.normalize_angle(
            heading + b * self.hip_steering_dot(heading, hip) * dt
        )

        d_px_d_v = np.cos(heading_new) * dt
        d_px_d_psi = -speed * np.sin(heading_new) * dt * d_heading_new_d_heading
        d_py_d_v = np.sin(heading_new) * dt
        d_py_d_psi = speed * np.cos(heading_new) * dt * d_heading_new_d_heading

        d_omega_d_psi = -b * np.cos(hip - heading)

        F = np.array([
            [1, 0, d_px_d_v, d_px_d_psi, 0],
            [0, 1, d_py_d_v, d_py_d_psi, 0],
            [0, 0, 1,        0,           0],
            [0, 0, 0,        d_heading_new_d_heading, 0],
            [0, 0, 0,        d_omega_d_psi,           0]
        ], dtype=float)

        return F

    def motion_model_profidea3(self, x):
        """
        Prof idea 3 — hip-steering with turn-coupled speed reduction.

        Humans slow down when turning: the larger the angle between hip heading
        and velocity heading, the more they decelerate.

            Δθ     = b · sin(θ_hip − θ) · Δt        (same as profidea1)
            turn   = |sin(θ_hip − θ)|                 (0=straight, 1=90° turn)
            v_pred = v · (1 − k · turn)               (speed reduced by turn magnitude)
            x_new  = x + v_pred · cos(θ_new) · Δt
            y_new  = y + v_pred · sin(θ_new) · Δt

        k = profidea3_k (deceleration gain). k=0 collapses to profidea1.
        Typical range: 0.2–0.5 (30° turn reduces speed 10–25%).
        """
        px      = x[0]
        py      = x[1]
        speed   = x[2]
        heading = x[3]

        hip = self.hip_heading
        dt  = self.dt
        b   = self.steering_gain_b
        k   = self.profidea3_k

        steer       = b * self.hip_steering_dot(heading, hip) * dt
        heading_new = self.normalize_angle(heading + steer)
        heading_rate_new = steer / dt if dt > 0 else 0.0

        turn   = abs(np.sin(hip - heading))
        v_pred = max(speed * (1.0 - k * turn), 0.0)

        px_new = px + v_pred * np.cos(heading_new) * dt
        py_new = py + v_pred * np.sin(heading_new) * dt

        return np.array([
            px_new,
            py_new,
            speed,       # EKF state speed unchanged — only prediction uses v_pred
            heading_new,
            heading_rate_new
        ], dtype=float)

    def F_jacobian_profidea3(self, x):
        """
        Jacobian for profidea3.
        Same structure as hip-steering Jacobian but with v_pred instead of v
        in the position partial derivatives.
        """
        speed   = x[2]
        heading = x[3]
        hip     = self.hip_heading
        dt      = self.dt
        b       = self.steering_gain_b
        k       = self.profidea3_k

        turn   = abs(np.sin(hip - heading))
        v_pred = max(speed * (1.0 - k * turn), 0.0)

        d_heading_new_d_heading = 1.0 - b * dt * np.cos(hip - heading)
        heading_new = self.normalize_angle(
            heading + b * self.hip_steering_dot(heading, hip) * dt
        )

        d_px_d_v    = np.cos(heading_new) * dt * (1.0 - k * turn)
        d_px_d_psi  = -v_pred * np.sin(heading_new) * dt * d_heading_new_d_heading
        d_py_d_v    = np.sin(heading_new) * dt * (1.0 - k * turn)
        d_py_d_psi  =  v_pred * np.cos(heading_new) * dt * d_heading_new_d_heading

        d_omega_d_psi = -b * np.cos(hip - heading)

        F = np.array([
            [1, 0, d_px_d_v, d_px_d_psi, 0],
            [0, 1, d_py_d_v, d_py_d_psi, 0],
            [0, 0, 1,        0,           0],
            [0, 0, 0,        d_heading_new_d_heading, 0],
            [0, 0, 0,        d_omega_d_psi,           0]
        ], dtype=float)

        return F

    def motion_model_profidea2(self, x):
        """
        Prof idea 2 — position-correction model.

        Step 1: predict position via straight-line CTRV step (r_p).
        Step 2: add correction vector Δr' = α·v̂ + C·n̂⊥
                where C = β·(1 − v̂·n̂)  and n̂⊥ is n̂ rotated 90° left.

        v̂  = unit vector of velocity heading (from EKF state)
        n̂  = unit vector of hip orientation (from ZED sensor)
        n̂⊥ = left-perpendicular to n̂: (-sin θ_hip, cos θ_hip)

        C = 0 when velocity and hips are aligned (no lateral correction needed).
        C > 0 when misaligned — pushes position toward where hips are pointing.
        """
        px      = x[0]
        py      = x[1]
        speed   = x[2]
        heading = x[3]
        heading_rate = x[4]
        dt = self.dt

        # --- Step 1: standard CTRV prediction ---
        heading_new = self.normalize_angle(heading + heading_rate * dt)
        if abs(heading_rate) > 1e-5:
            px_p = px + (speed / heading_rate) * (np.sin(heading_new) - np.sin(heading))
            py_p = py + (speed / heading_rate) * (-np.cos(heading_new) + np.cos(heading))
        else:
            px_p = px + speed * np.cos(heading) * dt
            py_p = py + speed * np.sin(heading) * dt

        # --- Step 2: position correction (scaled by dt = one EKF step) ---
        # Only apply when moving; correction is a rate so multiply by dt.
        if speed > 0.05:
            theta_h    = self.hip_heading
            v_hat      = np.array([np.cos(heading_new), np.sin(heading_new)])
            n_hat      = np.array([np.cos(theta_h), np.sin(theta_h)])
            n_hat_perp = np.array([-np.sin(theta_h), np.cos(theta_h)])
            C          = self.profidea2_beta * (1.0 - np.dot(v_hat, n_hat))
            delta_r    = (self.profidea2_alpha * v_hat + C * n_hat_perp) * dt
            px_p += delta_r[0]
            py_p += delta_r[1]

        return np.array([
            px_p,
            py_p,
            speed,
            heading_new,
            heading_rate
        ], dtype=float)

    def F_jacobian_profidea2(self, x):
        """
        Jacobian for prof idea 2.
        The correction Δr' depends on heading through v̂ and the CTRV step,
        but not on px/py directly — so rows 0,1 carry the same heading
        partials as CTRV plus the correction partials.
        We approximate by using the CTRV Jacobian; the correction is small
        and its heading partial is O(α, β·dt), acceptable for EKF linearisation.
        """
        speed        = x[2]
        heading      = x[3]
        heading_rate = x[4]

        heading_new = self.normalize_angle(heading + heading_rate * self.dt)

        if abs(heading_rate) > 1e-5:
            w  = heading_rate
            dt = self.dt
            d_px_d_v     =  (np.sin(heading_new) - np.sin(heading)) / w
            d_px_d_psi   =  (speed / w) * (np.cos(heading_new) - np.cos(heading))
            d_px_d_omega = -(speed / w**2) * (np.sin(heading_new) - np.sin(heading)) \
                           + (speed / w) * np.cos(heading_new) * dt
            d_py_d_v     =  (-np.cos(heading_new) + np.cos(heading)) / w
            d_py_d_psi   =  (speed / w) * (np.sin(heading_new) - np.sin(heading))
            d_py_d_omega =  (speed / w**2) * (np.cos(heading_new) - np.cos(heading)) \
                           + (speed / w) * np.sin(heading_new) * dt
        else:
            dt = self.dt
            d_px_d_v     =  np.cos(heading) * dt
            d_px_d_psi   = -speed * np.sin(heading) * dt
            d_px_d_omega =  0.0
            d_py_d_v     =  np.sin(heading) * dt
            d_py_d_psi   =  speed * np.cos(heading) * dt
            d_py_d_omega =  0.0

        F = np.array([
            [1, 0, d_px_d_v,  d_px_d_psi,  d_px_d_omega],
            [0, 1, d_py_d_v,  d_py_d_psi,  d_py_d_omega],
            [0, 0, 1,         0,            0           ],
            [0, 0, 0,         1,            self.dt     ],
            [0, 0, 0,         0,            1           ]
        ], dtype=float)

        return F

    def motion_model(self, x):
        """
        Nonlinear motion model — closed-form CTRV (standard).

        state = [px, py, speed, heading, heading_rate]

        When heading_rate is non-negligible, the exact arc integral is:
            px_new = px + (v/omega) * ( sin(heading + omega*dt) - sin(heading) )
            py_new = py + (v/omega) * (-cos(heading + omega*dt) + cos(heading) )
        When heading_rate ~ 0, fall back to straight-line motion to avoid division by zero.
        """
        if self.motion_model_type == "hip_steering":
            return self.motion_model_hip_steering(x)
        if self.motion_model_type == "profidea3":
            return self.motion_model_profidea3(x)
        if self.motion_model_type == "profidea2":
            return self.motion_model_profidea2(x)

        px           = x[0]
        py           = x[1]
        speed        = x[2]
        heading      = x[3]
        heading_rate = x[4]

        heading_new = self.normalize_angle(heading + heading_rate * self.dt)

        if abs(heading_rate) > 1e-5:
            px_new = px + (speed / heading_rate) * ( np.sin(heading_new) - np.sin(heading))
            py_new = py + (speed / heading_rate) * (-np.cos(heading_new) + np.cos(heading))
        else:
            px_new = px + speed * np.cos(heading) * self.dt
            py_new = py + speed * np.sin(heading) * self.dt

        return np.array([
            px_new,
            py_new,
            speed,
            heading_new,
            heading_rate
        ], dtype=float)

    def F_jacobian(self, x):
        """
        Jacobian of the closed-form CTRV motion model.

        Matches the motion_model above exactly.
        When heading_rate ~ 0, uses the straight-line Jacobian.
        """
        if self.motion_model_type == "hip_steering":
            return self.F_jacobian_hip_steering(x)
        if self.motion_model_type == "profidea3":
            return self.F_jacobian_profidea3(x)
        if self.motion_model_type == "profidea2":
            return self.F_jacobian_profidea2(x)

        speed        = x[2]
        heading      = x[3]
        heading_rate = x[4]

        heading_new = self.normalize_angle(heading + heading_rate * self.dt)

        if abs(heading_rate) > 1e-5:
            w  = heading_rate
            dt = self.dt

            # Exact partial derivatives of the closed-form arc integral
            d_px_d_v     =  (np.sin(heading_new) - np.sin(heading)) / w
            d_px_d_psi   =  (speed / w) * ( np.cos(heading_new) - np.cos(heading))
            d_px_d_omega = -(speed / w**2) * (np.sin(heading_new) - np.sin(heading)) \
                           + (speed / w) * np.cos(heading_new) * dt

            d_py_d_v     =  (-np.cos(heading_new) + np.cos(heading)) / w
            d_py_d_psi   =  (speed / w) * ( np.sin(heading_new) - np.sin(heading))
            d_py_d_omega =  (speed / w**2) * (np.cos(heading_new) - np.cos(heading)) \
                           + (speed / w) * np.sin(heading_new) * dt
        else:
            dt = self.dt
            d_px_d_v     =  np.cos(heading) * dt
            d_px_d_psi   = -speed * np.sin(heading) * dt
            d_px_d_omega =  0.0

            d_py_d_v     =  np.sin(heading) * dt
            d_py_d_psi   =  speed * np.cos(heading) * dt
            d_py_d_omega =  0.0

        F = np.array([
            [1, 0, d_px_d_v,  d_px_d_psi,  d_px_d_omega],
            [0, 1, d_py_d_v,  d_py_d_psi,  d_py_d_omega],
            [0, 0, 1,         0,            0           ],
            [0, 0, 0,         1,            self.dt     ],
            [0, 0, 0,         0,            1           ]
        ], dtype=float)

        return F

    def measurement_model(self, x):
        """
        Measurement model.

        state = [px, py, speed, heading, heading_rate]
        measurement = [px, py, heading]
        """

        px = x[0]
        py = x[1]
        heading = x[3]

        return np.array([
            px,
            py,
            heading
        ], dtype=float)

    def measurement_jacobian(self, x):
        """
        Measurement Jacobian.

        Sensor measures:
        px, py, heading

        Sensor does not directly measure:
        speed, heading_rate
        """

        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0]
        ], dtype=float)

        return H

    def residual(self, z, z_pred):
        """
        Difference between real measurement and predicted measurement.

        For heading, normalize the angle difference.
        """

        y = z - z_pred
        y[2] = self.normalize_angle(y[2])

        return y

    def predict(self):
        """
        EKF predict step.
        """

        # Jacobian, like F matrix for EKF
        F = self.F_jacobian(self.ekf.x)

        # Predict next state
        self.ekf.x = self.motion_model(self.ekf.x)

        # Predict uncertainty
        self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q

        # Keep heading angle clean
        self.ekf.x[3] = self.normalize_angle(self.ekf.x[3])
        self.ekf.x[2] = max(self.ekf.x[2], 0.0)

        return self.ekf.x

    def update(self, measured_px, measured_py, measured_heading):
        """
        EKF update/correction step.

        z = [measured_px, measured_py, measured_heading]
        """

        z = np.array([
            measured_px,
            measured_py,
            measured_heading
        ], dtype=float)

        self.ekf.update(
            z,
            HJacobian=self.measurement_jacobian,
            Hx=self.measurement_model,
            residual=self.residual
        )

        self.ekf.x[3] = self.normalize_angle(self.ekf.x[3])
        self.ekf.x[2] = max(self.ekf.x[2], 0.0)

        return self.ekf.x

    def process_measurement(self, measured_px, measured_py, measured_heading, dt):
        """
        Full EKF step:
        1. update dt
        2. predict
        3. update with ZED position + hip heading
        """

        self.update_dt(dt)
        self.hip_heading = measured_heading

        self.predict()

        self.update(
            measured_px,
            measured_py,
            measured_heading
        )

        px = float(self.ekf.x[0])
        py = float(self.ekf.x[1])
        speed = float(self.ekf.x[2])
        heading = float(self.ekf.x[3])
        heading_rate = float(self.ekf.x[4])

        return px, py, speed, heading, heading_rate

    def _integrate_future_step(self, px, py, speed, heading, heading_rate, dt_step):
        """
        One open-loop integration step for future trajectory prediction.
        """
        if self.motion_model_type in ("hip_steering", "profidea3"):
            steer = (
                self.steering_gain_b
                * self.hip_steering_dot(heading, self.hip_heading)
                * dt_step
            )
            heading = self.normalize_angle(heading + steer)
            if self.motion_model_type == "profidea3":
                turn   = abs(np.sin(self.hip_heading - heading))
                v_step = max(speed * (1.0 - self.profidea3_k * turn), 0.0)
            else:
                v_step = speed
            px = px + v_step * np.cos(heading) * dt_step
            py = py + v_step * np.sin(heading) * dt_step

        elif self.motion_model_type == "profidea2":
            heading_new = self.normalize_angle(heading + heading_rate * dt_step)
            if abs(heading_rate) > 1e-5:
                px_p = px + (speed / heading_rate) * (np.sin(heading_new) - np.sin(heading))
                py_p = py + (speed / heading_rate) * (-np.cos(heading_new) + np.cos(heading))
            else:
                px_p = px + speed * np.cos(heading) * dt_step
                py_p = py + speed * np.sin(heading) * dt_step

            # Only apply correction when moving — correction is a rate (per second),
            # so scale by dt_step so 20 sub-steps sum to the intended 1-second effect.
            if speed > 0.05:
                theta_h    = self.hip_heading
                v_hat      = np.array([np.cos(heading_new), np.sin(heading_new)])
                n_hat      = np.array([np.cos(theta_h), np.sin(theta_h)])
                n_hat_perp = np.array([-np.sin(theta_h), np.cos(theta_h)])
                C          = self.profidea2_beta * (1.0 - np.dot(v_hat, n_hat))
                delta_r    = (self.profidea2_alpha * v_hat + C * n_hat_perp) * dt_step
                px_p += delta_r[0]
                py_p += delta_r[1]

            px      = px_p
            py      = py_p
            heading = heading_new

        else:
            heading = self.normalize_angle(heading + heading_rate * dt_step)
            px = px + speed * np.cos(heading) * dt_step
            py = py + speed * np.sin(heading) * dt_step

        return px, py, heading

    def predictFuture(self, seconds_ahead, steps=20):
        """
        Predict one future point without changing the EKF state.
        """

        px = self.ekf.x[0]
        py = self.ekf.x[1]
        speed = self.ekf.x[2]
        heading = self.ekf.x[3]
        heading_rate = self.ekf.x[4]

        dt_step = seconds_ahead / steps

        for _ in range(steps):
            px, py, heading = self._integrate_future_step(
                px, py, speed, heading, heading_rate, dt_step
            )

        return px, py

    def predictFutureTrajectory(self, seconds_ahead, steps=20):
        """
        Predict multiple future points without changing the EKF state.
        This is used to draw a future trajectory curve.
        """

        px = self.ekf.x[0]
        py = self.ekf.x[1]
        speed = self.ekf.x[2]
        heading = self.ekf.x[3]
        heading_rate = self.ekf.x[4]

        dt_future = seconds_ahead / steps

        trajectory = []

        for _ in range(steps):
            px, py, heading = self._integrate_future_step(
                px, py, speed, heading, heading_rate, dt_future
            )
            trajectory.append((px, py))

        return trajectory