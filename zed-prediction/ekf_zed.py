from filterpy.kalman import ExtendedKalmanFilter
import numpy as np


class Ekf:
    def __init__(self, initial_px, initial_py, initial_speed,
                 initial_heading, initial_heading_rate, dt=0.033):

        self.initial_px = initial_px
        self.initial_py = initial_py
        self.initial_speed = initial_speed
        self.initial_heading = initial_heading
        self.initial_heading_rate = initial_heading_rate
        self.dt = dt

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
            [0,   0,   0,   0, 100]
        ], dtype=float)

        # Define R: measurement noise
        # measurement = [px_zed, py_zed, hip_heading]
        self.ekf.R = np.array([
            [0.1, 0,   0],
            [0,   0.1, 0],
            [0,   0,   0.3]
        ], dtype=float)

        # Define Q: process noise
        # model uncertainty for [px, py, speed, heading, heading_rate]
        self.ekf.Q = np.array([
            [0.01, 0,    0,   0,    0],
            [0,    0.01, 0,   0,    0],
            [0,    0,    0.1, 0,    0],
            [0,    0,    0,   0.05, 0],
            [0,    0,    0,   0,    0.1]
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

    def motion_model(self, x):
        """
        Nonlinear motion model.

        state = [px, py, speed, heading, heading_rate]
        """

        px = x[0]
        py = x[1]
        speed = x[2]
        heading = x[3]
        heading_rate = x[4]

        # heading new
        heading_new = heading + heading_rate * self.dt
        heading_new = self.normalize_angle(heading_new)

        # position update
        px_new = px + speed * np.cos(heading_new) * self.dt
        py_new = py + speed * np.sin(heading_new) * self.dt

        # assume speed and heading_rate stay the same during tiny dt
        speed_new = speed
        heading_rate_new = heading_rate

        return np.array([
            px_new,
            py_new,
            speed_new,
            heading_new,
            heading_rate_new
        ], dtype=float)

    def F_jacobian(self, x):
        """
        Jacobian of the nonlinear motion model.

        This replaces the F matrix from normal KF.
        """

        speed = x[2]
        heading = x[3]
        heading_rate = x[4]

        heading_new = heading + heading_rate * self.dt
        heading_new = self.normalize_angle(heading_new)

        F = np.array([
            [1, 0, np.cos(heading_new) * self.dt,
             -speed * np.sin(heading_new) * self.dt,
             -speed * np.sin(heading_new) * self.dt * self.dt],

            [0, 1, np.sin(heading_new) * self.dt,
             speed * np.cos(heading_new) * self.dt,
             speed * np.cos(heading_new) * self.dt * self.dt],

            [0, 0, 1, 0, 0],

            [0, 0, 0, 1, self.dt],

            [0, 0, 0, 0, 1]
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

        return self.ekf.x

    def process_measurement(self, measured_px, measured_py, measured_heading, dt):
        """
        Full EKF step:
        1. update dt
        2. predict
        3. update with ZED position + hip heading
        """

        self.update_dt(dt)

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

    def predictFuture(self, seconds_ahead):
        """
        Predict one future point without changing the EKF state.
        """

        px = self.ekf.x[0]
        py = self.ekf.x[1]
        speed = self.ekf.x[2]
        heading = self.ekf.x[3]
        heading_rate = self.ekf.x[4]

        future_heading = heading + heading_rate * seconds_ahead
        future_heading = self.normalize_angle(future_heading)

        future_px = px + speed * np.cos(future_heading) * seconds_ahead
        future_py = py + speed * np.sin(future_heading) * seconds_ahead

        return future_px, future_py

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

        for i in range(steps):
            heading = heading + heading_rate * dt_future
            heading = self.normalize_angle(heading)

            px = px + speed * np.cos(heading) * dt_future
            py = py + speed * np.sin(heading) * dt_future

            trajectory.append((px, py))

        return trajectory