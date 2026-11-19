"""
Extended Kalman Filter for vehicle localisation.

State      x = [X, Y, yaw, v, b_g]   (rear-axle pose, speed, gyro bias)
Inputs     gyro yaw rate w_m and longitudinal acceleration a_m (IMU)
Process    X' = v cos(yaw), Y' = v sin(yaw), yaw' = w_m - b_g, v' = a_m, b_g' = 0 (+ noise)
Updates    GNSS position (X, Y) and wheel speed v

Estimating the gyro bias is what keeps the heading from drifting during a GNSS
outage - in the demo the filter is run through a simulated tunnel.
"""

from __future__ import annotations

import numpy as np

from ..vehicle.models import VehicleState, wrap_angle
from .sensors import SensorConfig, SensorReading

NX = 5


class EKF:
    """Extended Kalman Filter with state ``[x, y, yaw, v, gyro_bias]`` (see module docstring)."""

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray | None = None,
        cfg: SensorConfig | None = None,
        q_bias: float = 1e-4,
        q_pos: float = 0.1,
    ) -> None:
        cfg = cfg or SensorConfig()
        self.x = np.asarray(x0, dtype=float).copy()
        if self.x.size == 4:
            self.x = np.r_[self.x, 0.0]
        self.P = P0 if P0 is not None else np.diag([1.0, 1.0, np.deg2rad(5) ** 2, 0.5, 0.05**2])
        # continuous-time process noise densities
        self.q_gyro = cfg.imu_gyro_sigma**2
        self.q_acc = cfg.imu_acc_sigma**2
        self.q_bias = q_bias**2
        # unmodelled lateral motion of the rear axle (tyre slip) -> position noise
        self.q_pos = q_pos
        self.R_gps = np.eye(2) * cfg.gps_sigma**2
        self.R_ws = cfg.wheel_speed_sigma**2
        self.last_innovation = None
        self.nis_history: list[float] = []

    # ------------------------------------------------------------------ #
    def predict(self, gyro: float, acc: float, dt: float) -> None:
        """Propagate state and covariance with the IMU measurements."""
        X, Y, yaw, v, bg = self.x
        c, s = np.cos(yaw), np.sin(yaw)
        w = gyro - bg
        self.x = np.array(
            [X + v * c * dt, Y + v * s * dt, wrap_angle(yaw + w * dt), v + acc * dt, bg]
        )
        F = np.eye(NX)
        F[0, 2] = -v * s * dt
        F[0, 3] = c * dt
        F[1, 2] = v * c * dt
        F[1, 3] = s * dt
        F[2, 4] = -dt
        # noise enters through yaw (gyro), v (accelerometer) and the bias random walk
        Q = np.diag(
            [self.q_pos * dt, self.q_pos * dt, self.q_gyro * dt, self.q_acc * dt, self.q_bias * dt]
        )
        self.P = F @ self.P @ F.T + Q

    def _update(
        self,
        z: np.ndarray,
        h: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        angle_idx: int | None = None,
    ):
        y = z - h
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        # Joseph form for numerical robustness
        I_KH = np.eye(NX) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        nis = float(y @ np.linalg.solve(S, y))
        return y, nis

    def update_gps(self, z: np.ndarray) -> None:
        """Correct the state with a GNSS position fix ``z = (x, y)``."""
        H = np.zeros((2, NX))
        H[0, 0] = H[1, 1] = 1.0
        y, nis = self._update(np.asarray(z), self.x[:2], H, self.R_gps)
        self.last_innovation = y
        self.nis_history.append(nis)

    def update_wheel_speed(self, v_meas: float) -> None:
        """Correct the speed with the wheel-speed measurement."""
        H = np.zeros((1, NX))
        H[0, 3] = 1.0
        self._update(np.array([v_meas]), self.x[3:4], H, np.array([[self.R_ws]]))

    def step(self, reading: SensorReading, dt: float) -> VehicleState:
        """One full filter cycle: IMU prediction followed by all available updates."""
        self.predict(reading.gyro, reading.acc, dt)
        self.update_wheel_speed(reading.wheel_speed)
        if reading.gps is not None:
            self.update_gps(reading.gps)
        return self.estimate(reading.gyro)

    def estimate(self, gyro: float = 0.0) -> VehicleState:
        """Current estimate as a ``VehicleState`` (yaw rate = bias-corrected gyro)."""
        X, Y, yaw, v, bg = self.x
        return VehicleState(X, Y, yaw, v, gyro - bg)

    @property
    def std(self) -> np.ndarray:
        """Standard deviations of the state estimate (square root of diag(P))."""
        return np.sqrt(np.diag(self.P))
