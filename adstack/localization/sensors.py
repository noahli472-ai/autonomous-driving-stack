"""Simulated sensors: GNSS, IMU (gyro + accelerometer) and wheel-speed sensor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vehicle.models import VehicleState


@dataclass
class SensorConfig:
    """Sensor rates, noise levels, gyro bias and GNSS outage windows."""

    gps_rate: float = 10.0  # [Hz]
    gps_sigma: float = 0.8  # [m] per axis
    imu_gyro_sigma: float = 0.01  # [rad/s] white noise
    imu_gyro_bias: float = 0.02  # [rad/s] constant bias (~1.1 deg/s)
    imu_acc_sigma: float = 0.15  # [m/s^2]
    wheel_speed_sigma: float = 0.05  # [m/s]
    gps_outages: tuple = ()  # ((t_start, t_end), ...) e.g. a tunnel


@dataclass
class SensorReading:
    """All sensor measurements of one time step."""

    t: float
    gyro: float
    acc: float
    wheel_speed: float
    gps: np.ndarray | None  # (x, y) or None if no fix in this cycle


class SensorSuite:
    """Generates noisy sensor readings from the true vehicle state."""

    def __init__(self, cfg: SensorConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or SensorConfig()
        self.rng = np.random.default_rng(seed)
        self._next_gps = 0.0

    def gps_available(self, t: float) -> bool:
        """False inside a configured GNSS outage window."""
        return not any(a <= t < b for a, b in self.cfg.gps_outages)

    def read(self, t: float, s: VehicleState) -> SensorReading:
        """Noisy measurements for the true state ``s`` at time ``t``."""
        c, rng = self.cfg, self.rng
        gyro = s.yaw_rate + c.imu_gyro_bias + c.imu_gyro_sigma * rng.standard_normal()
        acc = s.ax + c.imu_acc_sigma * rng.standard_normal()
        ws = s.v + c.wheel_speed_sigma * rng.standard_normal()
        gps = None
        if t + 1e-9 >= self._next_gps:
            self._next_gps += 1.0 / c.gps_rate
            if self.gps_available(t):
                gps = np.array([s.x, s.y]) + c.gps_sigma * rng.standard_normal(2)
        return SensorReading(t, gyro, acc, ws, gps)
