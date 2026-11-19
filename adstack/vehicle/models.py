"""
Vehicle models.

* :class:`KinematicBicycle` - kinematic single-track model, reference point at the
  rear axle. Used inside the controllers / MPC and as plant for low-speed manoeuvres.
* :class:`DynamicBicycle` - dynamic single-track model with linear tyres
  (lateral slip, yaw dynamics). Used as the "real" vehicle at higher speeds so that
  controllers designed on the kinematic model face a realistic model mismatch.

Both plants include simple actuator dynamics (first-order lag + rate limit for the
steering, first-order lag for the longitudinal acceleration) and saturation.

All models expose the pose of the **rear axle centre** via :meth:`observe`, which is
the convention used by every planner and controller in this package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class VehicleParams:
    """Geometry, mass properties and actuator limits (mid-size passenger car)."""

    wheelbase: float = 2.85  # [m] L = lf + lr
    lf: float = 1.25  # [m] CG -> front axle
    length: float = 4.70  # [m] overall length
    width: float = 1.85  # [m] overall width
    rear_overhang: float = 0.95  # [m] rear bumper -> rear axle
    mass: float = 1700.0  # [kg]
    Iz: float = 2900.0  # [kg m^2] yaw moment of inertia
    Cf: float = 95_000.0  # [N/rad] cornering stiffness front axle
    Cr: float = 110_000.0  # [N/rad] cornering stiffness rear axle
    max_steer: float = np.deg2rad(35.0)  # [rad]
    max_steer_rate: float = np.deg2rad(45.0)  # [rad/s]
    steer_tau: float = 0.08  # [s] steering actuator time constant
    max_accel: float = 3.0  # [m/s^2]
    max_decel: float = 7.0  # [m/s^2] (positive number)
    accel_tau: float = 0.15  # [s] powertrain / brake time constant
    max_speed: float = 40.0  # [m/s]

    @property
    def lr(self) -> float:
        """Distance centre of gravity -> rear axle [m]."""
        return self.wheelbase - self.lf

    @property
    def front_overhang(self) -> float:
        """Distance front axle -> front bumper [m]."""
        return self.length - self.wheelbase - self.rear_overhang

    @property
    def min_turning_radius(self) -> float:
        """Turning radius of the rear axle at full steering lock [m]."""
        return self.wheelbase / np.tan(self.max_steer)


@dataclass
class VehicleState:
    """Observable state, pose given at the rear axle centre."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0  # longitudinal speed [m/s] (negative = reversing)
    yaw_rate: float = 0.0  # [rad/s]
    ax: float = 0.0  # longitudinal acceleration [m/s^2]
    ay: float = 0.0  # lateral acceleration [m/s^2]
    steer: float = 0.0  # actual road wheel angle [rad]

    def as_array(self) -> np.ndarray:
        """``[x, y, yaw, v]`` as an array."""
        return np.array([self.x, self.y, self.yaw, self.v])


def wrap_angle(a: ArrayLike) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class _Actuators:
    steer: float = 0.0
    accel: float = 0.0


class _PlantBase:
    """Shared actuator model and integration helpers."""

    def __init__(self, params: VehicleParams | None = None) -> None:
        self.p = params or VehicleParams()
        self.act = _Actuators()
        self.t = 0.0

    def _update_actuators(self, accel_cmd: float, steer_cmd: float, dt: float):
        p = self.p
        steer_cmd = float(np.clip(steer_cmd, -p.max_steer, p.max_steer))
        accel_cmd = float(np.clip(accel_cmd, -p.max_decel, p.max_accel))
        # Steering: first-order lag, then rate limit
        d_steer = (steer_cmd - self.act.steer) * min(dt / p.steer_tau, 1.0)
        max_d = p.max_steer_rate * dt
        self.act.steer = float(
            np.clip(self.act.steer + np.clip(d_steer, -max_d, max_d), -p.max_steer, p.max_steer)
        )
        # Longitudinal: first-order lag
        self.act.accel += (accel_cmd - self.act.accel) * min(dt / p.accel_tau, 1.0)

    @staticmethod
    def _rk4(
        f: Callable[[np.ndarray, Sequence[float]], np.ndarray],
        x: np.ndarray,
        u: Sequence[float],
        dt: float,
    ) -> np.ndarray:
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class KinematicBicycle(_PlantBase):
    """Kinematic single-track model, state ``[x, y, yaw, v]`` at the rear axle.

    x' = v cos(yaw),  y' = v sin(yaw),  yaw' = v tan(delta) / L,  v' = a
    """

    def __init__(
        self, state: VehicleState | None = None, params: VehicleParams | None = None
    ) -> None:
        super().__init__(params)
        s = state or VehicleState()
        self.x = np.array([s.x, s.y, s.yaw, s.v], dtype=float)
        self.act.steer = s.steer

    @staticmethod
    def derivatives(x: np.ndarray, u: Sequence[float], wheelbase: float) -> np.ndarray:
        """Time derivative of the state for the input ``u = (a, delta)``."""
        _, _, yaw, v = x
        a, delta = u
        return np.array([v * np.cos(yaw), v * np.sin(yaw), v * np.tan(delta) / wheelbase, a])

    def step(
        self, accel_cmd: float, steer_cmd: float, dt: float, substeps: int = 2
    ) -> VehicleState:
        """Advance the vehicle by ``dt`` and return the observed state."""
        h = dt / substeps
        for _ in range(substeps):
            self._update_actuators(accel_cmd, steer_cmd, h)
            u = (self.act.accel, self.act.steer)
            self.x = self._rk4(lambda x, u: self.derivatives(x, u, self.p.wheelbase), self.x, u, h)
            self.x[3] = np.clip(self.x[3], -self.p.max_speed, self.p.max_speed)
        self.x[2] = wrap_angle(self.x[2])
        self.t += dt
        return self.observe()

    def observe(self) -> VehicleState:
        """Current state, pose at the rear axle."""
        x, y, yaw, v = self.x
        r = v * np.tan(self.act.steer) / self.p.wheelbase
        return VehicleState(x, y, yaw, v, r, self.act.accel, v * r, self.act.steer)


class DynamicBicycle(_PlantBase):
    """Dynamic single-track model with linear tyres.

    Internal state ``[X, Y, yaw, vx, vy, r]`` at the centre of gravity.
    Below ``v_blend`` the lateral dynamics are blended into the kinematic model,
    because the tyre slip angles are ill-defined at standstill.
    """

    def __init__(
        self,
        state: VehicleState | None = None,
        params: VehicleParams | None = None,
        v_blend: tuple[float, float] = (1.0, 3.0),
    ) -> None:
        super().__init__(params)
        s = state or VehicleState()
        p = self.p
        X = s.x + p.lr * np.cos(s.yaw)
        Y = s.y + p.lr * np.sin(s.yaw)
        self.x = np.array([X, Y, s.yaw, s.v, 0.0, 0.0], dtype=float)
        self.act.steer = s.steer
        self.v_blend = v_blend
        self._ay = 0.0
        self._ax = 0.0

    def derivatives(self, x: np.ndarray, u: Sequence[float]) -> np.ndarray:
        """Time derivative of the state for the input ``u = (a, delta)``."""
        p = self.p
        _, _, yaw, vx, vy, r = x
        a, delta = u

        # Kinematic reference (CG point) for low speed
        beta_kin = np.arctan(p.lr * np.tan(delta) / p.wheelbase)
        r_kin = vx * np.tan(delta) / p.wheelbase
        vy_kin = vx * np.tan(beta_kin)

        # Dynamic model
        vx_s = vx if abs(vx) > 0.1 else np.copysign(0.1, vx if vx != 0 else 1.0)
        alpha_f = np.arctan2(vy + p.lf * r, abs(vx_s)) * np.sign(vx_s) - delta
        alpha_r = np.arctan2(vy - p.lr * r, abs(vx_s)) * np.sign(vx_s)
        Fyf = -p.Cf * alpha_f
        Fyr = -p.Cr * alpha_r
        dvx = a + r * vy - Fyf * np.sin(delta) / p.mass
        dvy = (Fyf * np.cos(delta) + Fyr) / p.mass - r * vx
        dr = (p.lf * Fyf * np.cos(delta) - p.lr * Fyr) / p.Iz

        # Blend factor 0 (kinematic) .. 1 (dynamic)
        lo, hi = self.v_blend
        w = float(np.clip((abs(vx) - lo) / (hi - lo), 0.0, 1.0))
        if w < 1.0:
            # pull lateral states towards the kinematic solution
            k = 20.0
            dvy = w * dvy + (1 - w) * k * (vy_kin - vy)
            dr = w * dr + (1 - w) * k * (r_kin - r)
            dvx = w * dvx + (1 - w) * a

        dX = vx * np.cos(yaw) - vy * np.sin(yaw)
        dY = vx * np.sin(yaw) + vy * np.cos(yaw)
        return np.array([dX, dY, r, dvx, dvy, dr])

    def step(
        self, accel_cmd: float, steer_cmd: float, dt: float, substeps: int = 4
    ) -> VehicleState:
        """Advance the vehicle by ``dt`` and return the observed state."""
        h = dt / substeps
        for _ in range(substeps):
            self._update_actuators(accel_cmd, steer_cmd, h)
            u = (self.act.accel, self.act.steer)
            dx = self.derivatives(self.x, u)
            self._ay = dx[4] + self.x[5] * self.x[3]
            self._ax = dx[3]
            self.x = self._rk4(self.derivatives, self.x, u, h)
            self.x[3] = np.clip(self.x[3], -self.p.max_speed, self.p.max_speed)
        self.x[2] = wrap_angle(self.x[2])
        self.t += dt
        return self.observe()

    def observe(self) -> VehicleState:
        """Current state, pose at the rear axle."""
        p = self.p
        X, Y, yaw, vx, _vy, r = self.x
        xr = X - p.lr * np.cos(yaw)
        yr = Y - p.lr * np.sin(yaw)
        return VehicleState(xr, yr, yaw, vx, r, self._ax, self._ay, self.act.steer)

    @property
    def sideslip(self) -> float:
        """Body side-slip angle at the CG [rad]."""
        return float(np.arctan2(self.x[4], max(abs(self.x[3]), 1e-3)))


def footprint(x: float, y: float, yaw: float, p: VehicleParams) -> np.ndarray:
    """Corner points (4, 2) of the vehicle body for a rear-axle pose."""
    rear = -p.rear_overhang
    front = p.wheelbase + p.front_overhang
    hw = p.width / 2
    local = np.array([[rear, -hw], [front, -hw], [front, hw], [rear, hw]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([x, y])
