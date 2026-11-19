"""Common controller interface and the longitudinal speed controller."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..planning.path import ReferencePath
from ..vehicle.models import VehicleParams, VehicleState


@dataclass
class ControlOutput:
    """Commands of one control cycle plus diagnostic information."""

    accel: float
    steer: float
    info: dict = field(default_factory=dict)


class Controller:
    """Base class. ``control`` is called once per control cycle."""

    name = "controller"

    def __init__(self, params: VehicleParams | None = None, dt: float = 0.05) -> None:
        self.p = params or VehicleParams()
        self.dt = dt
        self.s_hint: float | None = None

    def reset(self) -> None:
        """Reset the internal state (call before a new run or path segment)."""
        self.s_hint = None

    def control(self, state: VehicleState, path: ReferencePath) -> ControlOutput:
        """Compute acceleration and steering commands for the current state."""
        raise NotImplementedError

    def _progress(self, state: VehicleState, path: ReferencePath):
        s, e_y, idx = path.project(state.x, state.y, self.s_hint)
        self.s_hint = s
        return s, e_y, idx


class SpeedController:
    """PI speed controller with acceleration feed-forward from the speed profile.

    The target speed is taken a short preview distance ahead so that the car
    starts braking before a corner rather than inside it.
    """

    def __init__(
        self, kp: float = 1.2, ki: float = 0.3, preview_time: float = 0.6, i_limit: float = 2.0
    ) -> None:
        self.kp, self.ki = kp, ki
        self.preview_time = preview_time
        self.i_limit = i_limit
        self.integral = 0.0

    def reset(self) -> None:
        """Reset the internal state (call before a new run or path segment)."""
        self.integral = 0.0

    def __call__(self, v: float, s: float, path: ReferencePath, dt: float) -> tuple[float, float]:
        """Return ``(acceleration command, signed target speed)``."""
        d = path.direction
        speed = abs(v)
        s_prev = s + max(speed, 1.0) * self.preview_time
        v_t = float(path.speed_at(s_prev))
        v_now = float(path.speed_at(s))
        # feed-forward: a = d(v^2/2)/ds
        ds = max(s_prev - s, 1e-3)
        a_ff = (v_t**2 - v_now**2) / (2 * ds)
        err = v_t - speed
        self.integral = float(np.clip(self.integral + err * dt, -self.i_limit, self.i_limit))
        a = a_ff + self.kp * err + self.ki * self.integral
        return d * a, d * v_t
