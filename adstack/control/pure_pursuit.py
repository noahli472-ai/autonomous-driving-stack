"""Pure Pursuit lateral controller (geometric, rear-axle reference)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..planning.path import ReferencePath
from ..vehicle.models import VehicleState
from .base import Controller, ControlOutput, SpeedController


class PurePursuit(Controller):
    """Steer along the circular arc through the rear axle and a look-ahead point.

    delta = atan(2 L sin(alpha) / L_d),  L_d = k_v * v + L_min

    Simple and robust, but it cuts corners at large look-ahead distances and has
    no explicit notion of the path heading.
    """

    name = "Pure Pursuit"

    def __init__(
        self, k_v: float = 0.15, ld_min: float = 3.0, ld_max: float = 25.0, **kw: Any
    ) -> None:
        super().__init__(**kw)
        self.k_v, self.ld_min, self.ld_max = k_v, ld_min, ld_max
        self.speed = SpeedController()

    def reset(self) -> None:
        """Reset the internal state, including the speed controller."""
        super().reset()
        self.speed.reset()

    def control(self, state: VehicleState, path: ReferencePath) -> ControlOutput:
        """Steer towards the look-ahead point on the path."""
        s, e_y, _ = self._progress(state, path)
        v = abs(state.v)
        ld = float(np.clip(self.k_v * v + self.ld_min, self.ld_min, self.ld_max))

        # look-ahead point: first path point whose Euclidean distance exceeds L_d
        s_t = s + ld
        for _ in range(4):
            tx, ty, _, _ = path.interpolate(s_t)
            dist = np.hypot(tx - state.x, ty - state.y)
            if abs(dist - ld) < 0.05 or (not path.closed and s_t >= path.s[-1]):
                break
            s_t += ld - dist
        tx, ty, _, _ = path.interpolate(s_t)

        heading = state.yaw if path.direction > 0 else state.yaw + np.pi
        alpha = np.arctan2(ty - state.y, tx - state.x) - heading
        dist = max(np.hypot(tx - state.x, ty - state.y), 1e-3)
        steer = path.direction * np.arctan2(2.0 * self.p.wheelbase * np.sin(alpha), dist)

        accel, v_ref = self.speed(state.v, s, path, self.dt)
        return ControlOutput(
            accel, float(steer), {"s": s, "e_y": e_y, "v_ref": v_ref, "target": (tx, ty)}
        )
