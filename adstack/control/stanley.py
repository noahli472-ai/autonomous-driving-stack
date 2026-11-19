"""Stanley lateral controller (front-axle reference, Thrun et al. 2006)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..planning.path import ReferencePath
from ..vehicle.models import VehicleState, wrap_angle
from .base import Controller, ControlOutput, SpeedController


class Stanley(Controller):
    """delta = theta_e + atan(k e_fa / (k_soft + v)) + k_r (r_ref - r) + delta_ff

    * theta_e: heading error between path and vehicle
    * e_fa: cross-track error of the front axle (positive = vehicle left of path)
    * yaw-rate damping and curvature feed-forward improve behaviour at speed
    """

    name = "Stanley"

    def __init__(
        self,
        k: float = 4.0,
        k_soft: float = 1.0,
        k_yaw: float = 0.05,
        use_feedforward: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.k, self.k_soft, self.k_yaw = k, k_soft, k_yaw
        self.use_ff = use_feedforward
        self.speed = SpeedController()
        self.s_hint_front: float | None = None

    def reset(self) -> None:
        """Reset the internal state, including the speed controller."""
        super().reset()
        self.s_hint_front = None
        self.speed.reset()

    def control(self, state: VehicleState, path: ReferencePath) -> ControlOutput:
        """Stanley steering law evaluated at the front axle."""
        p = self.p
        s_rear, e_rear, _ = self._progress(state, path)
        L = p.wheelbase
        fx = state.x + L * np.cos(state.yaw)
        fy = state.y + L * np.sin(state.yaw)
        hint = self.s_hint_front if self.s_hint_front is not None else s_rear + L * path.direction
        s_f, e_fa, _ = path.project(fx, fy, hint)
        self.s_hint_front = s_f
        _, _, yaw_ref, kappa = path.interpolate(s_f)

        theta_e = wrap_angle(yaw_ref - state.yaw)
        v = abs(state.v)
        steer = theta_e - np.arctan2(self.k * e_fa, self.k_soft + v)
        if self.use_ff:
            steer += np.arctan(L * kappa)
            steer += self.k_yaw * (state.v * kappa - state.yaw_rate)

        accel, v_ref = self.speed(state.v, s_rear, path, self.dt)
        return ControlOutput(accel, float(steer), {"s": s_rear, "e_y": e_rear, "v_ref": v_ref})
