"""
Reference paths: spline interpolation, projection (Frenet frame) and speed profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

from ..vehicle.models import wrap_angle


@dataclass
class ReferencePath:
    """Densely sampled path ``(s, x, y, yaw, kappa)`` with optional speed profile.

    ``direction`` is +1 for forward driving and -1 for reversing (yaw is always the
    vehicle heading, not the direction of travel).

    ``kappa`` is the curvature as seen by the vehicle, i.e. the steady-state
    steering angle is ``atan(wheelbase * kappa)`` in both driving directions.
    """

    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    kappa: np.ndarray
    closed: bool = False
    direction: int = 1
    v_ref: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_waypoints(
        cls,
        xs: ArrayLike,
        ys: ArrayLike,
        ds: float = 0.25,
        closed: bool = False,
        direction: int = 1,
    ) -> ReferencePath:
        """Cubic spline through waypoints, parametrised by (approximate) arc length."""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if closed and np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) > 1e-9:
            xs = np.r_[xs, xs[0]]
            ys = np.r_[ys, ys[0]]
        seg = np.hypot(np.diff(xs), np.diff(ys))
        keep = np.r_[True, seg > 1e-6]
        xs, ys = xs[keep], ys[keep]
        t = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
        bc = "periodic" if closed else "natural"
        sx = CubicSpline(t, xs, bc_type=bc)
        sy = CubicSpline(t, ys, bc_type=bc)

        # Re-sample with true arc length
        tt = np.linspace(0, t[-1], max(int(t[-1] / (ds / 4)), 50))
        dx, dy = sx(tt, 1), sy(tt, 1)
        arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(sx(tt)), np.diff(sy(tt))))]
        n = max(int(arc[-1] / ds), 2)
        s = np.linspace(0, arc[-1], n + 1)
        if closed:
            s = s[:-1]
        tq = np.interp(s, arc, tt)
        x, y = sx(tq), sy(tq)
        dx, dy = sx(tq, 1), sy(tq, 1)
        ddx, ddy = sx(tq, 2), sy(tq, 2)
        heading = np.arctan2(dy, dx)
        kappa = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
        yaw = heading if direction > 0 else wrap_angle(heading + np.pi)
        # kappa is stored as *vehicle* curvature: steering feed-forward = atan(L * kappa).
        # When reversing, the vehicle curvature has the opposite sign of the
        # curvature of the travelled curve.
        kappa = kappa * direction
        path = cls(s=s, x=x, y=y, yaw=yaw, kappa=kappa, closed=closed, direction=direction)
        path.total_length = float(arc[-1])
        return path

    @classmethod
    def from_poses(
        cls, xs: ArrayLike, ys: ArrayLike, yaws: ArrayLike, direction: int = 1
    ) -> ReferencePath:
        """Path directly from a dense pose sequence (e.g. output of Hybrid A*)."""
        xs, ys, yaws = (np.asarray(a, dtype=float) for a in (xs, ys, yaws))
        s = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
        yaw_u = np.unwrap(yaws)
        if len(s) > 2:
            kappa = np.gradient(yaw_u, s) * direction
        else:
            kappa = np.zeros_like(s)
        path = cls(s=s, x=xs, y=ys, yaw=wrap_angle(yaw_u), kappa=kappa, direction=direction)
        path.total_length = float(s[-1])
        return path

    # ------------------------------------------------------------------ #
    @property
    def length(self) -> float:
        """Total path length [m]."""
        return getattr(self, "total_length", float(self.s[-1]))

    def __len__(self) -> int:
        return len(self.s)

    def wrap_s(self, s: ArrayLike) -> np.ndarray:
        """Wrap (closed path) or clip (open path) arc lengths to the valid range."""
        return np.mod(s, self.length) if self.closed else np.clip(s, 0.0, self.s[-1])

    def index_at(self, s: ArrayLike) -> np.ndarray:
        """Index of the sample at or before arc length(s) ``s``."""
        s = self.wrap_s(np.asarray(s, dtype=float))
        idx = np.searchsorted(self.s, s, side="right") - 1
        return np.clip(idx, 0, len(self.s) - 1)

    def interpolate(self, s: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpolated (x, y, yaw, kappa) at arc length(s) ``s``."""
        s = self.wrap_s(np.asarray(s, dtype=float))
        if self.closed:
            sp = np.r_[self.s, self.length]
            xp, yp = np.r_[self.x, self.x[0]], np.r_[self.y, self.y[0]]
            yawp = np.unwrap(np.r_[self.yaw, self.yaw[0]])
            kp = np.r_[self.kappa, self.kappa[0]]
        else:
            sp, xp, yp, yawp, kp = self.s, self.x, self.y, np.unwrap(self.yaw), self.kappa
        return (
            np.interp(s, sp, xp),
            np.interp(s, sp, yp),
            wrap_angle(np.interp(s, sp, yawp)),
            np.interp(s, sp, kp),
        )

    def speed_at(self, s: ArrayLike) -> np.ndarray:
        """Speed profile interpolated at arc length(s) ``s``."""
        if self.v_ref is None:
            raise ValueError("No speed profile; call set_speed_profile first")
        s = self.wrap_s(np.asarray(s, dtype=float))
        if self.closed:
            return np.interp(s, np.r_[self.s, self.length], np.r_[self.v_ref, self.v_ref[0]])
        return np.interp(s, self.s, self.v_ref)

    def project(
        self, x: float, y: float, s_hint: float | None = None, window: float = 15.0
    ) -> tuple[float, float, int]:
        """Closest point on the path.

        Returns ``(s, e_y, idx)`` where ``e_y`` is the signed lateral offset
        (positive = left of the path, w.r.t. the direction of increasing s).
        A local search window around ``s_hint`` avoids jumping between path
        segments that are close in space (loops, hairpins).
        """
        if s_hint is None:
            cand = np.arange(len(self.s))
        else:
            ds = self.s[1] - self.s[0] if len(self.s) > 1 else 1.0
            i0 = int(self.index_at(s_hint))
            w = int(window / max(ds, 1e-6)) + 1
            cand = np.arange(i0 - w, i0 + w + 1)
            cand = (
                np.mod(cand, len(self.s))
                if self.closed
                else cand[(cand >= 0) & (cand < len(self.s))]
            )
        d2 = (self.x[cand] - x) ** 2 + (self.y[cand] - y) ** 2
        i = int(cand[np.argmin(d2)])

        # refine on the segment(s) adjacent to i
        best = (np.inf, self.s[i])
        for j in (i - 1, i):
            if self.closed:
                j0, j1 = j % len(self.s), (j + 1) % len(self.s)
            else:
                if j < 0 or j + 1 >= len(self.s):
                    continue
                j0, j1 = j, j + 1
            px, py = self.x[j0], self.y[j0]
            qx, qy = self.x[j1], self.y[j1]
            vx, vy = qx - px, qy - py
            seg2 = vx * vx + vy * vy
            tproj = 0.0 if seg2 < 1e-12 else np.clip(((x - px) * vx + (y - py) * vy) / seg2, 0, 1)
            cx, cy = px + tproj * vx, py + tproj * vy
            dist = (x - cx) ** 2 + (y - cy) ** 2
            if dist < best[0]:
                seg_len = np.sqrt(seg2)
                best = (dist, self.s[j0] + tproj * seg_len)
        s = float(self.wrap_s(best[1]))
        px, py, yaw, _ = self.interpolate(s)
        heading = yaw if self.direction > 0 else yaw + np.pi
        e_y = -np.sin(heading) * (x - px) + np.cos(heading) * (y - py)
        return s, float(e_y), int(self.index_at(s))

    def to_cartesian(self, s: ArrayLike, d: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Frenet (s, d) -> cartesian (x, y)."""
        x, y, yaw, _ = self.interpolate(s)
        heading = yaw if self.direction > 0 else yaw + np.pi
        return x - np.sin(heading) * d, y + np.cos(heading) * d

    # ------------------------------------------------------------------ #
    def set_speed_profile(
        self,
        v_max: float,
        a_lat_max: float = 4.0,
        a_acc: float = 2.0,
        a_dec: float = 3.0,
        v_start: float | None = None,
        v_end: float | None = None,
        v_limit: np.ndarray | None = None,
    ) -> np.ndarray:
        """Curvature-limited speed profile with forward/backward acceleration passes.

        ``v = min(v_max, sqrt(a_lat_max / |kappa|))``, then
        ``v_i+1 <= sqrt(v_i^2 + 2 a_acc ds)`` (forward) and the mirror image with
        ``a_dec`` (backward). For closed paths the passes are repeated so the
        profile is consistent across the start/finish line.
        Speeds are non-negative magnitudes; multiply by ``direction`` for signed speeds.
        ``v_limit`` is an optional additional point-wise speed limit.
        """
        k = np.abs(self.kappa)
        v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(k, 1e-9)))
        if v_limit is not None:
            v = np.minimum(v, v_limit)
        n = len(v)
        ds = np.diff(self.s)
        if self.closed:
            ds = np.r_[ds, self.length - self.s[-1]]
            for _ in range(2):
                for i in range(n):  # forward
                    j = (i + 1) % n
                    v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_acc * ds[i]))
                for i in range(n - 1, -1, -1):  # backward
                    j = (i + 1) % n
                    v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_dec * ds[i]))
        else:
            if v_start is not None:
                v[0] = min(v[0], v_start)
            if v_end is not None:
                v[-1] = min(v[-1], v_end)
            for i in range(n - 1):
                v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_acc * ds[i]))
            for i in range(n - 2, -1, -1):
                v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_dec * ds[i]))
        self.v_ref = v
        return v

    def steering_rate_limit(
        self, wheelbase: float, max_steer_rate: float, v_min: float = 0.1, factor: float = 0.7
    ) -> np.ndarray:
        """Speed limit so that the required steering rate stays below the actuator limit.

        delta(s) = atan(L kappa(s));  d(delta)/dt = d(delta)/ds * v  <=  factor * rate_max
        Paths with curvature jumps (Reeds-Shepp, Hybrid A*) therefore slow down at the
        jumps, like a driver steering almost at standstill.
        """
        delta = np.arctan(wheelbase * self.kappa)
        if len(self.s) < 3:
            return np.full(len(self.s), np.inf)
        ddelta = np.abs(np.gradient(delta, self.s))
        # widen the slow-down zone around each jump a little
        ddelta = np.maximum.reduce([ddelta, np.r_[ddelta[1:], 0], np.r_[0, ddelta[:-1]]])
        return np.maximum(factor * max_steer_rate / np.maximum(ddelta, 1e-9), v_min)
