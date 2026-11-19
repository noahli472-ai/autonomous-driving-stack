"""
Frenet-frame local planner for lane changes / overtaking and obstacle avoidance
(sampling-based, in the spirit of Werling et al., 2010).

In every planning cycle a set of candidate paths is generated in the Frenet
frame of the lane centre line: the lateral offset d(s) follows a quintic
polynomial from the current state (d, d', d'') to a target offset with zero
slope and curvature. Candidates are checked against the predicted positions of
(static or constant-velocity) obstacles and scored by comfort, lane preference,
speed and consistency. Each lateral candidate is combined with several target
speeds, so "change lanes now", "slow down and wait behind" and "stop" are all
part of the same search. If nothing is collision-free, the planner brakes
to a standstill in its lane.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from ..vehicle.models import VehicleParams, VehicleState, wrap_angle
from .path import ReferencePath


@dataclass
class Obstacle:
    """Rectangular obstacle moving along the reference line with constant speed."""

    s: float  # arc length on the reference line at t = 0
    d: float  # lateral offset
    length: float = 4.6
    width: float = 1.9
    v: float = 0.0  # speed along the reference line [m/s]
    name: str = "obstacle"

    def s_at(self, t: ArrayLike) -> np.ndarray:
        """Arc length of the obstacle at time(s) ``t``."""
        return self.s + self.v * np.asarray(t)


@dataclass
class FrenetConfig:
    """Sampling set, safety margins and cost weights of the Frenet planner."""

    lane_width: float = 3.5
    n_lanes: int = 2  # lane 0 = right lane (d = 0)
    target_offsets: tuple = (-0.4, 0.0, 0.6, 1.75, 2.9, 3.5, 3.9)
    maneuver_times: tuple = (2.0, 3.0, 4.0, 5.5)  # [s] lateral manoeuvre duration
    horizon_time: float = 6.0  # [s] collision check horizon
    min_horizon: float = 40.0  # [m]
    ds: float = 1.0  # sampling of candidates [m]
    safety_margin: float = 0.5  # extra lateral clearance [m]
    long_margin: float = 2.0  # extra longitudinal clearance [m]
    reinit_threshold: float = 0.5  # [m] re-plan from the measured state above this
    w_center: float = 2.0  # deviation from the nearest lane centre
    w_left: float = 1.5  # keep-right rule: penalty per lane to the left
    w_jerk: float = 300.0  # lateral "jerk" d'''(s)^2 integral
    w_time: float = 2.0  # prefer short manoeuvres
    w_consistency: float = 0.15  # stay with the previous target
    max_lat_acc: float = 3.0  # [m/s^2] reject too aggressive candidates
    speed_factors: tuple = (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.0)  # x v_des
    a_acc: float = 1.5  # [m/s^2] assumed for speed changes
    a_dec: float = 3.0
    w_speed: float = 12.0  # deviation from the desired speed
    w_speed_consistency: float = 1.0
    comfort_gap: float = 1.3  # [m] lateral gap below which a proximity cost applies
    w_proximity: float = 4.0
    hard_margin: float = 0.2  # [m] minimum margins used if nothing else is free


@dataclass
class Candidate:
    """One candidate path (lateral profile + target speed) with its evaluation."""

    d_target: float
    length: float
    s: np.ndarray
    d: np.ndarray
    x: np.ndarray
    y: np.ndarray
    cost: float = np.inf
    feasible: bool = True
    collision: bool = False
    v_target: float = 0.0
    gap: float = np.inf


@dataclass
class PlanResult:
    """Output of one planning cycle: path for the controller, chosen and all candidates."""

    path: ReferencePath
    best: Candidate
    candidates: list = field(default_factory=list)
    v_target: float = 0.0
    mode: str = "cruise"


def _quintic(d0: float, d1: float, d2: float, dT: float, L: float):
    """Coefficients of d(s) on [0, L] with d(0)=d0, d'(0)=d1, d''(0)=d2, d(L)=dT, d'(L)=d''(L)=0."""
    A = np.array(
        [[L**3, L**4, L**5], [3 * L**2, 4 * L**3, 5 * L**4], [6 * L, 12 * L**2, 20 * L**3]]
    )
    b = np.array([dT - d0 - d1 * L - 0.5 * d2 * L**2, -d1 - d2 * L, -d2])
    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([d0, d1, 0.5 * d2, c3, c4, c5])


class FrenetPlanner:
    """Sampling-based local planner in the Frenet frame of a reference line."""

    def __init__(
        self,
        ref: ReferencePath,
        params: VehicleParams | None = None,
        config: FrenetConfig | None = None,
    ) -> None:
        self.ref = ref
        self.p = params or VehicleParams()
        self.cfg = config or FrenetConfig()
        self.prev: Candidate | None = None
        self.prev_v: float | None = None
        self.s_hint: float | None = None
        hw = self.p.width / 2
        c = self.cfg
        self.d_min = -c.lane_width / 2 + hw + 0.15
        self.d_max = c.lane_width * (c.n_lanes - 0.5) - hw - 0.15

    # ------------------------------------------------------------------ #
    def frenet_state(self, st: VehicleState) -> tuple[float, float, float, float]:
        """Start state (s, d, d', d'') for the next planning cycle.

        The lateral state is taken from the previous plan at the current arc length
        as long as the vehicle tracks that plan closely ("replanning from the
        reference"). This keeps consecutive plans consistent; re-initialising from
        the measured state in every cycle turns the planner into an additional
        feedback loop and causes overshoot. Only if the deviation exceeds
        ``reinit_threshold`` the measured state is used.
        """
        # use the vehicle centre (not the rear axle) for the Frenet state
        cx = st.x + 0.5 * self.p.wheelbase * np.cos(st.yaw)
        cy = st.y + 0.5 * self.p.wheelbase * np.sin(st.yaw)
        s, d, _ = self.ref.project(cx, cy, self.s_hint)
        self.s_hint = s
        _, _, yaw_ref, kappa = self.ref.interpolate(s)
        dpsi = wrap_angle(st.yaw - yaw_ref)
        d1 = (1 - kappa * d) * np.tan(dpsi)
        d2 = 0.0
        if self.prev is not None and self.prev.s[0] <= s <= self.prev.s[-1]:
            ps, pd, h = self.prev.s, self.prev.d, self.cfg.ds
            d_plan = float(np.interp(s, ps, pd))
            if abs(d_plan - d) < self.cfg.reinit_threshold:
                d1_arr = np.gradient(pd, h)
                d2_arr = np.gradient(d1_arr, h)
                return s, d_plan, float(np.interp(s, ps, d1_arr)), float(np.interp(s, ps, d2_arr))
            d2 = float(np.interp(s, ps, np.gradient(np.gradient(pd, h), h)))
        return s, d, d1, d2

    def _time_at(self, s_rel: np.ndarray, v0: float, v_c: float) -> np.ndarray:
        """Time at which the ego reaches s_rel when changing speed from v0 to v_c
        with constant acceleration/deceleration. Unreachable points -> inf."""
        c = self.cfg
        tg = np.linspace(0.0, c.horizon_time * 2, 400)
        a = c.a_acc if v_c >= v0 else -c.a_dec
        t_ch = abs(v_c - v0) / abs(a) if v_c != v0 else 0.0
        v_t = np.where(tg < t_ch, v0 + a * tg, v_c)
        s_t = np.r_[0.0, np.cumsum(0.5 * (v_t[1:] + v_t[:-1]) * np.diff(tg))]
        t = np.interp(s_rel, s_t, tg, right=np.inf)
        if v_c < 0.1:
            t[s_rel > s_t[-1] - 1e-6] = np.inf  # never reached (vehicle stops before)
        return t

    def _check(
        self,
        cand: Candidate,
        t: np.ndarray,
        obstacles: list[Obstacle],
        lat_margin: float,
        lon_margin: float,
    ) -> tuple[bool, float]:
        """Collision test in the Frenet frame against the predicted obstacles.

        Returns ``(collision, min_lateral_gap)`` where the gap is measured only where
        the vehicle overlaps an obstacle longitudinally.
        """
        p = self.p
        reach = np.isfinite(t) & (t <= self.cfg.horizon_time)
        if not obstacles or not np.any(reach):
            return False, np.inf
        s, d, tt = cand.s[reach], cand.d[reach], t[reach]
        gap_min = np.inf
        for ob in obstacles:
            lon = np.abs(s - ob.s_at(tt)) < (p.length / 2 + ob.length / 2 + lon_margin)
            if not np.any(lon):
                continue
            gap = np.abs(d[lon] - ob.d) - (p.width / 2 + ob.width / 2)
            gmin = float(gap.min())
            gap_min = min(gap_min, gmin)
            if gmin < lat_margin:
                return True, gap_min
        return False, gap_min

    def plan(self, st: VehicleState, obstacles: list[Obstacle], v_des: float) -> PlanResult:
        """Generate, check and score all candidates and return the best plan."""
        c = self.cfg
        s0, d0, d1, d2 = self.frenet_state(st)
        v_now = abs(st.v)
        v = max(v_now, 1.0)
        horizon = max(c.min_horizon, max(v, v_des) * c.horizon_time)
        n = int(horizon / c.ds) + 1
        s_rel = np.arange(n) * c.ds
        s_abs = s0 + s_rel
        speeds = sorted({round(v_des * f, 2) for f in c.speed_factors})
        times = {vc: self._time_at(s_rel, v_now, vc) for vc in speeds}
        prev_d = self.prev.d_target if self.prev is not None else 0.0
        prev_v = self.prev_v if self.prev_v is not None else v_des

        cands = []
        for dT in c.target_offsets:
            if not (self.d_min - 1e-9 <= dT <= self.d_max + 1e-9):
                continue
            for T in c.maneuver_times:
                L = max(v * T, 8.0)
                coef = _quintic(d0, d1, d2, dT, L)
                sr = np.minimum(s_rel, L)
                pw = np.vstack([sr**k for k in range(6)])
                d = np.where(s_rel <= L, coef @ pw, dT)
                dd2 = np.polyval(np.polyder(coef[::-1], 2), sr) * (s_rel <= L)
                dd3 = np.polyval(np.polyder(coef[::-1], 3), sr) * (s_rel <= L)
                x, y = self.ref.to_cartesian(s_abs, d)
                # lateral acceleration of the manoeuvre ~ v^2 * d''
                feasible = np.max(np.abs(dd2)) * v**2 <= c.max_lat_acc
                lane = int(np.clip(round(dT / c.lane_width), 0, c.n_lanes - 1))
                base = (
                    c.w_center * (dT - lane * c.lane_width) ** 2
                    + c.w_left * lane
                    + c.w_jerk * np.sum(dd3**2) * c.ds
                    + c.w_time * T / 5.0
                    + c.w_consistency * abs(dT - prev_d)
                )
                for vc in speeds:
                    cand = Candidate(dT, L, s_abs, d, x, y)
                    cand.v_target = vc
                    cand.feasible = feasible
                    cand.collision, gap = self._check(
                        cand, times[vc], obstacles, c.safety_margin, c.long_margin
                    )
                    cand.gap = gap
                    cand.cost = (
                        base
                        + c.w_speed * ((v_des - vc) / max(v_des, 1e-3)) ** 2
                        + c.w_proximity * max(0.0, c.comfort_gap - gap) ** 2
                        + c.w_speed_consistency * abs(vc - prev_v) / max(v_des, 1e-3)
                    )
                    cands.append(cand)

        free = [cd for cd in cands if cd.feasible and not cd.collision]
        if not free:
            # second pass with the hard (minimum) safety margins
            for cd in cands:
                cd.collision, _ = self._check(
                    cd, times[cd.v_target], obstacles, c.hard_margin, c.hard_margin
                )
            free = [cd for cd in cands if not cd.collision]
        if free:
            best = min(free, key=lambda cd: cd.cost)
            v_target = best.v_target
            if v_target < 0.95 * v_des:
                mode = "follow / slow down"
            elif abs(best.d_target) > 0.5:
                mode = "overtake / avoid"
            else:
                mode = "cruise"
        else:
            # emergency: keep the lane and brake to standstill
            lane_d = float(np.clip(round(d0 / c.lane_width) * c.lane_width, self.d_min, self.d_max))
            pool = [
                cd for cd in cands if abs(cd.d_target - lane_d) < 1e-6 and cd.v_target == 0.0
            ] or cands
            best = min(pool, key=lambda cd: (abs(cd.d_target - lane_d), -cd.length))
            v_target = 0.0
            mode = "emergency brake"
        self.prev = best
        self.prev_v = v_target

        # extend the path a few metres backwards so that the rear axle (which is
        # behind the vehicle centre) projects onto the path, too
        s_back = s0 + np.arange(-8.0, 0.0, c.ds)
        xb, yb = self.ref.to_cartesian(s_back, d0 + d1 * (s_back - s0))
        path = ReferencePath.from_waypoints(np.r_[xb, best.x], np.r_[yb, best.y], ds=0.5)
        path.set_speed_profile(max(v_target, 0.0), a_lat_max=4.0, a_acc=c.a_acc, a_dec=c.a_dec)
        if v_target < 0.1:
            # stop before the first obstacle point that the candidate would hit at full speed
            path.v_ref = self._stopping_profile(path, s0, best, obstacles, v_now)
        return PlanResult(path, best, cands, v_target=v_target, mode=mode)

    def _stopping_profile(
        self,
        path: ReferencePath,
        s0: float,
        cand: Candidate,
        obstacles: list[Obstacle],
        v_now: float,
    ):
        """Speed profile that brings the vehicle to a halt behind the closest obstacle."""
        c, p = self.cfg, self.p
        s_stop = np.inf
        for ob in obstacles:
            if abs(ob.d - cand.d_target) < (p.width + ob.width) / 2 + c.safety_margin and ob.s > s0:
                s_stop = min(s_stop, ob.s - s0 - (p.length + ob.length) / 2 - c.long_margin)
        if not np.isfinite(s_stop):
            s_stop = v_now**2 / (2 * c.a_dec)
        s_path = path.s - 8.0  # path starts 8 m behind the vehicle centre
        remaining = np.maximum(s_stop - s_path, 0.0)
        return np.sqrt(2 * c.a_dec * remaining)
