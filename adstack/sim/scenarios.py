"""
Ready-made scenarios: race track, parking lot (perpendicular + parallel) and a
two-lane road with obstacles for the local planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..control.base import Controller
from ..control.mpc import MPC
from ..planning.frenet import FrenetConfig, FrenetPlanner, Obstacle
from ..planning.grid import OccupancyGrid
from ..planning.hybrid_astar import HybridAStar, HybridAStarConfig, HybridAStarResult
from ..planning.path import ReferencePath
from ..vehicle.models import VehicleParams, VehicleState
from .simulator import SimLog, make_plant, run_closed_loop

# --------------------------------------------------------------------------- #
# Race track
# --------------------------------------------------------------------------- #
TRACK_WAYPOINTS = np.array(
    [
        [0, 0],
        [120, 0],
        [160, 8],
        [178, 35],
        [170, 62],
        [140, 70],
        [115, 58],
        [95, 62],
        [80, 85],
        [95, 110],
        [130, 120],
        [160, 140],
        [150, 170],
        [110, 175],
        [60, 165],
        [20, 150],
        [-15, 120],
        [-30, 80],
        [-28, 40],
        [-18, 12],
    ],
    dtype=float,
)


def racetrack(v_max: float = 25.0, a_lat_max: float = 5.0) -> ReferencePath:
    """~790 m closed circuit with straights, a hairpin, an S-section and fast sweepers."""
    path = ReferencePath.from_waypoints(
        TRACK_WAYPOINTS[:, 0], TRACK_WAYPOINTS[:, 1], closed=True, ds=0.5
    )
    path.set_speed_profile(v_max, a_lat_max=a_lat_max, a_acc=2.5, a_dec=4.0)
    return path


def run_racetrack(controller: Controller, path: ReferencePath | None = None, **kw: Any) -> SimLog:
    """One lap on the race track with the dynamic vehicle model."""
    path = path or racetrack()
    x0 = VehicleState(path.x[0], path.y[0], path.yaw[0], 5.0)
    kw.setdefault("t_max", 150.0)
    return run_closed_loop(path, controller, x0=x0, plant="dynamic", **kw)


# --------------------------------------------------------------------------- #
# Parking
# --------------------------------------------------------------------------- #
@dataclass
class ParkingScenario:
    """Parking lot map with start and goal pose and planner settings."""

    grid: OccupancyGrid
    start: tuple
    goal: tuple
    name: str
    parked: list = field(default_factory=list)  # (cx, cy, yaw) of parked cars
    config: HybridAStarConfig = field(default_factory=HybridAStarConfig)


def perpendicular_parking(params: VehicleParams | None = None) -> ParkingScenario:
    """Parking lot with two rows of perpendicular slots; reverse into a free slot."""
    p = params or VehicleParams()
    g = OccupancyGrid(0, 0, 40, 22, 0.1)
    g.add_border()
    slot_w = 2.7
    parked = []
    for i in range(14):
        cx = 1.5 + slot_w / 2 + i * slot_w
        if i != 7:
            parked.append((cx, 2.9, np.pi / 2))
        if i != 3:
            parked.append((cx, 22 - 2.9, np.pi / 2))
    for cx, cy, yaw in parked:
        g.add_rectangle(cx, cy, 4.7, 1.85, yaw, label="car")
    target_cx = 1.5 + slot_w / 2 + 7 * slot_w
    start = (4.0, 11.0, 0.0)
    goal = (target_cx, 0.8 + p.rear_overhang, np.pi / 2)
    return ParkingScenario(g, start, goal, "perpendicular", parked)


def parallel_parking(params: VehicleParams | None = None, gap: float = 7.5) -> ParkingScenario:
    """Street with parallel parked cars and a gap of ``gap`` metres at the kerb."""
    p = params or VehicleParams()
    g = OccupancyGrid(0, 0, 45, 12, 0.1)
    g.add_border()
    # kerb / pavement
    g.add_polygon([[0, 0], [45, 0], [45, 0.8], [0, 0.8]], label="kerb")
    car_len, car_w = 4.6, 1.85
    y_park = 0.8 + 0.25 + car_w / 2
    parked = []
    x = 2.0 + car_len / 2
    gap_start = 20.0
    while x + car_len / 2 < 44:
        if gap_start - 0.01 < x - car_len / 2 < gap_start + gap:
            x = gap_start + gap + car_len / 2 + 0.01
            continue
        if x + car_len / 2 > gap_start and x - car_len / 2 < gap_start:
            x = gap_start + gap + car_len / 2
            continue
        parked.append((x, y_park, 0.0))
        x += car_len + 1.0
    for cx, cy, yaw in parked:
        g.add_rectangle(cx, cy, car_len, car_w, yaw, label="car")
    start = (5.0, 6.5, 0.0)
    # rear axle position of the parked ego vehicle, centred in the gap
    centre_x = gap_start + gap / 2
    rear_axle_x = centre_x - (p.length / 2 - p.rear_overhang)
    goal = (rear_axle_x, y_park, 0.0)
    # tighter manoeuvre: (almost) minimum turning radius and a smaller safety margin
    cfg = HybridAStarConfig(radius_scale=1.05, margin=0.1, max_expansions=20_000)
    return ParkingScenario(g, start, goal, "parallel", parked, cfg)


def plan_parking(
    sc: ParkingScenario,
    params: VehicleParams | None = None,
    config: HybridAStarConfig | None = None,
) -> HybridAStarResult | None:
    """Plan the parking manoeuvre with Hybrid A*."""
    return HybridAStar(sc.grid, params, config or sc.config).plan(sc.start, sc.goal)


def run_parking(
    sc: ParkingScenario,
    result: HybridAStarResult,
    controller: Controller | None = None,
    v_max: float = 1.5,
    params: VehicleParams | None = None,
) -> SimLog:
    """Drive the planned manoeuvre segment by segment with the MPC (kinematic plant)."""
    p = params or VehicleParams()
    segs = result.segments()
    for sg in segs:
        lim = sg.steering_rate_limit(p.wheelbase, p.max_steer_rate)
        sg.set_speed_profile(
            v_max, a_lat_max=1.0, a_acc=0.8, a_dec=0.8, v_start=0.0, v_end=0.0, v_limit=lim
        )
    controller = controller or MPC(horizon=20, dt_pred=0.1)
    x0 = VehicleState(*sc.start, 0.0)
    return run_closed_loop(
        segs,
        controller,
        x0=x0,
        params=params,
        plant="kinematic",
        t_max=120.0,
        stop_tol=0.05,
        steer_at_standstill=True,
    )


# --------------------------------------------------------------------------- #
# Two-lane road with obstacles (local planning)
# --------------------------------------------------------------------------- #
def two_lane_road() -> ReferencePath:
    """~620 m road (centre line of the right lane) with gentle curves."""
    xs = np.linspace(0, 600, 31)
    ys = 12.0 * np.sin(xs / 95.0) + 4.0 * np.sin(xs / 37.0)
    return ReferencePath.from_waypoints(xs, ys, ds=0.5)


def default_obstacles() -> list[Obstacle]:
    """Slow truck to overtake, cones narrowing the lane, a broken-down car that has
    to be passed while another car is driving in the left lane."""
    return [
        Obstacle(s=60.0, d=0.0, length=12.0, width=2.5, v=8.0, name="slow truck"),
        Obstacle(s=245.0, d=-1.45, length=8.0, width=1.2, v=0.0, name="cones"),
        Obstacle(s=100.0, d=3.5, length=4.6, width=1.9, v=13.0, name="car (left lane)"),
        Obstacle(s=400.0, d=0.1, length=4.6, width=1.9, v=0.0, name="broken-down car"),
    ]


@dataclass
class AvoidanceLog:
    """Simulation log plus planner snapshots and obstacle positions."""

    sim: SimLog
    plans: list  # (t, PlanResult) snapshots
    obstacles: list  # (t, [(s, d, length, width), ...])
    ref: ReferencePath
    min_clearance: float = np.inf


def run_avoidance(
    controller: Controller | None = None,
    obstacles: list[Obstacle] | None = None,
    v_des: float = 18.0,
    dt: float = 0.05,
    replan_every: int = 4,
    t_max: float = 45.0,
    params: VehicleParams | None = None,
) -> AvoidanceLog:
    """Closed loop: Frenet planner (re-planning at 5 Hz) + path-following controller."""
    params = params or VehicleParams()
    ref = two_lane_road()
    obstacles = obstacles if obstacles is not None else default_obstacles()
    controller = controller or MPC()
    planner = FrenetPlanner(ref, params, FrenetConfig())

    x0 = VehicleState(ref.x[0], ref.y[0], ref.yaw[0], v_des * 0.8)
    veh = make_plant("dynamic", x0, params)
    controller.reset()
    log = SimLog()
    plans, obs_log = [], []
    state = veh.observe()
    plan = None
    min_clear = np.inf
    t = 0.0
    for k in range(int(t_max / dt)):
        if k % replan_every == 0:
            plan = planner.plan(state, obstacles, v_des)
            controller.s_hint = None
            plans.append((t, plan))
        obs_log.append((t, [(o.s, o.d, o.length, o.width, o.name) for o in obstacles]))

        out = controller.control(state, plan.path)
        s_c, d_c, _ = ref.project(
            state.x + 0.5 * params.wheelbase * np.cos(state.yaw),
            state.y + 0.5 * params.wheelbase * np.sin(state.yaw),
            planner.s_hint,
        )
        # clearance to obstacles (rectangles in Frenet coordinates)
        for o in obstacles:
            ds_ = max(abs(s_c - o.s) - (params.length + o.length) / 2, 0.0)
            dd_ = max(abs(d_c - o.d) - (params.width + o.width) / 2, 0.0)
            min_clear = min(min_clear, np.hypot(ds_, dd_))

        log.t.append(t)
        log.x.append(state.x)
        log.y.append(state.y)
        log.yaw.append(state.yaw)
        log.v.append(state.v)
        log.steer.append(state.steer)
        log.steer_cmd.append(out.steer)
        log.accel_cmd.append(out.accel)
        log.ax.append(state.ax)
        log.ay.append(state.ay)
        log.s.append(s_c)
        log.e_y.append(d_c)
        log.e_yaw.append(out.info.get("e_y", np.nan))
        log.v_ref.append(plan.v_target)
        log.solve_time.append(out.info.get("solve_time", np.nan))
        log.path_id.append(len(plans) - 1)

        if s_c > ref.length - 30:
            log.finished = True
            break
        state = veh.step(out.accel, out.steer, dt)
        for o in obstacles:
            o.s += o.v * dt
        t += dt
    log.collision = min_clear <= 0.0
    return AvoidanceLog(log, plans, obs_log, ref, min_clear)
