"""
Closed-loop simulation: plant -> sensors -> (EKF) -> controller -> plant.

The plant runs with the control period ``dt``; the plant itself integrates with
several RK4 sub-steps and contains actuator dynamics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..control.base import Controller, ControlOutput
from ..localization.ekf import EKF
from ..localization.sensors import SensorConfig, SensorSuite
from ..planning.path import ReferencePath
from ..vehicle.models import (
    DynamicBicycle,
    KinematicBicycle,
    VehicleParams,
    VehicleState,
    wrap_angle,
)


@dataclass
class SimLog:
    """Time series of one closed-loop run (all arrays have the same length)."""

    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    yaw: list = field(default_factory=list)
    v: list = field(default_factory=list)
    steer: list = field(default_factory=list)  # actual road wheel angle
    steer_cmd: list = field(default_factory=list)
    accel_cmd: list = field(default_factory=list)
    ax: list = field(default_factory=list)
    ay: list = field(default_factory=list)
    s: list = field(default_factory=list)
    e_y: list = field(default_factory=list)  # true lateral error
    e_yaw: list = field(default_factory=list)  # true heading error
    v_ref: list = field(default_factory=list)
    solve_time: list = field(default_factory=list)
    est: list = field(default_factory=list)  # EKF estimate [x, y, yaw, v, b_g]
    est_std: list = field(default_factory=list)
    gps: list = field(default_factory=list)  # (t, x, y) of GNSS fixes
    prediction: list = field(default_factory=list)  # MPC predicted trajectories
    path_id: list = field(default_factory=list)  # segment index (multi-segment runs)
    finished: bool = False
    collision: bool = False

    def arrays(self) -> dict:
        """All logged time series as NumPy arrays."""
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, list) and k not in ("gps", "prediction"):
                out[k] = np.asarray(v)
        out["gps"] = np.asarray(self.gps).reshape(-1, 3)
        return out


def make_plant(
    kind: str, state: VehicleState, params: VehicleParams
) -> KinematicBicycle | DynamicBicycle:
    """Create the vehicle model (``'dynamic'`` or ``'kinematic'``)."""
    if kind == "dynamic":
        return DynamicBicycle(state, params)
    if kind == "kinematic":
        return KinematicBicycle(state, params)
    raise ValueError(kind)


def run_closed_loop(
    paths: ReferencePath | list[ReferencePath],
    controller: Controller,
    x0: VehicleState | None = None,
    params: VehicleParams | None = None,
    plant: str = "dynamic",
    dt: float = 0.05,
    t_max: float = 120.0,
    laps: int = 1,
    use_ekf: bool = False,
    sensor_cfg: SensorConfig | None = None,
    seed: int = 0,
    stop_tol: float = 0.4,
    store_predictions_every: int = 0,
    collision_fn: Callable[[VehicleState], bool] | None = None,
    steer_at_standstill: bool = False,
) -> SimLog:
    """Run a controller along one path or a sequence of path segments.

    * Closed paths: finish after ``laps`` laps.
    * Open paths: a segment is done when the vehicle is within ``stop_tol`` of its
      end (along the path) and almost stopped; then the next segment starts.
      This is how the multi-segment (forward / reverse) parking manoeuvre is driven.
    * ``steer_at_standstill``: at the start of every segment, first turn the wheels
      to the initial path curvature while the car is stopped.
    """
    params = params or VehicleParams()
    segments = paths if isinstance(paths, list) else [paths]
    seg = 0
    path = segments[0]
    if x0 is None:
        x0 = VehicleState(path.x[0], path.y[0], path.yaw[0], 0.0)
    veh = make_plant(plant, x0, params)
    controller.reset()

    sensors = ekf = None
    if use_ekf:
        sensor_cfg = sensor_cfg or SensorConfig()
        sensors = SensorSuite(sensor_cfg, seed=seed)
        x_init = np.array([x0.x, x0.y, x0.yaw, x0.v, 0.0])
        ekf = EKF(x_init, cfg=sensor_cfg)

    log = SimLog()
    s_prev, progress = None, 0.0
    stalled_steps = 0
    s_true_hint = None
    aligning = steer_at_standstill
    t = 0.0
    truth = veh.observe()
    est_state = truth
    n_steps = int(t_max / dt)
    for step in range(n_steps):
        # --- perception / localisation ---
        if ekf is not None:
            reading = sensors.read(t, truth)
            est_state = ekf.step(reading, dt) if step > 0 else ekf.estimate(reading.gyro)
            if reading.gps is not None:
                log.gps.append((t, *reading.gps))
        else:
            est_state = truth

        # --- control ---
        if aligning:
            # steer at standstill towards the initial curvature of the new segment
            # (what a parking assistant does at a cusp) before driving off
            d0 = float(np.arctan(params.wheelbase * path.kappa[0]))
            brake = -np.sign(truth.v) * min(2.0, abs(truth.v) / dt)
            out = ControlOutput(brake, d0, {"s": 0.0, "e_y": 0.0, "v_ref": 0.0})
            if abs(truth.steer - d0) < np.deg2rad(1.0) and abs(truth.v) < 0.02:
                aligning = False
                if hasattr(controller, "u_prev"):
                    controller.u_prev = np.array([0.0, truth.steer])
        else:
            out = controller.control(est_state, path)
        info = out.info

        # --- logging (true errors) ---
        s_true, e_true, _ = path.project(truth.x, truth.y, s_true_hint)
        s_true_hint = s_true
        _, _, yaw_ref, _ = path.interpolate(s_true)
        log.t.append(t)
        log.x.append(truth.x)
        log.y.append(truth.y)
        log.yaw.append(truth.yaw)
        log.v.append(truth.v)
        log.steer.append(truth.steer)
        log.steer_cmd.append(out.steer)
        log.accel_cmd.append(out.accel)
        log.ax.append(truth.ax)
        log.ay.append(truth.ay)
        log.s.append(s_true)
        log.e_y.append(e_true)
        log.e_yaw.append(float(wrap_angle(truth.yaw - yaw_ref)))
        log.v_ref.append(info.get("v_ref", np.nan))
        log.solve_time.append(info.get("solve_time", np.nan))
        log.path_id.append(seg)
        if ekf is not None:
            log.est.append(ekf.x.copy())
            log.est_std.append(ekf.std.copy())
        if (
            store_predictions_every
            and step % store_predictions_every == 0
            and info.get("prediction") is not None
        ):
            log.prediction.append((t, info["prediction"]))

        if collision_fn is not None and collision_fn(truth):
            log.collision = True
            break

        # --- termination / segment switching ---
        if path.closed:
            if s_prev is not None:
                ds = s_true - s_prev
                if ds < -0.5 * path.length:
                    ds += path.length
                elif ds > 0.5 * path.length:
                    ds -= path.length
                progress += ds
                if progress >= laps * path.length - 0.5:
                    log.finished = True
                    break
            s_prev = s_true
        else:
            near_end = s_true >= path.length - stop_tol and abs(truth.v) < 0.25
            # robustness: accept a segment end if the car has come to rest close to it
            stalled_steps = (
                stalled_steps + 1 if (abs(truth.v) < 0.05 and s_true >= path.length - 0.3) else 0
            )
            if near_end or stalled_steps > int(2.0 / dt):
                stalled_steps = 0
                seg += 1
                if seg >= len(segments):
                    log.finished = True
                    break
                path = segments[seg]
                controller.reset()
                s_true_hint = None
                aligning = steer_at_standstill

        # --- plant ---
        truth = veh.step(out.accel, out.steer, dt)
        t += dt
    return log
