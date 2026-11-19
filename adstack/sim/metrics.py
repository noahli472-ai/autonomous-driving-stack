"""Evaluation metrics for closed-loop runs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..planning.path import ReferencePath
from .simulator import SimLog


def tracking_metrics(log: SimLog, path: ReferencePath | None = None) -> dict:
    """Accuracy, comfort and computational metrics of a path-tracking run."""
    a = log.arrays()
    t = a["t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05
    m = {
        "finished": bool(log.finished),
        "time_s": float(t[-1]) if len(t) else np.nan,
        "rms_lateral_error_m": float(np.sqrt(np.mean(a["e_y"] ** 2))),
        "max_lateral_error_m": float(np.max(np.abs(a["e_y"]))),
        "rms_heading_error_deg": float(np.rad2deg(np.sqrt(np.mean(a["e_yaw"] ** 2)))),
        "max_lat_accel_mps2": float(np.max(np.abs(a["ay"]))),
        "rms_long_jerk_mps3": float(np.sqrt(np.mean((np.diff(a["ax"]) / dt) ** 2)))
        if len(t) > 2
        else np.nan,
        "rms_steer_rate_degps": float(np.rad2deg(np.sqrt(np.mean((np.diff(a["steer"]) / dt) ** 2))))
        if len(t) > 2
        else np.nan,
        "mean_speed_mps": float(np.mean(np.abs(a["v"]))),
    }
    if path is not None and path.v_ref is not None:
        v_profile = path.speed_at(a["s"])
        m["rms_speed_error_mps"] = float(np.sqrt(np.mean((np.abs(a["v"]) - v_profile) ** 2)))
    st = a["solve_time"]
    if np.any(np.isfinite(st)):
        m["mean_solve_ms"] = float(np.nanmean(st) * 1e3)
        m["max_solve_ms"] = float(np.nanmax(st) * 1e3)
    return m


def localization_metrics(log: SimLog, outages: Sequence[tuple[float, float]] = ()) -> dict:
    """Accuracy and consistency metrics of an EKF run (``use_ekf=True``)."""
    a = log.arrays()
    est, sd, t = a["est"], a["est_std"], a["t"]
    pos_err = np.hypot(est[:, 0] - a["x"], est[:, 1] - a["y"])
    yaw_err = np.angle(np.exp(1j * (est[:, 2] - a["yaw"])))
    g = a["gps"]
    gi = np.clip(np.searchsorted(t, g[:, 0] - 1e-9), 0, len(t) - 1)
    gps_err = np.hypot(g[:, 1] - a["x"][gi], g[:, 2] - a["y"][gi])
    in_out = np.zeros_like(t, dtype=bool)
    for t0, t1 in outages:
        in_out |= (t >= t0) & (t < t1)
    warm = t > 5.0
    nees = ((est[:, 0] - a["x"]) / sd[:, 0]) ** 2 + ((est[:, 1] - a["y"]) / sd[:, 1]) ** 2
    res = {
        "gps_rmse_m": float(np.sqrt(np.mean(gps_err**2))),
        "ekf_rmse_m": float(np.sqrt(np.mean(pos_err[warm & ~in_out] ** 2))),
        "ekf_heading_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(yaw_err[warm] ** 2)))),
        "speed_rmse_mps": float(np.sqrt(np.mean((est[:, 3] - a["v"]) ** 2))),
        "gyro_bias_estimate_degps": float(np.rad2deg(est[-1, 4])),
        "mean_position_nees": float(np.mean(nees[warm])),
    }
    if np.any(in_out):
        res["max_error_during_outage_m"] = float(pos_err[in_out].max())
    return res
