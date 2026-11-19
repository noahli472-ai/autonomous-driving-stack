"""
Run every scenario, write the benchmark tables to ``results/`` and the figures /
animations to ``assets/``.

    python scripts/run_all.py              # everything (a few minutes)
    python scripts/run_all.py --no-gif     # skip the animations
    python scripts/run_all.py --only racetrack parking
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from adstack.control import MPC, PurePursuit, Stanley  # noqa: E402
from adstack.localization.sensors import SensorConfig  # noqa: E402
from adstack.sim import scenarios as S  # noqa: E402
from adstack.sim.metrics import localization_metrics, tracking_metrics  # noqa: E402
from adstack.vehicle.models import VehicleParams  # noqa: E402
from adstack.viz import plots  # noqa: E402

ASSETS = os.path.join(ROOT, "assets")
RESULTS = os.path.join(ROOT, "results")


def md_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    """cols: (key, header, format)"""
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for k, _, f in cols:
            v = r.get(k, "")
            cells.append(format(v, f) if isinstance(v, (int, float, np.floating)) and f else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def controllers():
    return [PurePursuit(), Stanley(), MPC()]


# --------------------------------------------------------------------------- #
def racetrack(gif: bool):
    print("== Race track: controller benchmark")
    path = S.racetrack()
    nominal = VehicleParams()
    # heavier car with less grip than the controllers were tuned for
    mismatch = replace(
        nominal,
        mass=nominal.mass * 1.25,
        Iz=nominal.Iz * 1.25,
        Cf=nominal.Cf * 0.7,
        Cr=nominal.Cr * 0.7,
    )
    conditions = [
        ("nominal", dict()),
        ("EKF in the loop", dict(use_ekf=True, sensor_cfg=SensorConfig())),
        ("+25% mass, -30% tyre stiffness", dict(params=mismatch)),
    ]
    rows, logs_nominal = [], {}
    for cond, kw in conditions:
        for c in controllers():
            t0 = time.perf_counter()
            log = S.run_racetrack(
                c,
                path,
                store_predictions_every=2 if (cond == "nominal" and c.name == "MPC") else 0,
                **kw,
            )
            m = tracking_metrics(log, path)
            m.update(controller=c.name, condition=cond, wall_s=time.perf_counter() - t0)
            rows.append(m)
            if cond == "nominal":
                logs_nominal[c.name] = log
            print(
                f"  {cond:32s} {c.name:12s} rms e_y={m['rms_lateral_error_m']:.3f} m  "
                f"max={m['max_lateral_error_m']:.3f} m  lap={m['time_s']:.1f} s"
            )
    cols = [
        ("condition", "Condition", ""),
        ("controller", "Controller", ""),
        ("rms_lateral_error_m", "RMS lateral error [m]", ".3f"),
        ("max_lateral_error_m", "Max lateral error [m]", ".3f"),
        ("rms_heading_error_deg", "RMS heading error [deg]", ".2f"),
        ("rms_speed_error_mps", "RMS speed error [m/s]", ".2f"),
        ("time_s", "Lap time [s]", ".1f"),
        ("max_lat_accel_mps2", "Max lat. accel [m/s²]", ".2f"),
        ("rms_steer_rate_degps", "RMS steer rate [deg/s]", ".2f"),
        ("mean_solve_ms", "Mean / max solve [ms]", ""),
    ]
    for r in rows:
        if "mean_solve_ms" in r:
            r["mean_solve_ms"] = f"{r['mean_solve_ms']:.1f} / {r['max_solve_ms']:.1f}"
        else:
            r["mean_solve_ms"] = "< 0.1"
    with open(os.path.join(RESULTS, "racetrack.md"), "w") as f:
        f.write("# Race track benchmark\n\n" + md_table(rows, cols) + "\n")
    with open(os.path.join(RESULTS, "racetrack.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)
    plots.plot_racetrack_comparison(
        path, logs_nominal, os.path.join(ASSETS, "racetrack_comparison.png")
    )
    if gif:
        plots.animate_racetrack(
            path, logs_nominal["MPC"], os.path.join(ASSETS, "racetrack_mpc.gif"), t_end=30.0
        )
    return rows


def localization(gif: bool):
    print("== Localisation: EKF with GNSS outage")
    path = S.racetrack()
    outages = ((20.0, 30.0),)
    cfg = SensorConfig(gps_outages=outages)
    log = S.run_racetrack(MPC(), path, use_ekf=True, sensor_cfg=cfg)
    m = localization_metrics(log, outages)
    m["tracking_rms_lateral_error_m"] = tracking_metrics(log, path)["rms_lateral_error_m"]
    for k, v in m.items():
        print(f"  {k:32s} {v:.3f}")
    rows = [{"metric": k, "value": v} for k, v in m.items()]
    with open(os.path.join(RESULTS, "localization.md"), "w") as f:
        f.write(
            "# EKF localisation (10 s GNSS outage at t = 20-30 s)\n\n"
            + md_table(rows, [("metric", "Metric", ""), ("value", "Value", ".3f")])
            + "\n"
        )
    plots.plot_localization(path, log, outages, os.path.join(ASSETS, "localization.png"))
    return m


def parking(gif: bool):
    print("== Parking: Hybrid A* + MPC")
    rows, items = [], []
    p = VehicleParams()
    from adstack.planning.grid import FootprintChecker

    for make in (S.perpendicular_parking, S.parallel_parking):
        sc = make()
        res = S.plan_parking(sc)
        log = S.run_parking(sc, res)
        a = log.arrays()
        free = FootprintChecker(sc.grid, p, margin=0.0).collision_free(a["x"], a["y"], a["yaw"])
        row = {
            "scenario": sc.name,
            "planning_time_s": res.time,
            "expansions": res.expansions,
            "path_length_m": res.length,
            "direction_changes": res.n_switches,
            "manoeuvre_time_s": float(a["t"][-1]),
            "final_position_error_cm": 100
            * float(np.hypot(a["x"][-1] - sc.goal[0], a["y"][-1] - sc.goal[1])),
            "final_heading_error_deg": float(
                abs(np.rad2deg(np.angle(np.exp(1j * (a["yaw"][-1] - sc.goal[2])))))
            ),
            "max_tracking_error_cm": 100 * float(np.max(np.abs(a["e_y"]))),
            "collision_free": "yes" if free else "NO",
        }
        rows.append(row)
        items.append((sc, res, log))
        print(
            "  "
            + ", ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            )
        )
        if gif:
            plots.animate_parking(sc, res, log, os.path.join(ASSETS, f"parking_{sc.name}.gif"))
    cols = [
        ("scenario", "Scenario", ""),
        ("planning_time_s", "Planning time [s]", ".2f"),
        ("expansions", "Node expansions", "d"),
        ("path_length_m", "Path length [m]", ".1f"),
        ("direction_changes", "Direction changes", "d"),
        ("manoeuvre_time_s", "Manoeuvre time [s]", ".1f"),
        ("final_position_error_cm", "Final position error [cm]", ".1f"),
        ("final_heading_error_deg", "Final heading error [deg]", ".2f"),
        ("max_tracking_error_cm", "Max tracking error [cm]", ".1f"),
        ("collision_free", "Collision-free", ""),
    ]
    with open(os.path.join(RESULTS, "parking.md"), "w") as f:
        f.write("# Automated parking\n\n" + md_table(rows, cols) + "\n")
    plots.plot_parking(items, os.path.join(ASSETS, "parking.png"))
    return rows


def avoidance(gif: bool):
    print("== Local planning: Frenet planner on a two-lane road")
    rows, mpc_run = [], None
    for c in controllers():
        av = S.run_avoidance(controller=c)
        a = av.sim.arrays()
        modes = [p.mode for _, p in av.plans]
        row = {
            "controller": c.name,
            "finished": "yes" if av.sim.finished else "no",
            "collision": "yes" if av.sim.collision else "no",
            "min_clearance_m": av.min_clearance,
            "time_s": float(a["t"][-1]),
            "max_lat_accel_mps2": float(np.max(np.abs(a["ay"]))),
            "max_plan_deviation_m": float(np.nanmax(np.abs(a["e_yaw"]))),
            "lane_changes": int(np.sum(np.abs(np.diff(np.round(a["e_y"] / 3.5))) > 0)),
            "replans": len(av.plans),
            "slow_down_cycles": sum(m.startswith("follow") for m in modes),
        }
        rows.append(row)
        if c.name == "MPC":
            mpc_run = av
        print(
            "  "
            + ", ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            )
        )
    cols = [
        ("controller", "Tracking controller", ""),
        ("finished", "Finished", ""),
        ("collision", "Collision", ""),
        ("min_clearance_m", "Min. clearance [m]", ".2f"),
        ("time_s", "Time [s]", ".1f"),
        ("lane_changes", "Lane changes", "d"),
        ("max_lat_accel_mps2", "Max lat. accel [m/s²]", ".2f"),
        ("max_plan_deviation_m", "Max deviation from plan [m]", ".3f"),
    ]
    with open(os.path.join(RESULTS, "avoidance.md"), "w") as f:
        f.write("# Frenet local planner + tracking controller\n\n" + md_table(rows, cols) + "\n")
    plots.plot_avoidance(mpc_run, os.path.join(ASSETS, "avoidance.png"))
    if gif:
        plots.animate_avoidance(mpc_run, os.path.join(ASSETS, "avoidance.gif"))
    return rows


TASKS = {
    "racetrack": racetrack,
    "localization": localization,
    "parking": parking,
    "avoidance": avoidance,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--no-gif", action="store_true", help="skip GIF animations")
    ap.add_argument("--only", nargs="*", choices=list(TASKS), help="run only these scenarios")
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.perf_counter()
    for name in args.only or TASKS:
        TASKS[name](gif=not args.no_gif)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
