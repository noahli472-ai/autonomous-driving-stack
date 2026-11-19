import numpy as np
import pytest

from adstack.control import MPC, PurePursuit, Stanley
from adstack.localization.sensors import SensorConfig
from adstack.planning.grid import FootprintChecker
from adstack.sim import scenarios as S
from adstack.sim.metrics import localization_metrics, tracking_metrics
from adstack.vehicle.models import VehicleParams


@pytest.fixture(scope="module")
def track():
    return S.racetrack()


@pytest.mark.parametrize("ctrl", [PurePursuit, Stanley, MPC])
def test_controllers_complete_a_lap(track, ctrl):
    log = S.run_racetrack(ctrl(), track)
    m = tracking_metrics(log, track)
    assert m["finished"]
    assert m["rms_lateral_error_m"] < 0.3
    assert m["max_lateral_error_m"] < 0.6


def test_mpc_respects_constraints_and_runs_in_real_time(track):
    p = VehicleParams()
    log = S.run_racetrack(MPC(), track)
    a = log.arrays()
    assert np.max(np.abs(a["steer_cmd"])) <= p.max_steer + 1e-6
    assert np.max(a["accel_cmd"]) <= p.max_accel + 1e-3
    # 20 Hz control loop -> one solve must be well below 50 ms on average
    assert np.nanmean(a["solve_time"]) < 0.05


def test_ekf_beats_gnss_and_estimates_gyro_bias(track):
    cfg = SensorConfig()
    log = S.run_racetrack(MPC(), track, use_ekf=True, sensor_cfg=cfg)
    m = localization_metrics(log)
    assert m["ekf_rmse_m"] < 0.6 * m["gps_rmse_m"]
    assert m["gyro_bias_estimate_degps"] == pytest.approx(np.rad2deg(cfg.imu_gyro_bias), rel=0.3)
    assert 0.5 < m["mean_position_nees"] < 5.0  # filter is (roughly) consistent


def test_parking_execution_reaches_slot_without_collision():
    sc = S.perpendicular_parking()
    res = S.plan_parking(sc)
    log = S.run_parking(sc, res)
    a = log.arrays()
    assert log.finished
    assert np.hypot(a["x"][-1] - sc.goal[0], a["y"][-1] - sc.goal[1]) < 0.15
    assert FootprintChecker(sc.grid, VehicleParams(), margin=0.0).collision_free(
        a["x"], a["y"], a["yaw"]
    )


def test_obstacle_avoidance_scenario_is_safe():
    av = S.run_avoidance()
    assert av.sim.finished and not av.sim.collision
    assert av.min_clearance > 0.5
