import numpy as np
import pytest

from adstack.planning import reeds_shepp as rs
from adstack.planning.frenet import FrenetPlanner, Obstacle
from adstack.planning.grid import FootprintChecker, OccupancyGrid, astar_grid
from adstack.sim.scenarios import perpendicular_parking, plan_parking, two_lane_road
from adstack.vehicle.models import VehicleParams, VehicleState


def test_reeds_shepp_reaches_goal():
    rng = np.random.default_rng(1)
    for _ in range(300):
        start = (*rng.uniform(-5, 5, 2), rng.uniform(-np.pi, np.pi))
        goal = (*rng.uniform(-12, 12, 2), rng.uniform(-np.pi, np.pi))
        radius = rng.uniform(1, 6)
        paths = rs.all_paths(start, goal, radius)
        assert paths, "Reeds-Shepp must always find a path in free space"
        best = paths[0]
        x, y, th, _ = rs.sample_path(best, start, radius, 0.05)
        assert abs(x[-1] - goal[0]) < 1e-6 and abs(y[-1] - goal[1]) < 1e-6
        assert abs(np.angle(np.exp(1j * (th[-1] - goal[2])))) < 1e-6
        assert best.length * radius >= np.hypot(goal[0] - start[0], goal[1] - start[1]) - 1e-9


def test_grid_astar_avoids_wall():
    g = OccupancyGrid(0, 0, 20, 10, 0.1)
    g.add_polygon([[9, 0], [11, 0], [11, 7], [9, 7]])
    path = astar_grid(g, (2, 2), (18, 2), clearance=0.3)
    assert path is not None
    assert np.all(g.distance_at(path[:, 0], path[:, 1]) > 0.3)
    assert path[:, 1].max() > 7.0  # had to go around the wall
    length = np.sum(np.hypot(*np.diff(path, axis=0).T))
    assert length < 1.15 * (2 * np.hypot(7, 5.3) + 2)


def test_hybrid_astar_parking_is_collision_free():
    sc = perpendicular_parking()
    res = plan_parking(sc)
    assert res is not None
    chk = FootprintChecker(sc.grid, VehicleParams(), margin=0.0)
    assert chk.collision_free(res.x, res.y, res.yaw)
    assert np.hypot(res.x[-1] - sc.goal[0], res.y[-1] - sc.goal[1]) < 1e-6
    assert res.n_switches >= 1  # reverse parking needs a direction change
    segs = res.segments()
    assert {s.direction for s in segs} == {1, -1}


@pytest.fixture
def road():
    return two_lane_road()


def _state_on(road, s=20.0, v=18.0):
    x, y, yaw, _ = road.interpolate(s)
    p = VehicleParams()
    # FrenetPlanner works with the vehicle centre; place the rear axle accordingly
    return VehicleState(
        x - 0.5 * p.wheelbase * np.cos(yaw), y - 0.5 * p.wheelbase * np.sin(yaw), yaw, v
    )


def test_frenet_keeps_lane_when_free(road):
    plan = FrenetPlanner(road).plan(_state_on(road), [], v_des=18.0)
    assert plan.best.d_target == 0.0 and plan.v_target == 18.0


def test_frenet_changes_lane_around_static_obstacle(road):
    obs = [Obstacle(s=60.0, d=0.0, v=0.0)]
    plan = FrenetPlanner(road).plan(_state_on(road), obs, v_des=18.0)
    assert plan.best.d_target > 2.5 and not plan.best.collision


def test_frenet_slows_down_when_both_lanes_blocked(road):
    obs = [Obstacle(s=70.0, d=0.0, v=0.0), Obstacle(s=70.0, d=3.5, v=0.0)]
    plan = FrenetPlanner(road).plan(_state_on(road), obs, v_des=18.0)
    assert plan.v_target < 18.0
    assert plan.path.v_ref[-1] == pytest.approx(0.0, abs=1e-6) or plan.v_target == 0.0
