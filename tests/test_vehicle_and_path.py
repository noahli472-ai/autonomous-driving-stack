import numpy as np
import pytest

from adstack.planning.path import ReferencePath
from adstack.vehicle.models import (
    DynamicBicycle,
    KinematicBicycle,
    VehicleParams,
    VehicleState,
    wrap_angle,
)


def _fit_circle(P):
    A = np.c_[2 * P, np.ones(len(P))]
    c = np.linalg.lstsq(A, (P**2).sum(1), rcond=None)[0]
    return np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)


def test_kinematic_steady_state_radius():
    p = VehicleParams()
    car = KinematicBicycle(VehicleState(v=5.0), p)
    delta = 0.1
    P = np.array([[(s := car.step(0.0, delta, 0.01)).x, s.y] for _ in range(6000)])[1000:]
    assert _fit_circle(P) == pytest.approx(p.wheelbase / np.tan(delta), rel=1e-4)


def test_dynamic_matches_linear_single_track_theory():
    """Steady-state yaw rate r = v delta / (L + K_us v^2) of the linear bicycle model."""
    p = VehicleParams()
    v, delta = 25.0, 0.02
    car = DynamicBicycle(VehicleState(v=v), p)
    for _ in range(600):
        s = car.step(0.0, delta, 0.01)
    K_us = p.mass / p.wheelbase * (p.lr / p.Cf - p.lf / p.Cr)
    r_theory = s.v * delta / (p.wheelbase + K_us * s.v**2)
    assert s.yaw_rate == pytest.approx(r_theory, rel=0.02)
    assert K_us > 0  # the test vehicle understeers


def test_dynamic_close_to_kinematic_at_low_speed():
    p = VehicleParams()
    car = DynamicBicycle(VehicleState(v=2.0), p)
    for _ in range(1000):
        s = car.step(0.0, 0.2, 0.01)
    assert s.yaw_rate == pytest.approx(s.v * np.tan(0.2) / p.wheelbase, rel=0.05)


def test_steering_rate_limit():
    p = VehicleParams()
    car = KinematicBicycle(VehicleState(v=3.0), p)
    steer = [car.step(0.0, p.max_steer, 0.01).steer for _ in range(50)]
    assert np.max(np.diff(steer)) <= p.max_steer_rate * 0.01 + 1e-12
    assert np.max(steer) <= p.max_steer + 1e-12


def test_circle_path_geometry_and_projection():
    R = 20.0
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    path = ReferencePath.from_waypoints(R * np.cos(th), R * np.sin(th), closed=True)
    assert path.length == pytest.approx(2 * np.pi * R, rel=1e-4)
    assert np.allclose(path.kappa, 1 / R, rtol=5e-3)  # spline through 40 points
    s, e_y, _ = path.project(R + 1.0, 0.5)  # outside the CCW circle -> right
    assert e_y == pytest.approx(-(np.hypot(R + 1.0, 0.5) - R), abs=2e-3)
    assert s == pytest.approx(R * np.arctan2(0.5, R + 1.0), abs=1e-2)


def test_speed_profile_limits():
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    path = ReferencePath.from_waypoints(30 * np.cos(th), 15 * np.sin(th), closed=True)
    v = path.set_speed_profile(30.0, a_lat_max=4.0, a_acc=2.0, a_dec=3.0)
    assert np.all(v**2 * np.abs(path.kappa) <= 4.0 + 1e-6)
    ds = np.diff(path.s)
    acc = np.diff(v**2) / (2 * ds)
    assert acc.max() <= 2.0 + 1e-6 and acc.min() >= -3.0 - 1e-6


def test_reverse_path_curvature_convention():
    """Driving backwards with delta = atan(L * kappa) reproduces a reverse path."""
    p = VehicleParams()
    R = 8.0
    th = np.linspace(0, 1.2, 50)
    # vehicle reverses along a left-curving arc (heading = tangent + pi)
    xs, ys = R * np.sin(th), R * (1 - np.cos(th))
    path = ReferencePath.from_waypoints(xs, ys, ds=0.1, direction=-1)
    delta = np.arctan(p.wheelbase * np.median(path.kappa))
    car = KinematicBicycle(VehicleState(path.x[0], path.y[0], path.yaw[0], -1.0, steer=delta), p)
    for _ in range(int(path.length / 0.01)):
        s = car.step(0.0, delta, 0.01)
    assert np.hypot(s.x - path.x[-1], s.y - path.y[-1]) < 0.1
    assert abs(wrap_angle(s.yaw - path.yaw[-1])) < 0.02
