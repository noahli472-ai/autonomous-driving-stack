"""
Reeds-Shepp curves: shortest paths for a car that can drive forwards and backwards
with a minimum turning radius (Reeds & Shepp, 1990).

The formulas follow the formulation used in OMPL (formulas 8.1 - 8.11 of the
paper, combined with the time-flip / reflection / backwards symmetries).
Used by Hybrid A* for the analytic expansion to the exact goal pose.

Segment lengths are given in units of the turning radius; a negative length
means driving backwards. Segment types: 'L' (left), 'S' (straight), 'R' (right).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

PI = np.pi
ZERO = 1e-9


def _mod2pi(x: float) -> float:
    """Wrap to [-pi, pi] (OMPL convention)."""
    v = np.fmod(x, 2 * PI)
    if v < -PI:
        v += 2 * PI
    elif v > PI:
        v -= 2 * PI
    return v


def _polar(x: float, y: float):
    return np.hypot(x, y), np.arctan2(y, x)


def _tau_omega(u: float, v: float, xi: float, eta: float, phi: float):
    delta = _mod2pi(u - v)
    A = np.sin(u) - np.sin(delta)
    B = np.cos(u) - np.cos(delta) - 1.0
    t1 = np.arctan2(eta * A - xi * B, xi * A + eta * B)
    t2 = 2.0 * (np.cos(delta) - np.cos(v) - np.cos(u)) + 3.0
    tau = _mod2pi(t1 + PI) if t2 < 0 else _mod2pi(t1)
    omega = _mod2pi(tau - u + v - phi)
    return tau, omega


# ---------------------------------------------------------------- formulas
def _LpSpLp(x: float, y: float, phi: float):  # 8.1
    u, t = _polar(x - np.sin(phi), y - 1.0 + np.cos(phi))
    if t >= -ZERO:
        v = _mod2pi(phi - t)
        if v >= -ZERO:
            return t, u, v
    return None


def _LpSpRp(x: float, y: float, phi: float):  # 8.2
    u1, t1 = _polar(x + np.sin(phi), y - 1.0 - np.cos(phi))
    u1 = u1 * u1
    if u1 >= 4.0:
        u = np.sqrt(u1 - 4.0)
        theta = np.arctan2(2.0, u)
        t = _mod2pi(t1 + theta)
        v = _mod2pi(t - phi)
        if t >= -ZERO and v >= -ZERO:
            return t, u, v
    return None


def _LpRmL(x: float, y: float, phi: float):  # 8.3 / 8.4
    xi, eta = x - np.sin(phi), y - 1.0 + np.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 <= 4.0:
        u = -2.0 * np.arcsin(0.25 * u1)
        t = _mod2pi(theta + 0.5 * u + PI)
        v = _mod2pi(phi - t + u)
        if t >= -ZERO and u <= ZERO:
            return t, u, v
    return None


def _LpRupLumRm(x: float, y: float, phi: float):  # 8.7
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho = 0.25 * (2.0 + np.hypot(xi, eta))
    if rho <= 1.0:
        u = np.arccos(rho)
        t, v = _tau_omega(u, -u, xi, eta, phi)
        if t >= -ZERO and v <= ZERO:
            return t, u, v
    return None


def _LpRumLumRp(x: float, y: float, phi: float):  # 8.8
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
    if 0.0 <= rho <= 1.0:
        u = -np.arccos(rho)
        if u >= -0.5 * PI:
            t, v = _tau_omega(u, u, xi, eta, phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


def _LpRmSmLm(x: float, y: float, phi: float):  # 8.9
    xi, eta = x - np.sin(phi), y - 1.0 + np.cos(phi)
    rho, theta = _polar(xi, eta)
    if rho >= 2.0:
        r = np.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = _mod2pi(theta + np.arctan2(r, -2.0))
        v = _mod2pi(phi - 0.5 * PI - t)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _LpRmSmRm(x: float, y: float, phi: float):  # 8.10
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho, theta = _polar(-eta, xi)
    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = _mod2pi(t + 0.5 * PI - phi)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _LpRmSLmRp(x: float, y: float, phi: float):  # 8.11
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho, _ = _polar(xi, eta)
    if rho >= 2.0:
        u = 4.0 - np.sqrt(rho * rho - 4.0)
        if u <= ZERO:
            t = _mod2pi(np.arctan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = _mod2pi(t - phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


def _flip(types: str):
    return "".join({"L": "R", "R": "L", "S": "S"}[c] for c in types)


@dataclass
class RSPath:
    """Reeds-Shepp path: segment types and signed lengths (units of the turning radius)."""

    types: str
    lengths: tuple  # signed, in units of the turning radius

    @property
    def length(self) -> float:
        """Total length in units of the turning radius."""
        return float(sum(abs(l) for l in self.lengths))


def _all_normalised(x: float, y: float, phi: float) -> list[RSPath]:
    """All candidate paths from (0,0,0) to (x, y, phi) with unit turning radius."""
    out: list[RSPath] = []

    def add(types: str, lengths: Sequence[float]):
        if all(abs(l) < 1e-10 for l in lengths):
            return
        out.append(RSPath(types, tuple(lengths)))

    # --- CSC ---
    for f, ty in ((_LpSpLp, "LSL"), (_LpSpRp, "LSR")):
        if r := f(x, y, phi):
            add(ty, r)
        if r := f(-x, y, -phi):
            add(ty, [-a for a in r])
        if r := f(x, -y, -phi):
            add(_flip(ty), r)
        if r := f(-x, -y, phi):
            add(_flip(ty), [-a for a in r])

    # --- CCC ---
    if r := _LpRmL(x, y, phi):
        add("LRL", r)
    if r := _LpRmL(-x, y, -phi):
        add("LRL", [-a for a in r])
    if r := _LpRmL(x, -y, -phi):
        add("RLR", r)
    if r := _LpRmL(-x, -y, phi):
        add("RLR", [-a for a in r])
    xb = x * np.cos(phi) + y * np.sin(phi)
    yb = x * np.sin(phi) - y * np.cos(phi)
    if r := _LpRmL(xb, yb, phi):
        t, u, v = r
        add("LRL", (v, u, t))
    if r := _LpRmL(-xb, yb, -phi):
        t, u, v = r
        add("LRL", (-v, -u, -t))
    if r := _LpRmL(xb, -yb, -phi):
        t, u, v = r
        add("RLR", (v, u, t))
    if r := _LpRmL(-xb, -yb, phi):
        t, u, v = r
        add("RLR", (-v, -u, -t))

    # --- CCCC ---
    if r := _LpRupLumRm(x, y, phi):
        t, u, v = r
        add("LRLR", (t, u, -u, v))
    if r := _LpRupLumRm(-x, y, -phi):
        t, u, v = r
        add("LRLR", (-t, -u, u, -v))
    if r := _LpRupLumRm(x, -y, -phi):
        t, u, v = r
        add("RLRL", (t, u, -u, v))
    if r := _LpRupLumRm(-x, -y, phi):
        t, u, v = r
        add("RLRL", (-t, -u, u, -v))
    if r := _LpRumLumRp(x, y, phi):
        t, u, v = r
        add("LRLR", (t, u, u, v))
    if r := _LpRumLumRp(-x, y, -phi):
        t, u, v = r
        add("LRLR", (-t, -u, -u, -v))
    if r := _LpRumLumRp(x, -y, -phi):
        t, u, v = r
        add("RLRL", (t, u, u, v))
    if r := _LpRumLumRp(-x, -y, phi):
        t, u, v = r
        add("RLRL", (-t, -u, -u, -v))

    # --- CCSC ---
    h = 0.5 * PI
    for f, ty in ((_LpRmSmLm, "LRSL"), (_LpRmSmRm, "LRSR")):
        if r := f(x, y, phi):
            t, u, v = r
            add(ty, (t, -h, u, v))
        if r := f(-x, y, -phi):
            t, u, v = r
            add(ty, (-t, h, -u, -v))
        if r := f(x, -y, -phi):
            t, u, v = r
            add(_flip(ty), (t, -h, u, v))
        if r := f(-x, -y, phi):
            t, u, v = r
            add(_flip(ty), (-t, h, -u, -v))
    # backwards versions (CSCC)
    for f, ty in ((_LpRmSmLm, "LSRL"), (_LpRmSmRm, "RSRL")):
        if r := f(xb, yb, phi):
            t, u, v = r
            add(ty, (v, u, -h, t))
        if r := f(-xb, yb, -phi):
            t, u, v = r
            add(ty, (-v, -u, h, -t))
        if r := f(xb, -yb, -phi):
            t, u, v = r
            add(_flip(ty), (v, u, -h, t))
        if r := f(-xb, -yb, phi):
            t, u, v = r
            add(_flip(ty), (-v, -u, h, -t))

    # --- CCSCC ---
    if r := _LpRmSLmRp(x, y, phi):
        t, u, v = r
        add("LRSLR", (t, -h, u, -h, v))
    if r := _LpRmSLmRp(-x, y, -phi):
        t, u, v = r
        add("LRSLR", (-t, h, -u, h, -v))
    if r := _LpRmSLmRp(x, -y, -phi):
        t, u, v = r
        add("RLSRL", (t, -h, u, -h, v))
    if r := _LpRmSLmRp(-x, -y, phi):
        t, u, v = r
        add("RLSRL", (-t, h, -u, h, -v))
    return out


def all_paths(start: Sequence[float], goal: Sequence[float], radius: float) -> list[RSPath]:
    """All Reeds-Shepp candidates between two poses, sorted by length (in metres)."""
    x0, y0, th0 = start
    dx, dy = goal[0] - x0, goal[1] - y0
    c, s = np.cos(th0), np.sin(th0)
    x = (c * dx + s * dy) / radius
    y = (-s * dx + c * dy) / radius
    phi = _mod2pi(goal[2] - th0)
    paths = _all_normalised(x, y, phi)
    paths.sort(key=lambda p: p.length)
    return paths


def sample_path(
    path: RSPath, start: Sequence[float], radius: float, step: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Poses along a Reeds-Shepp path.

    Returns arrays ``x, y, yaw, direction`` (direction = +1 forward / -1 backward).
    """
    x, y, th = map(float, start)
    xs, ys, ths, dirs = [x], [y], [th], [1 if path.lengths[0] >= 0 else -1]
    for typ, l in zip(path.types, path.lengths, strict=False):
        seg_len = abs(l) * radius
        if seg_len < 1e-9:
            continue
        d = 1.0 if l >= 0 else -1.0
        kappa = {"L": 1.0 / radius, "R": -1.0 / radius, "S": 0.0}[typ]
        n = max(int(np.ceil(seg_len / step)), 1)
        h = seg_len / n
        x0, y0, th0 = x, y, th
        for i in range(1, n + 1):
            si = i * h * d  # signed distance along the segment
            if kappa == 0.0:
                x = x0 + si * np.cos(th0)
                y = y0 + si * np.sin(th0)
                th = th0
            else:
                th = th0 + kappa * si
                x = x0 + (np.sin(th) - np.sin(th0)) / kappa
                y = y0 - (np.cos(th) - np.cos(th0)) / kappa
            xs.append(x)
            ys.append(y)
            ths.append(th)
            dirs.append(int(d))
    return np.array(xs), np.array(ys), np.array(ths), np.array(dirs)
