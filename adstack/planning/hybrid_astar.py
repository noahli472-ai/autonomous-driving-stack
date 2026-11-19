"""
Hybrid A* (Dolgov et al., 2008) for kinematically feasible paths in tight spaces,
e.g. parking manoeuvres with direction changes.

* Search over continuous poses (x, y, yaw) stored in a discretised 3D grid
* Motion primitives: arcs with several steering angles, forwards and backwards
* Cost: path length + penalties for reversing, direction switches and steering
* Heuristic: obstacle-aware 2D Dijkstra distance to the goal
* Analytic expansion: Reeds-Shepp shot to the exact goal pose, checked for collisions
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..vehicle.models import VehicleParams, wrap_angle
from . import reeds_shepp as rs
from .grid import FootprintChecker, OccupancyGrid, dijkstra_cost_to_goal
from .path import ReferencePath


@dataclass
class HybridAStarConfig:
    """Resolution, motion primitives and cost weights of Hybrid A*."""

    xy_res: float = 0.5
    yaw_res: float = np.deg2rad(7.5)
    step: float = 1.2  # arc length per expansion [m]
    sample: float = 0.2  # collision-check spacing [m]
    n_steer: int = 5  # steering angles per direction (odd)
    radius_scale: float = 1.15  # use a turning radius larger than the minimum
    reverse_penalty: float = 1.5  # cost multiplier for reversing
    switch_penalty: float = 5.0  # cost per change of driving direction
    steer_penalty: float = 0.2
    steer_change_penalty: float = 0.4
    analytic_every: int = 5  # try a Reeds-Shepp shot every n expansions
    max_expansions: int = 40_000
    margin: float = 0.15  # safety margin of the footprint [m]


@dataclass
class _Node:
    x: float
    y: float
    yaw: float
    g: float
    direction: int
    steer: float
    parent: int
    traj: tuple = field(repr=False, default=())  # (xs, ys, yaws) from parent to this node


@dataclass
class HybridAStarResult:
    """Planned poses with driving direction and search statistics."""

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    direction: np.ndarray
    expansions: int
    time: float

    def segments(self, eps: float = 0.01) -> list[ReferencePath]:
        """Split the path at the cusps into forward / reverse ReferencePath segments.

        ``direction[i]`` is the driving direction used to reach pose ``i``; the cusp
        pose is shared by the two adjacent segments.
        """
        d = self.direction
        cuts = [0] + [i - 1 for i in range(1, len(d)) if d[i] != d[i - 1]] + [len(d) - 1]
        segs = []
        for a, b in zip(cuts[:-1], cuts[1:], strict=False):
            if b <= a:
                continue
            idx = np.arange(a, b + 1)
            xs, ys, yaws = self.x[idx], self.y[idx], self.yaw[idx]
            keep = np.r_[True, np.hypot(np.diff(xs), np.diff(ys)) > eps]
            if keep.sum() >= 2:
                segs.append(
                    ReferencePath.from_poses(
                        xs[keep], ys[keep], yaws[keep], direction=int(d[min(a + 1, len(d) - 1)])
                    )
                )
        return segs

    @property
    def length(self) -> float:
        """Path length [m]."""
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))

    @property
    def n_switches(self) -> int:
        """Number of changes of the driving direction."""
        return int(np.sum(np.diff(self.direction) != 0))


class HybridAStar:
    """Hybrid A* planner on an occupancy grid (see module docstring)."""

    def __init__(
        self,
        grid: OccupancyGrid,
        params: VehicleParams | None = None,
        config: HybridAStarConfig | None = None,
    ) -> None:
        self.grid = grid
        self.p = params or VehicleParams()
        self.cfg = config or HybridAStarConfig()
        self.checker = FootprintChecker(grid, self.p, margin=self.cfg.margin)
        self.radius = self.p.min_turning_radius * self.cfg.radius_scale
        max_steer = np.arctan(self.p.wheelbase / self.radius)
        self.steers = np.linspace(-max_steer, max_steer, self.cfg.n_steer)
        self.n_yaw = int(round(2 * np.pi / self.cfg.yaw_res))

    # ------------------------------------------------------------------ #
    def _key(self, x: float, y: float, yaw: float):
        c = self.cfg
        return (
            int(np.floor(x / c.xy_res)),
            int(np.floor(y / c.xy_res)),
            int(np.floor((wrap_angle(yaw) + np.pi) / c.yaw_res)) % self.n_yaw,
        )

    def _heuristic(self, x: float, y: float):
        ix = int((x - self.grid.x_min) / self.h_res)
        iy = int((y - self.grid.y_min) / self.h_res)
        if 0 <= iy < self.h_field.shape[0] and 0 <= ix < self.h_field.shape[1]:
            return self.h_field[iy, ix]
        return np.inf

    def _primitive(self, node: _Node, steer: float, direction: int):
        c = self.cfg
        n = max(int(np.ceil(c.step / c.sample)), 1)
        h = c.step / n
        kappa = np.tan(steer) / self.p.wheelbase
        si = np.arange(1, n + 1) * h * direction
        if abs(kappa) < 1e-9:
            xs = node.x + si * np.cos(node.yaw)
            ys = node.y + si * np.sin(node.yaw)
            yaws = np.full(n, node.yaw)
        else:
            yaws = node.yaw + kappa * si
            xs = node.x + (np.sin(yaws) - np.sin(node.yaw)) / kappa
            ys = node.y - (np.cos(yaws) - np.cos(node.yaw)) / kappa
        return xs, ys, yaws

    def _step_cost(self, node: _Node, steer: float, direction: int) -> float:
        c = self.cfg
        cost = c.step * (c.reverse_penalty if direction < 0 else 1.0)
        if node.parent >= 0 and direction != node.direction:
            cost += c.switch_penalty
        cost += c.steer_penalty * abs(steer) * c.step
        cost += c.steer_change_penalty * abs(steer - node.steer)
        return cost

    def _analytic(self, node: _Node, goal: Sequence[float]):
        for path in rs.all_paths((node.x, node.y, node.yaw), goal, self.radius)[:6]:
            xs, ys, yaws, dirs = rs.sample_path(
                path, (node.x, node.y, node.yaw), self.radius, self.cfg.sample
            )
            if self.checker.collision_free(xs, ys, yaws):
                return xs, ys, yaws, dirs
        return None

    # ------------------------------------------------------------------ #
    def plan(self, start: Sequence[float], goal: Sequence[float]) -> HybridAStarResult | None:
        """Plan from ``start`` to ``goal`` (poses ``(x, y, yaw)``); ``None`` if no path is found."""
        t0 = time.perf_counter()
        c = self.cfg
        if not self.checker.collision_free(*start) or not self.checker.collision_free(*goal):
            raise ValueError("start or goal pose is in collision")
        self.h_field, self.h_res = dijkstra_cost_to_goal(
            self.grid,
            goal,
            clearance=self.p.width / 2 - 0.2,
            coarse=max(int(round(0.5 / self.grid.res)), 1),
        )

        nodes = [_Node(*start, g=0.0, direction=1, steer=0.0, parent=-1)]
        open_heap = [(self._heuristic(start[0], start[1]), 0)]
        best_g = {self._key(*start): 0.0}
        closed = set()
        expansions = 0

        while open_heap and expansions < c.max_expansions:
            _, idx = heapq.heappop(open_heap)
            node = nodes[idx]
            key = self._key(node.x, node.y, node.yaw)
            if key in closed:
                continue
            closed.add(key)
            expansions += 1

            if expansions % c.analytic_every == 0 or self._heuristic(node.x, node.y) < 6.0:
                shot = self._analytic(node, goal)
                if shot is not None:
                    return self._reconstruct(nodes, idx, shot, expansions, time.perf_counter() - t0)

            for direction in (1, -1):
                for steer in self.steers:
                    xs, ys, yaws = self._primitive(node, steer, direction)
                    k2 = self._key(xs[-1], ys[-1], yaws[-1])
                    if k2 in closed:
                        continue
                    if not self.checker.collision_free(xs, ys, yaws):
                        continue
                    g2 = node.g + self._step_cost(node, steer, direction)
                    if g2 >= best_g.get(k2, np.inf):
                        continue
                    h = self._heuristic(xs[-1], ys[-1])
                    if not np.isfinite(h):
                        continue
                    best_g[k2] = g2
                    nodes.append(
                        _Node(xs[-1], ys[-1], yaws[-1], g2, direction, steer, idx, (xs, ys, yaws))
                    )
                    heapq.heappush(open_heap, (g2 + h, len(nodes) - 1))
        return None

    def _reconstruct(
        self,
        nodes: list[_Node],
        idx: int,
        shot: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        expansions: int,
        elapsed: float,
    ) -> HybridAStarResult:
        chain = []
        while idx >= 0:
            chain.append(nodes[idx])
            idx = nodes[idx].parent
        chain.reverse()
        xs, ys, yaws, dirs = (
            [chain[0].x],
            [chain[0].y],
            [chain[0].yaw],
            [chain[1].direction if len(chain) > 1 else int(shot[3][0])],
        )
        for nd in chain[1:]:
            tx, ty, tyaw = nd.traj
            xs.extend(tx)
            ys.extend(ty)
            yaws.extend(tyaw)
            dirs.extend([nd.direction] * len(tx))
        sx, sy, syaw, sdir = shot
        xs.extend(sx[1:])
        ys.extend(sy[1:])
        yaws.extend(syaw[1:])
        dirs.extend(sdir[1:])
        if len(chain) == 1:
            dirs[0] = int(sdir[1]) if len(sdir) > 1 else 1
        return HybridAStarResult(
            np.array(xs),
            np.array(ys),
            wrap_angle(np.array(yaws)),
            np.array(dirs, dtype=int),
            expansions,
            elapsed,
        )
