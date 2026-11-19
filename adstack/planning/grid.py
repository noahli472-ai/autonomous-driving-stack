"""
Occupancy grid, distance map, footprint collision checking and grid A*.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence

import numpy as np
from matplotlib.path import Path as MplPath
from numpy.typing import ArrayLike
from scipy.ndimage import distance_transform_edt

from ..vehicle.models import VehicleParams


class OccupancyGrid:
    """Binary occupancy grid with a Euclidean distance map.

    ``data[iy, ix]`` is True for occupied cells. ``dist`` holds the distance (m)
    of every cell centre to the nearest occupied cell.
    """

    def __init__(
        self, x_min: float, y_min: float, width: float, height: float, resolution: float = 0.1
    ) -> None:
        self.x_min, self.y_min = x_min, y_min
        self.res = resolution
        self.nx = int(np.ceil(width / resolution))
        self.ny = int(np.ceil(height / resolution))
        self.data = np.zeros((self.ny, self.nx), dtype=bool)
        self.dist = None
        self.polygons: list[np.ndarray] = []
        self.labels: list[str] = []

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Map bounds ``(x_min, x_max, y_min, y_max)``."""
        return (
            self.x_min,
            self.x_min + self.nx * self.res,
            self.y_min,
            self.y_min + self.ny * self.res,
        )

    def add_polygon(self, pts: ArrayLike, label: str = "obstacle") -> None:
        """Mark all cells inside the polygon as occupied."""
        pts = np.asarray(pts, dtype=float)
        self.polygons.append(pts)
        self.labels.append(label)
        xs = self.x_min + (np.arange(self.nx) + 0.5) * self.res
        ys = self.y_min + (np.arange(self.ny) + 0.5) * self.res
        X, Y = np.meshgrid(xs, ys)
        inside = MplPath(pts).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
        self.data |= inside
        self.dist = None

    def add_rectangle(
        self,
        cx: float,
        cy: float,
        length: float,
        width: float,
        yaw: ArrayLike = 0.0,
        label: str = "obstacle",
    ) -> None:
        """Add a rotated rectangle centred at ``(cx, cy)``."""
        c, s = np.cos(yaw), np.sin(yaw)
        hl, hw = length / 2, width / 2
        local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
        self.add_polygon(local @ np.array([[c, s], [-s, c]]) + [cx, cy], label)

    def add_border(self, thickness: float = 0.3) -> None:
        """Add walls along the map boundary."""
        x0, x1, y0, y1 = self.extent
        t = thickness
        for a, b, c, d in (
            (x0, y0, x1, y0 + t),
            (x0, y1 - t, x1, y1),
            (x0, y0, x0 + t, y1),
            (x1 - t, y0, x1, y1),
        ):
            self.add_polygon([[a, b], [c, b], [c, d], [a, d]], "wall")

    def compute_distance(self) -> np.ndarray:
        """Compute the distance map (Euclidean distance transform)."""
        self.dist = distance_transform_edt(~self.data) * self.res
        return self.dist

    def world_to_index(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Convert world coordinates to grid indices ``(ix, iy)``."""
        ix = np.floor((np.asarray(x) - self.x_min) / self.res).astype(int)
        iy = np.floor((np.asarray(y) - self.y_min) / self.res).astype(int)
        return ix, iy

    def distance_at(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Distance to the closest obstacle; 0 outside the map."""
        if self.dist is None:
            self.compute_distance()
        ix, iy = self.world_to_index(x, y)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        out = np.zeros(np.shape(ix), dtype=float)
        out[valid] = self.dist[iy[valid], ix[valid]]
        return out


class FootprintChecker:
    """Collision check of the vehicle rectangle against the distance map.

    Points are sampled on the outline of the vehicle body (every ``spacing`` m);
    a pose is collision-free if every outline point is further than ``margin``
    (+ half a grid cell) from the nearest obstacle. Obstacles in this package are
    larger than the sample spacing, so they cannot slip between two outline points.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        params: VehicleParams | None = None,
        spacing: float = 0.25,
        margin: float = 0.1,
    ) -> None:
        p = params or VehicleParams()
        self.grid = grid
        rear, front = -p.rear_overhang, p.wheelbase + p.front_overhang
        hw = p.width / 2
        nl = max(int(np.ceil(p.length / spacing)), 1)
        nw = max(int(np.ceil(p.width / spacing)), 1)
        xs = np.linspace(rear, front, nl + 1)
        ys = np.linspace(-hw, hw, nw + 1)
        pts = np.vstack(
            (
                np.c_[xs, np.full_like(xs, -hw)],
                np.c_[xs, np.full_like(xs, hw)],
                np.c_[np.full_like(ys, rear), ys],
                np.c_[np.full_like(ys, front), ys],
            )
        )
        self.local = pts
        self.margin = margin + 0.5 * grid.res
        if grid.dist is None:
            grid.compute_distance()

    def collision_free(self, x: ArrayLike, y: ArrayLike, yaw: ArrayLike) -> bool:
        """True if all given poses are collision-free."""
        x, y, yaw = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (x, y, yaw))
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        lx, ly = self.local[:, 0][None, :], self.local[:, 1][None, :]
        px = x[:, None] + c * lx - s * ly
        py = y[:, None] + s * lx + c * ly
        return bool(np.all(self.grid.distance_at(px, py) > self.margin))


def astar_grid(
    grid: OccupancyGrid,
    start: Sequence[float],
    goal: Sequence[float],
    clearance: float = 0.0,
    coarse: int = 1,
) -> np.ndarray | None:
    """8-connected A* on the occupancy grid (optionally on a coarsened grid).

    Cells closer than ``clearance`` to an obstacle are treated as occupied.
    Returns the path as (M, 2) world coordinates or ``None``.
    """
    if grid.dist is None:
        grid.compute_distance()
    free = grid.dist[::coarse, ::coarse] > clearance
    res = grid.res * coarse
    ny, nx = free.shape

    def to_idx(p: Sequence[float]):
        return (int((p[0] - grid.x_min) / res), int((p[1] - grid.y_min) / res))

    sx, sy = to_idx(start)
    gx, gy = to_idx(goal)
    if not (0 <= sx < nx and 0 <= sy < ny and 0 <= gx < nx and 0 <= gy < ny):
        return None
    if not free[sy, sx] or not free[gy, gx]:
        return None

    moves = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, np.sqrt(2)),
        (1, -1, np.sqrt(2)),
        (-1, 1, np.sqrt(2)),
        (-1, -1, np.sqrt(2)),
    ]
    g = np.full((ny, nx), np.inf)
    parent = -np.ones((ny, nx), dtype=np.int64)
    closed = np.zeros((ny, nx), dtype=bool)
    g[sy, sx] = 0.0
    heap = [(np.hypot(gx - sx, gy - sy), 0.0, sx, sy)]
    while heap:
        _, gc, x, y = heapq.heappop(heap)
        if closed[y, x]:
            continue
        closed[y, x] = True
        if (x, y) == (gx, gy):
            break
        for dx, dy, c in moves:
            x2, y2 = x + dx, y + dy
            if 0 <= x2 < nx and 0 <= y2 < ny and free[y2, x2] and not closed[y2, x2]:
                # no corner cutting through obstacles
                if dx and dy and not (free[y, x2] and free[y2, x]):
                    continue
                g2 = gc + c
                if g2 < g[y2, x2]:
                    g[y2, x2] = g2
                    parent[y2, x2] = y * nx + x
                    heapq.heappush(heap, (g2 + np.hypot(gx - x2, gy - y2), g2, x2, y2))
    if not closed[gy, gx]:
        return None
    cells = []
    x, y = gx, gy
    while (x, y) != (sx, sy):
        cells.append((x, y))
        p = parent[y, x]
        y, x = divmod(int(p), nx)
    cells.append((sx, sy))
    cells = np.array(cells[::-1], dtype=float)
    return np.c_[grid.x_min + (cells[:, 0] + 0.5) * res, grid.y_min + (cells[:, 1] + 0.5) * res]


def dijkstra_cost_to_goal(
    grid: OccupancyGrid, goal: Sequence[float], clearance: float, coarse: int = 5
) -> tuple:
    """Obstacle-aware distance-to-goal field on a coarse grid (Hybrid A* heuristic).

    Returns ``(cost_field, resolution)``; unreachable cells are ``inf``.
    """
    if grid.dist is None:
        grid.compute_distance()
    free = grid.dist[::coarse, ::coarse] > clearance
    res = grid.res * coarse
    ny, nx = free.shape
    gx = int((goal[0] - grid.x_min) / res)
    gy = int((goal[1] - grid.y_min) / res)
    cost = np.full((ny, nx), np.inf)
    gx, gy = np.clip(gx, 0, nx - 1), np.clip(gy, 0, ny - 1)
    cost[gy, gx] = 0.0
    heap = [(0.0, gx, gy)]
    moves = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, np.sqrt(2)),
        (1, -1, np.sqrt(2)),
        (-1, 1, np.sqrt(2)),
        (-1, -1, np.sqrt(2)),
    ]
    while heap:
        c, x, y = heapq.heappop(heap)
        if c > cost[y, x]:
            continue
        for dx, dy, w in moves:
            x2, y2 = x + dx, y + dy
            if 0 <= x2 < nx and 0 <= y2 < ny and free[y2, x2]:
                c2 = c + w * res
                if c2 < cost[y2, x2]:
                    cost[y2, x2] = c2
                    heapq.heappush(heap, (c2, x2, y2))
    return cost, res
