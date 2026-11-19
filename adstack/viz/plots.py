"""Static figures and GIF animations for all scenarios (matplotlib only)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib
import numpy as np
from matplotlib.axes import Axes

from ..planning.hybrid_astar import HybridAStarResult
from ..sim.scenarios import AvoidanceLog, ParkingScenario
from ..sim.simulator import SimLog

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from ..planning.path import ReferencePath  # noqa: E402
from ..vehicle.models import VehicleParams, footprint  # noqa: E402

COLORS = {"Pure Pursuit": "#4C72B0", "Stanley": "#DD8452", "MPC": "#55A868"}
EGO = "#2F6DB5"
OBST = "#C44E52"
REF = "#9A9A9A"

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "legend.frameon": False,
    }
)


# --------------------------------------------------------------------------- #
# Vehicle drawing
# --------------------------------------------------------------------------- #
class CarArtist:
    """Vehicle body + four wheels (front wheels turned by the steering angle)."""

    def __init__(
        self,
        ax: Axes,
        params: VehicleParams | None = None,
        color: str = EGO,
        alpha: float = 0.9,
        zorder: int = 5,
    ) -> None:
        self.p = params or VehicleParams()
        self.body = Polygon(
            np.zeros((4, 2)), closed=True, fc=color, ec="k", lw=0.8, alpha=alpha, zorder=zorder
        )
        ax.add_patch(self.body)
        self.wheels = [
            Polygon(np.zeros((4, 2)), closed=True, fc="k", ec="none", zorder=zorder + 1)
            for _ in range(4)
        ]
        for w in self.wheels:
            ax.add_patch(w)

    @staticmethod
    def _rect(cx: float, cy: float, yaw: float, length: float, width: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        hl, hw = length / 2, width / 2
        loc = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
        return loc @ np.array([[c, s], [-s, c]]) + [cx, cy]

    def update(self, x: float, y: float, yaw: float, steer: float = 0.0) -> list:
        """Move the vehicle drawing to a new pose and steering angle."""
        p = self.p
        self.body.set_xy(footprint(x, y, yaw, p))
        c, s = np.cos(yaw), np.sin(yaw)
        tw = p.width / 2 - 0.25
        for i, (lx, ly, st) in enumerate(
            ((0, -tw, 0), (0, tw, 0), (p.wheelbase, -tw, steer), (p.wheelbase, tw, steer))
        ):
            wx = x + c * lx - s * ly
            wy = y + s * lx + c * ly
            self.wheels[i].set_xy(self._rect(wx, wy, yaw + st, 0.7, 0.25))
        return [self.body, *self.wheels]


def _save_gif(anim: FuncAnimation, filename: str, fps: int):
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(anim._fig)


# --------------------------------------------------------------------------- #
# Race track
# --------------------------------------------------------------------------- #
def _track_edges(path: ReferencePath, half_width: float = 4.0):
    nx, ny = -np.sin(path.yaw), np.cos(path.yaw)
    left = np.c_[path.x + half_width * nx, path.y + half_width * ny]
    right = np.c_[path.x - half_width * nx, path.y - half_width * ny]
    return np.vstack((left, left[:1])), np.vstack((right, right[:1]))


def _break_at(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Insert NaN before the given indices (interrupts a plotted line)."""
    return np.insert(y.astype(float), idx, np.nan)


def plot_racetrack_comparison(path: ReferencePath, logs: dict, filename: str) -> None:
    """Trajectories, lateral error, speed and steering of several controllers."""
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.45, wspace=0.32)
    ax = fig.add_subplot(gs[:, 0])
    left, right = _track_edges(path)
    ax.fill(
        np.r_[left[:, 0], right[::-1, 0]], np.r_[left[:, 1], right[::-1, 1]], color="#EDEDED", lw=0
    )
    ax.plot(left[:, 0], left[:, 1], color="#BBBBBB", lw=1)
    ax.plot(right[:, 0], right[:, 1], color="#BBBBBB", lw=1)
    sc = ax.scatter(path.x, path.y, c=path.v_ref * 3.6, cmap="viridis", s=4, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("speed profile [km/h]")
    for name, log in logs.items():
        a = log.arrays()
        ax.plot(a["x"], a["y"], color=COLORS.get(name), lw=1.2, label=name, zorder=3)
    ax.set_aspect("equal")
    ax.set_title("Race track (~790 m): driven trajectories")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left")

    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    for name, log in logs.items():
        a = log.arrays()
        col = COLORS.get(name)
        s = a["s"].copy()
        # insert NaN where s wraps at the finish line so the lines are not joined
        wrap = np.where(np.diff(s) < -0.5 * path.length)[0] + 1
        s = _break_at(s, wrap)
        ax1.plot(s, _break_at(a["e_y"] * 100, wrap), color=col, lw=1, label=name)
        ax2.plot(s, _break_at(np.abs(a["v"]) * 3.6, wrap), color=col, lw=1)
        ax3.plot(s, _break_at(np.rad2deg(a["steer"]), wrap), color=col, lw=1)
    ax2.plot(path.s, path.v_ref * 3.6, color="k", ls="--", lw=1, label="profile")
    ax1.set_ylabel("lateral error [cm]")
    ax1.set_title("Tracking error, speed and steering along the lap")
    ax2.set_ylabel("speed [km/h]")
    ax2.legend(loc="lower right")
    ax3.set_ylabel("steer [deg]")
    ax3.set_xlabel("arc length s [m]")
    fig.savefig(filename, bbox_inches="tight", dpi=110)
    plt.close(fig)


def animate_racetrack(
    path: ReferencePath,
    log: SimLog,
    filename: str,
    fps: int = 15,
    every: int = 2,
    window: float = 45.0,
    t_end: float | None = None,
) -> None:
    """Chase-camera animation with the MPC prediction (green) and reference (dashed)."""
    a = log.arrays()
    preds = dict(log.prediction)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    fig.subplots_adjust(0, 0, 1, 1)
    left, right = _track_edges(path)
    ax.fill(
        np.r_[left[:, 0], right[::-1, 0]], np.r_[left[:, 1], right[::-1, 1]], color="#E6E6E6", lw=0
    )
    ax.plot(left[:, 0], left[:, 1], color="#888", lw=1.5)
    ax.plot(right[:, 0], right[:, 1], color="#888", lw=1.5)
    ax.plot(path.x, path.y, color="w", lw=1, ls=(0, (6, 6)))
    (trail,) = ax.plot([], [], color=EGO, lw=1.5, alpha=0.6)
    (pred_line,) = ax.plot([], [], color=COLORS["MPC"], lw=2.5, marker="o", ms=2.5)
    car = CarArtist(ax)
    txt = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
        bbox=dict(fc="w", ec="none", alpha=0.8),
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ts = list(preds.keys())
    n = len(a["t"]) if t_end is None else int(np.searchsorted(a["t"], t_end))
    frames = range(0, n, every)

    def upd(i: int):
        x, y = a["x"][i], a["y"][i]
        ax.set_xlim(x - window, x + window)
        ax.set_ylim(y - window * 0.69, y + window * 0.69)
        trail.set_data(a["x"][max(0, i - 200) : i + 1], a["y"][max(0, i - 200) : i + 1])
        if ts:
            k = int(np.argmin(np.abs(np.array(ts) - a["t"][i])))
            pr = preds[ts[k]]
            pred_line.set_data(pr[:, 0], pr[:, 1])
        car.update(x, y, a["yaw"][i], a["steer"][i])
        txt.set_text(
            f"MPC   t={a['t'][i]:5.1f}s\nv={abs(a['v'][i]) * 3.6:5.1f} km/h\n"
            f"e_y={a['e_y'][i] * 100:5.1f} cm"
        )
        return []

    anim = FuncAnimation(fig, upd, frames=frames, interval=1000 / fps)
    _save_gif(anim, filename, fps)


# --------------------------------------------------------------------------- #
# Localisation
# --------------------------------------------------------------------------- #
def plot_localization(
    path: ReferencePath, log: SimLog, outages: Sequence[tuple[float, float]], filename: str
) -> None:
    """EKF estimate vs. ground truth, errors with 3-sigma bounds and gyro bias."""
    a = log.arrays()
    est, sd, t, g = a["est"], a["est_std"], a["t"], a["gps"]
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.2], hspace=0.5)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(path.x, path.y, color=REF, lw=6, alpha=0.25, label="track")
    ax.scatter(g[:, 1], g[:, 2], s=3, color="#999", label="GNSS fixes (σ = 0.8 m)", zorder=2)
    ax.plot(a["x"], a["y"], color="k", lw=1, label="ground truth", zorder=3)
    ax.plot(est[:, 0], est[:, 1], color=COLORS["MPC"], lw=1.2, label="EKF estimate", zorder=4)
    for t0, t1 in outages:
        m = (t >= t0) & (t < t1)
        ax.plot(
            a["x"][m],
            a["y"][m],
            color=OBST,
            lw=4,
            alpha=0.5,
            label="GNSS outage (tunnel)",
            zorder=1,
        )
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("EKF localisation (GNSS + IMU + wheel speed)")

    pos_err = np.hypot(est[:, 0] - a["x"], est[:, 1] - a["y"])
    sig = np.hypot(sd[:, 0], sd[:, 1])
    ax1 = fig.add_subplot(gs[0, 1])
    gi = np.clip(np.searchsorted(t, g[:, 0] - 1e-9), 0, len(t) - 1)
    ax1.scatter(
        g[:, 0],
        np.hypot(g[:, 1] - a["x"][gi], g[:, 2] - a["y"][gi]),
        s=3,
        color="#999",
        label="GNSS error",
    )
    ax1.plot(t, pos_err, color=COLORS["MPC"], lw=1.2, label="EKF error")
    ax1.fill_between(t, 0, 3 * sig, color=COLORS["MPC"], alpha=0.15, label="EKF 3σ bound")
    ax1.set_ylabel("position error [m]")
    ax1.legend(loc="upper right", fontsize=8, ncol=3)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    yaw_err = np.rad2deg(np.angle(np.exp(1j * (est[:, 2] - a["yaw"]))))
    ax2.plot(t, yaw_err, color=COLORS["MPC"], lw=1)
    ax2.fill_between(
        t, -3 * np.rad2deg(sd[:, 2]), 3 * np.rad2deg(sd[:, 2]), color=COLORS["MPC"], alpha=0.15
    )
    ax2.set_ylabel("heading error [deg]")
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax3.plot(t, np.rad2deg(est[:, 4]), color=COLORS["MPC"], lw=1.2, label="estimated gyro bias")
    ax3.axhline(np.rad2deg(0.02), color="k", ls="--", lw=1, label="true bias")
    ax3.set_ylabel("bias [deg/s]")
    ax3.set_xlabel("t [s]")
    ax3.legend(loc="lower right", fontsize=8)
    for axx in (ax1, ax2, ax3):
        for t0, t1 in outages:
            axx.axvspan(t0, t1, color=OBST, alpha=0.12)
    fig.savefig(filename, bbox_inches="tight", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Parking
# --------------------------------------------------------------------------- #
def _draw_parking_scene(ax: Axes, sc: ParkingScenario):
    ax.add_patch(
        Rectangle(
            (sc.grid.x_min, sc.grid.y_min),
            sc.grid.nx * sc.grid.res,
            sc.grid.ny * sc.grid.res,
            fc="#E9E9E9",
            ec="none",
            zorder=0,
        )
    )
    face = {"car": "#9DB4C0", "kerb": "#BDB5A6", "wall": "#6E6E6E"}
    for poly, label in zip(sc.grid.polygons, sc.grid.labels, strict=False):
        ax.add_patch(
            Polygon(poly, closed=True, fc=face.get(label, "#7A7A7A"), ec="#555", lw=0.6, zorder=1)
        )
    p = VehicleParams()
    goal_fp = footprint(*sc.goal, p)
    ax.add_patch(
        Polygon(goal_fp, closed=True, fc="none", ec=COLORS["MPC"], lw=1.5, ls="--", zorder=2)
    )
    ax.set_xlim(sc.grid.extent[:2])
    ax.set_ylim(sc.grid.extent[2:])
    ax.set_aspect("equal")


def plot_parking(scenarios: list, filename: str) -> None:
    """Planned and driven paths of one or more parking scenarios."""
    fig, axes = plt.subplots(
        len(scenarios),
        1,
        figsize=(11, 4.6 * len(scenarios)),
        gridspec_kw={"height_ratios": [s[0].grid.ny for s in scenarios]},
    )
    axes = np.atleast_1d(axes)
    p = VehicleParams()
    for ax, (sc, res, log) in zip(axes, scenarios, strict=False):
        _draw_parking_scene(ax, sc)
        a = log.arrays()
        fwd = res.direction > 0
        ax.plot(
            np.where(fwd, res.x, np.nan),
            np.where(fwd, res.y, np.nan),
            color=EGO,
            lw=2,
            label="planned (forward)",
        )
        ax.plot(
            np.where(~fwd, res.x, np.nan),
            np.where(~fwd, res.y, np.nan),
            color=OBST,
            lw=2,
            label="planned (reverse)",
        )
        ax.plot(a["x"], a["y"], color="k", lw=1, ls="--", label="driven (MPC)")
        for i in np.linspace(0, len(a["x"]) - 1, 14).astype(int):
            ax.add_patch(
                Polygon(
                    footprint(a["x"][i], a["y"][i], a["yaw"][i], p),
                    closed=True,
                    fc=EGO,
                    alpha=0.12,
                    ec=EGO,
                    lw=0.5,
                    zorder=3,
                )
            )
        ax.set_title(
            f"{sc.name.capitalize()} parking - Hybrid A* + Reeds-Shepp, "
            f"{res.n_switches} direction change(s), path {res.length:.1f} m, "
            f"planning {res.time:.2f} s"
        )
        ax.legend(loc="center right", fontsize=8, framealpha=0.9, frameon=True)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=110)
    plt.close(fig)


def animate_parking(
    sc: ParkingScenario,
    res: HybridAStarResult,
    log: SimLog,
    filename: str,
    fps: int = 15,
    every: int = 3,
) -> None:
    """GIF of the parking manoeuvre."""
    a = log.arrays()
    w = sc.grid.nx * sc.grid.res
    h = sc.grid.ny * sc.grid.res
    fig, ax = plt.subplots(figsize=(6.4, 6.4 * h / w + 0.4))
    fig.subplots_adjust(0, 0, 1, 0.94)
    _draw_parking_scene(ax, sc)
    ax.set_axis_off()
    fwd = res.direction > 0
    ax.plot(
        np.where(fwd, res.x, np.nan), np.where(fwd, res.y, np.nan), color=EGO, lw=1.5, alpha=0.7
    )
    ax.plot(
        np.where(~fwd, res.x, np.nan), np.where(~fwd, res.y, np.nan), color=OBST, lw=1.5, alpha=0.7
    )
    (trail,) = ax.plot([], [], color="k", lw=1)
    car = CarArtist(ax)
    title = ax.set_title("", fontsize=10)
    frames = list(range(0, len(a["t"]), every)) + [len(a["t"]) - 1] * fps

    def upd(i: int):
        car.update(a["x"][i], a["y"][i], a["yaw"][i], a["steer"][i])
        trail.set_data(a["x"][: i + 1], a["y"][: i + 1])
        gear = "D" if a["v"][i] > 0.02 else ("R" if a["v"][i] < -0.02 else "-")
        title.set_text(
            f"{sc.name} parking   t={a['t'][i]:4.1f}s   gear {gear}   "
            f"v={a['v'][i]:+.2f} m/s   steer={np.rad2deg(a['steer'][i]):+5.1f}°"
        )
        return []

    anim = FuncAnimation(fig, upd, frames=frames, interval=1000 / fps)
    _save_gif(anim, filename, fps)


# --------------------------------------------------------------------------- #
# Obstacle avoidance
# --------------------------------------------------------------------------- #
def _road_lines(ref: ReferencePath, lane_width: float = 3.5, n_lanes: int = 2):
    lines = []
    for k, d in enumerate(np.arange(-0.5, n_lanes) * lane_width):
        x, y = ref.to_cartesian(ref.s, np.full_like(ref.s, d))
        lines.append((x, y, "solid" if k in (0, n_lanes) else "dashed"))
    return lines


def _obstacle_poly(ref: ReferencePath, s: float, d: float, length: float, width: float):
    ss = np.array([s - length / 2, s + length / 2, s + length / 2, s - length / 2])
    dd = np.array([d - width / 2, d - width / 2, d + width / 2, d + width / 2])
    # sample along s so the polygon follows curved roads
    sx = np.r_[np.linspace(ss[0], ss[1], 6), np.linspace(ss[2], ss[3], 6)]
    sd = np.r_[np.full(6, dd[0]), np.full(6, dd[2])]
    x, y = ref.to_cartesian(sx, sd)
    return np.c_[x, y]


def plot_avoidance(av: AvoidanceLog, filename: str, snapshot_every: float = 4.0) -> None:
    """Lateral offset along the road with time-synchronised snapshots of the ego
    vehicle and the obstacles (same colour = same time)."""
    a = av.sim.arrays()
    p = VehicleParams()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]}
    )
    for k, d in enumerate([-1.75, 1.75, 5.25]):
        ax1.axhline(d, color="#777", lw=1.2 if k != 1 else 1, ls="-" if k != 1 else "--")
    t = a["t"]
    snaps = np.arange(0.0, t[-1], snapshot_every)
    cmap = plt.get_cmap("viridis")
    for j, ts in enumerate(snaps):
        i = int(np.searchsorted(t, ts))
        col = cmap(j / max(len(snaps) - 1, 1))
        ax1.add_patch(
            Rectangle(
                (a["s"][i] - p.length / 2, a["e_y"][i] - p.width / 2),
                p.length,
                p.width,
                fc=col,
                ec="k",
                lw=0.6,
                alpha=0.9,
                zorder=4,
            )
        )
        ax1.text(
            a["s"][i],
            a["e_y"][i] - p.width / 2 - 0.45,
            f"{ts:.0f}s",
            ha="center",
            fontsize=7,
            color=col,
        )
        for (s_o, d_o, L, W, _name), (s_0, *_rest) in zip(
            av.obstacles[i][1], av.obstacles[0][1], strict=False
        ):
            moving = abs(s_o - s_0) > 0.5
            if moving:
                ax1.add_patch(
                    Rectangle(
                        (s_o - L / 2, d_o - W / 2),
                        L,
                        W,
                        fc=col,
                        ec=OBST,
                        lw=1.2,
                        alpha=0.35,
                        hatch="///",
                        zorder=3,
                    )
                )
    for s_o, d_o, L, W, name in av.obstacles[0][1]:
        moving = any(
            abs(o[1][k][0] - s_o) > 0.5
            for o in av.obstacles[-1:]
            for k in range(len(o[1]))
            if o[1][k][4] == name
        )
        if not moving:
            ax1.add_patch(
                Rectangle((s_o - L / 2, d_o - W / 2), L, W, fc=OBST, ec="k", lw=0.6, zorder=3)
            )
        ax1.text(
            s_o - L / 2,
            d_o + W / 2 + 0.15,
            name + (" (moving, hatched)" if moving else ""),
            fontsize=8,
            color=OBST,
        )
    ax1.plot(a["s"], a["e_y"], color=EGO, lw=1.5, label="ego path (vehicle centre)", zorder=2)
    ax1.set_ylabel("lateral offset d [m]")
    ax1.set_ylim(-2.4, 6.4)
    ax1.set_title("Frenet planner: ego (solid) and moving vehicles (hatched) at the same instants")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.plot(a["s"], a["v"] * 3.6, color=EGO, lw=1.5, label="speed")
    ax2.plot(a["s"], a["v_ref"] * 3.6, color="k", lw=1, ls="--", label="planner target speed")
    ax2.set_ylabel("speed [km/h]")
    ax2.set_xlabel("s [m]")
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=110)
    plt.close(fig)


def animate_avoidance(
    av: AvoidanceLog, filename: str, fps: int = 12, every: int = 3, window: float = 28.0
) -> None:
    """GIF of the avoidance scenario including the planner candidates."""
    a = av.sim.arrays()
    ref = av.ref
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=90)
    fig.subplots_adjust(0, 0, 1, 1)
    for x, y, ls in _road_lines(ref):
        ax.plot(
            x,
            y,
            color="#666",
            lw=1.5 if ls == "solid" else 1,
            ls="-" if ls == "solid" else (0, (6, 6)),
        )
    plan_ts = np.array([t for t, _ in av.plans])
    cand_lines = [ax.plot([], [], lw=0.6)[0] for _ in range(60)]
    (best_line,) = ax.plot([], [], color=COLORS["MPC"], lw=3, alpha=0.9)
    obst_patches = [
        Polygon(np.zeros((3, 2)), closed=True, fc=OBST, ec="k", lw=0.6, zorder=4)
        for _ in av.obstacles[0][1]
    ]
    for pch in obst_patches:
        ax.add_patch(pch)
    car = CarArtist(ax)
    txt = ax.text(
        0.01,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
        bbox=dict(fc="w", ec="none", alpha=0.85),
    )
    ax.set_aspect("equal")
    ax.set_axis_off()

    def upd(i: int):
        x, y = a["x"][i], a["y"][i]
        ax.set_xlim(x - window * 0.45, x + window * 1.55)
        ax.set_ylim(y - window * 0.5, y + window * 0.5)
        k = int(np.searchsorted(plan_ts, a["t"][i], side="right") - 1)
        plan = av.plans[max(k, 0)][1]
        # draw a subset of candidates: grey = free, red = colliding
        cands = [c for c in plan.candidates if c.v_target == plan.best.v_target]
        for line, cd in zip(cand_lines, cands + [None] * len(cand_lines), strict=False):
            if cd is None:
                line.set_data([], [])
                continue
            line.set_data(cd.x, cd.y)
            line.set_color(OBST if cd.collision else "#888")
            line.set_alpha(0.35 if cd.collision else 0.5)
        best_line.set_data(plan.best.x, plan.best.y)
        for pch, (s, d, L, W, _) in zip(obst_patches, av.obstacles[i][1], strict=False):
            pch.set_xy(_obstacle_poly(ref, s, d, L, W))
        car.update(x, y, a["yaw"][i], a["steer"][i])
        txt.set_text(
            f"t={a['t'][i]:5.1f}s  v={a['v'][i] * 3.6:5.1f} km/h  "
            f"target={plan.v_target * 3.6:5.1f} km/h\nmode: {plan.mode}"
        )
        return []

    anim = FuncAnimation(fig, upd, frames=range(0, len(a["t"]), every), interval=1000 / fps)
    _save_gif(anim, filename, fps)
