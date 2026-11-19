"""
Linear time-varying Model Predictive Control (LTV-MPC) for combined lateral and
longitudinal control, solved as a sparse QP with OSQP.

Prediction model: kinematic bicycle (rear axle), state ``x = [X, Y, yaw, v]``,
input ``u = [a, delta]``, linearised in every cycle along the previous optimal
trajectory and discretised with explicit Euler:

    x_{k+1} = A_k x_k + B_k u_k + c_k

Optimisation problem (horizon N):

    min   sum_k ||x_k - x_ref,k||^2_Q + ||u_k - u_ref,k||^2_R + ||u_k - u_{k-1}||^2_Rd
    s.t.  dynamics, |delta| <= delta_max, a_min <= a <= a_max,
          |delta_k - delta_{k-1}| <= delta_rate * dt,  |v| <= v_max

The sparsity pattern of the QP is fixed, so OSQP is set up once and afterwards
only updated (matrix values + vectors) and warm-started.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import osqp
import scipy.sparse as sp

from ..planning.path import ReferencePath
from ..vehicle.models import VehicleParams, VehicleState, wrap_angle
from .base import Controller, ControlOutput

NX, NU = 4, 2


class MPC(Controller):
    """LTV-MPC for combined lateral and longitudinal control (see module docstring)."""

    name = "MPC"

    def __init__(
        self,
        horizon: int = 20,
        dt_pred: float = 0.1,
        Q: Sequence[float] = (20.0, 20.0, 5.0, 0.6),
        Qf_scale: float = 4.0,
        R: Sequence[float] = (0.05, 2.0),
        Rd: Sequence[float] = (0.3, 100.0),
        a_ref_acc: float = 2.5,
        a_ref_dec: float = 5.0,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.N = horizon
        self.dtp = dt_pred
        self.Q = np.diag(Q)
        self.Qf = self.Q * Qf_scale
        self.R = np.diag(R)
        self.Rd = np.diag(Rd)
        self.a_ref_acc, self.a_ref_dec = a_ref_acc, a_ref_dec
        self._solver = None
        self._x_pred = None
        self._u_pred = None
        self.u_prev = np.zeros(NU)
        self._build_static()

    def reset(self) -> None:
        """Reset the internal state, including the QP solver and the warm start."""
        super().reset()
        self._solver = None
        self._x_pred = None
        self._u_pred = None
        self.u_prev = np.zeros(NU)

    # ------------------------------------------------------------------ #
    # QP structure
    # ------------------------------------------------------------------ #
    def _xi(self, k: int):  # index of state k in z
        return NX * k

    def _ui(self, k: int):  # index of input k in z
        return NX * (self.N + 1) + NU * k

    def _build_static(self):
        N = self.N
        nz = NX * (N + 1) + NU * N
        self.nz = nz

        # ---- Hessian P (constant) ----
        P = sp.lil_matrix((nz, nz))
        for k in range(1, N + 1):
            i = self._xi(k)
            P[i : i + NX, i : i + NX] = self.Qf if k == N else self.Q
        for k in range(N):
            i = self._ui(k)
            # R + Rd (u_k - u_{k-1}) terms
            n_rd = 2 if k < N - 1 else 1
            P[i : i + NU, i : i + NU] = self.R + n_rd * self.Rd
            if k > 0:
                j = self._ui(k - 1)
                P[i : i + NU, j : j + NU] = -self.Rd
                P[j : j + NU, i : i + NU] = -self.Rd
        self.P = sp.triu(2.0 * P.tocsc(), format="csc")

        # ---- constraint sparsity pattern (rows, cols) ----
        rows, cols = [], []
        r = 0
        # initial state
        for a in range(NX):
            rows.append(r + a)
            cols.append(self._xi(0) + a)
        r += NX
        # dynamics
        for k in range(N):
            for a in range(NX):
                for b in range(NX):  # -A_k
                    rows.append(r + a)
                    cols.append(self._xi(k) + b)
                for b in range(NU):  # -B_k
                    rows.append(r + a)
                    cols.append(self._ui(k) + b)
                rows.append(r + a)
                cols.append(self._xi(k + 1) + a)  # +I
            r += NX
        self.n_eq = r
        # input bounds
        for k in range(N):
            for b in range(NU):
                rows.append(r)
                cols.append(self._ui(k) + b)
                r += 1
        # steering rate
        for k in range(N):
            rows.append(r)
            cols.append(self._ui(k) + 1)
            if k > 0:
                rows.append(r)
                cols.append(self._ui(k - 1) + 1)
            r += 1
        # speed bounds
        for k in range(1, N + 1):
            rows.append(r)
            cols.append(self._xi(k) + 3)
            r += 1
        self.n_con = r
        self._rows = np.array(rows)
        self._cols = np.array(cols)
        # permutation that maps our COO order to CSC data order
        A0 = sp.csc_matrix(
            (np.arange(1, len(rows) + 1, dtype=float), (self._rows, self._cols)),
            shape=(self.n_con, nz),
        )
        self._perm = (A0.data - 1).astype(int)
        self._A_indices, self._A_indptr = A0.indices.copy(), A0.indptr.copy()

    # ------------------------------------------------------------------ #
    def _linearise(self, xb: np.ndarray, ub: np.ndarray):
        L = self.p.wheelbase
        _, _, yaw, v = xb
        _, d = ub
        c, s, t = np.cos(yaw), np.sin(yaw), np.tan(d)
        Jx = np.array([[0, 0, -v * s, c], [0, 0, v * c, s], [0, 0, 0, t / L], [0, 0, 0, 0]])
        Ju = np.array([[0, 0], [0, 0], [0, v / (L * np.cos(d) ** 2)], [1, 0]])
        f = np.array([v * c, v * s, v * t / L, ub[0]])
        A = np.eye(NX) + self.dtp * Jx
        B = self.dtp * Ju
        cvec = self.dtp * (f - Jx @ xb - Ju @ ub)
        return A, B, cvec

    def _reference(self, state: VehicleState, path: ReferencePath, s0: float):
        """Reference states/inputs along the path for the prediction horizon."""
        N, d = self.N, path.direction
        xr = np.zeros((N + 1, NX))
        ur = np.zeros((N, NU))
        s, v_prog = s0, abs(state.v)
        yaw_prev = state.yaw
        for k in range(N + 1):
            x, y, yaw, kappa = path.interpolate(s)
            yaw = yaw_prev + wrap_angle(yaw - yaw_prev)  # unwrap relative to vehicle
            yaw_prev = yaw
            xr[k] = (x, y, yaw, d * v_prog)
            if k < N:
                ur[k, 1] = np.arctan(self.p.wheelbase * kappa)
                v_t = float(path.speed_at(s))
                v_next = float(
                    np.clip(
                        v_t, v_prog - self.a_ref_dec * self.dtp, v_prog + self.a_ref_acc * self.dtp
                    )
                )
                ur[k, 0] = d * (v_next - v_prog) / self.dtp
                s = s + 0.5 * (v_prog + v_next) * self.dtp
                v_prog = v_next
                if not path.closed:
                    s = min(s, path.s[-1])
        return xr, ur

    # ------------------------------------------------------------------ #
    def control(self, state: VehicleState, path: ReferencePath) -> ControlOutput:
        """Solve the QP for the current state and return the first input of the plan."""
        t0 = time.perf_counter()
        p: VehicleParams = self.p
        N = self.N
        s0, e_y, _ = self._progress(state, path)
        x0 = np.array([state.x, state.y, state.yaw, state.v])
        xr, ur = self._reference(state, path, s0)

        # linearisation trajectory: shifted previous solution, else reference
        if self._x_pred is not None:
            xb = np.vstack((self._x_pred[1:], self._x_pred[-1:]))
            ub = np.vstack((self._u_pred[1:], self._u_pred[-1:]))
            xb[:, 2] = xr[:, 2] + wrap_angle(xb[:, 2] - xr[:, 2])
            xb[0] = x0
        else:
            xb, ub = xr.copy(), ur.copy()
            xb[0] = x0
            ub[:, 0] = 0.0

        # ---- constraint values ----
        vals = []
        l_eq = np.zeros(self.n_eq)
        l_eq[:NX] = x0
        vals.extend([1.0] * NX)
        for k in range(N):
            A, B, c = self._linearise(xb[k], ub[k])
            for a in range(NX):
                vals.extend(-A[a])
                vals.extend(-B[a])
                vals.append(1.0)
            l_eq[NX * (k + 1) : NX * (k + 2)] = c
        u_eq = l_eq.copy()

        vals.extend([1.0] * (NU * N))
        lu = np.tile([-p.max_decel, -p.max_steer], N)
        uu = np.tile([p.max_accel, p.max_steer], N)

        rate = p.max_steer_rate * self.dtp
        lr_, ur_ = np.empty(N), np.empty(N)
        for k in range(N):
            if k == 0:
                vals.append(1.0)
                lr_[k], ur_[k] = self.u_prev[1] - rate, self.u_prev[1] + rate
            else:
                vals.extend([1.0, -1.0])
                lr_[k], ur_[k] = -rate, rate
        vals.extend([1.0] * N)
        lv = np.full(N, -p.max_speed)
        uv = np.full(N, p.max_speed)

        l = np.concatenate((l_eq, lu, lr_, lv))
        u = np.concatenate((u_eq, uu, ur_, uv))
        Ax = np.asarray(vals, dtype=float)[self._perm]

        # ---- linear cost ----
        q = np.zeros(self.nz)
        for k in range(1, N + 1):
            W = self.Qf if k == N else self.Q
            i = self._xi(k)
            q[i : i + NX] = -2.0 * W @ xr[k]
        for k in range(N):
            i = self._ui(k)
            q[i : i + NU] = -2.0 * self.R @ ur[k]
        q[self._ui(0) : self._ui(0) + NU] += -2.0 * self.Rd @ self.u_prev

        if self._solver is None:
            A = sp.csc_matrix((Ax, self._A_indices, self._A_indptr), shape=(self.n_con, self.nz))
            self._solver = osqp.OSQP()
            self._solver.setup(
                self.P,
                q,
                A,
                l,
                u,
                verbose=False,
                warm_starting=True,
                polishing=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=4000,
            )
        else:
            self._solver.update(q=q, l=l, u=u, Ax=Ax)
        res = self._solver.solve(raise_error=False)

        status = res.info.status
        if res.x is None or not np.all(np.isfinite(res.x)):
            # fall back to the shifted previous plan (or the reference)
            u0 = ub[0].copy()
            ok = False
        else:
            z = res.x
            self._x_pred = z[: NX * (N + 1)].reshape(N + 1, NX)
            self._u_pred = z[NX * (N + 1) :].reshape(N, NU)
            u0 = self._u_pred[0].copy()
            ok = True
        self.u_prev = u0
        dt_solve = time.perf_counter() - t0
        return ControlOutput(
            float(u0[0]),
            float(u0[1]),
            {
                "s": s0,
                "e_y": e_y,
                "v_ref": xr[0, 3],
                "solve_time": dt_solve,
                "status": status,
                "ok": ok,
                "prediction": None if self._x_pred is None else self._x_pred.copy(),
                "reference": xr,
            },
        )
