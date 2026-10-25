"""
ctrex_t_omp.py - CT-Rex with complex orthogonal matching pursuit (t-OMP).

    from estimation.methods.ctrex import ctrex_t_omp
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.ctrex.trex import TerminatingSolver, run_ctrex


class TOMPSolver(TerminatingSolver):
    name = "t-OMP"

    def _setup(self, **kwargs):
        self.r = self.y.copy()
        self.Q = np.zeros((self.n_rows, 0), self.X.dtype)
        self.R = np.zeros((0, 0), self.X.dtype)
        self.is_active = np.zeros(self.n_cols, bool)
        self.y_norm = float(np.linalg.norm(self.y))

    def _scores(self, c):
        return np.abs(c)

    def _orthogonalise(self, j):
        x = self.X[:, j]
        proj = self.Q.conj().T @ x
        q = x - self.Q @ proj
        corr = self.Q.conj().T @ q
        q, proj = q - self.Q @ corr, proj + corr
        return q, proj, float(np.linalg.norm(q))

    def _step(self) -> bool:
        if len(self.active) >= min(self.n_rows, self.n_cols):
            return False
        c = self.XH @ self.r
        score = self._scores(c)
        score[self.is_active] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]) or abs(c[j]) <= 1e-12 * max(self.y_norm, 1e-300):
            return False
        q, proj, nq = self._orthogonalise(j)
        if nq < 1e-10:
            self.is_active[j] = True
            return True
        q = q / nq
        k = self.R.shape[0]
        R = np.zeros((k + 1, k + 1), self.X.dtype)
        R[:k, :k], R[:k, k], R[k, k] = self.R, proj, nq
        self.R, self.Q = R, np.column_stack([self.Q, q])
        self.r = self.r - q * (q.conj() @ self.r)
        self.active.append(j)
        self.is_active[j] = True
        self.entries.append((j, float(score[j])))
        self._after_add(q)
        return True

    def _after_add(self, q):
        pass

    def coefficients(self):
        b = np.zeros(self.n_cols, self.X.dtype)
        if self.active:
            b[self.active] = solve_triangular(self.R, self.Q.conj().T @ self.y)
        return b


def estimate(gate, array, settings, rng) -> DoAResult:
    angles, info = run_ctrex(TOMPSolver, gate, array, settings, rng)
    return DoAResult(angles, info)


METHODS = [MethodSpec("ctrex_tomp", r"CT-Rex t-OMP", "ctrex", estimate)]
