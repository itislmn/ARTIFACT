"""
ctrex_t_gp.py - CT-Rex with complex gradient pursuit (t-GP).

    from estimation.methods.ctrex import ctrex_t_gp
"""
from __future__ import annotations

import numpy as np

from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.ctrex.trex import TerminatingSolver, run_ctrex


class TGPSolver(TerminatingSolver):
    name = "t-GP"

    def _setup(self, **kwargs):
        self.r = self.y.copy()
        self.b = np.zeros(self.n_cols, self.X.dtype)
        self.is_active = np.zeros(self.n_cols, bool)
        self.y_norm = float(np.linalg.norm(self.y))
        self.max_steps = 5 * min(self.n_rows, self.n_cols) + 50

    def coefficients(self):
        return self.b

    def _direction(self, A, g):
        return g

    def _step(self) -> bool:
        if self.n_steps >= self.max_steps:
            return False
        c = self.XH @ self.r
        j = int(np.argmax(np.abs(c)))
        if abs(c[j]) <= 1e-12 * max(self.y_norm, 1e-300):
            return False
        if not self.is_active[j]:
            self.is_active[j] = True
            self.active.append(j)
            self.entries.append((j, float(abs(c[j]))))
        A = np.asarray(self.active)
        d = self._direction(A, c[A])
        cd = self.X[:, A] @ d
        nn = float(np.vdot(cd, cd).real)
        if nn <= 1e-300:
            return False
        a = np.vdot(cd, self.r) / nn
        self.b[A] += a * d
        self.r = self.r - a * cd
        self._last = (A, d, cd)
        return True


def estimate(gate, array, settings, rng) -> DoAResult:
    angles, info = run_ctrex(TGPSolver, gate, array, settings, rng)
    return DoAResult(angles, info)


METHODS = [MethodSpec("ctrex_tgp", r"CT-Rex t-GP", "ctrex", estimate)]
