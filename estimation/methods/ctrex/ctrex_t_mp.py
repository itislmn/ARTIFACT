"""
ctrex_t_mp.py - CT-Rex with complex matching pursuit (t-MP).

    from estimation.methods.ctrex import ctrex_t_mp
"""
from __future__ import annotations

import numpy as np

from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.ctrex.trex import TerminatingSolver, run_ctrex


class TMPSolver(TerminatingSolver):
    name = "t-MP"

    def _setup(self, **kwargs):
        self.r = self.y.copy()
        self.b = np.zeros(self.n_cols, self.X.dtype)
        self.is_active = np.zeros(self.n_cols, bool)
        self.y_norm = float(np.linalg.norm(self.y))
        self.max_steps = 20 * self.n_rows + 100

    def coefficients(self):
        return self.b

    def _step(self) -> bool:
        if self.n_steps >= self.max_steps:
            return False
        c = self.XH @ self.r
        j = int(np.argmax(np.abs(c)))
        if abs(c[j]) <= 1e-12 * max(self.y_norm, 1e-300):
            return False
        self.b[j] += c[j]
        self.r = self.r - self.X[:, j] * c[j]
        if not self.is_active[j]:
            self.is_active[j] = True
            self.active.append(j)
            self.entries.append((j, float(abs(c[j]))))
        return True


def estimate(gate, array, settings, rng) -> DoAResult:
    angles, info = run_ctrex(TMPSolver, gate, array, settings, rng)
    return DoAResult(angles, info)


METHODS = [MethodSpec("ctrex_tmp", r"CT-Rex t-MP", "ctrex", estimate)]
