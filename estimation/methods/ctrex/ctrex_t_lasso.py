"""
ctrex_t_lasso.py - CT-Rex with the LASSO modification of complex LARS (a variable leaves the path when its
coefficient modulus reaches zero).

    from estimation.methods.ctrex import ctrex_t_lasso
"""
from __future__ import annotations

import numpy as np

from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.ctrex.trex import run_ctrex
from estimation.methods.ctrex.ctrex_t_lars import TLARSSolver


class TLassoSolver(TLARSSolver):
    name = "t-LASSO"

    def _step(self) -> bool:
        if self.n_steps >= self.max_steps or self.C <= 1e-12 * max(1.0, np.linalg.norm(self.y)):
            return False
        if not self.active:
            self._add(int(np.argmax(np.abs(self.c))), self.C)
            return True
        gamma, j, A, w = self._knot()
        limit = gamma if (j is not None and np.isfinite(gamma)) else self.C
        bA = self.b[A]
        w2 = np.abs(w) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            g_min = np.where((np.abs(bA) > 0) & (w2 > 0), -np.real(np.conj(bA) * w) / w2, np.inf)
        g_min = np.where(g_min > 1e-12 * max(self.C, 1e-300), g_min, np.inf)
        k = int(np.argmin(g_min))
        if g_min[k] < min(limit, self.C):
            self._move(A, float(g_min[k]), w)
            self._drop(int(A[k]))
            return True
        if j is None or not np.isfinite(gamma) or gamma >= self.C:
            self._move(A, self.C, w)
            return False
        self._move(A, gamma, w)
        self._add(j, self.C)
        return True

    def _drop(self, k: int):
        self._remove(k)
        self._refresh()
        A = np.asarray(self.active)
        if A.size:
            self.C = float(np.max(np.abs(self.c[A])))


def estimate(gate, array, settings, rng) -> DoAResult:
    angles, info = run_ctrex(TLassoSolver, gate, array, settings, rng)
    return DoAResult(angles, info)


METHODS = [MethodSpec("ctrex_tlasso", r"CT-Rex t-LASSO", "ctrex", estimate)]
