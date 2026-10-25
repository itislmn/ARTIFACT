"""
ctrex_t_ols.py - CT-Rex with complex orthogonal least squares (t-OLS).

    from estimation.methods.ctrex import ctrex_t_ols
"""
from __future__ import annotations

import numpy as np

from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.ctrex.trex import run_ctrex
from estimation.methods.ctrex.ctrex_t_omp import TOMPSolver


class TOLSSolver(TOMPSolver):
    name = "t-OLS"

    def _setup(self, **kwargs):
        super()._setup(**kwargs)
        self.perp2 = np.ones(self.n_cols)

    def _scores(self, c):
        ok = self.perp2 > 1e-10
        s = np.full(c.shape, -np.inf)
        s[ok] = np.abs(c[ok]) / np.sqrt(self.perp2[ok])
        return s

    def _after_add(self, q):
        self.perp2 = np.maximum(self.perp2 - np.abs(self.XH @ q) ** 2, 0.0)


def estimate(gate, array, settings, rng) -> DoAResult:
    angles, info = run_ctrex(TOLSSolver, gate, array, settings, rng)
    return DoAResult(angles, info)


METHODS = [MethodSpec("ctrex_tols", r"CT-Rex t-OLS", "ctrex", estimate)]
