"""
ctrex_t_acgp.py - CT-Rex with complex approximate conjugate gradient pursuit (t-ACGP).

    from estimation.methods.ctrex import ctrex_t_acgp
"""
from __future__ import annotations

import numpy as np

from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.ctrex.trex import run_ctrex
from estimation.methods.ctrex.ctrex_t_gp import TGPSolver


class TACGPSolver(TGPSolver):
    name = "t-ACGP"

    def _setup(self, **kwargs):
        super()._setup(**kwargs)
        self._last = None

    def _direction(self, A, g):
        if self._last is None:
            return g
        A_prev, d_prev, c_prev = self._last
        nn = float(np.vdot(c_prev, c_prev).real)
        if nn <= 1e-300:
            return g
        b = -np.vdot(c_prev, self.X[:, A] @ g) / nn
        d_ext = np.zeros(len(A), self.X.dtype)
        pos = {int(k): i for i, k in enumerate(A)}
        for k, dk in zip(A_prev, d_prev):
            d_ext[pos[int(k)]] = dk
        return g + b * d_ext


def estimate(gate, array, settings, rng) -> DoAResult:
    angles, info = run_ctrex(TACGPSolver, gate, array, settings, rng)
    return DoAResult(angles, info)


METHODS = [MethodSpec("ctrex_tacgp", r"CT-Rex t-ACGP", "ctrex", estimate)]
