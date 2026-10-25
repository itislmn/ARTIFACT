"""
ctrex_dependency_aware.py - dependency-aware CT-Rex (CT-Rex+DA): the relative occurrences deflated within groups of
correlated grid angles before the voting.

    from estimation.methods.ctrex import ctrex_dependency_aware
"""
from __future__ import annotations

import numpy as np

from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.ctrex.trex import TRexSelector, deflated_relative_occurrence, fdp_hat, run_ctrex
from estimation.methods.ctrex.ctrex_t_lars import TLARSSolver
from estimation.methods.ctrex.ctrex_t_omp import TOMPSolver

RHO_GRID_LENGTH = 20


class TRexDASelector(TRexSelector):

    def _choose(self, cal):
        exps, T_stop, V, L, p = cal["exps"], cal["T_stop"], cal["V"], cal["L"], cal["p"]
        X = exps[0].X[:, :p]
        coh = np.abs(X.conj().T @ X)
        np.fill_diagonal(coh, 0.0)
        vals = np.unique(np.round(coh[np.triu_indices(p, 1)], 6))
        rho_full = np.append(vals, 1.0)
        idx = np.unique(np.round(np.linspace(0, rho_full.size - 1, min(RHO_GRID_LENGTH, p))).astype(int))
        rho_grid = rho_full[idx]
        T_max = T_stop - 1 if T_stop > 1 else 1
        phi = self.phi_mat(exps, T_max, p)
        best = None
        for ri, rho in enumerate(rho_grid):
            nbr = coh >= rho
            diff = np.abs(phi[:, None, :] - phi[None, :, :])
            diff = np.where(nbr[:, :, None], diff, np.inf)
            dmin = diff.min(axis=1)
            delta = np.where(np.isfinite(dmin), 2.0 - dmin, 2.0)
            phi_d = phi / delta
            for T in range(1, T_max + 1):
                Phi = phi_d[:, T - 1]
                fdp = fdp_hat(V, Phi, deflated_relative_occurrence(p, T, L, phi_d[:, :T], Phi))
                R = (Phi[:, None] > V[None, :]).sum(axis=0)
                for vi in np.flatnonzero(fdp <= self.tFDR):
                    key = (int(R[vi]), int(vi), ri, T)
                    if best is None or key > best:
                        best = key
                        sel = np.flatnonzero(Phi > V[vi])
        if best is None or best[0] == 0:
            return np.zeros(0, int), None, 1, {"rho": None}
        return sel, float(V[best[1]]), best[3], {"rho": float(rho_grid[best[2]])}


def estimate_lars(gate, array, settings, rng) -> DoAResult:
    angles, info = run_ctrex(TLARSSolver, gate, array, settings, rng, selector_cls=TRexDASelector)
    return DoAResult(angles, info)


def estimate_omp(gate, array, settings, rng) -> DoAResult:
    angles, info = run_ctrex(TOMPSolver, gate, array, settings, rng, selector_cls=TRexDASelector)
    return DoAResult(angles, info)


METHODS = [
    MethodSpec("ctrex_tlars_da", r"CT-Rex+DA t-LARS", "ctrex", estimate_lars),
    MethodSpec("ctrex_tomp_da", r"CT-Rex+DA t-OMP", "ctrex", estimate_omp),
]
