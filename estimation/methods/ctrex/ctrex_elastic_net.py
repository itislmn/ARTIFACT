"""
ctrex_elastic_net.py - CT-Rex with the complex elastic net (row-augmented CT-LARS, lambda_2 by ridge
cross-validation with the one-standard-error rule).

    from estimation.methods.ctrex import ctrex_elastic_net
"""
from __future__ import annotations

import numpy as np

from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.ctrex.trex import run_ctrex
from estimation.methods.ctrex.ctrex_t_lasso import TLassoSolver


class TENetSolver(TLassoSolver):
    name = "t-EN"

    def _setup(self, lambda2: float = 1.0, **kwargs):
        self.lam2 = float(lambda2)
        self.s = float(np.sqrt(1.0 + self.lam2))
        super()._setup(**kwargs)

    def _init_residual(self):
        self.r = self.y.copy()

    def _corr(self):
        return (self.XH @ self.r - self.lam2 * self.b / self.s) / self.s

    def _gram(self, A):
        return (self.XH[A] @ self.X[:, A] + self.lam2 * np.eye(len(A))) / self.s ** 2

    def _dir_corr(self, A, w):
        v = self.XH @ (self.X[:, A] @ w)
        v[A] += self.lam2 * w
        return v / self.s ** 2

    def _apply_step(self, A, gamma, w):
        self.b[A] += gamma * w
        self.r -= gamma * (self.X[:, A] @ w) / self.s

    def _refresh(self):
        nz = np.flatnonzero(self.b)
        self.r = self.y - self.X[:, nz] @ self.b[nz] / self.s
        self.c = self._corr()

    def _candidates(self):
        return ~self.is_active


def ridge_cv_lambda2(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                     n_folds: int = 10, n_grid: int = 40) -> float:
    n = X.shape[0]
    Xn = X / np.linalg.norm(X, axis=0, keepdims=True)
    folds = np.array_split(rng.permutation(n), min(n_folds, n))
    grid = np.linalg.norm(Xn, 2) ** 2 * np.logspace(-4, 2, n_grid)
    err = np.zeros((len(folds), n_grid))
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test)
        U, s, Vh = np.linalg.svd(Xn[train], full_matrices=False)
        Uy = U.conj().T @ y[train]
        for i, lam in enumerate(grid):
            beta = Vh.conj().T @ (s / (s ** 2 + lam) * Uy)
            err[f, i] = np.mean(np.abs(y[test] - Xn[test] @ beta) ** 2)
    m = err.mean(axis=0)
    se = err.std(axis=0, ddof=1) / np.sqrt(len(folds))
    i_min = int(np.argmin(m))
    return float(grid[np.flatnonzero(m <= m[i_min] + se[i_min]).max()])


def estimate(gate, array, settings, rng) -> DoAResult:
    G = gate.groups(settings.n_groups)
    lam2 = ridge_cv_lambda2(array.dictionary(G), gate.stacked_snapshot(G), rng)
    angles, info = run_ctrex(TENetSolver, gate, array, settings, rng, solver_kwargs={"lambda2": lam2})
    info["stats"]["lambda2"] = lam2
    return DoAResult(angles, info)


METHODS = [MethodSpec("ctrex_ten", r"CT-Rex t-ElasticNet", "ctrex", estimate)]
