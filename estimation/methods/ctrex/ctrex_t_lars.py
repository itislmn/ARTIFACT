"""
ctrex_t_lars.py - CT-Rex with complex least angle regression (CT-LARS), the reference solver of CT-Rex.

    from estimation.methods.ctrex import ctrex_t_lars
"""
from __future__ import annotations

import numpy as np

from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.ctrex.trex import TerminatingSolver, run_ctrex


def step_to_entry(c: np.ndarray, a: np.ndarray, C: float) -> np.ndarray:
    A2 = np.abs(a) ** 2 - 1.0
    B = np.real(np.conj(c) * a) - C
    C0 = np.abs(c) ** 2 - C ** 2
    sq = np.sqrt(np.maximum(B ** 2 - A2 * C0, 0.0))
    q = B + np.where(B >= 0, sq, -sq)
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(np.abs(A2) > 1e-12, q / A2, np.inf)
        r2 = np.where(np.abs(q) > 1e-300, C0 / q, np.inf)
    roots = np.stack([r1, r2])
    roots = np.where((roots > 1e-14 * max(C, 1e-300)) & (roots <= C * (1 + 1e-9)), roots, np.inf)
    return roots.min(axis=0)


class TLARSSolver(TerminatingSolver):
    name = "t-LARS"

    def _setup(self, **kwargs):
        self.b = np.zeros(self.n_cols, self.X.dtype)
        self.is_active = np.zeros(self.n_cols, bool)
        self._init_residual()
        self.c = self._corr()
        self.C = float(np.max(np.abs(self.c))) if self.c.size else 0.0
        self.Q = np.zeros((self.n_rows, 0), self.X.dtype)
        self.in_span = np.zeros(self.p)
        self.max_steps = 5 * min(self.n_rows, self.n_cols) + 50
        self._a = None

    def _init_residual(self):
        self.r = self.y.copy()

    def _corr(self):
        return self.XH @ self.r

    def _gram(self, A):
        return self.XH[A] @ self.X[:, A]

    def _dir_corr(self, A, w):
        return self.XH @ (self.X[:, A] @ w)

    def _apply_step(self, A, gamma, w):
        self.b[A] += gamma * w
        self.r -= gamma * (self.X[:, A] @ w)

    def _refresh(self):
        nz = np.flatnonzero(self.b)
        self.r = self.y - self.X[:, nz] @ self.b[nz]
        self.c = self._corr()

    def coefficients(self):
        return self.b

    def _rebuild_basis(self):
        if self.active:
            self.Q = np.linalg.qr(self.X[:, self.active])[0]
            self.in_span = np.sum(np.abs(self.XH[:self.p] @ self.Q) ** 2, axis=1)
        else:
            self.Q = np.zeros((self.n_rows, 0), self.X.dtype)
            self.in_span = np.zeros(self.p)

    def _add(self, j, score):
        x = self.X[:, j]
        q = x - self.Q @ (self.Q.conj().T @ x)
        q = q - self.Q @ (self.Q.conj().T @ q)
        nq = np.linalg.norm(q)
        if nq > 1e-10:
            q = q / nq
            self.Q = np.column_stack([self.Q, q])
            self.in_span += np.abs(self.XH[:self.p] @ q) ** 2
        self.active.append(int(j))
        self.is_active[j] = True
        self.entries.append((int(j), float(score)))

    def _remove(self, j):
        self.active.remove(int(j))
        self.is_active[j] = False
        self.b[j] = 0
        self._rebuild_basis()

    def _candidates(self):
        ok = ~self.is_active
        ok[:self.p] &= self.in_span < 1.0 - 1e-8
        if len(self.active) >= self.n_rows:
            ok[:] = False
        return ok

    def _knot(self):
        A = np.asarray(self.active)
        cA = self.c[A]
        s = cA / np.maximum(np.abs(cA), 1e-300)
        G = self._gram(A)
        try:
            w = np.linalg.solve(G, s)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(G, s, rcond=None)[0]
        a = self._dir_corr(A, w)
        self._a = a
        cand = self._candidates()
        if not cand.any():
            return np.inf, None, A, w
        gam = step_to_entry(self.c[cand], a[cand], self.C)
        i = int(np.argmin(gam))
        return float(gam[i]), int(np.flatnonzero(cand)[i]), A, w

    def _move(self, A, gamma, w):
        self._apply_step(A, gamma, w)
        self.c = self.c - gamma * self._a
        self.C = max(self.C - gamma, 0.0)

    def _step(self) -> bool:
        if self.n_steps >= self.max_steps or self.C <= 1e-12 * max(1.0, np.linalg.norm(self.y)):
            return False
        if not self.active:
            self._add(int(np.argmax(np.abs(self.c))), self.C)
            return True
        gamma, j, A, w = self._knot()
        if j is None or not np.isfinite(gamma) or gamma >= self.C:
            self._move(A, self.C, w)
            return False
        self._move(A, gamma, w)
        self._add(j, self.C)
        return True


def lars_entry_lambdas(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    s = TLARSSolver(X, y, None)
    s.execute_step(0, early_stop=False)
    lam = np.zeros(X.shape[1])
    for j, score in s.entries:
        lam[j] = max(lam[j], score)
    return lam


def estimate(gate, array, settings, rng) -> DoAResult:
    angles, info = run_ctrex(TLARSSolver, gate, array, settings, rng)
    return DoAResult(angles, info)


METHODS = [MethodSpec("ctrex_tlars", r"CT-Rex t-LARS", "ctrex", estimate)]
