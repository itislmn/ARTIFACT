"""
fixed_x_knockoffs.py - fixed-X knockoffs as the R package knockoff runs them by default: SDP knockoffs with rows added
for p < n < 2p, cross-validated lasso coefficient difference, knockoff+ threshold (or the knockoff threshold, offset 0);
on the Nyquist grid as the real/imaginary split (n = 24 > 22 variables), so one snapshot is enough. The knockoffs depend
on the steering matrix only and are built once; the lasso fits are read off the exact lasso (LARS) path.

    from estimation.methods import fixed_x_knockoffs
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from estimation.core.array import local_maxima
from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.bh import locate
from estimation.methods.model_x_knockoffs import glmnet_coefdiff, real_split, solve_sdp, statistics, threshold

TOL = 1e-5


def canonical_svd(X):
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    for j in range(d.size):
        if U[np.argmax(np.abs(U[:, j])), j] < 0:
            U[:, j], Vt[j] = -U[:, j], -Vt[j]
    return U, d, Vt.T


@lru_cache(maxsize=4)
def fixed_design(X_bytes, shape):
    X = np.frombuffer(X_bytes).reshape(shape)
    n, p = X.shape
    u2 = np.linalg.svd(X, full_matrices=True)[0][:, p:n]
    X = np.vstack([X, np.zeros((2 * p - n, p))]) if n < 2 * p else X
    X = X / np.linalg.norm(X, axis=0)
    U, d, V = canonical_svd(X)
    Q = np.linalg.qr(np.hstack([U, np.zeros((X.shape[0], p))]))[0]
    u_perp = Q[:, p:2 * p]
    d_inv = np.where(d <= TOL * d.max(), 0.0, 1.0 / d)
    G = (V * d ** 2) @ V.T
    G_inv = (V * d_inv ** 2) @ V.T
    s = solve_sdp(G)
    s[s <= TOL] = 0.0
    _, c_d, c_v = canonical_svd(2 * np.diag(s) - (s[:, None] * G_inv * s[None, :]))
    Xk = X - (X @ G_inv) * s + (u_perp * np.sqrt(np.maximum(0.0, c_d))) @ c_v.T
    return X, Xk, u2


def create_fixed(X, y):
    n, p = X.shape
    Xa, Xk, u2 = fixed_design(np.ascontiguousarray(X, float).tobytes(), X.shape)
    if n < 2 * p:
        sigma = np.sqrt(np.mean((u2.T @ y) ** 2))
        y = np.r_[y, np.random.default_rng(0).normal(0.0, sigma, 2 * p - n)]
    return Xa, Xk, y


def knockoff_filter(gate, array, settings, rng, offset):
    u = array.nyquist_grid
    A = array.steering(u)
    y = gate.mean_snapshot()
    Xr, yr = real_split(A, y)
    p = u.size

    def make():
        Xa, Xk, ya = create_fixed(Xr, yr)
        return glmnet_coefdiff(Xa, Xk, ya, rng, "lars")

    W, Z = statistics(("fixed-X", yr.tobytes()), make)
    tau = threshold(W, settings.alpha, offset)
    is_sel = np.zeros(p, bool)
    is_sel[np.flatnonzero(W >= tau) % p] = True
    T = np.abs(A.conj().T @ y) ** 2
    peaks = [i for i in local_maxima(T) if is_sel[i]]
    u_hat = np.array([locate(array, y, u[i]) for i in peaks])
    info = {}
    if settings.keep_info:
        info = {"kind": "knockoffs", "u": np.r_[u, u], "W": W, "Z": np.abs(Z[:2 * p]), "Zk": np.abs(Z[2 * p:]),
                "tau": tau, "selected": np.flatnonzero(is_sel), "alpha": settings.alpha}
    return DoAResult(array.u_to_deg(u_hat), info)


def estimate(gate, array, settings, rng) -> DoAResult:
    return knockoff_filter(gate, array, settings, rng, offset=1)


def estimate_offset0(gate, array, settings, rng) -> DoAResult:
    return knockoff_filter(gate, array, settings, rng, offset=0)


METHODS = [MethodSpec("knockoffs_lowdim", r"Fixed-X knockoff+ (low-dim.)", "fdr", estimate),
           MethodSpec("knockoffs_lowdim_offset0", r"Fixed-X knockoff (offset 0, low-dim.)", "fdr", estimate_offset0)]
