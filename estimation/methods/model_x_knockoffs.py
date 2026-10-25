"""
model_x_knockoffs.py - Model-X knockoffs exactly as the R package knockoff runs them by default: second-order Gaussian
knockoffs with the SDP, cross-validated lasso coefficient difference and the knockoff+ threshold (offset 1, FDR) or the
knockoff threshold (offset 0, modified FDR only); complex data as the real/imaginary split; the lasso fits run by
coordinate descent as glmnet does (or, for the fixed-X knockoffs, read off the exact lasso path at the same lambdas).

    from estimation.methods import model_x_knockoffs
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import cvxpy as cp
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lars_path, lasso_path

from estimation.core.array import regression_detections
from estimation.core.interfaces import DoAResult, MethodSpec

NFOLDS = 10
NLAMBDA = 500
CHUNK = 25
TOL = 1e-7


def is_posdef(A, tol=1e-9):
    return float(np.linalg.eigvalsh(A)[0]) > tol * 10


def cov_shrink(X):
    n, p = X.shape
    xc = X - X.mean(axis=0)
    v = np.sum(xc ** 2, axis=0) / (n - 1)
    w = xc ** 2
    var_v = n / (n - 1) ** 3 * np.sum((w - w.mean(axis=0)) ** 2, axis=0)
    med = np.median(v)
    lam_v = float(np.clip(np.sum(var_v) / max(np.sum((v - med) ** 2), 1e-300), 0.0, 1.0))
    xs = xc / np.sqrt(v)
    r = xs.T @ xs / (n - 1)
    wk = xs[:, :, None] * xs[:, None, :]
    var_r = n / (n - 1) ** 3 * np.sum((wk - wk.mean(axis=0)) ** 2, axis=0)
    off = ~np.eye(p, dtype=bool)
    lam = float(np.clip(np.sum(var_r[off]) / max(np.sum(r[off] ** 2), 1e-300), 0.0, 1.0))
    R = (1 - lam) * r
    np.fill_diagonal(R, 1.0)
    sd = np.sqrt(lam_v * med + (1 - lam_v) * v)
    return R * np.outer(sd, sd)


def solve_sdp(Sigma):
    d = np.sqrt(np.diag(Sigma))
    G = Sigma / np.outer(d, d)
    p = G.shape[0]
    s = cp.Variable(p)
    cp.Problem(cp.Maximize(cp.sum(s)), [s >= 0, s <= 1, 2 * G - cp.diag(s) >> 0]).solve(solver=cp.SCS, eps=1e-6, max_iters=100000)
    s = np.clip(np.asarray(s.value, float), 0.0, 1.0)
    eps = 1e-8
    while eps <= 0.1 and not is_posdef(2 * G - np.diag(s * (1 - eps))):
        eps *= 10
    return np.clip(s * (1 - eps), 0.0, None) * np.diag(Sigma)


@lru_cache(maxsize=8)
def second_order_model(X_bytes, shape):
    X = np.frombuffer(X_bytes).reshape(shape)
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False)
    if not is_posdef(Sigma):
        Sigma = cov_shrink(X)
    diag_s = np.diag(solve_sdp(Sigma))
    SigmaInv_s = np.linalg.solve(Sigma, diag_s)
    Sigma_k = 2 * diag_s - diag_s @ SigmaInv_s
    return mu, SigmaInv_s, np.linalg.cholesky(0.5 * (Sigma_k + Sigma_k.T)).T


def create_second_order(X, rng):
    mu, SigmaInv_s, chol = second_order_model(np.ascontiguousarray(X, float).tobytes(), X.shape)
    return X - (X - mu) @ SigmaInv_s + rng.standard_normal(X.shape) @ chol


def glmnet_path(A, b, lam):
    xm, ym = A.mean(axis=0), b.mean()
    Ac, bc = A - xm, b - ym
    tss = float(bc @ bc)
    out, c0, dev0 = [], None, 0.0
    for start in range(0, lam.size, CHUNK):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            c = lasso_path(Ac, bc, alphas=lam[start:start + CHUNK], coef_init=c0, tol=TOL, max_iter=100000)[1]
        for k in range(c.shape[1]):
            out.append(c[:, k])
            dev = 1.0 - float(np.sum((bc - Ac @ c[:, k]) ** 2)) / tss
            if len(out) >= 5 and (dev > 0.999 or dev - dev0 < 1e-5 * dev):
                return np.array(out).T, xm, ym
            dev0 = dev
        c0 = c[:, -1].copy()
    return np.array(out).T, xm, ym


def lars_glmnet_path(A, b, lam):
    xm, ym = A.mean(axis=0), b.mean()
    Ac, bc = A - xm, b - ym
    tss = float(bc @ bc)
    knots, _, path = lars_path(Ac, bc, method="lasso", alpha_min=float(lam[-1]), eps=np.finfo(float).eps)
    k = np.clip(np.searchsorted(-knots, -lam, side="right") - 1, 0, max(knots.size - 2, 0))
    a0, a1 = knots[k], knots[np.minimum(k + 1, knots.size - 1)]
    w = np.where(a0 > a1, (a0 - lam) / np.where(a0 > a1, a0 - a1, 1.0), 0.0)
    coefs = path[:, k] + np.clip(w, 0.0, 1.0) * (path[:, np.minimum(k + 1, knots.size - 1)] - path[:, k])
    coefs[:, lam >= knots[0]] = 0.0
    coefs[:, lam < knots[-1]] = path[:, -1:]
    dev = 1.0 - np.sum((bc[:, None] - Ac @ coefs) ** 2, axis=0) / tss
    stop = (np.arange(lam.size) >= 4) & ((dev > 0.999) | (np.diff(dev, prepend=0.0) < 1e-5 * dev))
    m = int(np.argmax(stop)) + 1 if stop.any() else lam.size
    return coefs[:, :m], xm, ym


PATHS = {"cd": glmnet_path, "lars": lars_glmnet_path}


def cv_coeffs_glmnet(X, y, rng, path="cd"):
    n = X.shape[0]
    Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    lam_max = np.max(np.abs(Xs.T @ y)) / n
    lam = lam_max * (1 / 2e3) ** (np.arange(NLAMBDA) / NLAMBDA)
    folds = rng.permutation(np.resize(np.arange(NFOLDS), n))
    fit = PATHS[path]
    full = fit(Xs, y, lam)[0]
    m = full.shape[1]
    err = np.zeros(m)
    for f in range(NFOLDS):
        tr = folds != f
        coefs, xm, ym = fit(Xs[tr], y[tr], lam[:m])
        coefs = np.hstack([coefs, np.repeat(coefs[:, -1:], m - coefs.shape[1], axis=1)])
        err += np.sum((y[~tr, None] - ym - (Xs[~tr] - xm) @ coefs) ** 2, axis=0)
    cvm = err / n
    return full[:, int(np.flatnonzero(cvm <= cvm.min())[0])]


def glmnet_coefdiff(X, Xk, y, rng, path="cd"):
    p = X.shape[1]
    swap = rng.binomial(1, 0.5, p).astype(float)
    Xsw = X * (1 - swap) + Xk * swap
    Xksw = X * swap + Xk * (1 - swap)
    Z = cv_coeffs_glmnet(np.hstack([Xsw, Xksw]), y, rng, path)
    W = (np.abs(Z[:p]) - np.abs(Z[p:])) * (1 - 2 * swap)
    return W, Z


def threshold(W, alpha, offset):
    for t in np.sort(np.r_[0.0, np.abs(W)]):
        if (offset + np.sum(W <= -t)) / max(1, np.sum(W >= t)) <= alpha:
            return float(t)
    return np.inf


def real_split(X, y):
    return np.block([[X.real, -X.imag], [X.imag, X.real]]), np.r_[y.real, y.imag]


_STATS = {}


def statistics(key, make):
    if key not in _STATS:
        if len(_STATS) > 64:
            _STATS.clear()
        _STATS[key] = make()
    return _STATS[key]


def knockoff_filter(gate, array, settings, rng, offset, path="cd"):
    X = array.dictionary(1)
    y = gate.mean_snapshot()
    p = X.shape[1]
    Xr, yr = real_split(X, y)
    W, Z = statistics(("model-X", path, yr.tobytes()),
                      lambda: glmnet_coefdiff(Xr, create_second_order(Xr, rng), yr, rng, path))
    tau = threshold(W, settings.alpha, offset)
    chosen = np.flatnonzero(W >= tau)
    angles = np.unique(chosen % p)
    u_hat = regression_detections(array, X, y, angles)
    info = {}
    if settings.keep_info:
        info = {"kind": "knockoffs", "u": np.r_[array.u_grid, array.u_grid], "W": W, "Z": np.abs(Z[:2 * p]),
                "Zk": np.abs(Z[2 * p:]), "tau": tau, "selected": angles, "alpha": settings.alpha}
    return DoAResult(array.u_to_deg(u_hat), info)


def estimate(gate, array, settings, rng) -> DoAResult:
    return knockoff_filter(gate, array, settings, rng, offset=1)


def estimate_offset0(gate, array, settings, rng) -> DoAResult:
    return knockoff_filter(gate, array, settings, rng, offset=0)


METHODS = [MethodSpec("knockoffs", r"Model-X knockoff+", "fdr", estimate),
           MethodSpec("knockoffs_offset0", r"Model-X knockoff (offset 0)", "fdr", estimate_offset0)]
