"""
infinirex.py - Infini-Rex: the T-Rex selector with infinitely many random experiments in closed form; the probability
that a dummy enters at each greedy step gives every grid angle an e-value, selected by e-BH at level alpha. Radar
variant: the greedy path runs on the continuous steering manifold (each entered angle refined before the next
residual) on per-channel noise-normalised channels, so the residual stays noise as the dummies require.

    from estimation.methods import infinirex
"""
from __future__ import annotations

import numpy as np
from scipy.special import betainc

from estimation.core.array import regression_detections
from estimation.core.interfaces import DoAResult, MethodSpec

L_MULTIPLIERS = (1, 2, 5, 10, 20, 50, 100)


def dummy_win_prob(rho, r, complex_valued=True):
    rho = np.clip(rho, 0.0, 1.0)
    if complex_valued:
        return (1.0 - rho) ** (r - 1)
    return betainc((r - 1) / 2.0, 0.5, 1.0 - rho)


def omp_path(X, y, s_max, complex_valued=True):
    n, p = X.shape
    XH = X.conj().T
    r = y.copy()
    Q = np.zeros((n, 0), X.dtype)
    path, rho = [], []
    S = 0.0
    for _ in range(s_max):
        if S >= 1.0:
            break
        rr = float(np.vdot(r, r).real)
        if rr <= 1e-24:
            break
        c = np.abs(XH @ r) ** 2
        c[path] = -1.0
        j = int(np.argmax(c))
        q = X[:, j] - Q @ (Q.conj().T @ X[:, j])
        nq = np.linalg.norm(q)
        if nq < 1e-8:
            break
        path.append(j)
        rho.append(min(c[j] / rr, 1.0))
        S += float(dummy_win_prob(rho[-1], n, complex_valued))
        q = q / nq
        Q = np.column_stack([Q, q])
        r = r - q * np.vdot(q, r)
    return np.array(path, int), np.array(rho)


def infinirex(X, y, alpha, s_max=None):
    n, p = X.shape
    Xn = X / np.linalg.norm(X, axis=0, keepdims=True)
    cplx = np.iscomplexobj(Xn) or np.iscomplexobj(y)
    U, sv, _ = np.linalg.svd(Xn, full_matrices=False)
    r = int(np.sum(sv > sv[0] * max(n, p) * np.finfo(float).eps * 10))
    U = U[:, :r]
    Xc, yc = U.conj().T @ Xn, U.conj().T @ y
    path, rho = omp_path(Xc, yc, s_max or max(1, r // 2), cplx)
    S = np.cumsum(dummy_win_prob(rho, r, cplx))
    e = np.zeros(p)
    e[path] = e_values(S, p)
    sel, k_hat, thr = e_bh(e, alpha)
    infinirex.last = {"S": S, "rank": r, "k_hat": k_hat, "threshold": thr}
    return sel, e, path


def e_values(S, p):
    Ls = p * np.asarray(L_MULTIPLIERS, float)
    return np.mean((Ls[:, None] + 1) * np.clip(1.0 - np.asarray(S)[None, :], 0.0, None) ** Ls[:, None], axis=0)


def e_bh(e, alpha):
    p = e.size
    order = np.argsort(-e)
    k_hat = 0
    for k in range(1, p + 1):
        if e[order[k - 1]] >= p / (alpha * k):
            k_hat = k
    return order[:k_hat], k_hat, p / (alpha * max(k_hat, 1))


def whitening(gate):
    s = getattr(gate, "channel_noise", None)
    if s is None:
        return np.eye(gate.snapshots.shape[0])
    s = np.asarray(s, float)
    return np.diag(1.0 / np.sqrt(s / s.mean()))


def refine_angles(us, y, W, pos_c, az_sign, u_lo, u_hi, du, cycles=50, steps=20, tol=1e-6):
    w = 2 * np.pi * az_sign * pos_c
    steer = lambda u: W @ np.exp(1j * w[:, None] * np.atleast_1d(u)[None, :])
    us = list(us)
    lo, hi = np.broadcast_to(u_lo, len(us)), np.broadcast_to(u_hi, len(us))
    for _ in range(cycles):
        before = np.array(us, float)
        for i in range(len(us)):
            A = steer(us)
            b = np.linalg.lstsq(A, y, rcond=None)[0]
            r = y - np.delete(A, i, axis=1) @ np.delete(b, i)
            u = us[i]
            c = W.conj().T @ r
            for _ in range(steps):
                e = c * np.exp(-1j * w * u)
                g, g1, g2 = e.sum(), (-1j * w * e).sum(), (-(w ** 2) * e).sum()
                d1 = 2 * float(np.real(np.conj(g) * g1))
                d2 = 2 * float(abs(g1) ** 2 + np.real(np.conj(g) * g2))
                if d2 >= 0:
                    break
                step = float(np.clip(-d1 / d2, -du, du))
                u = float(np.clip(u + step, lo[i], hi[i]))
                if abs(step) < 1e-7:
                    break
            us[i] = u
        if np.max(np.abs(np.array(us) - before), initial=0.0) < tol:
            break
    return us


def estimate(gate, array, settings, rng) -> DoAResult:
    G = gate.groups(settings.n_groups)
    X = array.dictionary(G)
    y = gate.stacked_snapshot(G)
    sel, e, path = infinirex(X, y, settings.alpha)
    u_hat = regression_detections(array, X, y, sel)
    return DoAResult(array.u_to_deg(u_hat), {"stats": {"steps": int(path.size), "L": 0},
                                             **_info(array, X, y, e, path, sel, settings.alpha)})


def _info(array, X, y, e, path, sel, alpha):
    Xn = X / np.linalg.norm(X, axis=0, keepdims=True)
    bart = np.abs(Xn.conj().T @ y) ** 2
    p = array.u_grid.size
    bart = bart.reshape(-1, p).sum(axis=0) if bart.size != p else bart
    return {"kind": "infinirex", "u": array.u_grid, "e": e[:p] if e.size >= p else e, "path": path % p,
            "selected": np.asarray(sel, int) % p, "bartlett": bart, "alpha": alpha, **infinirex.last}


def continuous_path(yw, W, array, s_max, alpha=None):
    pos_c = array.positions - array.positions.mean()
    w = 2 * np.pi * array.az_sign * pos_c
    steer = lambda u: W @ np.exp(1j * w[:, None] * np.atleast_1d(u)[None, :])
    G = steer(array.u_grid)
    Gn = G / np.linalg.norm(G, axis=0, keepdims=True)
    du = float(array.u_grid[1] - array.u_grid[0])
    half = 0.5 * array.rayleigh_u
    n = yw.size
    r, us, path, rho, S = yw.copy(), [], [], [], 0.0
    for _ in range(s_max):
        rr = float(np.vdot(r, r).real)
        if S >= 1.0 or rr <= 1e-24:
            break
        c = np.abs(Gn.conj().T @ r) ** 2
        c[path] = -1.0
        j = int(np.argmax(c))
        g = Gn[:, j]
        if us:
            Q, _ = np.linalg.qr(steer(us))
            g = g - Q @ (Q.conj().T @ g)
        ng = np.linalg.norm(g)
        if ng < 1e-8:
            break
        path.append(j)
        rho.append(min(abs(np.vdot(g / ng, r)) ** 2 / rr, 1.0))
        S += float(dummy_win_prob(rho[-1], n))
        if alpha is not None and e_values([S], array.u_grid.size)[0] < array.u_grid.size / (alpha * s_max):
            us = us + [float(array.u_grid[j])]
            break
        centre = array.u_grid[path]
        us = refine_angles(us + [float(array.u_grid[j])], yw, W, pos_c, array.az_sign,
                           np.maximum(centre - half, array.u_grid[0]), np.minimum(centre + half, array.u_grid[-1]), du)
        A = steer(us)
        r = yw - A @ np.linalg.lstsq(A, yw, rcond=None)[0]
    return np.array(path, int), np.array(rho), np.array(us), steer


def estimate_radar(gate, array, settings, rng) -> DoAResult:
    W = whitening(gate)
    yw = W @ gate.mean_snapshot()
    p, n = array.u_grid.size, yw.size
    path, rho, us, steer = continuous_path(yw, W, array, max(1, n // 2), settings.alpha)
    S = np.cumsum(dummy_win_prob(rho, n))
    e = np.zeros(p)
    e[path] = e_values(S, p)
    sel, k_hat, thr = e_bh(e, settings.alpha)
    infinirex.last = {"S": S, "rank": n, "k_hat": k_hat, "threshold": thr}
    keep = np.isin(path, sel)
    u_sel = us[keep]
    u_hat = np.zeros(0)
    if u_sel.size:
        A = steer(u_sel)
        w = np.abs(np.linalg.lstsq(A, yw, rcond=None)[0]) ** 2
        left = np.ones(u_sel.size, bool)
        out = []
        while left.any():
            i0 = int(np.flatnonzero(left)[np.argmax(w[left])])
            out.append(float(u_sel[i0]))
            left &= np.abs(u_sel - u_sel[i0]) > array.u_grid[1] - array.u_grid[0]
        u_hat = np.asarray(out)
    A = W @ array.steering(array.u_grid)
    info = _info(array, A, yw, e, path, sel, settings.alpha)
    info["u_refined"] = us
    return DoAResult(array.u_to_deg(u_hat), {"stats": {"steps": int(path.size), "L": 0}, **info})


METHODS = [
    MethodSpec("infinirex", r"Infini-Rex (general)", "custom", estimate),
    MethodSpec("infinirex_radar", r"Infini-Rex (radar)", "custom", estimate_radar),
]
