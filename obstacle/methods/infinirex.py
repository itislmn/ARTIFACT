"""
infinirex.py - Infini-Rex e-values of every gate and weighted e-BH over the whole frame, with the tracks' prediction
regions as prior weights: FDR at most alpha per look (estimation's statistic, tracking's frame selector).

    from methods.infinirex import FrameSelector, prior_weights
"""
from __future__ import annotations

import time

import numpy as np
from scipy.special import betainc

from methods.array_model import elevation_deg, regression_detections


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


L_MULT = 10.0 ** np.arange(7)


class FrameSelector:
    def __init__(self, array, alpha: float):
        self.array, self.alpha = array, float(alpha)
        self.u_grid = array.u_grid
        self.p = array.p
        self.A = array.steering(array.u_grid)
        self.L = self.p * L_MULT
        self.fast = True

    def _basis(self, noise_ch):
        az = self.array.channels
        s = noise_ch[az] / noise_ch[az].mean()
        W = np.diag(1.0 / np.sqrt(s))
        Aw = W @ self.A
        Xn = Aw / np.linalg.norm(Aw, axis=0, keepdims=True)
        U, sv, _ = np.linalg.svd(Xn, full_matrices=False)
        r = int(np.sum(sv > sv[0] * 1e-10))
        return W, Xn, U[:, :r], r

    def evalues(self, rd, gates) -> dict:
        t0 = time.perf_counter()
        W, Xn, U, r = self._basis(rd.noise_ch)
        Xc = U.conj().T @ Xn
        az = self.array.channels
        E = np.zeros((len(gates), self.p))
        paths_, ys = [], []
        for i, g in enumerate(gates):
            y = W @ rd.D[g.k, g.d, az].astype(np.complex128)
            path, rho = omp_path(Xc, U.conj().T @ y, max(1, r // 2), True)
            if path.size:
                S = np.cumsum(dummy_win_prob(rho, r, True))
                logv = np.log1p(self.L[:, None]) + self.L[:, None] * np.log1p(-np.clip(S[None, :], 0, 1 - 1e-16))
                E[i, path] = np.mean(np.exp(logv), axis=0) * (S < 1)
            paths_.append(path)
            ys.append(y)
        return {"E": E, "paths": paths_, "y": ys, "W": W, "Xn": Xn, "rank": r, "ms": 1e3 * (time.perf_counter() - t0)}

    def _refine(self, u0: float, y, W, du: float) -> float:
        u = float(u0)
        for _ in range(3):
            us = np.array([u - du, u, u + du])
            A = W @ self.array.steering(us)
            f = np.abs(A.conj().T @ y) ** 2 / np.sum(np.abs(A) ** 2, axis=0)
            den = f[0] - 2 * f[1] + f[2]
            if den < 0:
                u = u + float(np.clip(0.5 * (f[0] - f[2]) / den, -1, 1)) * du
            du /= 2
        return float(np.clip(u, self.u_grid[0], self.u_grid[-1]))

    def weighted_ebh(self, E: np.ndarray, Wt: np.ndarray, alpha: float | None = None) -> np.ndarray:
        alpha = self.alpha if alpha is None else alpha
        m = E.size
        if m == 0:
            return np.zeros_like(E, bool)
        ew = (E * Wt).ravel()
        order = np.argsort(-ew)
        k = np.arange(1, m + 1)
        ok = ew[order] >= m / (alpha * k)
        k_hat = int(k[ok].max()) if ok.any() else 0
        rej = np.zeros(m, bool)
        rej[order[:k_hat]] = True
        return rej.reshape(E.shape)

    def detections(self, rd, gates, picked, fe, ev, weights, cfg) -> list[dict]:
        arr = self.array
        pos_c = arr.positions - arr.positions.mean()
        du = float(self.u_grid[1] - self.u_grid[0])
        out = []
        for i, g in enumerate(gates):
            sel = np.flatnonzero(picked[i])
            if sel.size == 0:
                continue
            y = ev["y"][i]
            us = regression_detections(arr, ev["Xn"], y, sel)
            if self.fast:
                us = [self._refine(u, y, ev["W"], du) for u in us]
            else:
                us = refine_angles(list(us), y, ev["W"], pos_c, arr.az_sign, float(self.u_grid[0]), float(self.u_grid[-1]), du)
            Aw = ev["W"] @ arr.steering(us)
            beta = np.linalg.lstsq(Aw, y, rcond=None)[0]
            y16 = rd.D[g.k, g.d].astype(np.complex128)
            th = arr.u_to_deg(np.asarray(us))
            try:
                el, el_fit = elevation_deg(y16, cfg, th, arr.az_sign)
            except np.linalg.LinAlgError:
                el, el_fit = np.full(len(us), np.nan), np.full(len(us), np.nan)
            noise = float(np.mean(rd.noise_ch[arr.channels]))
            for j, u in enumerate(us):
                r, v, bp = fe.refine(rd, g.k, g.d, float(u))
                jj = int(sel[np.argmin(np.abs(self.u_grid[sel] - u))])
                amp2 = float(np.abs(beta[j]) ** 2)
                out.append({"range_m": r, "theta_deg": float(th[j]), "u": float(u), "v_mps": v,
                            "elev_deg": float(el[j]), "elev_fit": float(el_fit[j]),
                            "snr_db": float(10 * np.log10(max(amp2 * arr.n / noise, 1e-12))),
                            "power_db": float(10 * np.log10(max(bp, 1e-30))),
                            "e": float(ev["E"][i, jj]), "w": float(weights[i, jj]), "gate": i,
                            "cell": [int(g.k), int(g.d)], "source": g.source})
        return out


def prior_weights(gates, u_grid: np.ndarray, regions: list[dict], lam: float, merge_u: float) -> np.ndarray:
    G, p = len(gates), u_grid.size
    m = G * p
    if m == 0:
        return np.zeros((0, p))
    w = np.full((G, p), lam)
    masks = []
    for reg in regions:
        inside_gate = np.array([reg["r_lo"] <= g.range_m <= reg["r_hi"] and _v_in(g.v_mps, reg) for g in gates])
        ang = (u_grid >= reg["u_lo"] - merge_u) & (u_grid <= reg["u_hi"] + merge_u)
        M = inside_gate[:, None] & ang[None, :]
        if M.any():
            masks.append(M)
    if masks and lam < 1:
        for M in masks:
            w += (1 - lam) / len(masks) * M * (m / M.sum())
    else:
        w[:] = 1.0
    return w * (m / w.sum())


def _v_in(v: float, reg: dict) -> bool:
    span = reg.get("v_span", np.inf)
    lo, hi = reg["v_lo"], reg["v_hi"]
    if not np.isfinite(span) or hi - lo >= span:
        return lo <= v <= hi or hi - lo >= span
    x = (v - lo) % span
    return x <= hi - lo
