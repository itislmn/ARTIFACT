"""
kalman_cqr.py - KalmanCQR: an extended Kalman filter per object and conformalised quantile regression of its
innovations, adapted online (ACI), for distribution-free prediction regions (shared with obstacle/).

    from methods.kalman_cqr import EKF, ConformalRegion
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

DIMS = ("range_m", "theta_rad", "v_mps")


def rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


@dataclass
class EKF:
    s: np.ndarray
    P: np.ndarray

    @classmethod
    def from_measurement(cls, r, th, vr, sig_r, sig_th, sig_v, v_tangential_max: float):
        e_r, e_t = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        p = r * e_r
        v = vr * e_r
        J = np.c_[e_r, r * e_t]
        Pp = J @ np.diag([sig_r ** 2, sig_th ** 2]) @ J.T
        Pv = np.outer(e_r, e_r) * sig_v ** 2 + np.outer(e_t, e_t) * v_tangential_max ** 2
        P = np.zeros((4, 4))
        P[:2, :2], P[2:, 2:] = Pp, Pv
        return cls(np.r_[p, v], P)

    @staticmethod
    def F_Q(dt: float, sigma_a: float, dpsi: float = 0.0):
        Rt = rot(-dpsi)
        F = np.zeros((4, 4))
        F[:2, :2], F[:2, 2:], F[2:, 2:] = Rt, Rt * dt, Rt
        G = np.r_[np.eye(2) * dt ** 2 / 2, np.eye(2) * dt]
        Q = sigma_a ** 2 * G @ G.T
        B = np.zeros((4, 4))
        B[:2, :2], B[2:, 2:] = Rt, Rt
        return F, B @ Q @ B.T

    def predicted(self, dt: float, sigma_a: float, v_ego=(0.0, 0.0), dpsi: float = 0.0):
        F, Q = self.F_Q(dt, sigma_a, dpsi)
        s = F @ self.s
        s[:2] -= rot(-dpsi) @ np.asarray(v_ego) * dt
        return s, F @ self.P @ F.T + Q

    @staticmethod
    def h(s, v_ego=(0.0, 0.0)):
        x, y = s[0], s[1]
        r = max(np.hypot(x, y), 1e-6)
        vx, vy = s[2] - v_ego[0], s[3] - v_ego[1]
        z = np.array([r, np.arctan2(y, x), (x * vx + y * vy) / r])
        H = np.zeros((3, 4))
        H[0, :2] = [x / r, y / r]
        H[1, :2] = [-y / r ** 2, x / r ** 2]
        H[2, 0] = y * (vx * y - vy * x) / r ** 3
        H[2, 1] = x * (vy * x - vx * y) / r ** 3
        H[2, 2:] = [x / r, y / r]
        return z, H


def wrap_v(dv: float, v_max: float) -> float:
    return (dv + v_max) % (2 * v_max) - v_max


def weighted_quantile(u: np.ndarray, w: np.ndarray, tau: float) -> float:
    o = np.argsort(u)
    cw = np.cumsum(w[o])
    return float(u[o][min(np.searchsorted(cw, tau * cw[-1]), len(u) - 1)])


@dataclass
class ConformalRegion:
    alpha: float
    window: int = 300
    gamma: float = 0.02
    dims: int = 3
    u: deque = field(default_factory=deque)
    sig: deque = field(default_factory=deque)
    alpha_t: float = -1.0
    misses: list = field(default_factory=list)
    _fit: tuple | None = None

    def __post_init__(self):
        self.u, self.sig = deque(maxlen=self.window), deque(maxlen=self.window)
        self.alpha_t = self.alpha if self.alpha_t < 0 else self.alpha_t

    def add(self, u: np.ndarray, sig: np.ndarray, missed: bool | None):
        self.u.append(np.asarray(u, float))
        self.sig.append(np.asarray(sig, float))
        if missed is not None:
            self.misses.append((int(missed), self.alpha_t))
            self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.alpha - missed), 1e-3, 0.5))
        self._fit = None

    def fit(self):
        if self._fit is not None:
            return self._fit
        n = len(self.u)
        U, S = np.array(self.u).reshape(-1, self.dims), np.array(self.sig).reshape(-1, self.dims)
        tr, ca = np.arange(0, n, 2), np.arange(1, n, 2)
        need = int(np.ceil(1 / self.alpha_t)) - 1
        if len(ca) < max(need, 5):
            z = norm.isf(self.alpha_t / (2 * self.dims))
            self._fit = (-z * np.ones(self.dims), z * np.ones(self.dims), 0.0, False)
            return self._fit
        b_lo = np.array([weighted_quantile(U[tr, d], S[tr, d], self.alpha / 2) for d in range(self.dims)])
        b_hi = np.array([weighted_quantile(U[tr, d], S[tr, d], 1 - self.alpha / 2) for d in range(self.dims)])
        sc = np.max(np.maximum(b_lo[None] - U[ca], U[ca] - b_hi[None]), axis=1)
        k = int(np.ceil((1 - self.alpha_t) * (len(ca) + 1)))
        q = float(np.sort(sc)[k - 1]) if k <= len(ca) else float(np.max(sc)) + 1.0
        self._fit = (b_lo, b_hi, q, True)
        return self._fit

    def box(self, z: np.ndarray, sigma: np.ndarray):
        b_lo, b_hi, q, ok = self.fit()
        return z + (b_lo - q) * sigma, z + (b_hi + q) * sigma, ok

    def box_at(self, z: np.ndarray, sigma: np.ndarray, alpha: float):
        b_lo, b_hi, q, ok = self.fit()
        U = np.array(self.u).reshape(-1, self.dims)
        ca = np.arange(1, len(U), 2)
        k = int(np.ceil((1 - alpha) * (len(ca) + 1)))
        if not ok or k > len(ca):
            zq = norm.isf(alpha / (2 * self.dims))
            return z - zq * sigma, z + zq * sigma, False
        sc = np.max(np.maximum(b_lo[None] - U[ca], U[ca] - b_hi[None]), axis=1)
        qa = float(np.sort(sc)[k - 1])
        return z + (b_lo - qa) * sigma, z + (b_hi + qa) * sigma, True

    def coverage(self, last: int = 200) -> float:
        m = [x for x, _ in self.misses[-last:]]
        return 1 - float(np.mean(m)) if m else float("nan")
