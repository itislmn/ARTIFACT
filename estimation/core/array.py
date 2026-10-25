"""
array.py - the 12-channel azimuth ULA (steering vectors on a grid uniform in sin theta), covariance,
forward-backward smoothing, MDL order and the grouping of selected grid angles into detections.

    from estimation.core.array import ArrayModel
"""
from __future__ import annotations

import numpy as np


class ArrayModel:

    def __init__(self, cfg, oversampling: int = 8, fov_deg: float = 60.0, az_sign: int = 1):
        self.channels = np.asarray(cfg.azimuth_channels)
        self.n = int(self.channels.size)
        self.d = float(cfg.d_az)
        self.positions = cfg.x_lam[self.channels]
        self.az_sign = int(az_sign)
        self.oversampling = int(oversampling)
        self.fov_deg = float(fov_deg)
        self.du = 1.0 / (self.oversampling * self.n * self.d)
        k_max = int(np.floor(np.sin(np.radians(fov_deg)) / self.du + 1e-9))
        self.u_grid = np.arange(-k_max, k_max + 1) * self.du
        self.theta_grid_deg = np.degrees(np.arcsin(self.u_grid))
        n_circle = int(round(1.0 / (self.d * self.du)))
        self.u_circle = (np.arange(n_circle) - n_circle // 2) * self.du
        self.fov_slice = slice(n_circle // 2 - k_max, n_circle // 2 + k_max + 1)

    @property
    def p(self) -> int:
        return int(self.u_grid.size)

    @property
    def rayleigh_u(self) -> float:
        return 1.0 / (self.n * self.d)

    @property
    def nyquist_grid(self) -> np.ndarray:
        k_max = int(np.floor(np.sin(np.radians(self.fov_deg)) * self.n * self.d + 1e-9))
        return np.arange(-k_max, k_max + 1) / (self.n * self.d)

    @property
    def merge_gap(self) -> int:
        return max(self.oversampling - 1, 1)

    @staticmethod
    def u_to_deg(u):
        return np.degrees(np.arcsin(np.clip(u, -1.0, 1.0)))

    @staticmethod
    def deg_to_u(theta_deg):
        return np.sin(np.radians(theta_deg))

    def steering(self, u, n_elements: int | None = None) -> np.ndarray:
        pos = self.positions if n_elements is None else self.positions[:n_elements]
        u = np.atleast_1d(np.asarray(u, float))
        return np.exp(2j * np.pi * self.az_sign * pos[:, None] * u[None, :])

    def dictionary(self, n_groups: int = 1) -> np.ndarray:
        A = self.steering(self.u_grid)
        X = np.tile(A, (n_groups, 1))
        return (X / np.linalg.norm(X, axis=0, keepdims=True)).astype(np.complex128)

    def beam_power(self, y, circle: bool = False, taper=None) -> np.ndarray:
        u = self.u_circle if circle else self.u_grid
        w = np.ones(self.n) if taper is None else np.asarray(taper, float)
        z = (self.steering(u) * w[:, None]).conj().T @ np.asarray(y)
        return np.abs(z) ** 2 / float(np.sum(w ** 2))


def sample_covariance(Y: np.ndarray) -> np.ndarray:
    return (Y @ Y.conj().T) / Y.shape[1]


def smoothing_subarray_size(n: int) -> int:
    return int(round(2 * n / 3))


def fb_smoothed_covariance(R: np.ndarray, m: int) -> np.ndarray:
    n = R.shape[0]
    L = n - m + 1
    Rf = sum(R[l:l + m, l:l + m] for l in range(L)) / L
    J = np.eye(m)[::-1]
    return 0.5 * (Rf + J @ Rf.conj() @ J)


def mdl_order(eigvals_desc: np.ndarray, n_snapshots: int, forward_backward: bool = True):
    lam = np.clip(np.asarray(eigvals_desc, float), 1e-300, None)
    m, K = lam.size, float(n_snapshots)
    mdl = np.empty(m)
    for k in range(m):
        tail = lam[k:]
        g = np.exp(np.mean(np.log(tail)))
        a = np.mean(tail)
        pen = (0.25 * k * (2 * m - k + 1) if forward_backward else 0.5 * k * (2 * m - k)) * np.log(K)
        mdl[k] = -K * (m - k) * np.log(g / a) + pen
    return int(np.argmin(mdl)), mdl


def classical_subspace(Y: np.ndarray) -> dict:
    n, K = Y.shape
    R = sample_covariance(Y)
    m = smoothing_subarray_size(n)
    Rfb = fb_smoothed_covariance(R, m)
    lam, U = np.linalg.eigh(Rfb)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order].real, U[:, order]
    k_hat, mdl = mdl_order(lam, 2 * (n - m + 1) * K, forward_backward=True)
    k_hat = min(k_hat, m - 1)
    noise = float(np.mean(lam[k_hat:]))
    return {"R": R, "Rfb": Rfb, "m": m, "eigvals": lam, "eigvecs": U, "k_hat": k_hat,
            "mdl": mdl, "noise_level": noise, "n_snapshots": K}


def local_maxima(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    left = np.r_[-np.inf, x[:-1]]
    right = np.r_[x[1:], -np.inf]
    return np.flatnonzero((x >= left) & (x > right))


def largest_peaks(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros(0, int)
    idx = local_maxima(x)
    return idx[np.argsort(x[idx])[::-1][:k]]


def refine_peak(grid: np.ndarray, values: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(values) - 1:
        return float(grid[i])
    ym, y0, yp = values[i - 1], values[i], values[i + 1]
    den = ym - 2 * y0 + yp
    delta = 0.0 if den >= 0 else float(np.clip(0.5 * (ym - yp) / den, -0.5, 0.5))
    return float(grid[i] + delta * (grid[i + 1] - grid[i]))


def regression_detections(array, X, y, selected) -> np.ndarray:
    selected = np.sort(np.asarray(selected, int))
    if selected.size == 0:
        return np.zeros(0)
    beta = np.linalg.lstsq(X[:, selected], y, rcond=None)[0]
    w = np.abs(beta) ** 2 + 1e-30
    remaining = np.ones(selected.size, bool)
    out = []
    while remaining.any():
        j0 = int(np.flatnonzero(remaining)[np.argmax(w[remaining])])
        group = remaining & (np.abs(selected - selected[j0]) <= array.merge_gap)
        out.append(float(np.sum(w[group] * array.u_grid[selected[group]]) / np.sum(w[group])))
        remaining &= ~group
    return np.asarray(out)
