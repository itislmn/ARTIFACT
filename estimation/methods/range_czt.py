"""
range_czt.py - the range from a chirp Z-transform zoom of the spectrum around the detected bin (fine grid, same
window).

    from estimation.methods import range_czt
"""
from __future__ import annotations

import numpy as np
from scipy.signal import zoom_fft

from estimation.core.radar import C0, RadarConfig, range_window

NAME = "czt"
LABEL = r"CZT zoom"
N_POINTS = 257


def zoom_profile(x: np.ndarray, cfg: RadarConfig, r_lo: float, r_hi: float, n_points: int = N_POINTS):
    x = np.asarray(x, float)
    w = range_window(cfg.n_samples)
    xw = (x - x.mean(axis=0, keepdims=True)) * (w if x.ndim == 1 else w[:, None])
    to_f = 2.0 * cfg.slope_hz_per_s / C0
    Z = zoom_fft(xw, [r_lo * to_f, r_hi * to_f], m=n_points, fs=cfg.fs_hz, endpoint=True, axis=0)
    P = np.abs(Z) ** 2
    if P.ndim > 1:
        P = P.sum(axis=1)
    return np.linspace(r_lo, r_hi, n_points), P


def estimate(x: np.ndarray, cfg: RadarConfig, peak_bins, half_width_bins: float = 1.0,
             n_points: int = N_POINTS) -> np.ndarray:
    out = []
    for k in np.atleast_1d(peak_bins).astype(int):
        lo = max(0.0, (k - half_width_bins) * cfg.range_res_m)
        hi = (k + half_width_bins) * cfg.range_res_m
        r, P = zoom_profile(x, cfg, lo, hi, n_points)
        out.append(r[int(np.argmax(P))])
    return np.asarray(out, float)
