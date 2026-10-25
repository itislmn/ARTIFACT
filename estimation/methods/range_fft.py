"""
range_fft.py - the range of a reflector from the Hann-windowed range FFT: the bin centre, or a parabola through the
peak bin and its neighbours.

    from estimation.methods import range_fft
"""
from __future__ import annotations

import numpy as np

from estimation.core.array import refine_peak
from estimation.core.radar import RadarConfig, range_window

NAME = "fft"
LABEL = r"FFT (bin centre)"


def range_profile(x: np.ndarray, cfg: RadarConfig, zero_pad: int = 1):
    x = np.asarray(x, float)
    x = (x - x.mean(axis=0, keepdims=True)) * (range_window(cfg.n_samples)
                                               if x.ndim == 1 else range_window(cfg.n_samples)[:, None])
    n_fft = int(zero_pad) * cfg.n_samples
    X = np.fft.rfft(x, n=n_fft, axis=0)[: n_fft // 2]
    P = np.abs(X) ** 2
    if P.ndim > 1:
        P = P.sum(axis=1)
    return np.arange(n_fft // 2) * cfg.range_res_m / zero_pad, P


def estimate(x: np.ndarray, cfg: RadarConfig, peak_bins, interpolation: str = "none") -> np.ndarray:
    r_axis, P = range_profile(x, cfg)
    p_db = 10 * np.log10(P + 1e-30)
    out = []
    for k in np.atleast_1d(peak_bins).astype(int):
        out.append(refine_peak(r_axis, p_db, k) if interpolation == "parabolic" else r_axis[k])
    return np.asarray(out, float)
