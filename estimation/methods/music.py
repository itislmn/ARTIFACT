"""
music.py - MUSIC: the noise subspace of the forward-backward smoothed covariance of the K chirp snapshots (MDL
order) and the peaks of the pseudo-spectrum 1 / ||U_n^H a||^2.

    from estimation.methods import music
"""
from __future__ import annotations

import numpy as np

from estimation.core.array import classical_subspace, largest_peaks, refine_peak
from estimation.core.interfaces import DoAResult, MethodSpec


def spectrum(Un: np.ndarray, array) -> np.ndarray:
    m = Un.shape[0]
    A = array.steering(array.u_grid, n_elements=m)
    proj = np.sum(np.abs(Un.conj().T @ A) ** 2, axis=0)
    return m / np.maximum(proj, 1e-300)


def estimate(gate, array, settings, rng=None) -> DoAResult:
    sub = classical_subspace(gate.snapshots)
    k = sub["k_hat"]
    P = spectrum(sub["eigvecs"][:, k:], array)
    p_db = 10 * np.log10(P)
    peaks = largest_peaks(p_db, k)
    u_hat = np.array([refine_peak(array.u_grid, p_db, i) for i in peaks])
    info = {}
    if settings.keep_info:
        info = {"kind": "spectrum", "u": array.u_grid, "spectrum": P, "peaks": peaks, "k_hat": k,
                "eigvals": sub["eigvals"], "mdl": sub["mdl"],
                "ylabel": r"$P_\mathrm{MUSIC}(\theta)$ [dB]"}
    return DoAResult(array.u_to_deg(u_hat), info)


METHODS = [MethodSpec("music", r"MUSIC + MDL", "classical", estimate, snapshot="multi")]
