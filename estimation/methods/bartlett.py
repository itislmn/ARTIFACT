"""
bartlett.py - the conventional (delay-and-sum) beamformer: peaks of the Bartlett spectrum of the K chirp snapshots,
as many as the MDL order.

    from estimation.methods import bartlett
"""
from __future__ import annotations

import numpy as np

from estimation.core.array import classical_subspace, largest_peaks, refine_peak
from estimation.core.interfaces import DoAResult, MethodSpec


def spectrum(R: np.ndarray, array) -> np.ndarray:
    A = array.steering(array.u_grid)
    return np.real(np.sum(A.conj() * (R @ A), axis=0)) / array.n


def estimate(gate, array, settings, rng=None) -> DoAResult:
    sub = classical_subspace(gate.snapshots)
    P = spectrum(sub["R"], array)
    p_db = 10 * np.log10(np.maximum(P, 1e-300))
    peaks = largest_peaks(p_db, sub["k_hat"])
    u_hat = np.array([refine_peak(array.u_grid, p_db, i) for i in peaks])
    info = {}
    if settings.keep_info:
        info = {"kind": "spectrum", "u": array.u_grid, "spectrum": P, "peaks": peaks,
                "k_hat": sub["k_hat"], "eigvals": sub["eigvals"], "mdl": sub["mdl"],
                "ylabel": r"$P_\mathrm{Bartlett}(\theta)$ [dB]"}
    return DoAResult(array.u_to_deg(u_hat), info)


METHODS = [MethodSpec("bartlett", r"Bartlett + MDL", "beamforming", estimate, snapshot="multi")]
