"""
capon.py - the Capon / MVDR beamformer on the forward-backward smoothed covariance, as many peaks as the MDL order.

    from estimation.methods import capon
"""
from __future__ import annotations

import numpy as np

from estimation.core.array import classical_subspace, largest_peaks, refine_peak
from estimation.core.interfaces import DoAResult, MethodSpec


def spectrum(Rfb: np.ndarray, delta: float, array) -> np.ndarray:
    m = Rfb.shape[0]
    A = array.steering(array.u_grid, n_elements=m) / np.sqrt(m)
    Ri = np.linalg.inv(Rfb + delta * np.eye(m))
    return 1.0 / np.maximum(np.real(np.sum(A.conj() * (Ri @ A), axis=0)), 1e-300)


def estimate(gate, array, settings, rng=None) -> DoAResult:
    sub = classical_subspace(gate.snapshots)
    P = spectrum(sub["Rfb"], sub["noise_level"], array)
    p_db = 10 * np.log10(P)
    peaks = largest_peaks(p_db, sub["k_hat"])
    u_hat = np.array([refine_peak(array.u_grid, p_db, i) for i in peaks])
    info = {}
    if settings.keep_info:
        info = {"kind": "spectrum", "u": array.u_grid, "spectrum": P, "peaks": peaks,
                "k_hat": sub["k_hat"], "eigvals": sub["eigvals"], "mdl": sub["mdl"],
                "ylabel": r"$P_\mathrm{Capon}(\theta)$ [dB]"}
    return DoAResult(array.u_to_deg(u_hat), info)


METHODS = [MethodSpec("capon", r"Capon/MVDR + MDL", "beamforming", estimate, snapshot="multi")]
