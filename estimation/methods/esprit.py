"""
esprit.py - ESPRIT: the angles from the eigenvalues of the rotation between two shifted subarrays of the MDL signal
subspace (forward-backward smoothed, K chirp snapshots).

    from estimation.methods import esprit
"""
from __future__ import annotations

import numpy as np

from estimation.core.array import classical_subspace
from estimation.core.interfaces import DoAResult, MethodSpec


def rotation_eigenvalues(Us: np.ndarray) -> np.ndarray:
    Psi = np.linalg.lstsq(Us[:-1], Us[1:], rcond=None)[0]
    return np.linalg.eigvals(Psi)


def estimate(gate, array, settings, rng=None) -> DoAResult:
    sub = classical_subspace(gate.snapshots)
    k = sub["k_hat"]
    phi = rotation_eigenvalues(sub["eigvecs"][:, :k]) if k > 0 else np.zeros(0, complex)
    u = array.az_sign * np.angle(phi) / (2 * np.pi * array.d)
    inside = np.abs(u) <= np.sin(np.radians(array.fov_deg))
    info = {}
    if settings.keep_info:
        info = {"kind": "esprit", "phi": phi, "u_all": u, "inside": inside, "k_hat": k,
                "eigvals": sub["eigvals"], "mdl": sub["mdl"], "d": array.d}
    return DoAResult(array.u_to_deg(u[inside]), info)


METHODS = [MethodSpec("esprit", r"ESPRIT + MDL", "classical", estimate, snapshot="multi")]
