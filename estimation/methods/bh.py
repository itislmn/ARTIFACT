"""
bh.py - Benjamini-Hochberg on the low-dimensional model (orthogonal Nyquist grid, p = 11 <= n): OLS z-tests of every
grid coefficient, step-up at level alpha, angle refined inside the cell.

    from estimation.methods import bh
"""
from __future__ import annotations

import numpy as np

from estimation.core.array import local_maxima, refine_peak
from estimation.core.interfaces import DoAResult, MethodSpec


def ols_pvalues(gate, array):
    u = array.nyquist_grid
    A = array.steering(u)
    G_inv = np.linalg.inv(A.conj().T @ A)
    beta = G_inv @ (A.conj().T @ gate.mean_snapshot())
    T = np.abs(beta) ** 2 / (gate.noise_var_of_mean() * np.real(np.diag(G_inv)))
    return u, T, np.exp(-T)


def step_up(p: np.ndarray, level: float) -> np.ndarray:
    m = p.size
    order = np.argsort(p)
    below = np.flatnonzero(p[order] <= np.arange(1, m + 1) * level / m)
    if below.size == 0:
        return np.zeros(0, int)
    return np.sort(order[: below[-1] + 1])


def locate(array, y, u_k):
    half = 0.5 / (array.n * array.d)
    uu = np.linspace(u_k - half, u_k + half, 41)
    P = 10 * np.log10(np.abs(array.steering(uu).conj().T @ y) ** 2 + 1e-300)
    return refine_peak(uu, P, int(np.argmax(P)))


def select(gate, array, level: float, keep_info: bool, name: str) -> DoAResult:
    u, T, p = ols_pvalues(gate, array)
    rejected = step_up(p, level)
    is_rej = np.zeros(u.size, bool)
    is_rej[rejected] = True
    peaks = [i for i in local_maxima(T) if is_rej[i]]
    y = gate.mean_snapshot()
    u_hat = np.array([locate(array, y, u[i]) for i in peaks])
    info = {}
    if keep_info:
        info = {"kind": "pvalues", "rule": name, "u": u, "T": T, "p": p,
                "rejected": rejected, "level": level, "m": p.size}
    return DoAResult(array.u_to_deg(u_hat), info)


def estimate(gate, array, settings, rng=None) -> DoAResult:
    return select(gate, array, settings.alpha, settings.keep_info, "BH")


METHODS = [MethodSpec("bh", r"Benjamini-Hochberg (OLS)", "fdr", estimate)]
