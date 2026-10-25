"""
cfar.py - CA- and OS-CFAR: a detector, not an estimator; in angle it thresholds the Bartlett power of the
chirp-averaged snapshot cell by cell and the angle of a detection is its peak position (parabola through the peak
cell and its two neighbours, in dB).

    from estimation.methods import cfar
"""
from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np
from scipy import integrate, optimize, special, stats

from estimation.core.interfaces import DoAResult, MethodSpec

GUARD_BINS = 1


@lru_cache(maxsize=512)
def ca_factor(n_train: int, pfa: float, n_looks: int = 1) -> float:
    q = stats.beta.isf(pfa, n_looks, n_train * n_looks)
    return float(n_train * q / (1.0 - q))


@lru_cache(maxsize=512)
def os_factor(n_train: int, k: int, pfa: float, n_looks: int = 1) -> float:
    if n_looks == 1:
        f = lambda a: sum(np.log((n_train - i) / (n_train - i + a)) for i in range(k)) - np.log(pfa)
    else:
        L = float(n_looks)

        def log_pfa(a):
            def dens(z):
                F = special.gammainc(L, z)
                return (k * comb(n_train, k) * F ** (k - 1) * (1 - F) ** (n_train - k)
                        * stats.gamma.pdf(z, L) * special.gammaincc(L, a * z))
            return np.log(integrate.quad(dens, 0, np.inf, limit=200)[0] + 1e-300)
        f = lambda a: log_pfa(a) - np.log(pfa)
    return float(optimize.brentq(f, 1e-9, 1e9))


def cfar_threshold(power: np.ndarray, train_offsets, pfa: float, kind: str = "ca",
                   n_looks: int = 1, circular: bool = False, os_rank: float = 0.75):
    power = np.asarray(power, float)
    n = power.size
    offs = np.asarray(train_offsets, int)
    rel = np.concatenate([-offs[::-1], offs])
    thr = np.full(n, np.inf)
    noise = np.full(n, np.nan)
    for i in range(n):
        idx = i + rel
        idx = idx % n if circular else idx[(idx >= 0) & (idx < n)]
        nt = idx.size
        if nt < max(4, offs.size):
            continue
        cells = power[idx]
        if kind == "ca":
            noise[i] = cells.mean()
            thr[i] = ca_factor(nt, pfa, n_looks) * noise[i]
        else:
            k = int(np.ceil(os_rank * nt))
            noise[i] = np.sort(cells)[k - 1]
            thr[i] = os_factor(nt, k, pfa, n_looks) * noise[i]
    return thr, noise


def local_maxima_mask(power: np.ndarray, circular: bool = False) -> np.ndarray:
    p = np.asarray(power, float)
    if circular:
        left, right = np.roll(p, 1), np.roll(p, -1)
    else:
        left, right = np.r_[-np.inf, p[:-1]], np.r_[p[1:], -np.inf]
    return (p >= left) & (p > right)


def range_cfar(profile: np.ndarray, n_guard: int, n_train: int, pfa: float, n_looks: int, kind: str = "os",
               os_rank: float = 0.5):
    thr, _ = cfar_threshold(profile, np.arange(n_guard + 1, n_guard + n_train + 1), pfa, kind, n_looks,
                            os_rank=os_rank)
    hits = np.flatnonzero((profile > thr) & local_maxima_mask(profile))
    return hits, thr


def _refine_circular(u, p_db, i):
    n = p_db.size
    ym, y0, yp = p_db[(i - 1) % n], p_db[i], p_db[(i + 1) % n]
    den = ym - 2 * y0 + yp
    delta = 0.0 if den >= 0 else float(np.clip(0.5 * (ym - yp) / den, -0.5, 0.5))
    return float(u[i] + delta * (u[1] - u[0]))


def spatial_cfar(gate, array, settings, kind: str) -> DoAResult:
    y = gate.mean_snapshot()
    P = array.beam_power(y, circle=True)
    n_side = (array.n - 1 - 2 * GUARD_BINS) // 2
    offsets = array.oversampling * (GUARD_BINS + np.arange(1, n_side + 1))
    thr, noise = cfar_threshold(P, offsets, settings.pfa, kind, n_looks=1, circular=True)
    det = (P > thr) & local_maxima_mask(P, circular=True)
    fov = np.zeros(P.size, bool)
    fov[array.fov_slice] = True
    peaks = np.flatnonzero(det & fov)
    p_db = 10 * np.log10(P + 1e-30)
    u_hat = np.array([_refine_circular(array.u_circle, p_db, i) for i in peaks])
    info = {}
    if settings.keep_info:
        info = {"kind": "cfar", "variant": kind, "u": array.u_circle, "power": P, "threshold": thr,
                "noise": noise, "peaks": peaks, "fov": array.fov_slice, "n_train": 2 * n_side,
                "factor": float(np.nanmedian(thr / noise))}
    return DoAResult(array.u_to_deg(u_hat), info)


def estimate_ca(gate, array, settings, rng=None) -> DoAResult:
    return spatial_cfar(gate, array, settings, "ca")


def estimate_os(gate, array, settings, rng=None) -> DoAResult:
    return spatial_cfar(gate, array, settings, "os")


METHODS = [
    MethodSpec("ca_cfar", r"CA-CFAR", "cfar", estimate_ca),
    MethodSpec("os_cfar", r"OS-CFAR", "cfar", estimate_os),
]
