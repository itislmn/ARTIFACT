"""
radar_gates.py - range gates by range-angle OS-CFAR (one CFAR per beam, Bonferroni-corrected), so a weak object next to
a strong one in another direction is not masked.

    from backend.radar_gates import install
"""
from __future__ import annotations

import numpy as np


def install(process_frame) -> None:
    g = process_frame.__globals__
    if g.get("_cv_gates_installed"):
        return
    orig = g["propose_gates"]
    range_cfar, Gate, range_czt = g["range_cfar"], g["Gate"], g["range_czt"]
    GUARD, TRAIN = g["GATE_GUARD"], g["GATE_TRAIN"]
    offs = np.r_[-np.arange(GUARD + 1, GUARD + TRAIN + 1), np.arange(GUARD + 1, GUARD + TRAIN + 1)]
    scale = {}

    def propose_gates(X, x_avg, cfg, array, settings):
        r_axis = cfg.range_axis_m
        Xm = X[:, :, array.channels].mean(axis=1)
        A = array.steering(array.u_grid)
        P = np.abs(Xm @ A.conj()) ** 2 / array.n
        pfa = settings.gate_pfa / P.shape[1]
        if pfa not in scale:
            prof = np.random.default_rng(0).exponential(1.0, 512)
            _, thr = range_cfar(prof, GUARD, TRAIN, pfa, n_looks=1)
            k = 256
            scale[pfa] = float(thr[k] / np.median(prof[k + offs]))
        idx = np.clip(np.arange(len(P))[:, None] + offs[None, :], 0, len(P) - 1)
        above = P > scale[pfa] * np.median(P[idx], axis=1)
        Q = np.pad(P, 1, constant_values=-np.inf)
        peak = np.ones(P.shape, bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    peak &= P >= Q[1 + di:Q.shape[0] - 1 + di, 1 + dj:Q.shape[1] - 1 + dj]
        hits = set(int(k) for k in np.flatnonzero((above & peak).any(axis=1)))
        hits.update(gate.bin for gate in orig(X, x_avg, cfg, array, settings))
        hits = np.array(sorted(k for k in hits if settings.r_min_m <= r_axis[k] <= settings.r_max_m
                               and k < cfg.n_range_bins - GUARD - 1), int)
        if hits.size == 0:
            return []
        profile = np.sum(np.abs(Xm) ** 2, axis=1)
        keep = []
        for k in hits:
            if keep and k - keep[-1] <= 1:
                if profile[k] > profile[keep[-1]]:
                    keep[-1] = k
            else:
                keep.append(k)
        hits = np.array(keep, int)
        ranges = range_czt.estimate(x_avg[:, array.channels], cfg, hits)
        K, gates = X.shape[1], []
        k0 = int(np.ceil(settings.r_min_m / cfg.range_res_m))
        power = np.abs(X[k0:, :, array.channels]) ** 2
        channel_noise = np.median(power.reshape(-1, array.n), axis=0) / np.log(2.0)
        floor = float(np.median(power[:, 0]) / np.log(2.0))
        for k, r in zip(hits, ranges):
            Y = X[k][:, array.channels].T.astype(np.complex128)
            noise_var = float(np.mean(np.var(Y, axis=1, ddof=1))) if K >= 2 else floor
            snr = profile[k] / (array.n * noise_var / max(K, 1))
            gates.append(Gate(int(k), float(r), Y, noise_var, float(10 * np.log10(max(snr, 1e-12))), channel_noise))
        return gates

    g["propose_gates"] = propose_gates
    g["_cv_gates_installed"] = True
