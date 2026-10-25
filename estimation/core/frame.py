"""
frame.py - one radar frame through the chain: range FFT, range gates by OS-CFAR (common to every method), CZT range,
then every DoA method gate by gate; a detection that is only the range sidelobe of another detection at the same
angle is dropped (the Hann window's own leakage), and so are detections outside the evaluation region.

    from estimation.core.frame import process_frame
"""
from __future__ import annotations

import time
import tracemalloc
import zlib
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from estimation.core.interfaces import Gate, Settings
from estimation.core.radar import range_fft, range_window
from estimation.methods import range_czt
from estimation.methods.cfar import range_cfar

GATE_GUARD = 2
GATE_TRAIN = 8


@lru_cache(maxsize=4)
def leakage_envelope(n_samples: int, n_bins: int) -> np.ndarray:
    over = 64
    W = np.abs(np.fft.fft(range_window(n_samples).astype(float), n_samples * over)) ** 2
    P = lambda x: W[np.round(np.abs(x) * over).astype(int) % W.size]
    delta = np.linspace(-0.5, 0.5, over + 1)
    return np.array([np.max(P(d - delta) / P(delta)) for d in range(n_bins)])


def sidelobe_blanking(dets, X, cfg, array):
    if not dets:
        return dets
    Ybar = X[:, :, array.channels].mean(axis=1)
    H = leakage_envelope(cfg.n_samples, Ybar.shape[0])
    kk = np.arange(Ybar.shape[0])
    keep = []
    for k in sorted({d[0] for d in dets}):
        in_gate = [d for d in dets if d[0] == k]
        A = array.steering(np.sin(np.radians([d[2] for d in in_gate])))
        beta = np.linalg.lstsq(A, Ybar[k], rcond=None)[0]
        far = np.abs(kk - k) >= 2
        for (_, r, th), bk, a in zip(in_gate, beta, A.T):
            b = np.abs(Ybar @ a.conj()) ** 2 / array.n ** 2
            if abs(bk) ** 2 > np.max(b[far] * H[np.abs(kk[far] - k)]):
                keep.append((k, r, th))
    return keep


def range_profile(X, array):
    return np.sum(np.abs(X[:, :, array.channels].mean(axis=1)) ** 2, axis=1)


def in_region(r_m, theta_deg, settings: Settings):
    r = np.asarray(r_m, float)
    y = r * np.sin(np.radians(np.asarray(theta_deg, float)))
    return (r >= settings.r_min_m) & (r <= settings.r_max_m) & (y <= settings.y_max_m)


def propose_gates(X, x_avg, cfg, array, settings: Settings):
    profile = range_profile(X, array)
    hits, _ = range_cfar(profile, GATE_GUARD, GATE_TRAIN, settings.gate_pfa, n_looks=array.n)
    r_axis = cfg.range_axis_m
    hits = hits[(r_axis[hits] >= settings.r_min_m) & (r_axis[hits] <= settings.r_max_m)
                & (hits < cfg.n_range_bins - GATE_GUARD - 1)]
    if hits.size == 0:
        return []
    ranges = range_czt.estimate(x_avg[:, array.channels], cfg, hits)
    K = X.shape[1]
    k0 = int(np.ceil(settings.r_min_m / cfg.range_res_m))
    power = np.abs(X[k0:, :, array.channels]) ** 2
    channel_noise = np.median(power.reshape(-1, array.n), axis=0) / np.log(2.0)
    floor = float(np.median(power[:, 0]) / np.log(2.0))
    gates = []
    for k, r in zip(hits, ranges):
        Y = X[k][:, array.channels].T.astype(np.complex128)
        noise_var = float(np.mean(np.var(Y, axis=1, ddof=1))) if K >= 2 else floor
        snr = profile[k] / (array.n * noise_var / K)
        gates.append(Gate(int(k), float(r), Y, noise_var, float(10 * np.log10(max(snr, 1e-12))), channel_noise))
    return gates


@dataclass
class FrameResult:
    gates: list
    detections: dict = field(default_factory=dict)
    applicable: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    peak_mb: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    rfft: np.ndarray | None = None


def method_rng(seed, name):
    base = list(seed) if isinstance(seed, (list, tuple)) else [int(seed)]
    return np.random.default_rng(base + [zlib.crc32(name.encode())])


def background_from_recording(n_frames, get_frame, cfg, channel_cal=None):
    acc_X, acc_x = None, None
    for i in range(n_frames):
        f = get_frame(i)
        X = range_fft(f, cfg, channel_cal).mean(axis=1)
        x = np.asarray(f, float).mean(axis=1)
        acc_X = X if acc_X is None else acc_X + X
        acc_x = x if acc_x is None else acc_x + x
    return {"rfft": acc_X / n_frames, "adc": acc_x / n_frames}


def process_frame(frame, cfg, array, methods, settings: Settings, seed, channel_cal=None, background=None,
                  profile_memory=False):
    X = range_fft(frame, cfg, channel_cal)
    x_avg = np.asarray(frame, float).mean(axis=1)
    if background is not None:
        X = X - background["rfft"][:, None, :].astype(X.dtype)
        x_avg = x_avg - background["adc"]
    gates = propose_gates(X, x_avg, cfg, array, settings)
    out = FrameResult(gates, rfft=X if settings.keep_info else None)
    for m in methods:
        rng = method_rng(seed, m.name)
        t0 = time.perf_counter()
        dets, res, stats, ok = [], [], [], True
        for g in gates:
            r = m.fn(g, array, settings, rng)
            ok &= r.info.get("applicable", True)
            dets.extend((g.bin, g.range_m, float(a)) for a in np.atleast_1d(r.angles_deg))
            stats.append(r.info.get("stats", {}))
            if settings.keep_info:
                res.append(r)
        out.seconds[m.name] = time.perf_counter() - t0
        d = np.array([(r, a) for _, r, a in sidelobe_blanking(dets, X, cfg, array)], float).reshape(-1, 2)
        out.detections[m.name] = d[in_region(d[:, 0], d[:, 1], settings)]
        out.applicable[m.name] = bool(ok)
        out.stats[m.name] = stats
        if settings.keep_info:
            out.results[m.name] = res
        if profile_memory and gates:
            rng = method_rng(seed, m.name)
            tracemalloc.start()
            for g in gates:
                m.fn(g, array, settings, rng)
            out.peak_mb[m.name] = tracemalloc.get_traced_memory()[1] / 2 ** 20
            tracemalloc.stop()
    return out
