"""
dsp.py - one frame of the cube to a point cloud: range FFT over the samples, Doppler FFT over the loops, a 2-D
cell-averaging CFAR at a stated false-alarm probability, and the azimuth of every detection from the 12-channel row
(a second angle peak above half the first is a second object, as multiObjBeamForming does on the chip).

    from radar_tutorial.dsp import point_cloud, range_fft, range_doppler
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def range_fft(frame: np.ndarray, radar) -> np.ndarray:
    x = frame - frame.mean(axis=-1, keepdims=True)
    X = np.fft.fft(x * np.hanning(radar.n_samples), axis=-1)
    return X[..., : radar.n_range_bins]


def doppler_fft(X: np.ndarray, static: bool = True) -> np.ndarray:
    if not static:
        X = X - X.mean(axis=0, keepdims=True)
    w = np.hanning(X.shape[0])[:, None, None]
    return np.fft.fftshift(np.fft.fft(X * w, axis=0), axes=0)


def range_doppler(frame: np.ndarray, radar, static: bool = True) -> np.ndarray:
    return np.sum(np.abs(doppler_fft(range_fft(frame, radar), static)) ** 2, axis=1)


def cfar_2d(power: np.ndarray, guard=(1, 2), train=(4, 8), pfa=1e-4) -> np.ndarray:
    outer = (2 * (guard[0] + train[0]) + 1, 2 * (guard[1] + train[1]) + 1)
    inner = (2 * guard[0] + 1, 2 * guard[1] + 1)
    n = outer[0] * outer[1] - inner[0] * inner[1]
    mode = ("wrap", "nearest")
    s_out = uniform_filter(power, outer, mode=mode) * outer[0] * outer[1]
    s_in = uniform_filter(power, inner, mode=mode) * inner[0] * inner[1]
    noise = (s_out - s_in) / n
    alpha = n * (pfa ** (-1 / n) - 1)
    peak = power >= np.maximum(np.roll(power, 1, 0), np.roll(power, -1, 0))
    peak &= power >= np.maximum(np.pad(power, ((0, 0), (1, 0)), mode="edge")[:, :-1],
                                np.pad(power, ((0, 0), (0, 1)), mode="edge")[:, 1:])
    return (power > alpha * noise) & peak


def steering(radar, u: np.ndarray) -> np.ndarray:
    m = np.arange(radar.azimuth_channels.size)
    return np.exp(2j * np.pi * radar.d_az * np.outer(m, u))


def tdm_compensation(radar, velocity: float) -> np.ndarray:
    slot = np.repeat(radar.tx_reorder, radar.n_rx)
    return np.exp(-4j * np.pi * velocity * slot * (radar.idle_s + radar.ramp_end_s) / radar.wavelength_m)


def azimuth_spectrum(snapshot: np.ndarray, radar, u: np.ndarray) -> np.ndarray:
    return np.abs(steering(radar, u).conj().T @ snapshot[radar.azimuth_channels]) ** 2


def range_azimuth(frame: np.ndarray, radar, u: np.ndarray, static: bool = True) -> np.ndarray:
    X = range_fft(frame, radar)
    if not static:
        X = X - X.mean(axis=0, keepdims=True)
    A = steering(radar, u).conj().T
    Y = X[:, radar.azimuth_channels, :]
    return np.mean(np.abs(np.einsum("um,lmk->lku", A, Y)) ** 2, axis=0)


def refine(p: np.ndarray, i: int) -> float:
    if 0 < i < len(p) - 1:
        a, b, c = np.log(p[i - 1:i + 2] + 1e-30)
        den = a - 2 * b + c
        if den < 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def angle_peaks(spec: np.ndarray, second: float) -> list[int]:
    peaks = [i for i in range(len(spec)) if spec[i] >= spec[i - 1] and spec[i] >= spec[(i + 1) % len(spec)]]
    top = max(spec[i] for i in peaks)
    return [i for i in peaks if spec[i] >= second * top]


def point_cloud(frame: np.ndarray, radar, static: bool = True, pfa=1e-4, second=0.5, n_u=256) -> np.ndarray:
    D = doppler_fft(range_fft(frame, radar), static)
    power = np.sum(np.abs(D) ** 2, axis=1)
    hits = np.argwhere(cfar_2d(power, pfa=pfa))
    u = np.linspace(-1, 1, n_u, endpoint=False)
    noise = np.median(power)
    out = []
    for d, k in hits:
        v = (d - radar.n_loops // 2) * radar.v_res_mps
        spec = azimuth_spectrum(D[d, :, k] * tdm_compensation(radar, v), radar, u)
        r = refine(power[d], k) * radar.range_res_m
        for j in angle_peaks(spec, second):
            th = np.degrees(np.arcsin(np.clip(np.interp(refine(spec, j), np.arange(n_u), u), -1, 1)))
            out.append((r, v, th, r * np.cos(np.radians(th)), r * np.sin(np.radians(th)),
                        10 * np.log10(power[d, k] * spec[j] / spec.max() / noise)))
    return np.array(out, float).reshape(-1, 6)
