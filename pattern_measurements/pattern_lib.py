"""
pattern_lib.py - one gain per channel and frame: Hann range FFT, coherent mean over the chirps, empty-room spectrum
subtracted, target bin in a window around the target range, energy of it and its neighbours, noise floor from far bins.

    from pattern_lib import measure_static, static_spectrum
"""
import numpy as np

from capture_lib import CaptureError

MIN_BIN = 2
NOISE_START_FRAC = 0.6


def frame_cube(raw_frame, p):
    per_frame = p["numAdcSamples"] * p["numRx"] * p["chirpsPerFrame"]
    if raw_frame.size != per_frame:
        raise CaptureError(f"frame has {raw_frame.size} samples, expected {per_frame}")
    data = raw_frame.astype(np.complex64)
    cube = data.reshape(p["chirpsPerFrame"], p["numRx"], p["numAdcSamples"])
    if p["numTx"] > 1:
        cube = cube.reshape(-1, p["numTx"], p["numRx"], p["numAdcSamples"])
        cube = cube[:, p["tx_reorder"], :, :]
        cube = cube.reshape(-1, p["numTx"] * p["numRx"], p["numAdcSamples"])
    return cube


def frame_spectrum(cube_frame):
    n = cube_frame.shape[-1]
    spec = np.fft.fft(cube_frame * np.hanning(n), axis=-1)[..., : n // 2]
    return spec.mean(axis=0)


def static_spectrum(cube):
    acc = None
    for f in range(cube.shape[0]):
        s = frame_spectrum(cube[f])
        acc = s if acc is None else acc + s
    return acc / cube.shape[0]


def range_window(range_axis, n_half, center_m, half_width_m):
    dr = range_axis[1] - range_axis[0]
    lo = max(MIN_BIN, int(np.floor((center_m - half_width_m) / dr)))
    hi = min(n_half - 1, int(np.ceil((center_m + half_width_m) / dr)) + 1)
    if hi - lo < 3:
        raise CaptureError(f"Range window {center_m - half_width_m:.2f}-"
                           f"{center_m + half_width_m:.2f} m contains fewer than 3 range bins.")
    return lo, hi


def find_target_bin(Y, range_axis, center_m, half_width_m):
    lo, hi = range_window(range_axis, Y.shape[1], center_m, half_width_m)
    profile = np.sqrt((np.abs(Y) ** 2).mean(axis=0))
    peak = lo + int(np.argmax(profile[lo:hi]))
    floor = np.median(profile[MIN_BIN:])
    return peak, float(20 * np.log10(profile[peak] / (floor + 1e-12) + 1e-12))


def bin_energy(Z, peak):
    sl = slice(max(0, peak - 1), peak + 2)
    return np.sqrt((np.abs(Z[:, sl]) ** 2).sum(axis=1))


def noise_floor(Z):
    far = Z[:, int(NOISE_START_FRAC * Z.shape[1]):]
    p_med = np.median(np.abs(far) ** 2, axis=1)
    return np.sqrt(3.0 * p_med / np.log(2.0))


def measure_static(S, B, range_axis, center_m, half_width_m):
    Y = S - B if B is not None else S
    peak, prom = find_target_bin(Y, range_axis, center_m, half_width_m)
    return {"gains": bin_energy(Y, peak), "gains_raw": bin_energy(S, peak),
            "range_m": float(range_axis[peak]), "prominence_db": prom,
            "noise": noise_floor(Y), "bin": peak}


def reference_looks_wrong(prom_clean_db, min_prom_db=10.0):
    return prom_clean_db < min_prom_db
