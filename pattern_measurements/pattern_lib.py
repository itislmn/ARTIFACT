"""
pattern_lib.py - the signal processing shared by measure_interactive.py and
measure_continuous.py. Pure numpy, no hardware, unit-tested on synthetic data.

HOW ONE GAIN NUMBER IS OBTAINED (per virtual channel, per capture/frame)
-----------------------------------------------------------------------
1. Range FFT (Hann window) of every chirp -> complex spectrum.
2. COHERENT mean over the chirps (loops) of the frame: a static reflector has
   the same complex value every chirp, so its amplitude is kept in full while
   noise (random phase) and anything moving average down. This is the
   "processing gain" that pushes the noise floor far below what a magnitude
   average can reach (an average of |X| never goes below the noise level - it
   just flattens out there, which is the -25...-30 dB shelf seen before).
3. Optional: subtract the empty-room spectrum (also complex) - removes TX-RX
   leakage and static clutter. It is LINEAR: measured = leakage + target,
   reference = leakage  ->  difference = target exactly. It only removes what
   is in the reference; if the target was still in the reference it would be
   removed too, which is why a safety check (reference_looks_wrong) exists.
4. Pick the target's range bin inside a window around the target range.
5. Amplitude = sqrt(sum of |X|^2 over that bin and its two neighbours). One bin
   alone under-reads by up to 1.4 dB depending on where the target falls
   between bins; three bins keep the error under 0.1 dB.
6. Noise floor, measured on the same 3-bin scale from range bins far beyond the
   target, so a curve that flattens can be compared to it (the pattern is only
   trustworthy well above that floor).
Gain in dB is later 20*log10(amplitude / reference amplitude).
"""
import numpy as np

from capture_lib import CaptureError

MIN_BIN = 2              # ignore range bins 0-1 (ADC DC offset)
NOISE_START_FRAC = 0.6   # noise floor from the top 40 % of the range bins


def frame_cube(raw_frame, p):
    """One frame of raw int16 (real sampling) -> complex64 (loops, channels, samples).
    Identical reshaping/TX un-scrambling to capture_lib.load_complex_cube."""
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
    """(loops, channels, samples) -> (channels, n_half) complex: Hann range FFT,
    coherent mean over loops."""
    n = cube_frame.shape[-1]
    spec = np.fft.fft(cube_frame * np.hanning(n), axis=-1)[..., : n // 2]
    return spec.mean(axis=0)


def static_spectrum(cube):
    """(frames, loops, channels, samples) -> (channels, n_half): coherent mean
    over loops AND frames (used for the empty-room reference)."""
    acc = None
    for f in range(cube.shape[0]):
        s = frame_spectrum(cube[f])
        acc = s if acc is None else acc + s
    return acc / cube.shape[0]


def range_window(range_axis, n_half, center_m, half_width_m):
    """Bin indices [lo, hi) covering center +/- half_width (DC bins excluded)."""
    dr = range_axis[1] - range_axis[0]
    lo = max(MIN_BIN, int(np.floor((center_m - half_width_m) / dr)))
    hi = min(n_half - 1, int(np.ceil((center_m + half_width_m) / dr)) + 1)
    if hi - lo < 3:
        raise CaptureError(f"Range window {center_m - half_width_m:.2f}-"
                           f"{center_m + half_width_m:.2f} m contains fewer than 3 range bins.")
    return lo, hi


def find_target_bin(Y, range_axis, center_m, half_width_m):
    """Peak bin of the channel-averaged profile inside [center +/- half_width],
    plus its height above the noise floor in dB."""
    lo, hi = range_window(range_axis, Y.shape[1], center_m, half_width_m)
    profile = np.sqrt((np.abs(Y) ** 2).mean(axis=0))
    peak = lo + int(np.argmax(profile[lo:hi]))
    floor = np.median(profile[MIN_BIN:])
    return peak, float(20 * np.log10(profile[peak] / (floor + 1e-12) + 1e-12))


def bin_energy(Z, peak):
    """sqrt(sum |Z|^2 over bins peak-1..peak+1), per channel."""
    sl = slice(max(0, peak - 1), peak + 2)
    return np.sqrt((np.abs(Z[:, sl]) ** 2).sum(axis=1))


def noise_floor(Z):
    """Per-channel noise level on the same 3-bin scale as bin_energy().
    |X|^2 of complex Gaussian noise is exponentially distributed, whose mean is
    median/ln2; three bins -> x3."""
    far = Z[:, int(NOISE_START_FRAC * Z.shape[1]):]
    p_med = np.median(np.abs(far) ** 2, axis=1)
    return np.sqrt(3.0 * p_med / np.log(2.0))


def measure_static(S, B, range_axis, center_m, half_width_m):
    """S: spectrum of a capture/frame, B: empty-room spectrum or None.
    Returns dict with gains (clean), gains_raw, range_m, prominence_db,
    noise (linear, per channel)."""
    Y = S - B if B is not None else S
    peak, prom = find_target_bin(Y, range_axis, center_m, half_width_m)
    return {"gains": bin_energy(Y, peak), "gains_raw": bin_energy(S, peak),
            "range_m": float(range_axis[peak]), "prominence_db": prom,
            "noise": noise_floor(Y), "bin": peak}


def reference_looks_wrong(prom_clean_db, min_prom_db=10.0):
    """After subtracting the empty-room reference, the target (at boresight,
    where it is strongest) must still stand well above the noise floor. If it
    does not, the reference almost certainly already contained the target (or
    another return at that range) and subtracting it erased the real signal."""
    return prom_clean_db < min_prom_db
