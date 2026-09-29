r"""
LESSON 07 - Range FFT, Doppler FFT, and CFAR detection

    python 07_range_doppler_cfar.py

No hardware needed - runs against a synthetic target with a KNOWN range
and velocity (common.py's synthesize_raw_adc), and asserts the pipeline
actually finds it, at the end.

This lesson picks up exactly where lesson 06 left off: a correctly-
labeled (frame, loop, virtual_channel, sample) cube. Everything here
works on ONE virtual channel (TX0-RX0) for clarity - lesson 08 is where
using ALL 16 channels together becomes the point.
"""
import numpy as np

from common import synthesize_raw_adc

C = 2.99792458e8


def load_minimal_cfg():
    """The handful of numbers this lesson needs, straight from this
    project's real profileCfg/frameCfg (lesson 01 has the full,
    annotated parse of every field - duplicated minimally here so this
    file runs standalone)."""
    return {
        "numAdcSamples": 656, "sampleRate_ksps": 13349.0,
        "slope_MHz_us": 70.0, "startFreq_GHz": 77.0,
        "idleTime_us": 186.0, "rampEndTime_us": 57.14,
        "numRx": 4, "numTx": 4, "numLoops": 64, "numFrames": 5,
        "chirpsPerFrame": 4 * 64, "isComplex": False,
    }


# =======================================================================
# RANGE FFT
# =======================================================================
def range_fft(cube, window=True):
    """cube: (frames, loops, channels, samples), complex (see lesson 05 -
    real-format samples cast straight into the real part; the FFT below
    is what actually produces genuine phase/complex spectral content).
    Returns the same shape, FFT'd along the LAST (sample/fast-time) axis.
    """
    if window:
        # Hanning: tapers each chirp's samples toward zero at both ends
        # before the FFT. Without it, the sharp edges of a finite-length
        # sample block leak energy into neighboring range bins (spectral
        # leakage) and mask weaker targets sitting near a strong one.
        # Costs a little main-lobe width (slightly worse range
        # resolution) to buy a lot of sidelobe suppression - a standard,
        # near-always-worth-it trade.
        w = np.hanning(cube.shape[-1])
        cube = cube * w
    return np.fft.fft(cube, axis=-1)


def range_axis(p):
    N, fs = p["numAdcSamples"], p["sampleRate_ksps"] * 1e3
    slope = p["slope_MHz_us"] * 1e12
    return np.arange(N) * (C * fs) / (2 * slope * N)   # see lesson 01


# =======================================================================
# DOPPLER FFT
# =======================================================================
def doppler_fft(range_fft_cube):
    """Input: (frames, loops, channels, samples) - already range-FFT'd.
    We do the Doppler FFT along the LOOP axis (slow time) for one
    frame at a time, then fftshift so 0 velocity sits in the middle
    instead of wrapping at the edges (standard FFT output has "0, +1,
    +2, ..., -2, -1" bin order; fftshift reorders that to "..., -2, -1,
    0, +1, +2, ...", which is the order you actually want to look at
    or index by physical velocity)."""
    return np.fft.fftshift(np.fft.fft(range_fft_cube, axis=1), axes=1)


def velocity_axis(p, tx_count):
    lam = C / (p["startFreq_GHz"] * 1e9)
    chirp_period_s = (p["idleTime_us"] + p["rampEndTime_us"]) * 1e-6
    doppler_pri_s = tx_count * chirp_period_s   # TDM-MIMO effective PRI, lesson 06
    n_loops = p["numLoops"]
    v_max = lam / (4 * doppler_pri_s)
    return np.linspace(-v_max, v_max, n_loops, endpoint=False)


# =======================================================================
# CA-CFAR (cell-averaging constant false alarm rate) - 1D, along range,
# for one chosen Doppler bin. This project's own .cfg configures the
# chip's ON-CHIP detector to do CFAR twice, once per axis (its cfarCfg
# lines: one with procDirection=range, one with procDirection=Doppler,
# per lesson 01) - the 1D version here is that same idea, applied by
# hand to one axis so the mechanism is easy to see; chaining it a second
# time along Doppler for each range bin is the natural, direct extension
# to match the chip's own two-pass approach.
# =======================================================================
def ca_cfar_1d(power, num_guard, num_ref, threshold_scale):
    """power: 1D array of (linear) power values. For each cell, average
    the REFERENCE cells on either side (skipping a GUARD band right next
    to the cell under test, so the target's own energy doesn't leak into
    its own noise estimate), multiply by threshold_scale, and flag the
    cell if it exceeds that local threshold. Returns a boolean mask."""
    n = len(power)
    detections = np.zeros(n, dtype=bool)
    thresholds = np.zeros(n)
    win = num_guard + num_ref
    for i in range(win, n - win):
        leading = power[i - win:i - num_guard]
        trailing = power[i + num_guard + 1:i + win + 1]
        noise_level = np.mean(np.concatenate([leading, trailing]))
        thresholds[i] = noise_level * threshold_scale
        detections[i] = power[i] > thresholds[i]
    return detections, thresholds


# =======================================================================
# END TO END, against a target with a KNOWN range/velocity
# =======================================================================
def main():
    p = load_minimal_cfg()
    true_range_m, true_velocity_mps = 1.50, 0.30

    print(f"Synthesizing a target at {true_range_m} m, {true_velocity_mps} "
          "m/s (a small, known ground truth - real capture bytes would "
          "come from load_complex_cube() instead, see lesson 06)...")
    raw_bytes = synthesize_raw_adc(p, target_range_m=true_range_m,
                                    target_velocity_mps=true_velocity_mps,
                                    target_angle_deg=0.0, snr_db=15.0)
    raw = np.frombuffer(raw_bytes, dtype=np.int16)
    per_frame = p["numAdcSamples"] * p["numRx"] * p["chirpsPerFrame"]
    n_frames = raw.size // per_frame
    cube = raw.astype(np.complex64)[:n_frames * per_frame].reshape(
        n_frames, p["chirpsPerFrame"], p["numRx"], p["numAdcSamples"])
    cube = cube.reshape(n_frames, p["numLoops"], p["numTx"], p["numRx"],
                         p["numAdcSamples"])
    # (assume firing order already == TX id order for this synthetic
    # example - lesson 06 covers when/why that's NOT a safe assumption
    # for a real capture)
    cube = cube.reshape(n_frames, p["numLoops"], p["numTx"] * p["numRx"],
                         p["numAdcSamples"])

    ch = 0 * p["numRx"] + 0   # TX0-RX0
    single_channel = cube[:, :, ch, :]           # (frames, loops, samples)

    rfft = range_fft(single_channel)             # (frames, loops, samples)
    r_axis = range_axis(p)

    # Real sampling (lesson 05): only the first HALF of the FFT is
    # independent information - the second half is the Hermitian mirror
    # of the first (same magnitudes, negated phase) and carries no new
    # range information, just a redundant reflection. Search only that
    # first half, exactly like the real, working pipeline
    # (capture_lib.py's load_complex_cube usage) does - searching the
    # full spectrum would occasionally "detect" a target's own mirror
    # image out past the true unambiguous range as if it were a second,
    # separate target.
    n_half = p["numAdcSamples"] // 2
    r_axis = r_axis[:n_half]

    # Average magnitude over frames for a clean range profile.
    range_profile = np.abs(rfft[..., :n_half]).mean(axis=(0, 1))
    peak_bin = int(np.argmax(range_profile))
    print(f"\nRange FFT peak: bin {peak_bin} -> {r_axis[peak_bin]:.3f} m "
          f"(true: {true_range_m} m)")
    assert abs(r_axis[peak_bin] - true_range_m) < 0.10, \
        "range peak should land within one bin or so of the true range"

    # Doppler FFT on frame 0, all loops, at the range bin we just found.
    dfft = doppler_fft(rfft[0:1])[0][:, :n_half]  # (loops, samples<=n_half)
    v_axis = velocity_axis(p, tx_count=p["numTx"])
    doppler_slice = np.abs(dfft[:, peak_bin])
    v_peak_bin = int(np.argmax(doppler_slice))
    print(f"Doppler FFT peak at that range: bin {v_peak_bin} -> "
          f"{v_axis[v_peak_bin]:+.3f} m/s (true: {true_velocity_mps} m/s)")
    assert abs(v_axis[v_peak_bin] - true_velocity_mps) < 0.05, \
        "velocity peak should land close to the true velocity"

    # CA-CFAR along range, at the Doppler bin we just found.
    power = np.abs(dfft[v_peak_bin, :]) ** 2
    detections, thresholds = ca_cfar_1d(power, num_guard=6, num_ref=20,
                                         threshold_scale=12.0)
    detected_bins = np.where(detections)[0]
    near_target = [b for b in detected_bins if abs(b - peak_bin) <= 5]
    stray = [b for b in detected_bins if abs(b - peak_bin) > 5]
    print(f"\nCA-CFAR (guard=6, ref=20, scale=12x) flagged "
          f"{len(detected_bins)} range bin(s): {list(detected_bins)}")
    print(f"  {len(near_target)} of them are the target's own main lobe "
          f"(bins within 5 of {peak_bin} - windowing, lesson 07's top, "
          "widens a single target into a few adjacent bins).")
    if stray:
        print(f"  {len(stray)} stray bin(s) elsewhere ({stray}) are just "
              "CFAR's designed-in false alarm rate doing its job on pure "
              "noise - a CFAR threshold that NEVER false-alarms would "
              "also miss real weak targets. This is expected, not a "
              "bug; a real system deals with it via track confirmation "
              "over several frames, not by chasing a perfectly silent "
              "single-frame threshold.")
    assert peak_bin in detected_bins, \
        "CFAR should flag the bin the target is actually in"
    print(f"Target's true bin ({peak_bin}) is correctly among them.")

    print("\nEnd to end: synthesized a target at a known range/velocity,")
    print("recovered both from raw ADC bytes via range FFT + Doppler FFT,")
    print("and had CA-CFAR correctly flag it against the noise floor -")
    print("all three assertions above passed.")


if __name__ == "__main__":
    main()
    print("\nNext: 08_angle_estimation_basics.py - using all 16 virtual")
    print("channels together (not just TX0-RX0) to recover a bearing, "
          "not just range and velocity.")
