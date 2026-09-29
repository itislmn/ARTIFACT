r"""
LESSON 12 - Saving a capture to disk, and turning the cube into pictures

    python 12_saving_and_visualizing_the_cube.py --demo

No hardware needed - synthesizes a capture, saves it, reloads it, and
produces three real plot files you can open (a range-time waterfall, a
range-Doppler heatmap with CFAR detections marked, and a plain range
profile) - each checked for existing and being a real, non-trivial-size
file before this script calls itself done.

This is "using the data cube" in the everyday sense: not reprocessing
raw bytes every single time, and actually LOOKING at what you captured
rather than reading printed numbers.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")   # no display needed - write files directly
import matplotlib.pyplot as plt
import numpy as np

from common import synthesize_raw_adc

C = 2.99792458e8
OUT_DIR = "output"


def load_cfg_numbers():
    return {
        "numAdcSamples": 656, "sampleRate_ksps": 13349.0,
        "slope_MHz_us": 70.0, "startFreq_GHz": 77.0,
        "idleTime_us": 186.0, "rampEndTime_us": 57.14,
        "numRx": 4, "numTx": 4, "numLoops": 64, "numFrames": 10,
        "chirpsPerFrame": 4 * 64, "isComplex": False,
        "chirpTx": {0: 1, 1: 4, 2: 8, 3: 2},
    }


def compute_tx_reorder(chirp_tx, num_tx):
    perm = [None] * num_tx
    for slot, mask in chirp_tx.items():
        if mask:
            tx_id = mask.bit_length() - 1
            if tx_id < num_tx:
                perm[tx_id] = slot
    assert None not in perm
    return perm


def bytes_to_cube(raw_int16, p):
    per_frame = p["numAdcSamples"] * p["numRx"] * p["chirpsPerFrame"]
    n_frames = raw_int16.size // per_frame
    cube = raw_int16.astype(np.complex64)[:n_frames * per_frame].reshape(
        n_frames, p["chirpsPerFrame"], p["numRx"], p["numAdcSamples"])
    cube = cube.reshape(n_frames, p["numLoops"], p["numTx"], p["numRx"],
                         p["numAdcSamples"])
    perm = compute_tx_reorder(p["chirpTx"], p["numTx"])
    cube = cube[:, :, perm, :, :]
    cube = cube.reshape(n_frames, p["numLoops"], p["numTx"] * p["numRx"],
                         p["numAdcSamples"])
    return cube


# =======================================================================
# SAVE / LOAD - .npz bundles the cube AND the cfg numbers it needs to be
# interpreted, so a reload never has to guess or re-derive anything.
# =======================================================================
def save_cube_npz(path, cube, p):
    np.savez_compressed(path, cube=cube, **{f"cfg_{k}": v for k, v in p.items()
                                              if not isinstance(v, dict)})
    size_kb = os.path.getsize(path) / 1024
    print(f"Saved {path} ({size_kb:.1f} KB, cube shape {cube.shape})")


def load_cube_npz(path):
    data = np.load(path)
    cube = data["cube"]
    p = {k[4:]: data[k].item() if data[k].ndim == 0 else data[k]
         for k in data.files if k.startswith("cfg_")}
    return cube, p


# =======================================================================
# PLOTS
# =======================================================================
def range_axis(p):
    N, fs = p["numAdcSamples"], p["sampleRate_ksps"] * 1e3
    slope = p["slope_MHz_us"] * 1e12
    return (np.arange(N) * (C * fs) / (2 * slope * N))[:N // 2]


def plot_range_time(cube, p, out_path):
    """One channel, magnitude vs range, one line per frame - a 'waterfall'
    view that's the fastest way to eyeball whether ANYTHING is showing up
    at all, and whether a target is moving (its range bin drifting frame
    to frame) or stationary (a flat vertical band)."""
    ch = 0
    n_half = p["numAdcSamples"] // 2
    window = np.hanning(p["numAdcSamples"])
    r_axis = range_axis(p)

    fig, ax = plt.subplots(figsize=(8, 5))
    n_frames = cube.shape[0]
    for fr in range(n_frames):
        rfft = np.fft.fft(cube[fr, 0, ch, :] * window)
        mag_db = 20 * np.log10(np.abs(rfft[:n_half]) + 1e-6)
        color = plt.cm.viridis(fr / max(1, n_frames - 1))
        ax.plot(r_axis, mag_db, color=color, alpha=0.8, linewidth=1)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"Range-time (TX0-RX0, {n_frames} frames, dark->light = "
                 "early->late)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def ca_cfar_2d(power_2d, num_guard, num_ref, threshold_scale):
    """Same CA-CFAR idea as lesson 07, run along the RANGE axis for
    every Doppler row independently - the natural 1-axis-at-a-time
    extension lesson 07 pointed at."""
    n_dop, n_rng = power_2d.shape
    detections = np.zeros_like(power_2d, dtype=bool)
    win = num_guard + num_ref
    for d in range(n_dop):
        row = power_2d[d]
        for i in range(win, n_rng - win):
            leading = row[i - win:i - num_guard]
            trailing = row[i + num_guard + 1:i + win + 1]
            noise = np.mean(np.concatenate([leading, trailing]))
            if row[i] > noise * threshold_scale:
                detections[d, i] = True
    return detections


def plot_range_doppler(cube, p, out_path):
    """The 2D heatmap every radar demo you've ever seen a screenshot of
    is showing: range on one axis, velocity on the other, color = signal
    strength - with CA-CFAR detections marked on top.

    Two windows are applied here, not one: a Hanning window along the
    FAST-time (range/sample) axis, same as every other lesson - AND a
    second Hanning window along the SLOW-time (loop/Doppler) axis. Skip
    that second window and a real target's energy doesn't land cleanly
    in one Doppler bin - it splatters across many neighbouring Doppler
    bins (classic rectangular-window spectral leakage), and CA-CFAR
    then flags dozens of those leakage bins as if they were dozens of
    separate targets. Windowing both axes is what keeps a single real
    target looking like a single tight blob in the map below instead of
    a smear."""
    ch = 0
    n_half = p["numAdcSamples"] // 2
    range_window = np.hanning(p["numAdcSamples"])
    doppler_window = np.hanning(p["numLoops"])[:, None]
    r_axis = range_axis(p)

    rfft = np.fft.fft(cube[0, :, ch, :] * range_window, axis=-1)[:, :n_half]
    dfft = np.fft.fftshift(np.fft.fft(rfft * doppler_window, axis=0), axes=0)
    power_db = 20 * np.log10(np.abs(dfft) + 1e-6)

    lam = C / (p["startFreq_GHz"] * 1e9)
    chirp_period_s = (p["idleTime_us"] + p["rampEndTime_us"]) * 1e-6
    doppler_pri_s = p["numTx"] * chirp_period_s
    v_max = lam / (4 * doppler_pri_s)
    v_axis = np.linspace(-v_max, v_max, p["numLoops"], endpoint=False)

    # Wider guard/reference windows and a higher threshold than lesson
    # 07's 1-D range-only CFAR: a 2-D map has a much larger "noise"
    # population to average over, and the target's own mainlobe is
    # wider here (it now spans a handful of bins in BOTH dimensions),
    # so the CFAR window needs to stay clear of it in both directions
    # too or the mainlobe contaminates its own noise estimate.
    detections = ca_cfar_2d(np.abs(dfft) ** 2, num_guard=6, num_ref=20,
                             threshold_scale=15.0)
    det_v, det_r = np.where(detections)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(r_axis, v_axis, power_db, shading="auto", cmap="viridis")
    if len(det_r):
        ax.scatter(r_axis[det_r], v_axis[det_v], marker="x", color="red",
                   s=40, label=f"CA-CFAR detections ({len(det_r)})")
        ax.legend(loc="upper right")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Range-Doppler map (TX0-RX0, frame 0)")
    fig.colorbar(im, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return len(det_r)


def plot_range_profile(cube, p, out_path):
    """Simplest possible plot: magnitude vs range, averaged over
    everything (all frames, loops, channels) - a quick sanity check for
    'is there a clear peak at all', without any of the range-Doppler
    map's extra complexity."""
    n_half = p["numAdcSamples"] // 2
    window = np.hanning(p["numAdcSamples"])
    rfft = np.fft.fft(cube * window, axis=-1)[..., :n_half]
    profile_db = 20 * np.log10(np.abs(rfft).mean(axis=(0, 1, 2)) + 1e-6)
    r_axis = range_axis(p)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r_axis, profile_db)
    peak_bin = int(np.argmax(profile_db))
    ax.axvline(r_axis[peak_bin], color="red", linestyle="--", alpha=0.6,
               label=f"peak: {r_axis[peak_bin]:.2f} m")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Magnitude (dB), averaged over all frames/loops/channels")
    ax.set_title("Range profile")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return r_axis[peak_bin]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    p = load_cfg_numbers()
    true_range = 1.6

    print(f"Synthesizing a capture (target at {true_range} m)...")
    raw_bytes = synthesize_raw_adc(p, target_range_m=true_range,
                                    target_velocity_mps=0.4,
                                    target_angle_deg=10.0, snr_db=18.0)
    raw = np.frombuffer(raw_bytes, dtype=np.int16)
    cube = bytes_to_cube(raw, p)

    npz_path = os.path.join(OUT_DIR, "capture.npz")
    save_cube_npz(npz_path, cube, p)

    print("\nReloading from disk (proving the round-trip actually works,")
    print("not just that saving didn't crash)...")
    cube2, p2 = load_cube_npz(npz_path)
    assert cube2.shape == cube.shape
    assert np.allclose(cube2, cube)
    assert p2["numAdcSamples"] == p["numAdcSamples"]
    print(f"Reloaded cube shape {cube2.shape} - bit-for-bit identical to "
          "what was saved.")

    print("\nGenerating plots...")
    rt_path = os.path.join(OUT_DIR, "range_time.png")
    plot_range_time(cube2, p2, rt_path)
    print(f"  {rt_path}: {os.path.getsize(rt_path)} bytes")

    rd_path = os.path.join(OUT_DIR, "range_doppler.png")
    n_det = plot_range_doppler(cube2, p2, rd_path)
    print(f"  {rd_path}: {os.path.getsize(rd_path)} bytes "
          f"({n_det} CFAR detections marked)")

    rp_path = os.path.join(OUT_DIR, "range_profile.png")
    peak_range = plot_range_profile(cube2, p2, rp_path)
    print(f"  {rp_path}: {os.path.getsize(rp_path)} bytes "
          f"(peak at {peak_range:.2f} m, true range {true_range} m)")

    for path in (rt_path, rd_path, rp_path):
        assert os.path.getsize(path) > 5000, f"{path} looks suspiciously small/empty"
    assert abs(peak_range - true_range) < 0.10
    assert n_det > 0, "the range-Doppler map should have found at least one detection"

    print(f"\nAll three plots saved to {OUT_DIR}/, all non-trivial in size,")
    print("and the range profile's peak matches the synthesized target's")
    print("true range - the save/load/plot round trip is verified, not")
    print("just 'ran without crashing'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="(the only mode - "
                     "this lesson works on any cube, real or synthetic)")
    ap.parse_args()
    main()
    print("\nThat's every elemental piece this tutorial set out to cover:")
    print("config, both control protocols, raw capture AND point-cloud")
    print("acquisition, the cube, TDM-MIMO, range/Doppler/CFAR, angle")
    print("estimation, saving/visualizing, and a real-time active loop.")
