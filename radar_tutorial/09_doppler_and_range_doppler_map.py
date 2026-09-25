"""
LESSON 09 — Adding velocity: the second FFT, and an honest warning

Lesson 08 gave you range from ONE fft, along the samples axis. Velocity
comes from a SECOND fft, along the CHIRPS axis - watching how the range-FFT
peak's phase slowly rotates from chirp to chirp tells us how fast something
is moving.

Run it:
    python 09_doppler_and_range_doppler_map.py --cube ../output/07_build_and_explore_cube/cube.npy --cfg your_profile.cfg

READ THIS BEFORE YOU LOOK AT YOUR PLOT:
If you're using multiple TX antennas fired one-after-another (TDM - see
lesson 02's explanation of chirpCfg), you likely have FEWER usable chirps
per antenna than your total chirp count suggests, giving you a short,
coarse Doppler FFT. Short FFTs are more sensitive to tiny timing
inconsistencies between antenna switches, which commonly shows up as
energy pinned at the very edges of the velocity axis, spread across every
range - NOT a real moving target. This is a well-known TDM-MIMO artifact,
not a bug in this code. This lesson will point out explicitly whether your
result looks like this pattern.
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np

C = 299792458.0
OUT_DIR = "output/09_doppler_and_range_doppler_map"


def get_axes(cfg_path):
    p = {}
    for raw in open(cfg_path):
        line = raw.strip()
        v = re.split(r"\s+", line)[1:] if line else []
        if line.startswith("profileCfg"):
            p["startFreq_GHz"] = float(v[1])
            p["idleTime_us"] = float(v[2])
            p["rampEndTime_us"] = float(v[5])
            p["slope_MHz_us"] = float(v[7])
            p["numAdcSamples"] = int(v[9])
            p["sampleRate_ksps"] = float(v[10])
        elif line.startswith("chirpCfg"):
            p.setdefault("chirpTx", {})[int(v[0])] = int(v[7])
    p["numTx"] = len({m for m in p.get("chirpTx", {}).values() if m}) or 1
    N, fs = p["numAdcSamples"], p["sampleRate_ksps"] * 1e3
    slope = p["slope_MHz_us"] * 1e12
    bw = slope * N / fs
    range_axis = np.arange(N) * (C * fs) / (2 * slope * N)
    lam = C / (p["startFreq_GHz"] * 1e9 + bw / 2)
    tc = (p["idleTime_us"] + p["rampEndTime_us"]) * 1e-6 * p["numTx"]
    return range_axis, lam, tc, p["numTx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cube = np.load(args.cube)
    range_axis, lam, tc, num_tx = get_axes(args.cfg)
    f = cube[min(args.frame, cube.shape[0] - 1)]
    n_chirps = f.shape[0]

    v_max = lam / (4 * tc)
    v_axis = np.linspace(-v_max, v_max, n_chirps, endpoint=False)
    print(f"With {n_chirps} usable chirps for Doppler, velocity resolution is "
          f"{2*v_max/n_chirps:.3f} m/s, max +-{v_max:.2f} m/s")
    if n_chirps < 32:
        print(f"\n*** {n_chirps} chirps is a SHORT Doppler FFT. Watch for energy")
        print("*** pinned at the very top/bottom rows of the plot below, spread")
        print("*** across every range - that's the TDM artifact this lesson warned about.")

    # FFT 1: range (same as lesson 08)
    range_fft = np.fft.fft(f * np.hanning(f.shape[-1]), axis=-1)
    range_fft -= range_fft.mean(axis=0, keepdims=True)  # remove non-moving (static) energy

    # FFT 2: doppler, along the CHIRP axis this time
    dop_window = np.hanning(n_chirps)[:, None, None]
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft * dop_window, axis=0), axes=0)
    rd_map_db = 20 * np.log10(np.abs(doppler_fft).sum(axis=1) + 1e-9)

    n_half = len(range_axis)
    plt.figure(figsize=(8, 5))
    plt.imshow(rd_map_db[:, :n_half], aspect="auto", origin="lower", cmap="viridis",
               extent=[range_axis[0], range_axis[n_half-1], v_axis[0], v_axis[-1]])
    plt.colorbar(label="dB")
    plt.xlabel("Range [m]"); plt.ylabel("Velocity [m/s]")
    plt.title("Range-Doppler map (this frame)")
    out = f"{OUT_DIR}/range_doppler.png"
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    print("A REAL moving target looks like a compact bright blob at a specific")
    print("(range, velocity) pair. If instead you see bright horizontal bands")
    print("hugging the top and bottom edges across every range, that's the TDM")
    print("artifact - improving it needs TX phase calibration or a longer")
    print("Doppler FFT (more chirps per frame), both beyond this lesson's scope.")


if __name__ == "__main__":
    main()
