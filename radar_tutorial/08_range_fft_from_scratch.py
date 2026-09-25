"""
LESSON 08 — The range FFT, on YOUR real data this time

Lesson 04 proved the range-FFT idea works on fake data where we knew the
answer in advance. Now let's run the exact same operation on your real
radar cube from lesson 07, and look at what a real, messy, noisy
environment actually produces.

Run it:
    python 08_range_fft_from_scratch.py --cube ../output/07_build_and_explore_cube/cube.npy --cfg your_profile.cfg
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np

C = 299792458.0
OUT_DIR = "output/08_range_fft_from_scratch"


def get_range_axis(cfg_path):
    p = {}
    for raw in open(cfg_path):
        line = raw.strip()
        if line.startswith("profileCfg"):
            v = re.split(r"\s+", line)[1:]
            p["slope_MHz_us"] = float(v[7])
            p["numAdcSamples"] = int(v[9])
            p["sampleRate_ksps"] = float(v[10])
    N = p["numAdcSamples"]
    fs = p["sampleRate_ksps"] * 1e3
    slope = p["slope_MHz_us"] * 1e12
    return np.arange(N) * (C * fs) / (2 * slope * N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cube = np.load(args.cube)
    print(f"Loaded cube, shape {cube.shape} (frames, chirps, virtual_rx, samples)")

    frame = cube[min(args.frame, cube.shape[0] - 1)]
    print(f"Using frame {args.frame}, shape {frame.shape} (chirps, virtual_rx, samples)")

    # THE ENTIRE LESSON IS THIS ONE LINE:
    # a window function first (reduces spectral leakage - try removing it and
    # compare, you'll see messier, wider peaks without it)
    window = np.hanning(frame.shape[-1])
    range_fft = np.fft.fft(frame * window, axis=-1)
    print(f"\nAfter FFT along the LAST axis (samples), shape is still "
          f"{range_fft.shape} - the FFT doesn't change the shape, it")
    print("transforms what each number along that axis MEANS: from 'signal")
    print("strength at this moment in time' to 'signal strength at this distance'.")

    # average over chirps and antennas just to get one clean profile to look at
    magnitude_db = 20 * np.log10(np.abs(range_fft).mean(axis=(0, 1)) + 1e-9)
    range_axis = get_range_axis(args.cfg)
    n_half = len(magnitude_db) // 2   # FFT output is mirrored, only first half is physical

    peak_bin = np.argmax(magnitude_db[:n_half])
    print(f"\nStrongest reflection in this frame is at {range_axis[peak_bin]:.2f} m")
    print("(this is whatever was physically the strongest reflector during")
    print("capture - a wall, furniture, a person - not necessarily 'the' target)")

    plt.figure(figsize=(8, 4))
    plt.plot(range_axis[:n_half], magnitude_db[:n_half])
    plt.axvline(range_axis[peak_bin], color="red", linestyle="--",
                label=f"strongest peak: {range_axis[peak_bin]:.2f} m")
    plt.xlabel("Range [m]"); plt.ylabel("Magnitude [dB]")
    plt.title("Range FFT - your real captured data")
    plt.legend(); plt.grid(alpha=0.3)
    out = f"{OUT_DIR}/range_profile.png"
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    print("COMPARE this to lesson 04's clean single peak: real data is bumpy,")
    print("has many small reflections everywhere, not one obvious spike. That's")
    print("normal - a real room has many reflective surfaces.")


if __name__ == "__main__":
    main()
