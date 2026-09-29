"""
run_on_capture.py — the validated methods, applied to YOUR real captured data.

    python run_on_capture.py --bin path/to/raw_capture.bin --cfg pattern_awr2944P.cfg --n-targets 1

Pulls out the 12 real azimuth virtual channels at the strongest range bin
(or one you specify), uses each chirp as one snapshot, and runs all four
validated DOA methods on it - producing a spectrum plot with each method's
estimated angle marked.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "radar_tutorial"))
from pattern_measurements.visualize_pattern import parse_cfg, load_complex_cube          # noqa: E402
from doa_methods import (AZIMUTH_CHANNEL_INDICES, bartlett,    # noqa: E402
                          capon, music, esprit, find_peaks)

OUT_DIR = "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--n-targets", type=int, default=1)
    ap.add_argument("--target-range-m", type=float, default=None)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    p = parse_cfg(args.cfg)
    cube = load_complex_cube(args.bin, p)      # (frames, chirps, 16, samples)
    f = cube[min(args.frame, cube.shape[0] - 1)]

    window = np.hanning(f.shape[-1])
    range_fft = np.fft.fft(f * window, axis=-1)
    n_half = f.shape[-1] // 2
    mag = np.abs(range_fft[:, :, :n_half]).mean(axis=(0, 1))

    if args.target_range_m is not None:
        center = int(np.argmin(np.abs(p["range_axis"] - args.target_range_m)))
        tol = max(3, int(0.3 / (p["range_axis"][1] - p["range_axis"][0])))
        lo, hi = max(0, center - tol), min(n_half, center + tol)
        peak_bin = lo + int(np.argmax(mag[lo:hi]))
    else:
        peak_bin = int(np.argmax(mag))
    print(f"Using range bin {peak_bin} = {p['range_axis'][peak_bin]:.2f} m")

    # snapshots: 12 real azimuth channels, one column per chirp, at that range bin
    snapshots = range_fft[:, AZIMUTH_CHANNEL_INDICES, peak_bin].T   # (12, n_chirps)
    print(f"Using {snapshots.shape[1]} chirps as snapshots")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "0.25",
                          "grid.color": "0.85", "legend.frameon": False})

    for name, fn in [("Bartlett", bartlett), ("Capon", capon)]:
        ang, spec = fn(snapshots)
        spec_db = 20 * np.log10(spec / spec.max() + 1e-12)
        ax.plot(ang, spec_db, label=name, linewidth=1.3)
        est = find_peaks(ang, spec, args.n_targets)
        print(f"{name:10s} estimated angle(s): {est}")

    ang, spec = music(snapshots, args.n_targets)
    spec_db = 20 * np.log10(spec / spec.max() + 1e-12)
    ax.plot(ang, spec_db, label="MUSIC", linewidth=1.3)
    est_music = find_peaks(ang, spec, args.n_targets)
    print(f"{'MUSIC':10s} estimated angle(s): {est_music}")

    est_esprit = esprit(snapshots, args.n_targets)
    print(f"{'ESPRIT':10s} estimated angle(s): {est_esprit}  (no spectrum - closed form)")
    for a in np.atleast_1d(est_esprit):
        ax.axvline(a, color="red", linestyle="--", alpha=0.6)
    ax.plot([], [], color="red", linestyle="--", label="ESPRIT (closed-form)")

    ax.set_xlabel(r"Azimuth Angle $\alpha$ [$^\circ$]")
    ax.set_ylabel("Normalized spectrum [dB]")
    ax.set_title(f"DOA spectrum at {p['range_axis'][peak_bin]:.2f} m "
                 f"({args.n_targets} target(s) assumed)")
    ax.grid(True, alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = f"{OUT_DIR}/doa_real_capture.pdf"
    fig.savefig(out, format="pdf", dpi=300)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
