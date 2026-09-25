"""
LESSON 11 — A gallery of every way to LOOK at this data

Different questions need different pictures. This lesson takes one radar
cube and makes several genuinely different visualizations from it, side by
side, explaining what question each one answers.

Run it:
    python 11_visualization_gallery.py --cube ../output/07_build_and_explore_cube/cube.npy --cfg your_profile.cfg
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np

C = 299792458.0
OUT_DIR = "output/11_visualization_gallery"


def get_range_axis(cfg_path):
    for raw in open(cfg_path):
        line = raw.strip()
        if line.startswith("profileCfg"):
            v = re.split(r"\s+", line)[1:]
            slope = float(v[7]) * 1e12
            N = int(v[9])
            fs = float(v[10]) * 1e3
            return np.arange(N) * (C * fs) / (2 * slope * N)
    raise ValueError("no profileCfg line found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    cube = np.load(args.cube)  # (frames, chirps, vrx, samples)
    range_axis = get_range_axis(args.cfg)
    n_half = len(range_axis)
    n_frames = cube.shape[0]

    # ---- VIEW 1: single range profile, one frame, one antenna ----
    # Question: "what does the raw signal from ONE antenna look like right now?"
    one_chirp = cube[0, 0, 0, :]
    plt.figure(figsize=(7, 3.5))
    plt.plot(np.abs(one_chirp))
    plt.xlabel("sample index"); plt.ylabel("|amplitude|")
    plt.title("VIEW 1: raw magnitude of one single chirp, one antenna\n"
              "(the most zoomed-in view possible - just one waveform)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/view1_single_chirp.png", dpi=150)
    plt.close()

    # ---- VIEW 2: range profile averaged over everything ----
    # Question: "what's out there, distance-wise, on average?"
    rp = 20*np.log10(np.abs(np.fft.fft(cube*np.hanning(cube.shape[-1]),
                     axis=-1)).mean(axis=(0,1,2)) + 1e-9)
    plt.figure(figsize=(7, 3.5))
    plt.plot(range_axis[:n_half], rp[:n_half])
    plt.xlabel("Range [m]"); plt.ylabel("dB")
    plt.title("VIEW 2: average range profile across the WHOLE capture\n"
              "('what's persistently out there')")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/view2_avg_range_profile.png", dpi=150)
    plt.close()

    # ---- VIEW 3: range-time intensity ("micro-motion" view) ----
    # Question: "how did the scene change over the WHOLE capture, second by second?"
    n_show = min(n_frames, 400)
    rti = np.stack([
        20*np.log10(np.abs(np.fft.fft(cube[i]*np.hanning(cube.shape[-1]),
                    axis=-1)).mean(axis=(0,1))[:n_half] + 1e-9)
        for i in range(n_show)])
    rti -= rti.mean(axis=0, keepdims=True)
    plt.figure(figsize=(8, 4))
    plt.imshow(rti.T, aspect="auto", origin="lower", cmap="inferno",
               extent=[0, n_show, 0, range_axis[n_half-1]])
    plt.colorbar(label="dB rel. mean")
    plt.xlabel("frame number"); plt.ylabel("Range [m]")
    plt.title("VIEW 3: range-time intensity\n"
              "(a moving object draws a diagonal streak here - look for one)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/view3_range_time.png", dpi=150)
    plt.close()

    # ---- VIEW 4: per-antenna comparison ----
    # Question: "do all my receive antennas actually see the same thing?"
    # (a real sanity check - if one antenna looks totally different, it may
    # be wired/configured wrong)
    n_ant = cube.shape[2]
    fig, axes = plt.subplots(1, min(n_ant, 4), figsize=(4*min(n_ant,4), 3.5), sharey=True)
    if n_ant == 1:
        axes = [axes]
    for a in range(min(n_ant, 4)):
        rp_a = 20*np.log10(np.abs(np.fft.fft(
            cube[0,:,a,:]*np.hanning(cube.shape[-1]), axis=-1)).mean(axis=0)[:n_half]+1e-9)
        axes[a].plot(range_axis[:n_half], rp_a)
        axes[a].set_title(f"antenna {a}")
        axes[a].set_xlabel("Range [m]")
    axes[0].set_ylabel("dB")
    plt.suptitle("VIEW 4: same frame, each virtual antenna plotted separately\n"
                 "(they should look broadly similar - big differences hint at a config/wiring issue)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/view4_per_antenna.png", dpi=150)
    plt.close()

    print(f"Saved 4 different views into {OUT_DIR}/:")
    print("  view1_single_chirp.png       - the most raw, zoomed-in signal")
    print("  view2_avg_range_profile.png  - what's out there overall")
    print("  view3_range_time.png         - how the scene changed over time")
    print("  view4_per_antenna.png        - sanity-check that antennas agree")
    print("\nEach answers a genuinely different question - pick the view that")
    print("matches what you're actually trying to find out, don't default to")
    print("range-Doppler for everything just because it's the 'standard' plot.")


if __name__ == "__main__":
    main()
