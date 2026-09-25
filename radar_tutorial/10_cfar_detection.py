"""
LESSON 10 — Automatically finding targets: CFAR

Looking at a range-Doppler map by eye and pointing at "the bright spot" does
not scale to a real-time system. CFAR ("Constant False Alarm Rate") is the
standard algorithm that does this automatically: for every cell in the map,
compare it to the AVERAGE of its neighbors ("training cells"), and flag it
as a detection only if it's significantly brighter than that local average.

WHY "local" average and not one global threshold: a busy, cluttered range
bin might have a naturally higher noise floor than an empty one. Comparing
each cell only to its own neighborhood makes the algorithm adapt
automatically, which is what "constant false alarm rate" means - roughly
the same fraction of false detections everywhere, not more false alarms in
noisy regions.

Run it:
    python 10_cfar_detection.py --cube ../output/07_build_and_explore_cube/cube.npy --cfg your_profile.cfg

(needs scipy: pip install scipy - if missing, this lesson explains what
you're missing and stops cleanly rather than crashing confusingly)
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np

C = 299792458.0
OUT_DIR = "output/10_cfar_detection"


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
    return range_axis, lam, tc


def ca_cfar(rd_db, guard=(2, 2), train=(6, 6), offset_db=8.0):
    from scipy.ndimage import uniform_filter
    gw = (2*guard[0]+1, 2*guard[1]+1)
    tw = (2*(guard[0]+train[0])+1, 2*(guard[1]+train[1])+1)
    lin = 10 ** (rd_db / 20.0)
    big = uniform_filter(lin, size=tw, mode="nearest") * (tw[0]*tw[1])
    small = uniform_filter(lin, size=gw, mode="nearest") * (gw[0]*gw[1])
    n_train = tw[0]*tw[1] - gw[0]*gw[1]
    noise_floor = (big - small) / max(n_train, 1)
    threshold_db = 20 * np.log10(noise_floor + 1e-9) + offset_db
    print(f"CFAR: each cell compared against its {n_train} neighboring 'training'")
    print(f"cells (a {tw[0]}x{tw[1]} window with a {gw[0]}x{gw[1]} guard gap around")
    print(f"the cell itself, so the target's own energy doesn't pollute its own")
    print(f"noise estimate), flagged if it's >{offset_db} dB above that local average.")
    return rd_db > threshold_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        import scipy  # noqa
    except ImportError:
        print("scipy is not installed. Run: pip install scipy")
        print("(CFAR specifically needs it for a fast neighborhood-average")
        print("operation; nothing else in this whole tutorial does.)")
        return

    cube = np.load(args.cube)
    range_axis, lam, tc = get_axes(args.cfg)
    f = cube[min(args.frame, cube.shape[0]-1)]
    n_chirps = f.shape[0]
    v_max = lam / (4 * tc)
    v_axis = np.linspace(-v_max, v_max, n_chirps, endpoint=False)

    range_fft = np.fft.fft(f * np.hanning(f.shape[-1]), axis=-1)
    range_fft -= range_fft.mean(axis=0, keepdims=True)
    doppler_fft = np.fft.fftshift(
        np.fft.fft(range_fft * np.hanning(n_chirps)[:, None, None], axis=0), axes=0)
    rd_db = 20 * np.log10(np.abs(doppler_fft).sum(axis=1) + 1e-9)
    n_half = len(range_axis)
    rd_db = rd_db[:, :n_half]

    detections = ca_cfar(rd_db)
    yy, xx = np.nonzero(detections)
    print(f"\n{len(xx)} cells flagged as detections in this frame.")

    plt.figure(figsize=(8, 5))
    plt.imshow(rd_db, aspect="auto", origin="lower", cmap="gray",
               extent=[range_axis[0], range_axis[n_half-1], v_axis[0], v_axis[-1]])
    plt.scatter(range_axis[xx], v_axis[yy], s=8, c="red", label="CFAR detection")
    plt.xlabel("Range [m]"); plt.ylabel("Velocity [m/s]")
    plt.title(f"CFAR detections ({len(xx)} cells)")
    plt.legend()
    out = f"{OUT_DIR}/cfar.png"
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    print("\nEach red dot is now a candidate 'target': a (range, velocity) pair")
    print("your policy/ML code could act on directly, instead of the whole map.")


if __name__ == "__main__":
    main()
