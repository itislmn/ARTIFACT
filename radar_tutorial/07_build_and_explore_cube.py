"""
LESSON 07 — From a pile of bytes to a labeled "radar cube"

You have a .bin file full of raw int16 numbers (from lesson 06, or your
earlier capture). Right now it's just one giant flat list. This lesson
reshapes it into 4 clearly labeled dimensions - the "radar cube" - and,
critically, PRINTS its shape and explains what each axis means so you stop
treating it as a magic black box.

Run it:
    python 07_build_and_explore_cube.py --bin ../output/06_capture_raw_iq/raw_capture.bin --cfg your_profile.cfg

THE KEY IDEA: a .bin file is just samples written one after another, in a
fixed, predictable order: for each frame, for each chirp, for each receive
antenna, for each sample-in-the-chirp - one number (or one I+jQ pair) at a
time. If we know how many of each there are (from the .cfg, via lesson 02),
reshaping is just `.reshape(n_frames, n_chirps, n_rx, n_samples)` - genuinely
one line of numpy, once you know the four numbers to put in it.
"""
import argparse
import os
import re

import numpy as np

OUT_DIR = "output/07_build_and_explore_cube"


def parse_cfg(path):
    """Same minimal parser as lesson 02 - repeated here so this file stands alone."""
    p = {"chirpTx": {}}
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith(("%", "#")):
            continue
        t = re.split(r"\s+", line)
        k, v = t[0], t[1:]
        if k == "profileCfg":
            p["numAdcSamples"] = int(v[9])
        elif k == "frameCfg":
            p["chirpStartIdx"], p["chirpEndIdx"] = int(v[0]), int(v[1])
            p["numLoops"] = int(v[2])
        elif k == "channelCfg":
            p["rxMask"] = int(v[0])
        elif k == "adcCfg":
            p["isComplex"] = int(v[1]) != 0
        elif k == "chirpCfg":
            p["chirpTx"][int(v[0])] = int(v[7])
    p["numRx"] = bin(p.get("rxMask", 15)).count("1")
    p["numTx"] = len({m for m in p["chirpTx"].values() if m}) or 1
    chirps_per_loop = p.get("chirpEndIdx", 0) - p.get("chirpStartIdx", 0) + 1
    p["chirpsPerFrame"] = chirps_per_loop * p.get("numLoops", 1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    p = parse_cfg(args.cfg)
    print("From your .cfg, we know:")
    print(f"  numAdcSamples (per chirp) = {p['numAdcSamples']}")
    print(f"  numRx (antennas)          = {p['numRx']}")
    print(f"  chirpsPerFrame            = {p['chirpsPerFrame']}")
    print(f"  numTx (antennas)          = {p['numTx']}")
    print(f"  complex ADC output?       = {p['isComplex']}")

    raw = np.fromfile(args.bin, dtype=np.int16)
    print(f"\nThe file has {raw.size:,} raw int16 numbers.")

    if p["isComplex"]:
        print("Complex mode: every 4 numbers are [I0, I1, Q0, Q1] for two samples.")
        quads = raw.reshape(-1, 4)
        data = (quads[:, 0:2] + 1j * quads[:, 2:4]).reshape(-1)
    else:
        print("Real-only mode: every number IS one sample directly, no pairing needed.")
        data = raw.astype(np.complex64)

    per_frame = p["numAdcSamples"] * p["numRx"] * p["chirpsPerFrame"]
    n_frames = data.size // per_frame
    print(f"\nOne full frame needs {p['numAdcSamples']} x {p['numRx']} x "
          f"{p['chirpsPerFrame']} = {per_frame:,} samples.")
    print(f"Your file has enough data for {n_frames} COMPLETE frames "
          f"(dropping {data.size % per_frame} leftover samples that don't fill a whole frame).")

    if n_frames == 0:
        print("\nZero complete frames - the numbers above don't divide evenly into")
        print("your file size. This almost always means numTx, numRx, or")
        print("numAdcSamples doesn't actually match what was captured.")
        return

    cube = data[:n_frames * per_frame].reshape(
        n_frames, p["chirpsPerFrame"], p["numRx"], p["numAdcSamples"])
    print(f"\nRESHAPED CUBE SHAPE: {cube.shape}")
    print("  axis 0 (size {}) = which FRAME (a full radar 'snapshot')".format(cube.shape[0]))
    print("  axis 1 (size {}) = which CHIRP within that frame".format(cube.shape[1]))
    print("  axis 2 (size {}) = which RECEIVE ANTENNA".format(cube.shape[2]))
    print("  axis 3 (size {}) = which SAMPLE within that chirp".format(cube.shape[3]))

    if p["numTx"] > 1:
        print(f"\nYour setup uses {p['numTx']} TX antennas fired one after another")
        print("(TDM). Right now axis 1 mixes 'which chirp' with 'which TX antenna'")
        print("together. Separating them turns axis 1+2 into a clean VIRTUAL")
        print("antenna axis - this is what lesson 09 (Doppler) actually needs:")
        cube_sep = cube.reshape(n_frames, -1, p["numTx"], p["numRx"], p["numAdcSamples"])
        cube_sep = cube_sep.reshape(n_frames, -1, p["numTx"] * p["numRx"], p["numAdcSamples"])
        print(f"  after separating TX: {cube_sep.shape}  "
              f"(frames, doppler-chirps-per-antenna, virtual-antennas, samples)")
        cube = cube_sep

    np.save(f"{OUT_DIR}/cube.npy", cube)
    print(f"\nSaved the finished cube to {OUT_DIR}/cube.npy")
    print(f"Pass this file straight into lesson 08 and 09 with --cube {OUT_DIR}/cube.npy")


if __name__ == "__main__":
    main()
