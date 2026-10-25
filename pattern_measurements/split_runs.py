"""
split_runs.py - splits a log in which several sweeps were appended into one data set per run (by run id, backward
angle jumps or background on/off); the original log is never changed.

    python split_runs.py --cut azimuth --assign 1=air,2=pla,3=petg
"""
import argparse
import json
import os
import sys

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def frames_of(log):
    pairs = sorted({(r["tx"], r["rx"]) for r in log})
    n_ch = len(pairs)
    if len(log) % n_ch:
        print(f"note: {len(log)} readings is not a multiple of {n_ch}; the last incomplete frame is dropped.")
    frames = []
    for i in range(0, len(log) - n_ch + 1, n_ch):
        chunk = log[i:i + n_ch]
        if sorted((r["tx"], r["rx"]) for r in chunk) != pairs:
            raise SystemExit(f"Readings {i}..{i + n_ch - 1} are not one complete TX/RX set - the log is "
                             "not in the expected order, cannot split it automatically.")
        frames.append(chunk)
    return frames


def split(frames, jump_deg):
    runs, cur = [], [frames[0]]
    for prev, f in zip(frames, frames[1:]):
        new = False
        if prev[0].get("run") != f[0].get("run"):
            new = True
        elif f[0]["angle"] < prev[0]["angle"] - jump_deg:
            new = True
        elif prev[0].get("bg_subtracted") != f[0].get("bg_subtracted"):
            new = True
        if new:
            runs.append(cur)
            cur = []
        cur.append(f)
    runs.append(cur)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", required=True, choices=["azimuth", "elevation"])
    ap.add_argument("--tag", default="", help="split a tagged log instead of the untagged one")
    ap.add_argument("--log", default=None, help="explicit path of the merged log")
    ap.add_argument("--jump", type=float, default=45.0)
    ap.add_argument("--assign", default=None, help='e.g. "1=air,2=pla,3=petg"')
    args = ap.parse_args()

    name = args.cut + (f"_{args.tag}" if args.tag else "")
    path = args.log or os.path.join(OUT_DIR, f"{name}_measurements.json")
    if not os.path.exists(path):
        print(f"No log at {path}.")
        sys.exit(1)
    log = json.load(open(path))
    if not log:
        print("The log is empty.")
        sys.exit(1)
    runs = split(frames_of(log), args.jump)

    print(f"{len(log)} readings in {path} -> {len(runs)} run(s):\n")
    print("  run  frames  angle span          range   bg-sub  median level (dB, absolute units)")
    for i, r in enumerate(runs, 1):
        ang = np.array([f[0]["angle"] for f in r])
        lev = 20 * np.log10(np.median([x["gain_linear"] for f in r for x in f]) + 1e-12)
        rng = np.median([f[0].get("range_m", np.nan) for f in r])
        bg = r[0][0].get("bg_subtracted")
        print(f"  {i:3d}  {len(r):6d}  {ang.min():+7.1f} .. {ang.max():+7.1f}  {rng:6.2f}  "
              f"{str(bg):6s}  {lev:7.1f}")
    print()

    if not args.assign:
        print('Name the runs to write them out, e.g.:  python split_runs.py --cut ' + args.cut +
              ' --assign "1=air,2=pla,3=petg"')
        return
    for item in args.assign.split(","):
        k, v = item.split("=")
        k = int(k)
        if not 1 <= k <= len(runs):
            print(f"There is no run {k}.")
            sys.exit(1)
        out = os.path.join(OUT_DIR, f"{args.cut}_{v.strip()}_measurements.json")
        if os.path.exists(out):
            print(f"{out} already exists - not overwriting; pick another name or remove it.")
            sys.exit(1)
        flat = [x for f in runs[k - 1] for x in f]
        with open(out, "w") as fh:
            json.dump(flat, fh, separators=(",", ":"))
        print(f"run {k} -> {out} ({len(flat)} readings)")
    print("\nNext:  python run_all.py   (with the tags in dataset.TAGS)")


if __name__ == "__main__":
    main()
