"""
split_runs.py - recover separate data sets from a log in which several measurements
were APPENDED together (the measurement scripts always append to
output/<cut>[_tag]_measurements.json - that is what merged your air / PLA / PETG runs).

    python split_runs.py --cut azimuth                      # only list the runs it finds
    python split_runs.py --cut azimuth --assign 1=air,2=pla,3=petg   # write one file per run

Your original file is never modified. The new files are
output/<cut>_<name>_measurements.json and work with plot_pattern.py --tag <name> and
compare_patterns.py.

HOW RUNS ARE FOUND
  * Readings are grouped into frames/captures of 16 consecutive readings (one per TX-RX pair).
  * A new run starts when (a) the stored run id changes (logs written by the newer
    measure_continuous.py / assign_angles.py), or (b) the angle jumps BACKWARDS by more than
    --jump degrees from one frame to the next (default 45: a sweep that ended at +90 and a new
    one that starts at -90), or (c) background subtraction was switched on/off.
  * The table shows, per run, the number of frames, angle span, the range the target was found
    at and the median level in absolute units. Use it to decide which run is which: if you
    measured them in a known order (e.g. air, PLA, PETG), run 1, 2, 3 are in that order.
  * If two runs had overlapping angles in the same direction and nothing else separating them,
    they cannot be told apart automatically; the table will show one big run. In that case
    the --jump option cannot help - say so and re-measure that one.
"""
import argparse
import json
import os
import sys

import numpy as np

OUT_DIR = "output"


def frames_of(log):
    """Split the reading list into frames (one per TX-RX sweep of the 16 channels)."""
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
    print("\nNext:  python compare_patterns.py --cut " + args.cut + " --tags air pla petg --ref air")


if __name__ == "__main__":
    main()
