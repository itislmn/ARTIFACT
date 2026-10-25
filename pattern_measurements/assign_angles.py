"""
assign_angles.py - gives every frame of a continuous sweep its angle afterwards, linearly interpolated between
time:angle marks; the readings of a recording are replaced, never duplicated, in data/<cut>_<tag>_measurements.json.

    python assign_angles.py --cut azimuth --tag air --marks "0:-90, 15:0, 30:90"        (--show: the level over time)
"""
import argparse
import json
import os
import re
import sys

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_marks(text):
    pairs = []
    for part in re.split(r"[,;\n]+", text):
        part = part.split("#")[0].strip()
        if not part:
            continue
        nums = re.split(r"[:=\s]+", part)
        if len(nums) != 2:
            raise ValueError(f"cannot read mark '{part}' - use TIME:ANGLE, e.g. 15:0")
        pairs.append((float(nums[0]), float(nums[1])))
    if len(pairs) < 2:
        raise ValueError("need at least 2 marks (start and end of the sweep)")
    t = np.array([a for a, _ in pairs])
    ang = np.array([b for _, b in pairs])
    if np.any(np.diff(t) <= 0):
        raise ValueError("mark TIMES must be strictly increasing (angles may go up and down)")
    return t, ang


def angles_from_marks(frame_t, mark_t, mark_angle):
    frame_t = np.asarray(frame_t, float)
    ang = np.interp(frame_t, mark_t, mark_angle)
    inside = (frame_t >= mark_t[0]) & (frame_t <= mark_t[-1])
    return ang, inside


def write_log(run, ang, keep, log_path):
    run_id = int(run["run_id"])
    log = json.load(open(log_path)) if os.path.exists(log_path) else []
    before = len(log)
    log = [r for r in log if r.get("run") != run_id]
    replaced = before - len(log)
    n_rx = int(run["numRx"])
    gains, raw, noise = run["gains"], run["gains_raw"], run["noise"]
    rng = run["range_m"]
    bg = bool(run["bg_subtracted"])
    added = 0
    for f in np.nonzero(keep)[0]:
        for ch in range(gains.shape[1]):
            log.append({"angle": round(float(ang[f]), 4), "tx": ch // n_rx, "rx": ch % n_rx,
                        "gain_linear": float(gains[f, ch]),
                        "gain_raw_linear": float(raw[f, ch]),
                        "noise_linear": float(noise[f, ch]),
                        "range_m": float(rng[f]),
                        "bg_subtracted": bg, "run": run_id})
            added += 1
    with open(log_path, "w") as fh:
        json.dump(log, fh, separators=(",", ":"))
    return added, replaced, len(log)


def show(run):
    t = run["frame_time_rel"]
    level = np.median(run["gains"], axis=1)
    db = 20 * np.log10(level / level.max() + 1e-12)
    dur = float(t[-1])
    step = max(1, int(np.ceil(dur / 60)))
    print(f"{len(t)} frames, {t[0]:.1f} to {t[-1]:.1f} s ({len(t) / max(dur, 1e-9):.1f} frames/s), "
          f"target locked at {float(run['lock_range_m']):.2f} m.")
    print(f"Strongest frame at t = {t[int(np.argmax(level))]:.1f} s (0 dB) - normally the moment the "
          "target crossed boresight (0 deg).\n")
    print("  time     level (median of 16 channels, dB re strongest)")
    for s in range(0, int(dur) + 1, step):
        sel = (t >= s) & (t < s + step)
        if not sel.any():
            continue
        v = float(db[sel].max())
        bar = "#" * int(max(0, (v + 60) / 60 * 40))
        print(f"  {s:4d} s  {v:6.1f}  {bar}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", required=True, choices=["azimuth", "elevation"])
    ap.add_argument("--tag", default="", help="label for this data set (e.g. air, pla, petg): every file gets it in its name, so different radomes/setups never mix or overwrite each other")
    ap.add_argument("--marks", type=str, default=None, help='"time_s:angle_deg, ..." e.g. "0:-90, 15:0, 30:90"')
    ap.add_argument("--marks-file", type=str, default=None)
    ap.add_argument("--run", type=str, default=None,
                    help="recording to use (default data/<cut>_continuous_raw.npz)")
    ap.add_argument("--time-shift", type=float, default=0.0)
    ap.add_argument("--hold-ends", action="store_true")
    ap.add_argument("--show", action="store_true", help="only show the recording's level over time")
    args = ap.parse_args()

    name = args.cut + (f"_{args.tag}" if args.tag else "")
    run_path = args.run or os.path.join(OUT_DIR, f"{name}_continuous_raw.npz")
    if not os.path.exists(run_path):
        print(f"No recording at {run_path} - run measure_continuous.py first.")
        sys.exit(1)
    run = dict(np.load(run_path))
    show(run)
    if args.show:
        return

    if args.marks_file:
        text = open(args.marks_file).read()
    elif args.marks:
        text = args.marks
    else:
        print("Enter your marks as TIME:ANGLE pairs (seconds:degrees), comma-separated, e.g.")
        print("   0:-90, 6:-60, 15:0, 24:60, 30:90")
        text = input("marks> ")
    try:
        mt, ma = parse_marks(text)
    except ValueError as e:
        print(f"Marks problem: {e}")
        sys.exit(1)
    mt = mt + args.time_shift

    ang, inside = angles_from_marks(run["frame_time_rel"], mt, ma)
    keep = np.ones_like(inside) if args.hold_ends else inside
    if keep.sum() == 0:
        print("No frame lies between your first and last mark - check the times "
              "(seconds since GO) against the table above.")
        sys.exit(1)
    print(f"{inside.sum()} frames lie between the first ({mt[0]:.1f} s) and last ({mt[-1]:.1f} s) mark, "
          f"{(~inside).sum()} outside "
          f"({'kept at the end angles' if args.hold_ends else 'not logged; use --hold-ends to keep them'}).")
    if inside.sum():
        span = ma.max() - ma.min()
        print(f"Density: {inside.sum() / max(span, 1e-9):.1f} frames per degree over {span:g} deg.")

    log_path = os.path.join(OUT_DIR, f"{name}_measurements.json")
    added, replaced, total = write_log(run, ang, keep, log_path)
    print(f"Added {added} readings to {log_path}"
          + (f" (replaced {replaced} older readings of this recording)" if replaced else "")
          + f"; {total} in total.")
    print("\nNow run:  python run_all.py   (with the tag in dataset.TAGS)")


if __name__ == "__main__":
    main()
