"""
assign_angles.py - give the frames of a continuous sweep their ANGLES afterwards.

    1) record without pressing anything while sweeping:
         python measure_continuous.py --cli COM4 --cfg pattern_awr2944P.cfg ^
             --cut azimuth --range-m 0.30 --mark-later --background
    2) look at the recording (and read off where the pattern peaks in time):
         python assign_angles.py --cut azimuth --show
    3) enter your marks - pairs  TIME_IN_SECONDS:ANGLE_IN_DEGREES  - and the log is written:
         python assign_angles.py --cut azimuth --marks "0:-90, 6:-60, 15:0, 24:60, 30:90"
       (or run it without --marks and it will ask; or put one "time angle" pair per
        line in a text file and use --marks-file marks.txt)

TIME reference: seconds since the "GO" line / the on-screen clock of measure_continuous.py
started (the moment the radar started). Use a phone stopwatch started at GO, a video of
the sweep, or simply the times the on-screen clock showed.

No hardware is needed for this step; it can be re-run as often as you like with different
marks - readings of the same recording are REPLACED in the log, not duplicated (each
recording has a run id that is stored with its readings).

HOW THE ANGLE OF A FRAME IS COMPUTED
  Linear interpolation of your marks over time (np.interp). Between two marks the angle is
  assumed to change at constant speed, so more marks = better (marks at every 30 deg keep the
  error around a degree). The marks may go up and down (-90 -> +90 -> -90) as long as the
  TIMES strictly increase.

FRAMES OUTSIDE THE MARKS
  Frames recorded before your first mark or after your last mark have no known angle (you
  were not sweeping yet / any more), so by default they are NOT added to the log. With
  --hold-ends they are kept and assigned the first/last mark's angle (useful when you waited
  at the start position on purpose).

--time-shift S adds S seconds to every mark time (e.g. you started your stopwatch 0.4 s
after GO -> --time-shift 0.4).

--show prints the strength of the target return over time (median over the 16 channels,
0 dB = strongest frame). The main lobe of the antenna is where the target crosses boresight,
so the time of the strongest frame is a good sanity check for your "0 deg" mark.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

OUT_DIR = "output"


# ----------------------------------------------------------------------
def parse_marks(text):
    """'0:-90, 6:-60, 15:0' or one 'time angle' pair per line -> (times, angles)."""
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
    """Angle of every frame (linear interpolation) and a mask of the frames that lie
    between the first and the last mark."""
    frame_t = np.asarray(frame_t, float)
    ang = np.interp(frame_t, mark_t, mark_angle)
    inside = (frame_t >= mark_t[0]) & (frame_t <= mark_t[-1])
    return ang, inside


def write_log(run, ang, keep, log_path):
    """Append the readings of the kept frames to the measurement log. Readings of the
    same run (same run id) already in the log are removed first."""
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


# ----------------------------------------------------------------------
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
    ap.add_argument("--marks", type=str, default=None, help='"time_s:angle_deg, ..." e.g. "0:-90, 15:0, 30:90"')
    ap.add_argument("--marks-file", type=str, default=None)
    ap.add_argument("--run", type=str, default=None,
                    help="recording to use (default output/<cut>_continuous_raw.npz)")
    ap.add_argument("--time-shift", type=float, default=0.0)
    ap.add_argument("--hold-ends", action="store_true")
    ap.add_argument("--show", action="store_true", help="only show the recording's level over time")
    args = ap.parse_args()

    run_path = args.run or os.path.join(OUT_DIR, f"{args.cut}_continuous_raw.npz")
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

    log_path = os.path.join(OUT_DIR, f"{args.cut}_measurements.json")
    added, replaced, total = write_log(run, ang, keep, log_path)
    print(f"Added {added} readings to {log_path}"
          + (f" (replaced {replaced} older readings of this recording)" if replaced else "")
          + f"; {total} in total.")
    print(f"\nNow run:  python plot_pattern.py --cut {args.cut}")


if __name__ == "__main__":
    main()
