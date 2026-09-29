"""
measure_interactive.py - press Enter to advance, everything else is automatic.

    python measure_interactive.py --cli COM4 --cfg pattern_awr2944P.cfg ^
        --cut azimuth --range-m 0.30 --start -90 --stop 90 --step 5 --background

WHAT IS DIFFERENT FROM THE OLD VERSION (all aimed at short range in a messy lab)
--------------------------------------------------------------------------------
1. The old version took the strongest range bin of the WHOLE profile at every
   angle. At wide angles the target is weak, so it silently picked TX-RX
   leakage / walls / furniture instead - a flat "shelf" of angle-independent
   values that looks like antenna response but is not. Now the peak search is
   restricted to a window around the target range, and after two good captures
   the window LOCKS to +/- --gate-m around the range the target really showed
   up at (the radar's measured range can differ from your tape measure).
2. --background: you remove the target once, the script records the empty room
   (coherently: complex, static part only), and subtracts it from every
   measurement. TX-RX leakage and all static clutter cancel. At 30 cm the
   leakage bin (~0.17 m) is only ~3 range bins from the target, so this matters.
3. Energy of 3 range bins (peak +/-1) instead of a single bin, so a target that
   sits between two bins does not lose up to ~1.4 dB (Hann scalloping).
4. Coherent average over the chirps of every frame: only STATIC things survive,
   so a person walking around during the capture averages out.
5. The boresight angle is measured FIRST (target strongest -> reliable range
   lock), then the sweep runs in normal order.
6. Each reading also stores range_m, the prominence, the noise floor and the
   un-subtracted gain. Nothing is thrown away; plot_pattern.py can draw either
   version (--use raw) and shows the noise floor as a line.
7. Safety check: if, after subtracting the empty-room reference, the target is
   (almost) gone at boresight, the script warns that the reference probably
   contained the target and may be erasing the real signal.

Extra angles can be added any time: run again with a different --start/--step
(e.g. --start -87.5 --stop 87.5 --step 5); readings are appended to the same log
and plot_pattern.py merges them.
"""
import argparse
import json
import os
import sys

import numpy as np

from capture_lib import preflight_check, capture_once, load_complex_cube, CaptureError
from pattern_lib import static_spectrum, measure_static, reference_looks_wrong

OUT_DIR = "output"
SEARCH_HALF_WIDTH_M = 0.35     # initial search window around --range-m
GOOD_PROMINENCE_DB = 10.0      # a capture this far above the noise floor counts as "found the target"


def target_position(angle_deg, range_m):
    x = range_m * np.sin(np.radians(angle_deg))
    y = range_m * np.cos(np.radians(angle_deg))
    return x, y


def sweep_order(angles):
    """Boresight-nearest angle first (best range lock), then the rest ascending."""
    i0 = int(np.argmin(np.abs(angles)))
    return [angles[i0]] + [a for i, a in enumerate(angles) if i != i0]


def process_capture(bin_path, p, center_m, half_width_m, B=None):
    cube = load_complex_cube(bin_path, p)          # (frames, loops, 16, samples)
    return measure_static(static_spectrum(cube), B, p["range_axis"], center_m, half_width_m)


# ----------------------------------------------------------------------
def do_capture(args, bin_path):
    """One capture with a single retry, same policy as before."""
    try:
        return capture_once(args.cli, args.cfg, args.seconds, bin_path)
    except CaptureError as e:
        print(f"\nCAPTURE FAILED: {e}")
        print("Fix the issue above, don't move anything, then press Enter to retry.")
        input("Press Enter to retry...")
        return capture_once(args.cli, args.cfg, args.seconds, bin_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--cut", required=True, choices=["azimuth", "elevation"])
    ap.add_argument("--range-m", type=float, required=True,
                    help="Nominal target distance (tape measure). The real peak may "
                         "sit a few cm to ~20 cm away from this; the search window allows for it.")
    ap.add_argument("--start", type=float, default=-60)
    ap.add_argument("--stop", type=float, default=60)
    ap.add_argument("--step", type=float, default=10)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--background", action="store_true",
                    help="Record the EMPTY room now (target removed) and subtract it "
                         "from every measurement. Strongly recommended at short range.")
    ap.add_argument("--reuse-background", action="store_true",
                    help="Load the empty-room recording saved by an earlier --background "
                         "run instead of recording a new one (room must be unchanged).")
    ap.add_argument("--gate-m", type=float, default=0.08,
                    help="After the range lock, only accept peaks within this many metres "
                         "of the locked range (default 0.08 ~ 2 range bins).")
    ap.add_argument("--lock-range-m", type=float, default=None,
                    help="Skip the auto-lock: use this MEASURED peak range (e.g. the "
                         "'peak at X m' printed by an earlier run).")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Checking DCA1000 is reachable before asking you to move anything...")
    try:
        preflight_check()
    except CaptureError as e:
        print(f"\nSTOP - preflight check failed:\n{e}")
        sys.exit(1)
    print("DCA1000 is alive.\n")

    if args.cut == "elevation" and (args.stop - args.start) > 20 and args.step >= 3:
        print("NOTE: per TI's AWR2944(P)EVM antenna spec, the onboard array's")
        print("3 dB beamwidth is roughly +/-30 deg in AZIMUTH but only about")
        print("+/-3 deg in ELEVATION. A coarse step over a wide span mostly samples")
        print("noise with one narrow spike near 0 deg - that is physics, not a bug.")
        print("For a usable elevation cut try: --start -10 --stop 10 --step 1\n")

    bin_path = os.path.join(OUT_DIR, "_last_capture.bin")
    bg_path = os.path.join(OUT_DIR, f"{args.cut}_background.npy")

    # ---------------- empty-room background ----------------
    B = None
    if args.background:
        print("=" * 60)
        print("BACKGROUND: remove the target / reflector from the scene. Leave EVERYTHING")
        print("else exactly as it will be during the sweep (radar, table, cables, walls),")
        print("and stand where you will stand while measuring - or out of the way.")
        input("Press Enter when the scene is EMPTY of the target...")
        _, p = do_capture(args, bin_path)
        B = static_spectrum(load_complex_cube(bin_path, p))
        np.save(bg_path, B)
        print(f"  -> empty-room reference saved to {bg_path}\n")
    elif args.reuse_background:
        if not os.path.exists(bg_path):
            print(f"No saved background at {bg_path} - run once with --background first.")
            sys.exit(1)
        B = np.load(bg_path)
        print(f"Loaded empty-room reference from {bg_path}\n")
    elif args.range_m < 0.6:
        print("WARNING: target range < 0.6 m and no --background. The TX-RX leakage bin")
        print("(~0.17 m) is only a few range bins away and can dominate the target's bin,")
        print("especially at wide angles. Re-run with --background for clean results.\n")

    angles = np.arange(args.start, args.stop + 1e-9, args.step)
    log_path = os.path.join(OUT_DIR, f"{args.cut}_measurements.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else []

    print(f"Pivot point: mark a spot exactly {args.range_m} m in front of the radar's")
    print("antenna face, centered on boresight (0 deg). Every position below is an offset")
    print("from THAT spot. The boresight-nearest angle is measured first (range lock),")
    print("then the sweep continues in order.\n")

    lock_center = args.lock_range_m
    good_ranges = []
    checked_reference = False

    for angle in sweep_order(angles):
        x, y = target_position(angle, args.range_m)
        print("=" * 60)
        print(f"ANGLE {angle:+.1f} deg  ->  place target at x={x:+.3f} m, y={y:.3f} m from the pivot")
        input("Press Enter once the target is in position (and you are out of the beam)...")

        try:
            n_bytes, p = do_capture(args, bin_path)
        except CaptureError as e2:
            print(f"Failed again: {e2}\nSkipping this angle - re-run it later with --start/--stop.")
            continue

        if B is not None and B.shape[0] != p["numTx"] * p["numRx"]:
            print("Background channel count does not match this cfg - re-record with --background.")
            sys.exit(1)

        if lock_center is None:
            center, half = args.range_m, SEARCH_HALF_WIDTH_M
        else:
            center, half = lock_center, args.gate_m

        try:
            m = process_capture(bin_path, p, center, half, B)
        except CaptureError as e:
            print(f"Processing failed: {e}. Skipping this angle.")
            continue
        gains, gains_raw = m["gains"], m["gains_raw"]
        actual_range, prom = m["range_m"], m["prominence_db"]

        note = ""
        if lock_center is None and prom >= GOOD_PROMINENCE_DB:
            good_ranges.append(actual_range)
            if len(good_ranges) >= 2:
                lock_center = float(np.median(good_ranges))
                note = f"   [range LOCKED at {lock_center:.2f} m +/-{args.gate_m:.2f}]"
        clean_db = 20 * np.log10(np.median(gains) + 1e-12)
        raw_db = 20 * np.log10(np.median(gains_raw) + 1e-12)
        floor_db = 20 * np.log10(np.median(m["noise"]) + 1e-12)
        print(f"  -> {n_bytes} bytes, peak at {actual_range:.2f} m, {prom:.0f} dB above the noise "
              f"floor; level {clean_db:.1f} dB, noise floor {floor_db:.1f} dB (absolute units)"
              + (f"; without background removal {raw_db:.1f} dB" if B is not None else "")
              + note)
        if B is not None and not checked_reference and abs(angle) <= 5:
            checked_reference = True
            if reference_looks_wrong(prom):
                print("  !! WARNING: after subtracting the empty-room reference almost nothing is left")
                print("     at the target range, although the target should be strongest right here.")
                print("     The reference probably already contained the target (or a return at that")
                print("     range), so the subtraction may be erasing the REAL signal. Re-record it")
                print("     with the target truly removed - or plot the un-subtracted numbers, which")
                print("     are stored too:  python plot_pattern.py --cut " + args.cut + " --use raw")

        for ch in range(gains.shape[0]):
            tx, rx = ch // p["numRx"], ch % p["numRx"]
            log.append({"angle": float(angle), "tx": int(tx), "rx": int(rx),
                        "gain_linear": float(gains[ch]),
                        "gain_raw_linear": float(gains_raw[ch]),
                        "noise_linear": float(m["noise"][ch]),
                        "range_m": float(actual_range),
                        "prominence_db": float(prom),
                        "bg_subtracted": B is not None})
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  -> logged. Total readings for this cut so far: {len(log)}")

    print("\nSweep complete. Now run:")
    print(f"  python plot_pattern.py --cut {args.cut}")


if __name__ == "__main__":
    main()
