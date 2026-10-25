r"""
inspect_recording.py - lists the strongest reflectors of a recording; calibrates the virtual channels with a corner
reflector on boresight (--save-channel-cal) and sets mirrored angle or velocity signs (--set-signs).

    python inspect_recording.py data/raw/setup/corner_boresight.npy --save-channel-cal
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from material_classification.utils.micro_doppler import (CHANNEL_CAL_PATH, CONVENTIONS_PATH, FramePreprocessor,  # noqa: E402
                                 RadarParams, direction, load_channel_cal, pulse_pair)
from material_classification.utils.recording import last_stream_start, load_frame_times  # noqa: E402


def local_maxima(db):
    p = np.pad(db, 1, constant_values=-np.inf)
    c = p[1:-1, 1:-1]
    ok = np.ones_like(c, bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                ok &= c >= p[1 + di:p.shape[0] - 1 + di, 1 + dj:p.shape[1] - 1 + dj]
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", nargs="?")
    ap.add_argument("--cfg", default=str(HERE / "config" / "awr2944_material_spec.cfg"))
    ap.add_argument("--max-range", type=float, default=5.0)
    ap.add_argument("--min-range", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=5, help="number of reflectors to list")
    ap.add_argument("--save-channel-cal", action="store_true")
    ap.add_argument("--cal-range", type=float, default=3.0,
                    help="calibrate on the strongest reflector closer than this (m)")
    ap.add_argument("--set-signs", default=None, help="e.g. az=-1,el=1,doppler=1")
    args = ap.parse_args()

    if args.set_signs:
        conv = {"az_sign": 1, "el_sign": 1, "doppler_sign": 1}
        if CONVENTIONS_PATH.exists():
            conv.update(json.load(open(CONVENTIONS_PATH)))
        for part in args.set_signs.split(","):
            k, v = part.split("=")
            conv[f"{k.strip()}_sign"] = int(v)
        json.dump(conv, open(CONVENTIONS_PATH, "w"), indent=2)
        print(f"wrote {CONVENTIONS_PATH}: {conv}")
        if not args.recording:
            return
    if not args.recording:
        ap.error("give a recording")

    params = RadarParams.from_cfg(args.cfg)
    stack = np.load(args.recording, mmap_mode="r")
    if tuple(stack.shape[1:]) != (params.n_samples, params.n_chirps, params.n_virt):
        sys.exit(f"shape {stack.shape[1:]} does not match the cfg")
    frame_t = load_frame_times(args.recording)
    if frame_t is None:
        frame_t = np.arange(len(stack)) * params.frame_period_s
    k = last_stream_start(frame_t)
    if k:
        print(f"skipping the first {k} frame(s): they arrived before a pause (leftovers of an earlier run)")
    stack, frame_t = stack[k:], frame_t[k:]
    cal = None if args.save_channel_cal else load_channel_cal()
    pre = FramePreprocessor(params, max_range_m=args.max_range, channel_cal=cal)
    geo = pre.geo
    prods = [pre.process(c, t) for c, t in zip(stack, frame_t)]
    R = np.stack([p.rfft for p in prods])
    RA = np.mean([p.ra for p in prods], axis=0)
    rng = params.range_axis[:RA.shape[0]]
    noise = np.median(RA[:, geo.uy_valid])
    print(f"{args.recording}: {len(stack)} frames, {frame_t[-1] - frame_t[0]:.1f} s | "
          f"channel cal: {'applied' if cal is not None else 'none'} | {params.describe()}")

    db = 10 * np.log10(RA / noise + 1e-12)
    mask = (rng >= args.min_range)[:, None] & geo.uy_valid[None, :] & local_maxima(db)
    print(f"\n{'#':>2} {'range m':>8} {'az deg':>7} {'el deg':>7} {'SNR dB':>7} {'v meas m/s':>11}")
    first = None
    for i in range(args.n):
        s = np.where(mask, db, -np.inf)
        b, a = np.unravel_index(np.argmax(s), s.shape)
        if not np.isfinite(s[b, a]):
            break
        mask[max(b - 2, 0):b + 3, max(a - 3, 0):a + 4] = False
        snaps = R[:, b].reshape(-1, params.n_virt)
        w = np.linalg.eigh(snaps.conj().T @ snaps)[1][:, -1]
        v, _ = pulse_pair(R[:, b] @ np.conj(w), params)
        X = np.mean(R[:, b], axis=(0, 1))
        uz = float(np.nan_to_num(geo.uz_from(X)))
        u = direction(geo.uy_grid[a], uz)
        az, el = np.degrees(np.arctan2(u[1], u[0])), np.degrees(np.arcsin(u[2]))
        print(f"{i + 1:2d} {rng[b]:8.2f} {az:7.1f} {el:7.1f} {s[b, a]:7.1f} {v:11.3f}")
        if first is None and rng[b] <= args.cal_range:
            first = (b, X)

    if args.save_channel_cal and first is not None:
        b, X = first
        c = X / X[0]
        c = c / np.mean(np.abs(c))
        np.save(CHANNEL_CAL_PATH, c.astype(np.complex64))
        print(f"\nchannel calibration from the reflector at {rng[b]:.2f} m -> {CHANNEL_CAL_PATH}")
        print("phase (deg):", np.round(np.degrees(np.angle(c)), 1))


if __name__ == "__main__":
    main()
