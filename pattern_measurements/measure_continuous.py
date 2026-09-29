"""
measure_continuous.py - ONE long capture while you sweep smoothly from one angle
to the other. Every radar frame becomes a data point (~10 per second), which is
what gives the dense, continuous curves of a reference pattern plot - instead of
one point per minute of stopping, moving and re-capturing.

    python measure_continuous.py --cli COM4 --cfg pattern_awr2944P.cfg ^
        --cut azimuth --range-m 0.30 --start -90 --stop 90 --mark-step 30 --background

    python plot_pattern.py --cut azimuth

CAN'T PRESS ENTER WHILE MOVING THE TARGET?  Use --mark-later
------------------------------------------------------------
    python measure_continuous.py --cli COM4 --cfg pattern_awr2944P.cfg ^
        --cut azimuth --range-m 0.30 --mark-later --background
Both hands stay free: you only press Enter to start and once to stop. A clock shows the
seconds since GO; jot down (or film) the time at which the target passes landmark angles.
Afterwards, no hardware needed:
    python assign_angles.py --cut azimuth --show
    python assign_angles.py --cut azimuth --marks "0:-90, 6:-60, 15:0, 24:60, 30:90"
See assign_angles.py for details (--time-shift, --hold-ends, re-running with other marks).

HOW THE ANGLE OF EACH FRAME IS KNOWN (Enter-marks mode)
-------------------------------------------------------
The radar does not know the angle - you do. You give it "marks": after the
recording starts you press Enter at each waypoint (-90, -60, ..., +90) at the
instant the target passes that angle. Every frame has an arrival timestamp; its
angle is linearly interpolated between the two marks around it. Frames before
the first mark / after the last one are held at the end angles. Marking every
30 deg keeps the error to about a degree even if your speed is not constant.
Practical tips:
  * Easiest and most accurate: leave the TARGET fixed and ROTATE THE RADAR about
    its own vertical axis (range and geometry stay perfect; the angle you mark
    is the angle between the radar's boresight and the target).
  * Move at a steady, slow pace (e.g. 180 deg in 40-60 s).
  * Mark as the target passes each angle, not a moment later.

WHAT THIS CHANGES IN THE RADAR CONFIG (a temporary copy, your .cfg is untouched)
-----------------------------------------------------------------------------
  frameCfg: numFrames -> 10000 (runs until this script sends sensorStop) and
  numLoops -> --loops (default 32, half of 64: keeps the data rate near 7 MB/s
  so no UDP packets are lost; you still get 32 chirps x 4 TX of coherent gain
  per frame). Everything else (chirps, profile, TDM order) stays exactly as is.

Results: the raw frame-level table (no angles) is always saved to
output/<cut>_continuous_raw.npz; every frame x channel is appended to
output/<cut>_measurements.json (same format as measure_interactive.py, so both can be
mixed) - immediately in Enter-marks mode, via assign_angles.py with --mark-later.
Nothing is filtered.
"""
import argparse
import json
import os
import socket
import struct
import sys
import threading
import time

import numpy as np

import capture_lib as cl
from capture_lib import (preflight_check, arm_dca1000, stop_dca1000, force_sensor_stop,
                         send_sensor_config, parse_cfg, CaptureError, _read_reply)
from pattern_lib import (frame_cube, frame_spectrum, range_window, find_target_bin, bin_energy,
                         noise_floor, reference_looks_wrong)
from assign_angles import angles_from_marks, write_log

OUT_DIR = "output"
SEARCH_HALF_WIDTH_M = 0.35
MAX_SEQ_GAP = 2000          # larger jumps are treated as "field not interpretable", never zero-filled


# ----------------------------------------------------------------------
# config handling
# ----------------------------------------------------------------------
def make_continuous_cfgs(cfg_path, loops, out_dir):
    """Copy of the .cfg with frameCfg's numLoops/numFrames changed and the final
    sensorStart removed (this script sends sensorStart itself so it knows the
    exact moment). Returns the path of the copy."""
    out, found = [], False
    for raw in open(cfg_path):
        line = raw.strip()
        tok = line.split()
        if not tok or line.startswith(("%", "#")):
            continue
        if tok[0] == "sensorStart":
            continue
        if tok[0] == "frameCfg":
            if len(tok) < 5:
                raise CaptureError(f"unexpected frameCfg line: '{line}'")
            tok[3] = str(int(loops))       # numLoops   (parse_cfg: v[2])
            tok[4] = "10000"               # numFrames  (parse_cfg: v[3]); stopped by sensorStop
            line = " ".join(tok)
            found = True
        out.append(line)
    if not found:
        raise CaptureError("no frameCfg line in the .cfg")
    path = os.path.join(out_dir, "_continuous_nostart.cfg")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


# ----------------------------------------------------------------------
# UDP stream receiver: writes the payload to a file and time-stamps frames
# ----------------------------------------------------------------------
class StreamReceiver(threading.Thread):
    """DCA1000 data packet = 10-byte header (4-byte sequence number, 6-byte byte
    count) + payload. Payload is appended to `out_path`. Because every radar frame
    has a fixed size, the arrival time of the packet that completes frame k IS
    that frame's timestamp - no need to know the radar's frame period.

    Packet loss would shift every later frame; the sequence numbers are used to
    zero-fill small gaps so alignment survives (and the loss is reported). The
    sequence field is only trusted if the jump is 1..MAX_SEQ_GAP."""

    def __init__(self, out_path, frame_bytes, bind_ip=None, port=None):
        super().__init__(daemon=True)
        self.out_path, self.frame_bytes = out_path, frame_bytes
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
        except OSError:
            pass
        self.sock.settimeout(0.3)
        try:
            self.sock.bind((bind_ip or cl.PC_IP, port or cl.PC_DATA_PORT))
        except OSError as e:
            raise CaptureError(f"Cannot bind data port {bind_ip or cl.PC_IP}:{port or cl.PC_DATA_PORT} -> {e}")
        self._stop_evt = threading.Event()
        self.total = 0
        self.n_packets = 0
        self.lost_packets = 0
        self.seq_suspect = 0
        self.frame_times = []
        self._prev_seq = None

    def stop(self):
        self._stop_evt.set()

    def _advance(self, now):
        while self.total >= (len(self.frame_times) + 1) * self.frame_bytes:
            self.frame_times.append(now)

    def run(self):
        try:
            with open(self.out_path, "wb") as f:
                while True:
                    try:
                        pkt, _ = self.sock.recvfrom(65536)
                    except socket.timeout:
                        if self._stop_evt.is_set():
                            break
                        continue
                    now = time.time()
                    if len(pkt) <= 10:
                        continue
                    self.n_packets += 1
                    payload = pkt[10:]
                    seq = struct.unpack("<I", pkt[:4])[0]
                    if self._prev_seq is not None:
                        gap = seq - self._prev_seq - 1
                        if 0 < gap <= MAX_SEQ_GAP:
                            f.write(b"\x00" * (gap * len(payload)))
                            self.total += gap * len(payload)
                            self.lost_packets += gap
                            self._advance(now)
                        elif gap != 0:
                            self.seq_suspect += 1
                    self._prev_seq = seq
                    f.write(payload)
                    self.total += len(payload)
                    self._advance(now)
        finally:
            self.sock.close()


def record(cli, cfg_nostart, out_bin, frame_bytes, during, dca_timer=255):
    """arm DCA1000 -> start receiving -> send config -> sensorStart (timestamped)
    -> run `during(t_sensor_start)` (your interaction) -> sensorStop -> clean up.
    Returns (during's return value, the StreamReceiver with frame_times etc.)."""
    import serial
    force_sensor_stop(cli)
    dca_sock = arm_dca1000(4, timer=dca_timer)
    rx = None
    try:
        rx = StreamReceiver(out_bin, frame_bytes)
        rx.start()
        time.sleep(0.5)
        send_sensor_config(cli, cfg_nostart, strict=True)
        with serial.Serial(cli, 115200, timeout=0.05) as ser:
            ser.write(b"sensorStart\n")
            t_start = time.time()
            rx.t_start = t_start
            if "Done" not in _read_reply(ser):
                raise CaptureError("sensorStart was not acknowledged with 'Done'")
            try:
                result = during(t_start)
            finally:
                ser.write(b"sensorStop\n")
                time.sleep(0.3)
                _read_reply(ser)
        time.sleep(0.3)
    finally:
        if rx is not None:
            rx.stop()
            rx.join(timeout=10)
        stop_dca1000(dca_sock)
        force_sensor_stop(cli)
    return result, rx


# ----------------------------------------------------------------------
# processing (pure numpy)
# ----------------------------------------------------------------------
def spectra_from_bin(bin_path, p, n_frames=None):
    """Range spectrum of every frame -> (frames, channels, n_half) complex64.
    Frame by frame from a memory map, so long captures do not need much RAM."""
    per = p["numAdcSamples"] * p["numRx"] * p["chirpsPerFrame"]
    raw = np.memmap(bin_path, dtype=np.int16, mode="r")
    n = raw.size // per if n_frames is None else min(n_frames, raw.size // per)
    if n == 0:
        raise CaptureError("no complete frame in the recording")
    out = np.empty((n, p["numTx"] * p["numRx"], p["numAdcSamples"] // 2), dtype=np.complex64)
    for f in range(n):
        out[f] = frame_spectrum(frame_cube(np.asarray(raw[f * per:(f + 1) * per]), p))
    return out


def analyse_frames(Xs, B, p, center_m, half_m):
    """Xs: (frames, ch, n_half) raw spectra, B: reference or None.
    The target is physically at ONE range for the whole sweep, so the range bin is
    found once (from the strongest 10 % of the frames), then every frame is read at
    that bin (+/-1 for small drift) - a weak wide-angle frame can never jump to a
    clutter peak elsewhere. Returns dict of per-frame arrays."""
    ra = p["range_axis"]
    Ys = Xs - B if B is not None else Xs
    lo, hi = range_window(ra, Xs.shape[2], center_m, half_m)
    energy = (np.abs(Ys[:, :, lo:hi]) ** 2).mean(axis=1)                  # (frames, bins)
    strong = energy.max(axis=1) >= np.quantile(energy.max(axis=1), 0.9)
    lock_bin = lo + int(np.argmax(energy[strong].mean(axis=0)))
    best = int(np.argmax(energy.max(axis=1)))
    _, prom_best = find_target_bin(Ys[best], ra, ra[lock_bin], 0.09)

    n = Xs.shape[0]
    gains = np.empty((n, Xs.shape[1]))
    raw_g = np.empty_like(gains)
    noise = np.empty_like(gains)
    bins = np.empty(n, int)
    for f in range(n):
        prof = np.sqrt((np.abs(Ys[f, :, lock_bin - 1:lock_bin + 2]) ** 2).mean(axis=0))
        b = lock_bin - 1 + int(np.argmax(prof))
        bins[f] = b
        gains[f] = bin_energy(Ys[f], b)
        raw_g[f] = bin_energy(Xs[f], b)
        noise[f] = noise_floor(Ys[f])
    return {"gains": gains, "gains_raw": raw_g, "noise": noise, "bins": bins,
            "lock_bin": lock_bin, "lock_range_m": float(ra[lock_bin]),
            "prominence_best_db": prom_best}


def frame_angles(frame_times, marks, waypoints, active_s):
    """Angle of every frame: linear interpolation between the marks; frames outside
    the marked interval are held at the first/last waypoint."""
    t_center = np.asarray(frame_times) - 0.5 * active_s
    return np.interp(t_center, marks, waypoints)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--cut", required=True, choices=["azimuth", "elevation"])
    ap.add_argument("--range-m", type=float, required=True,
                    help="Nominal target distance (tape measure); the search window allows +/-35 cm.")
    ap.add_argument("--start", type=float, default=-90)
    ap.add_argument("--stop", type=float, default=90)
    ap.add_argument("--mark-step", type=float, default=30,
                    help="Waypoint spacing for the Enter-marks (default every 30 deg).")
    ap.add_argument("--waypoints", type=str, default=None,
                    help="Explicit comma-separated marks, e.g. -90,-45,0,45,90 (overrides start/stop/step).")
    ap.add_argument("--mark-later", action="store_true",
                    help="Do NOT press Enter at waypoints. Just sweep (both hands free); press Enter "
                         "once to stop. Afterwards enter your marks (time:angle) with assign_angles.py.")
    ap.add_argument("--loops", type=int, default=32)
    ap.add_argument("--background", action="store_true",
                    help="Record the EMPTY scene first (target removed) and subtract it.")
    ap.add_argument("--reuse-background", action="store_true")
    ap.add_argument("--background-seconds", type=float, default=4.0)
    ap.add_argument("--dca-timer", type=int, default=255,
                    help="Value of the timer byte in the DCA1000 CONFIG_FPGA_GEN command "
                         "(the older scripts send 30; 255 avoids a possible cut-off on long sweeps).")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.waypoints:
        waypoints = [float(x) for x in args.waypoints.split(",")]
    else:
        waypoints = list(np.arange(args.start, args.stop + 1e-9, args.mark_step))
        if abs(waypoints[-1] - args.stop) > 1e-6:
            waypoints.append(args.stop)
    if not args.mark_later and (len(waypoints) < 2 or any(b <= a for a, b in zip(waypoints, waypoints[1:]))):
        print("Waypoints must be at least 2 and strictly increasing.")
        sys.exit(1)

    print("Checking DCA1000 is reachable before asking you to move anything...")
    try:
        preflight_check()
    except CaptureError as e:
        print(f"\nSTOP - preflight check failed:\n{e}")
        sys.exit(1)
    print("DCA1000 is alive.\n")

    cfg2 = make_continuous_cfgs(args.cfg, args.loops, OUT_DIR)
    p = parse_cfg(cfg2)
    if p["isComplex"]:
        print("This script needs real sampling (adcCfg 2 0).")
        sys.exit(1)
    frame_bytes = 2 * p["numAdcSamples"] * p["numRx"] * p["chirpsPerFrame"]
    active_s = p["chirpsPerFrame"] * (p["idleTime_us"] + p["rampEndTime_us"]) * 1e-6
    bin_path = os.path.join(OUT_DIR, "_continuous_capture.bin")
    bg_path = os.path.join(OUT_DIR, f"{args.cut}_background_continuous.npy")
    print(f"Frame = {p['numLoops']} loops x {p['numTx']} TX = {p['chirpsPerFrame']} chirps, "
          f"{frame_bytes/1e3:.0f} kB; active time {active_s*1e3:.1f} ms per frame.\n")

    # ---------------- background ----------------
    B = None
    if args.background:
        print("=" * 60)
        print("BACKGROUND: REMOVE the target/reflector (and its holder if it is not part of what")
        print("you sweep). Leave the radar, table, cables and walls exactly as during the sweep.")
        input("Press Enter when the scene is EMPTY of the target...")
        _, rx = record(args.cli, cfg2, bin_path, frame_bytes,
                       lambda t0: time.sleep(args.background_seconds), args.dca_timer)
        Xb = spectra_from_bin(bin_path, p)
        B = Xb.mean(axis=0)
        np.save(bg_path, B)
        print(f"  -> empty-scene reference from {Xb.shape[0]} frames saved to {bg_path}\n")
    elif args.reuse_background:
        if not os.path.exists(bg_path):
            print(f"No saved background at {bg_path}; run once with --background first.")
            sys.exit(1)
        B = np.load(bg_path)
        print(f"Loaded empty-scene reference from {bg_path}\n")
    elif args.range_m < 0.6:
        print("WARNING: range < 0.6 m and no --background: TX-RX leakage (~0.17 m) is only a few")
        print("range bins from the target. Consider --background.\n")

    # ---------------- the sweep ----------------
    print("=" * 60)
    if args.mark_later:
        print(f"SWEEP (marks entered afterwards): put the target at {args.range_m} m from the radar at your")
        print("start angle. Sweep steadily to the end angle. Note the ON-SCREEN CLOCK time (or use a")
        print("stopwatch started at GO) when you pass landmark angles - you enter them afterwards.")
    else:
        print(f"SWEEP: waypoints {', '.join(f'{w:+g}' for w in waypoints)} deg.")
        print(f"Put the target at {args.range_m} m from the radar, at the FIRST waypoint ({waypoints[0]:+g} deg).")
    input("Press Enter to start recording (the sensor config takes several seconds; then you will see GO)...")

    def interaction_later(t_start):
        print("\n>>> GO - RECORDING (clock = seconds since GO). Sweep now. Press Enter ONCE when you are done.\n")
        stop = threading.Event()

        def clock():
            while not stop.is_set():
                print(f"\r    elapsed {time.time() - t_start:6.1f} s   ", end="", flush=True)
                time.sleep(0.1)
        th = threading.Thread(target=clock, daemon=True)
        th.start()
        try:
            input()
        finally:
            stop.set()
            th.join()
            print()
        time.sleep(0.5)
        return [time.time()]

    def interaction(t_start):
        marks = []
        print("\n>>> RECORDING. Stay at the first waypoint, press Enter, then sweep steadily.")
        print("    Press Enter at EACH waypoint the moment the target passes it. After the last")
        print("    Enter, hold still for a second.\n")
        for w in waypoints:
            input(f"    Enter when at {w:+g} deg ...")
            t = time.time()
            if marks and t <= marks[-1]:
                t = marks[-1] + 1e-3
            marks.append(t)
        time.sleep(1.0)
        return marks

    try:
        marks, rx = record(args.cli, cfg2, bin_path, frame_bytes,
                           interaction_later if args.mark_later else interaction, args.dca_timer)
    except CaptureError as e:
        print(f"\nCAPTURE FAILED: {e}")
        sys.exit(1)

    n_frames = len(rx.frame_times)
    print(f"\nReceived {rx.n_packets} packets, {rx.total/1e6:.1f} MB, {n_frames} complete frames.")
    if rx.lost_packets:
        print(f"WARNING: {rx.lost_packets} UDP packets were lost (zero-filled to keep alignment; the "
              "affected frames are noisy). Reduce --loops or close other network programs.")
    if rx.seq_suspect:
        print(f"note: {rx.seq_suspect} packets had a sequence number that could not be interpreted; ignored.")
    if n_frames < 10:
        print("Far too few frames - check the UART log above.")
        sys.exit(1)
    if rx.frame_times[-1] < marks[-1] - 1.5:   # in --mark-later mode marks[-1] is the moment you pressed stop
        print("WARNING: the data stream stopped BEFORE your last mark. The frames after that "
              "point are missing (DCA1000 timer / packet problem?) - the last angles are not measured.")

    Xs = spectra_from_bin(bin_path, p, n_frames)
    if B is not None and B.shape != Xs.shape[1:]:
        print("Background shape does not match this recording; re-record with --background.")
        sys.exit(1)
    res = analyse_frames(Xs, B, p, args.range_m, SEARCH_HALF_WIDTH_M)
    print(f"Target locked at {res['lock_range_m']:.2f} m; strongest frame is "
          f"{res['prominence_best_db']:.0f} dB above the noise floor.")
    if B is not None and reference_looks_wrong(res["prominence_best_db"]):
        print("!! WARNING: after subtracting the empty-scene reference almost nothing is left at the")
        print("   target range. The reference probably already contained the target, so the")
        print("   subtraction may have erased the REAL signal. Re-record it with the target truly")
        print(f"   removed, or plot the un-subtracted numbers:  python plot_pattern.py --cut {args.cut} --use raw")

    # Frame time relative to GO (sensorStart), referred to the CENTRE of the frame's active time.
    n = Xs.shape[0]
    t_rel = np.asarray(rx.frame_times[:n]) - rx.t_start - 0.5 * active_s
    ra = p["range_axis"]
    run_id = int(rx.t_start)

    # The raw recording (everything measured, no angles yet) is ALWAYS saved: with it the angles
    # can be assigned - or changed - at any time with assign_angles.py.
    raw_path = os.path.join(OUT_DIR, f"{args.cut}_continuous_raw.npz")
    np.savez_compressed(raw_path, frame_time_rel=t_rel, gains=res["gains"], gains_raw=res["gains_raw"],
                        noise=res["noise"], bins=res["bins"], range_m=ra[res["bins"]],
                        numRx=p["numRx"], bg_subtracted=B is not None, run_id=run_id,
                        lock_range_m=res["lock_range_m"], active_s=active_s)
    print(f"Recording saved to {raw_path} (run id {run_id}).")
    run = {"run_id": run_id, "numRx": p["numRx"], "gains": res["gains"], "gains_raw": res["gains_raw"],
           "noise": res["noise"], "range_m": ra[res["bins"]], "bg_subtracted": B is not None}
    log_path = os.path.join(OUT_DIR, f"{args.cut}_measurements.json")

    if args.mark_later:
        print(f"Recorded {t_rel[-1]:.1f} s. No angles assigned yet. Next:")
        print(f"  python assign_angles.py --cut {args.cut} --show      (level over time)")
        print(f"  python assign_angles.py --cut {args.cut} --marks \"0:-90, 15:0, 30:90\"   "
              "(your TIME:ANGLE marks)")
        return

    marks_rel = np.asarray(marks) - rx.t_start
    ang, inside = angles_from_marks(t_rel, marks_rel, waypoints)
    print(f"{inside.sum()} frames fall between the first and last mark "
          f"({inside.sum() / max(1e-9, waypoints[-1] - waypoints[0]):.1f} frames per degree); "
          f"{(~inside).sum()} more are held at the end angles.")
    added, replaced, total = write_log(run, ang, np.ones(n, bool), log_path)
    print(f"Appended {added} readings to {log_path} ({total} total).")
    print(f"\nNow run:  python plot_pattern.py --cut {args.cut}")


if __name__ == "__main__":
    main()
