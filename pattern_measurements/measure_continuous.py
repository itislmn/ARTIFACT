"""
measure_continuous.py - one long capture while the target sweeps from one angle to the other: every frame gives
one reading per channel (about 10 per second), its angle interpolated between Enter marks or assigned later.

    python measure_continuous.py --cli COM4 --cfg config/pattern_awr2944P.cfg --cut azimuth --tag air --range-m 0.30 --background
"""
import argparse
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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEARCH_HALF_WIDTH_M = 0.35
MAX_SEQ_GAP = 2000


def make_continuous_cfgs(cfg_path, loops, out_dir):
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
            tok[3] = str(int(loops))
            tok[4] = "10000"
            line = " ".join(tok)
            found = True
        out.append(line)
    if not found:
        raise CaptureError("no frameCfg line in the .cfg")
    path = os.path.join(out_dir, "_continuous_nostart.cfg")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


class StreamReceiver(threading.Thread):

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


def spectra_from_bin(bin_path, p, n_frames=None):
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
    ra = p["range_axis"]
    Ys = Xs - B if B is not None else Xs
    lo, hi = range_window(ra, Xs.shape[2], center_m, half_m)
    energy = (np.abs(Ys[:, :, lo:hi]) ** 2).mean(axis=1)
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
    t_center = np.asarray(frame_times) - 0.5 * active_s
    return np.interp(t_center, marks, waypoints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--cut", required=True, choices=["azimuth", "elevation"])
    ap.add_argument("--tag", default="", help="label for this data set (e.g. air, pla, petg): every file gets it in its name, so different radomes/setups never mix or overwrite each other")
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
    name = args.cut + (f"_{args.tag}" if args.tag else "")
    tagopt = f" --tag {args.tag}" if args.tag else ""
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
    bg_path = os.path.join(OUT_DIR, f"{name}_background_continuous.npy")
    print(f"Frame = {p['numLoops']} loops x {p['numTx']} TX = {p['chirpsPerFrame']} chirps, "
          f"{frame_bytes/1e3:.0f} kB; active time {active_s*1e3:.1f} ms per frame.\n")

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
    if rx.frame_times[-1] < marks[-1] - 1.5:
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
        print("   removed; the un-subtracted gains are stored too (gain_raw_linear)")

    n = Xs.shape[0]
    t_rel = np.asarray(rx.frame_times[:n]) - rx.t_start - 0.5 * active_s
    ra = p["range_axis"]
    run_id = int(rx.t_start)

    raw_path = os.path.join(OUT_DIR, f"{name}_continuous_raw.npz")
    np.savez_compressed(raw_path, frame_time_rel=t_rel, gains=res["gains"], gains_raw=res["gains_raw"],
                        noise=res["noise"], bins=res["bins"], range_m=ra[res["bins"]],
                        numRx=p["numRx"], bg_subtracted=B is not None, run_id=run_id,
                        lock_range_m=res["lock_range_m"], active_s=active_s)
    print(f"Recording saved to {raw_path} (run id {run_id}).")
    run = {"run_id": run_id, "numRx": p["numRx"], "gains": res["gains"], "gains_raw": res["gains_raw"],
           "noise": res["noise"], "range_m": ra[res["bins"]], "bg_subtracted": B is not None}
    log_path = os.path.join(OUT_DIR, f"{name}_measurements.json")

    if args.mark_later:
        print(f"Recorded {t_rel[-1]:.1f} s. No angles assigned yet. Next:")
        print(f"  python assign_angles.py --cut {args.cut}{tagopt} --show      (level over time)")
        print(f"  python assign_angles.py --cut {args.cut}{tagopt} --marks \"0:-90, 15:0, 30:90\"   "
              "(your TIME:ANGLE marks)")
        return

    marks_rel = np.asarray(marks) - rx.t_start
    ang, inside = angles_from_marks(t_rel, marks_rel, waypoints)
    print(f"{inside.sum()} frames fall between the first and last mark "
          f"({inside.sum() / max(1e-9, waypoints[-1] - waypoints[0]):.1f} frames per degree); "
          f"{(~inside).sum()} more are held at the end angles.")
    added, replaced, total = write_log(run, ang, np.ones(n, bool), log_path)
    print(f"Appended {added} readings to {log_path} ({total} total).")
    print("\nNow run:  python run_all.py   (with the tag in dataset.TAGS)")


if __name__ == "__main__":
    main()
