"""
capture.py - records one take with the AWR2944 and the DCA1000 into data/raw/<session>/ (raw .bin and ground-truth
.json); --update-truth recomputes the truth of recorded takes.

    python capture.py --cli COM4 --session S2 --plan config/session_plan.json --name single_m30 --target-xy "1.732:-1.000:small CR 1"
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from estimation.core.radar import RadarConfig, detect_padding
from estimation.core.truth import complete, heights

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
DCA_IP, PC_IP = "192.168.33.180", "192.168.33.30"
CONFIG_PORT, DATA_PORT = 4096, 4098
RESET_FPGA, CONFIG_FPGA_GEN, RECORD_START, RECORD_STOP = 0x01, 0x03, 0x05, 0x06
SYSTEM_CONNECT, CONFIG_PACKET_DATA = 0x09, 0x0B
HEADER_BYTES = 10


class CaptureError(RuntimeError):
    pass


def log(msg):
    print(f"  [capture] {msg}", flush=True)


def dca_frame(cmd: int, data: bytes = b"") -> bytes:
    return struct.pack("<HHH", 0xA55A, cmd, len(data)) + data + struct.pack("<H", 0xEEAA)


class DCA1000:
    def __init__(self, timeout_s: float = 3.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(timeout_s)
        try:
            self.sock.bind((PC_IP, CONFIG_PORT))
        except OSError as e:
            raise CaptureError(f"cannot bind {PC_IP}:{CONFIG_PORT} ({e}): is the Ethernet adapter set to the "
                               "static IP 192.168.33.30, and no other capture tool running?")

    def command(self, name: str, code: int, data: bytes = b""):
        self.sock.sendto(dca_frame(code, data), (DCA_IP, CONFIG_PORT))
        try:
            resp, _ = self.sock.recvfrom(2048)
        except socket.timeout:
            raise CaptureError(f"{name}: no reply from the DCA1000 (5 V power? Ethernet cable? "
                               "SW_CONFIG mode? press its reset button)")
        status = struct.unpack("<H", resp[4:6])[0] if len(resp) >= 8 else -1
        log(f"{name:20s} -> {'OK' if status == 0 else f'ERROR status={status}'}  ({resp.hex()})")
        if status != 0:
            raise CaptureError(f"{name} returned status {status}")

    def start(self, lvds_lanes: int = 4, packet_delay_us: int = 25):
        self.command("SYSTEM_CONNECT", SYSTEM_CONNECT)
        self.command("RESET_FPGA", RESET_FPGA)
        time.sleep(0.5)
        lanes = 1 if lvds_lanes == 4 else 2
        self.command("CONFIG_FPGA_GEN", CONFIG_FPGA_GEN, bytes([1, lanes, 1, 2, 3, 30]))
        self.command("CONFIG_PACKET_DATA", CONFIG_PACKET_DATA, struct.pack("<HHH", 1466, packet_delay_us, 0))
        self.command("RECORD_START", RECORD_START)

    def stop(self):
        try:
            self.sock.sendto(dca_frame(RECORD_STOP), (DCA_IP, CONFIG_PORT))
            self.sock.settimeout(1.0)
            try:
                self.sock.recvfrom(2048)
            except socket.timeout:
                pass
        finally:
            self.sock.close()


def _read_reply(ser, quiet_s: float = 0.15, max_s: float = 2.0) -> str:
    buf, t0 = b"", time.time()
    last = t0
    while time.time() - t0 < max_s:
        chunk = ser.read(256)
        if chunk:
            buf, last = buf + chunk, time.time()
        elif time.time() - last > quiet_s:
            break
    return buf.decode(errors="ignore")


def sensor_stop(port: str, baud: int = 115200):
    import serial
    try:
        with serial.Serial(port, baud, timeout=0.05) as ser:
            ser.write(b"\nsensorStop\n")
            _read_reply(ser)
    except serial.SerialException as e:
        log(f"could not open {port} for sensorStop: {e}")


def send_cfg(port: str, cfg_path: Path, baud: int = 115200):
    import serial
    with serial.Serial(port, baud, timeout=0.05) as ser:
        ser.write(b"\n")
        time.sleep(0.2)
        ser.reset_input_buffer()
        for raw in open(cfg_path, encoding="utf-8", errors="ignore"):
            line = raw.strip()
            if not line or line.startswith(("%", "#")):
                continue
            ser.write((line + "\n").encode())
            reply = _read_reply(ser)
            ok = "Done" in reply
            log(f"UART {line[:44]:44s} -> {'Done' if ok else (reply.strip()[:60] or 'NO REPLY')}")
            if not ok:
                raise CaptureError(f"the radar rejected '{line}' - fix the .cfg and retry")


class FrameAssembler:

    def __init__(self, frame_bytes: int):
        self.fb = frame_bytes
        self.buf = bytearray(frame_bytes)
        self.cur, self.filled = None, 0
        self.complete = self.incomplete = 0

    def push(self, byte_count: int, payload) -> list:
        out = []
        pos, end, src = byte_count, byte_count + len(payload), 0
        while pos < end:
            fidx, off = divmod(pos, self.fb)
            n = min(end - pos, self.fb - off)
            if self.cur is None or fidx > self.cur:
                if self.cur is not None and self.filled:
                    self.incomplete += 1
                self.cur, self.filled = fidx, 0
            if fidx == self.cur:
                self.buf[off:off + n] = payload[src:src + n]
                self.filled += n
                if self.filled >= self.fb:
                    out.append(bytes(self.buf))
                    self.complete += 1
                    self.cur, self.filled = fidx + 1, 0
            pos += n
            src += n
        return out


class Receiver(threading.Thread):
    def __init__(self, out_path: Path, frame_bytes: int, n_frames: int):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        self.sock.settimeout(0.5)
        try:
            self.sock.bind((PC_IP, DATA_PORT))
        except OSError as e:
            raise CaptureError(f"cannot bind the data port {PC_IP}:{DATA_PORT} ({e})")
        self.asm = FrameAssembler(frame_bytes)
        self.out_path, self.n_frames = out_path, n_frames
        self.stop_evt = threading.Event()
        self.done_evt = threading.Event()
        self.packets = self.lost = 0
        self.frames_written = 0
        self.first_frame = None
        self._last_seq = None

    def run(self):
        pkt = bytearray(2048)
        view = memoryview(pkt)
        quiet = 0
        with open(self.out_path, "wb") as f:
            while not self.stop_evt.is_set() and self.frames_written < self.n_frames:
                try:
                    n = self.sock.recv_into(pkt)
                except socket.timeout:
                    quiet += 1
                    if self.packets and quiet >= 4:
                        break
                    continue
                except OSError:
                    break
                if n <= HEADER_BYTES:
                    continue
                quiet = 0
                self.packets += 1
                seq = int.from_bytes(pkt[0:4], "little")
                if self._last_seq is not None and seq > self._last_seq + 1:
                    self.lost += seq - self._last_seq - 1
                self._last_seq = seq
                for frame in self.asm.push(int.from_bytes(pkt[4:10], "little"), view[HEADER_BYTES:n]):
                    if self.frames_written < self.n_frames:
                        f.write(frame)
                        self.frames_written += 1
                        if self.first_frame is None:
                            self.first_frame = frame[: 4 * 4096 * 2]
        self.sock.close()
        self.done_evt.set()


def parse_target(text: str) -> dict:
    parts = text.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("--target needs 'range_m:azimuth_deg[:label]'")
    return {"range_m": float(parts[0]), "azimuth_deg": float(parts[1]),
            "label": parts[2].strip() if len(parts) > 2 else ""}


def parse_target_xy(text: str) -> dict:
    parts = text.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("--target-xy needs 'x_m:y_m[:label]'")
    return {"x_m": float(parts[0]), "y_m": float(parts[1]), "label": parts[2].strip() if len(parts) > 2 else ""}


def show_target(t: dict) -> str:
    return (rf"$R = {t['range_m']:.3f}$ m, $\theta = {t['azimuth_deg']:+.2f}^\circ$, "
            rf"$\varphi = {t['elevation_deg']:+.2f}^\circ$ (x {t['x_m']:.3f}, y {t['y_m']:+.3f}, z {t['z_m']:+.3f} m)  {t['label']}")


def update_truth(folder: Path, plan: dict | None, name: str | None = None, targets: list = ()):
    if plan is None or heights(plan)[0] is None:
        print("note: no antenna_height_m in the plan - every reflector is taken at antenna height (z = 0)")
    for p in [folder / f"{name}.json"] if name else sorted(folder.glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        src = list(targets) if name and targets else doc.get("targets", [])
        doc["targets"] = [complete(t, plan) for t in src]
        doc["antenna_height_m"] = heights(plan)[0]
        p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"{p.name}: " + ("no reflectors" if not doc["targets"] else ""))
        for t in doc["targets"]:
            print("  " + show_target(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", help="radar CLI (UART) port, e.g. COM4")
    ap.add_argument("--cfg", default=str(ROOT / "config" / "awr2944_doa_perfect.cfg"))
    ap.add_argument("--session", default="", help="sub-folder of data/raw/ (one folder per session)")
    ap.add_argument("--name", default=None, help="file name stem (with --plan: the take's name in the plan)")
    ap.add_argument("--plan", default=None,
                    help="session plan (JSON, e.g. config/session_plan.json): scenario and planned targets of --name")
    ap.add_argument("--scenario", default=None,
                    choices=["single", "multi", "background", "calibration", "null"],
                    help="null: empty scene that is EVALUATED (every detection is false)")
    ap.add_argument("--target", action="append", type=parse_target, default=[],
                    help="MEASURED 'range_m:azimuth_deg:label' of a reflector apex (repeat; replaces the plan's)")
    ap.add_argument("--target-xy", action="append", type=parse_target_xy, default=[],
                    help="MEASURED 'x_m:y_m:label' (x along the boresight line, y sideways, + = right); "
                         "converted to range and azimuth (repeat; replaces the plan's)")
    ap.add_argument("--notes", default="", help="anything worth knowing later (room, stands, people ...)")
    ap.add_argument("--timeout", type=float, default=None, help="seconds (default: numFrames x period + 20 s)")
    ap.add_argument("--packet-delay-us", type=int, default=25,
                    help="DCA1000 gap between UDP packets (default 25); larger = lower burst rate on the Ethernet link")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="check arguments and paths, no hardware")
    ap.add_argument("--update-truth", action="store_true",
                    help="no recording: recompute the truth of the recorded takes from x, y and the plan's heights "
                         "(all takes of --session, or --name with new --target-xy)")
    args = ap.parse_args()
    args.target = args.target + args.target_xy
    plan_doc = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else None
    folder = RAW / args.session if args.session else RAW
    if args.update_truth:
        update_truth(folder, plan_doc, args.name, args.target)
        return
    if not args.name:
        ap.error("--name is required")
    if args.plan:
        plan = {t["name"]: t for t in plan_doc["takes"]}
        if args.name not in plan and not args.scenario:
            raise SystemExit(f"take '{args.name}' is not in {args.plan}: {sorted(plan)} "
                             "(an extra take needs --scenario and --target-xy; the plan then gives the heights)")
        entry = plan.get(args.name, {})
        args.scenario = args.scenario or entry["scenario"]
        if not args.target:
            args.target = [dict(t) for t in entry.get("targets", [])]
            if args.target:
                print("targets from the PLAN - enter the measured values with --target, or edit the .json later")
        args.notes = args.notes or entry.get("setup", "")
    args.scenario = args.scenario or "single"
    args.target = [complete(t, plan_doc) for t in args.target]

    cfg_path = Path(args.cfg)
    cfg = RadarConfig.from_cfg(cfg_path)
    folder.mkdir(parents=True, exist_ok=True)
    out_bin, out_json = folder / f"{args.name}.bin", folder / f"{args.name}.json"
    if out_bin.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"{out_bin} exists - choose another --name or pass --overwrite")
    if args.scenario not in ("background", "null") and not args.target:
        raise SystemExit("give the measured reflector position(s) with --target 'R:theta:label' "
                         "(or --scenario background / null for an empty-scene take)")
    n_frames = cfg.n_frames if cfg.n_frames > 0 else 50
    frame_bytes = cfg.wire_bytes_per_frame(padded=True)
    timeout = args.timeout or n_frames * cfg.frame_period_s + 20.0
    print(f"capture '{args.name}': {n_frames} frames x {frame_bytes / 2**20:.1f} MiB on the wire, timeout {timeout:.0f} s")
    print("radar: " + cfg.describe())
    for t in args.target:
        print("  target: " + show_target(t))
    meta = {"name": args.name, "scenario": args.scenario, "targets": args.target, "notes": args.notes,
            "antenna_height_m": heights(plan_doc)[0],
            "cfg_file": str(cfg_path), "cfg_text": cfg_path.read_text(encoding="utf-8", errors="ignore"),
            "frames_requested": n_frames, "frame_bytes_on_wire": frame_bytes, "packet_delay_us": args.packet_delay_us}
    if args.dry_run:
        print(f"dry run: would write {out_bin} and {out_json}")
        return
    if not args.cli:
        raise SystemExit("--cli PORT is required (or use --dry-run)")
    if out_bin.exists():
        prev = folder / "_previous"
        prev.mkdir(exist_ok=True)
        for p in (out_bin, out_json):
            if p.exists():
                p.replace(prev / p.name)
        log(f"previous {args.name} moved to {prev}")

    sensor_stop(args.cli)
    dca = DCA1000()
    rx = None
    t0 = time.time()
    try:
        dca.start(packet_delay_us=args.packet_delay_us)
        rx = Receiver(out_bin, frame_bytes, n_frames)
        rx.start()
        time.sleep(0.3)
        send_cfg(args.cli, cfg_path)
        log("streaming ...")
        while not rx.done_evt.wait(1.0):
            log(f"{rx.frames_written}/{n_frames} frames, {rx.packets} packets, {rx.lost} lost")
            if time.time() - t0 > timeout:
                log("timeout - stopping")
                break
    finally:
        sensor_stop(args.cli)
        if rx is not None:
            rx.stop_evt.set()
            rx.join(timeout=3.0)
        dca.stop()

    padded = detect_padding(np.frombuffer(rx.first_frame, dtype="<i2")) if rx and rx.first_frame else None
    meta.update(recorded=time.strftime("%Y-%m-%d %H:%M:%S"), frames=rx.frames_written if rx else 0,
                packets=rx.packets if rx else 0, lost_packets=rx.lost if rx else 0,
                incomplete_frames_dropped=rx.asm.incomplete if rx else 0, padding_detected=padded)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"{meta['frames']} frames -> {out_bin}  ({meta['lost_packets']} packets lost, "
          f"{meta['incomplete_frames_dropped']} incomplete frames dropped, padding {padded})")
    if meta["frames"] == 0:
        print("ZERO frames: check that adcCfg is '2 0' (real), the UART log shows 'Done' for every line, "
              "and the DCA1000 replied OK to every command.")
    if meta["lost_packets"]:
        print("PACKETS LOST: the Ethernet path cannot carry the DCA1000 burst rate. A USB Ethernet adapter must "
              "run at USB 3 speed (blue / 'SS' USB-A port or USB-C, directly on the laptop, not via a USB 2 hub). "
              "If no USB 3 port helps, retry with a larger --packet-delay-us (e.g. 40).")


if __name__ == "__main__":
    main()
