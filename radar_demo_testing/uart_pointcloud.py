"""

Flash the mmWave SDK out-of-box demo, send a .cfg over the CLI UART from
Python, and parse the TLV output stream yourself. You get a live point cloud
+ range profile, recorded to .npz, entirely in Python. If the DCA1000 refuses
to cooperate tonight, THIS is what you show tomorrow.

    python step1_uart_pointcloud.py --cli COM4 --data COM5 --cfg my_profile.cfg

Notes:
  - --databaud: the AWR294x demo does not always use 921600. If you get no
    magic word, run with --scanbaud to try the common rates automatically.
  - Point the EVM at a corridor and walk toward it. Moving targets are far
    more convincing to a room of professors than a static scene.
"""
import argparse
import struct
import time

import numpy as np
import serial

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_LEN = 40  # magic(8) ver,len,platform,frameNum,cpuCycles,numObj,numTLV,subFrame
COMMON_BAUDS = [921600, 1250000, 115200, 852272, 3125000]


def send_config(cli_port, cfg_path, baud=115200, verbose=True):
    """Stream a .cfg line by line to the sensor CLI. Returns the lines sent."""
    sent = []
    with serial.Serial(cli_port, baud, timeout=1) as ser:
        ser.write(b"\n")
        time.sleep(0.2)
        ser.reset_input_buffer()
        for raw in open(cfg_path):
            line = raw.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            ser.write((line + "\n").encode())
            time.sleep(0.05)
            reply = ser.read(512).decode(errors="ignore")
            sent.append(line)
            if verbose:
                status = "Done" if "Done" in reply else reply.strip()[:60]
                print(f"  > {line[:55]:55s} | {status}")
            if "Error" in reply or "not recognized" in reply:
                print(f"  !! Sensor rejected: {line}")
    return sent


def find_magic(buf, start=0):
    return buf.find(MAGIC, start)


def parse_frame(buf):
    """Parse one complete frame starting at a magic word. Returns (dict, bytes_consumed) or (None,0)."""
    if len(buf) < HEADER_LEN:
        return None, 0
    (version, total_len, platform, frame_num, cpu_cycles,
     num_obj, num_tlv, sub_frame) = struct.unpack("<8I", buf[8:HEADER_LEN])
    if total_len < HEADER_LEN or total_len > 200000:
        return None, 0
    if len(buf) < total_len:
        return None, 0

    out = {"frame": frame_num, "num_obj": num_obj, "points": None, "range_profile": None}
    idx = HEADER_LEN
    for _ in range(num_tlv):
        if idx + 8 > total_len:
            break
        tlv_type, tlv_len = struct.unpack("<2I", buf[idx:idx + 8])
        payload = buf[idx + 8: idx + 8 + tlv_len - 8] if tlv_len > 8 else b""
        # some SDK builds encode tlv_len as payload-only; handle both
        if idx + 8 + tlv_len <= total_len:
            payload = buf[idx + 8: idx + 8 + tlv_len]
            step = 8 + tlv_len
        else:
            step = 8 + max(0, tlv_len - 8)

        if tlv_type == 1 and len(payload) >= 16:  # DETECTED_POINTS: x,y,z,vel float32
            n = len(payload) // 16
            pts = np.frombuffer(payload[:n * 16], dtype=np.float32).reshape(n, 4)
            out["points"] = pts
        elif tlv_type == 2 and len(payload) >= 2:  # RANGE_PROFILE: uint16 log-mag
            out["range_profile"] = np.frombuffer(
                payload[: (len(payload) // 2) * 2], dtype=np.uint16).astype(np.float32)
        idx += step
    return out, total_len


def scan_baud(data_port, seconds=2.0):
    print("Scanning for the data-port baud rate...")
    for b in COMMON_BAUDS:
        try:
            with serial.Serial(data_port, b, timeout=0.2) as ser:
                t0 = time.time()
                buf = b""
                while time.time() - t0 < seconds:
                    buf += ser.read(4096)
                    if MAGIC in buf:
                        print(f"  MAGIC WORD FOUND at {b} baud.")
                        return b
        except serial.SerialException as e:
            print(f"  {b}: {e}")
    print("  No magic word at any common baud. Is the OOB demo actually running?")
    print("  Open the CLI port in a terminal at 115200, press NRST, and look for SBL logs.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, help="CLI/application COM port, e.g. COM4")
    ap.add_argument("--data", required=True, help="Data COM port, e.g. COM5")
    ap.add_argument("--cfg", required=True, help="Path to the .cfg from the SDK / visualizer")
    ap.add_argument("--clibaud", type=int, default=115200)
    ap.add_argument("--databaud", type=int, default=921600)
    ap.add_argument("--scanbaud", action="store_true")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="captures/uart_capture.npz")
    ap.add_argument("--noplot", action="store_true")
    args = ap.parse_args()

    print(f"\nSending config from {args.cfg} to {args.cli}...")
    sent_lines = send_config(args.cli, args.cfg, args.clibaud)
    if not any("sensorstart" in l.lower() for l in sent_lines):
        print("  (cfg had no sensorStart line - sending it now)")
        import serial as _serial
        with _serial.Serial(args.cli, args.clibaud, timeout=1) as ser:
            ser.write(b"sensorStart\n")
            time.sleep(0.3)
            print("  > sensorStart | " + ser.read(512).decode(errors="ignore").strip()[:60])
    print("Config sent, sensor started. NOW scanning/reading data stream...\n")

    databaud = scan_baud(args.data) if args.scanbaud else args.databaud
    if databaud is None:
        print("Still no magic word even after sensorStart. See troubleshooting")
        print("notes below before giving up - this is very fixable.")
        return

    frames, profiles = [], []
    plot = None
    if not args.noplot:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        sc = ax1.scatter([], [], s=18)
        ax1.set_xlim(-6, 6); ax1.set_ylim(0, 12)
        ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.set_title("Detected points")
        ax1.grid(alpha=0.3)
        (ln,) = ax2.plot([], [])
        ax2.set_xlabel("range bin"); ax2.set_ylabel("log magnitude"); ax2.set_title("Range profile")
        ax2.grid(alpha=0.3)
        plot = (plt, fig, ax1, ax2, sc, ln)

    buf = b""
    t0 = time.time()
    with serial.Serial(args.data, databaud, timeout=0.1) as ser:
        while time.time() - t0 < args.seconds:
            buf += ser.read(8192)
            while True:
                i = find_magic(buf)
                if i < 0:
                    if len(buf) > 1 << 20:
                        buf = buf[-4096:]
                    break
                buf = buf[i:]
                parsed, consumed = parse_frame(buf)
                if parsed is None:
                    break
                buf = buf[consumed:]
                frames.append(parsed)
                if parsed["range_profile"] is not None:
                    profiles.append(parsed["range_profile"])
                if len(frames) % 10 == 0:
                    print(f"  frame {parsed['frame']:6d}  objects={parsed['num_obj']:3d}")
                if plot and parsed["points"] is not None and len(parsed["points"]):
                    plt, fig, ax1, ax2, sc, ln = plot
                    p = parsed["points"]
                    sc.set_offsets(np.c_[p[:, 0], p[:, 1]])
                    if parsed["range_profile"] is not None:
                        rp = parsed["range_profile"]
                        ln.set_data(np.arange(len(rp)), rp)
                        ax2.relim(); ax2.autoscale_view()
                    fig.canvas.draw_idle(); fig.canvas.flush_events()

    print(f"\nCaptured {len(frames)} frames.")
    if frames:
        pts = [f["points"] for f in frames if f["points"] is not None]
        np.savez_compressed(
            args.out,
            num_obj=np.array([f["num_obj"] for f in frames]),
            frame_ids=np.array([f["frame"] for f in frames]),
            range_profiles=np.array(profiles) if profiles else np.array([]),
            points=np.array(pts, dtype=object) if pts else np.array([]),
        )
        print(f"Saved -> {args.out}")
    else:
        print("No frames parsed. Try --scanbaud, and confirm the demo is running.")


if __name__ == "__main__":
    main()