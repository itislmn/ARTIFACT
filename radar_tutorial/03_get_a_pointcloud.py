"""
LESSON 03 — The easy path: let the chip do the math, you just read the answer

The AWR2944P can do ALL the signal processing itself - range FFT, Doppler
FFT, CFAR detection, angle estimation - and hand you a short, tidy list of
"detected points": (x, y, z, velocity). This is the fastest way to prove
your whole chain works, because you don't have to understand any DSP yet.
You're just reading text-like commands to configure it, then reading its
answer.

Run it:
    python 03_get_a_pointcloud.py --cli COM4 --data COM5 --cfg your_profile.cfg --seconds 15

WHAT'S HAPPENING UNDER THE HOOD:
1. We open COM4 and type the same commands you'd type by hand in a
   terminal - each line of your .cfg, one at a time, waiting for "Done"
   after each. This is literally just automated typing.
2. The LAST command, sensorStart, is what makes the chip start transmitting
   chirps and computing results.
3. On a SECOND serial port (COM5, the "data port"), the chip streams its
   results as structured binary packets. Each packet starts with a fixed
   8-byte "magic word" so we can find where one result frame ends and the
   next begins, then a header telling us how many objects were found, then
   the (x,y,z,velocity) numbers themselves.
"""
import argparse
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import serial

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
OUT_DIR = "output/03_get_a_pointcloud"


def send_cfg(cli_port, cfg_path, baud=115200):
    print(f"Typing {cfg_path} into {cli_port}, one line at a time...")
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
            reply = ser.read(300).decode(errors="ignore")
            print(f"  > {line[:45]:45s} | {'Done' if 'Done' in reply else reply.strip()[:40]}")


def read_points(data_port, seconds, baud=921600):
    print(f"\nListening on {data_port} for detected-point frames...")
    buf = b""
    all_points = []
    t0 = time.time()
    with serial.Serial(data_port, baud, timeout=0.1) as ser:
        while time.time() - t0 < seconds:
            buf += ser.read(4096)
            i = buf.find(MAGIC)
            if i < 0:
                continue
            buf = buf[i:]
            if len(buf) < 40:
                continue
            total_len, num_obj, num_tlv = struct.unpack("<III", buf[12:24])[0:1] + \
                struct.unpack("<II", buf[24:32])
            # (kept simple on purpose - see step1_uart_pointcloud.py for the full robust parser)
            if len(buf) < total_len or total_len < 40 or total_len > 200000:
                continue
            idx = 40
            points_this_frame = []
            for _ in range(num_tlv):
                if idx + 8 > total_len:
                    break
                tlv_type, tlv_len = struct.unpack("<II", buf[idx:idx + 8])
                payload = buf[idx + 8: idx + tlv_len]
                if tlv_type == 1 and len(payload) >= 16:
                    n = len(payload) // 16
                    pts = np.frombuffer(payload[:n * 16], dtype=np.float32).reshape(n, 4)
                    points_this_frame.append(pts)
                idx += tlv_len
            if points_this_frame:
                all_points.append(np.vstack(points_this_frame))
                print(f"  frame with {len(all_points[-1])} detected point(s)")
            buf = buf[total_len:]
    return all_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--databaud", type=int, default=921600)
    args = ap.parse_args()

    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    send_cfg(args.cli, args.cfg)
    points = read_points(args.data, args.seconds, args.databaud)

    if not points:
        print("\nNo point-cloud frames parsed. Common reasons: wrong --databaud")
        print("(try --databaud 115200), or the .cfg doesn't match the flashed demo.")
        return

    allp = np.vstack(points)
    plt.figure(figsize=(6, 6))
    plt.scatter(allp[:, 0], allp[:, 1], c=allp[:, 3], cmap="coolwarm", s=15)
    plt.colorbar(label="velocity [m/s]")
    plt.xlabel("x [m]"); plt.ylabel("y [m] (range)")
    plt.title("Everything detected during this capture")
    plt.grid(alpha=0.3)
    out = f"{OUT_DIR}/pointcloud.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out} - {len(allp)} total points across {len(points)} frames.")
    print("WHAT TO LOOK FOR: a cluster of points roughly where you know something")
    print("physical was standing/moving. If you walked toward the sensor, points")
    print("should show up closer over time with negative velocity (approaching).")


if __name__ == "__main__":
    main()
