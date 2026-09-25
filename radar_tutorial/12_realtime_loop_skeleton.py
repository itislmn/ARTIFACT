"""
LESSON 12 — Putting it all together: a live, real-time loop

Every lesson before this processed a FILE, after the fact. Real, active
sensing means doing lesson 08+09+10's math INSIDE the receive loop, on each
frame as it arrives, and letting the result change what happens next - even
changing the sensor's own configuration mid-run.

This is a SKELETON, deliberately not a finished product: it prints the
strongest detection every frame, live, and shows exactly where your own
decision logic (ML model, tracker, rail-control code, whatever) plugs in.
There is no scipy CFAR here on purpose - simplicity over completeness, so
you can read the whole loop in one sitting.

Run it:
    python 12_realtime_loop_skeleton.py --cli COM4 --cfg your_profile_with_lvds.cfg

Stop it with Ctrl+C - it cleans up and reports what it saw.
"""
import argparse
import re
import socket
import struct
import threading
import time

import numpy as np
import serial

DCA_IP = "192.168.33.180"
PC_IP = "192.168.33.30"
PC_DATA_PORT = 4098
RESET_FPGA, CONFIG_FPGA_GEN, RECORD_START, RECORD_STOP = 0x01, 0x03, 0x05, 0x06
SYSTEM_CONNECT, CONFIG_PACKET_DATA = 0x09, 0x0B
C = 299792458.0


def frame_cmd(cmd, data=b""):
    return struct.pack("<HHH", 0xA55A, cmd, len(data)) + data + struct.pack("<H", 0xEEAA)


def parse_cfg(path):
    p = {"chirpTx": {}}
    for raw in open(path):
        line = raw.strip()
        v = re.split(r"\s+", line)[1:] if line else []
        if line.startswith("profileCfg"):
            p["slope_MHz_us"], p["numAdcSamples"] = float(v[7]), int(v[9])
            p["sampleRate_ksps"] = float(v[10])
        elif line.startswith("channelCfg"):
            p["numRx"] = bin(int(v[0])).count("1")
        elif line.startswith("chirpCfg"):
            p["chirpTx"][int(v[0])] = int(v[7])
        elif line.startswith("frameCfg"):
            loop = int(v[2])
            chirps = int(v[1]) - int(v[0]) + 1
            p["chirpsPerFrame"] = chirps * loop
    p["numTx"] = len({m for m in p["chirpTx"].values() if m}) or 1
    N, fs = p["numAdcSamples"], p["sampleRate_ksps"]*1e3
    slope = p["slope_MHz_us"]*1e12
    p["range_axis"] = np.arange(N) * (C*fs) / (2*slope*N)
    return p


def arm_dca1000():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(3.0)
    sock.bind((PC_IP, 4096))
    for code, data in [(SYSTEM_CONNECT, b""), (RESET_FPGA, b"")]:
        sock.sendto(frame_cmd(code, data), (DCA_IP, 4096))
        sock.recvfrom(2048)
    sock.sendto(frame_cmd(CONFIG_FPGA_GEN, bytes([1, 1, 1, 2, 3, 30])), (DCA_IP, 4096))
    sock.recvfrom(2048)
    sock.sendto(frame_cmd(CONFIG_PACKET_DATA, struct.pack("<HHH", 1466, 25, 0)), (DCA_IP, 4096))
    sock.recvfrom(2048)
    sock.sendto(frame_cmd(RECORD_START), (DCA_IP, 4096))
    sock.recvfrom(2048)
    return sock


def start_sensor(cli_port, cfg_path):
    with serial.Serial(cli_port, 115200, timeout=1) as ser:
        for raw in open(cfg_path):
            line = raw.strip()
            if line and not line.startswith(("%", "#")):
                ser.write((line+"\n").encode())
                time.sleep(0.05)
                ser.read(300)


# ============================================================
# YOUR DECISION LOGIC GOES HERE. This is the entire point of
# the tutorial - everything above is plumbing to get you a
# clean numpy array, once per frame, as fast as the sensor
# produces it. Replace this function with your own model.
# ============================================================
def policy(range_m, strength_db):
    """Given the single strongest detection this frame, decide what to do.
    Right now: just classify near/far. Replace with a real classifier,
    tracker, or whatever controls your rail/vision system."""
    if range_m < 1.0:
        return "CLOSE - would trigger rail stop / alert"
    elif range_m < 3.0:
        return "MID range - normal operation"
    else:
        return "FAR / background"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()

    p = parse_cfg(args.cfg)
    per_frame_floats = p["numAdcSamples"] * p["numRx"] * p["chirpsPerFrame"]
    n_half = p["numAdcSamples"] // 2

    print("Arming DCA1000 and starting sensor...")
    dca_sock = arm_dca1000()
    data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    data_sock.settimeout(1.0)
    data_sock.bind((PC_IP, PC_DATA_PORT))

    t = threading.Thread(target=start_sensor, args=(args.cli, args.cfg), daemon=True)
    t.start()

    print("Streaming live. Ctrl+C to stop.\n")
    buf = bytearray()
    frame_count = 0
    try:
        while True:
            try:
                pkt, _ = data_sock.recvfrom(65536)
            except socket.timeout:
                continue
            buf.extend(pkt[10:])
            if len(buf) < per_frame_floats * 2:  # int16 = 2 bytes each
                continue
            # we have enough bytes for one full frame - process it, live
            chunk = bytes(buf[:per_frame_floats*2])
            del buf[:per_frame_floats*2]
            raw = np.frombuffer(chunk, dtype=np.int16).astype(np.complex64)
            cube = raw.reshape(p["chirpsPerFrame"], p["numRx"], p["numAdcSamples"])

            range_fft = np.fft.fft(cube * np.hanning(cube.shape[-1]), axis=-1)
            rp_db = 20*np.log10(np.abs(range_fft).mean(axis=(0,1))[:n_half] + 1e-9)
            peak = np.argmax(rp_db)
            range_m, strength_db = p["range_axis"][peak], rp_db[peak]

            decision = policy(range_m, strength_db)
            frame_count += 1
            print(f"frame {frame_count:4d}  strongest @ {range_m:5.2f} m "
                  f"({strength_db:5.1f} dB)  -> {decision}")
    except KeyboardInterrupt:
        print(f"\nStopped after {frame_count} live frames.")
    finally:
        dca_sock.sendto(frame_cmd(RECORD_STOP), (DCA_IP, 4096))
        dca_sock.close()
        data_sock.close()
        print("Cleaned up. This loop is your starting point - swap policy()")
        print("for a real model, and apply_action() (not implemented here)")
        print("for actually moving your rail or triggering vision.")


if __name__ == "__main__":
    main()
