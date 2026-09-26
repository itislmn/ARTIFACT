"""
LESSON 06 — A real raw ADC / IQ capture, start to finish

This combines everything from lessons 02 and 05: configure and arm the
DCA1000 over UDP, THEN send your .cfg over UART to actually start the
sensor transmitting, and save whatever streams back.

Run it:
    python 06_capture_raw_iq.py --cli COM4 --cfg your_profile_WITH_lvdsStreamCfg.cfg --seconds 8

REQUIREMENT: your .cfg MUST contain a line like `lvdsStreamCfg -1 0 1 0`.
Without it, the chip never sends anything over LVDS, and the DCA1000 will
sit there listening to silence. Lesson 02 explains this line specifically.

WHAT "RAW" MEANS HERE: every single ADC sample, from every receive
antenna, for every chirp, completely unprocessed. No FFTs, no detection
logic, nothing. Just the numbers straight off the analog-to-digital
converter. This is the maximum amount of information you can get out of
the chip - lesson 07 onward is entirely about turning this pile of numbers
into something meaningful.

WHAT "IQ" MEANS: if your .cfg's adcCfg line has complex mode enabled, each
sample is actually TWO numbers (I and Q) representing one point on the
complex plane - both the strength AND the phase of the signal at that
instant. Real-only mode (what the example .cfg here likely uses) only
keeps one of those two numbers. This lesson saves whatever format your
.cfg actually configures, and tells you which one it detected.
"""
import argparse
import os
import socket
import struct
import threading
import time

import numpy as np
import serial

DCA_IP = "192.168.33.180"
PC_IP = "192.168.33.30"
PC_DATA_PORT = 4098
OUT_DIR = "output/06_capture_raw_iq"

RESET_FPGA, CONFIG_FPGA_GEN, RECORD_START, RECORD_STOP = 0x01, 0x03, 0x05, 0x06
SYSTEM_CONNECT, CONFIG_PACKET_DATA, READ_FPGA_VERSION = 0x09, 0x0B, 0x0E


def frame(cmd, data=b""):
    return struct.pack("<HHH", 0xA55A, cmd, len(data)) + data + struct.pack("<H", 0xEEAA)


def arm_dca1000(lvds_lanes=4):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(3.0)
    sock.bind((PC_IP, 4096))

    def cmd(name, code, data=b""):
        sock.sendto(frame(code, data), (DCA_IP, 4096))
        try:
            resp, _ = sock.recvfrom(2048)
            print(f"  {name:20s} -> {resp.hex()}")
        except socket.timeout:
            print(f"  {name:20s} -> TIMEOUT")

    print("Arming the DCA1000...")
    cmd("SYSTEM_CONNECT", SYSTEM_CONNECT)
    cmd("RESET_FPGA", RESET_FPGA)
    time.sleep(0.5)
    lvds_code = 1 if lvds_lanes == 4 else 2
    payload = bytes([1, lvds_code, 1, 2, 3, 30])  # raw, lanes, LVDS, ethernet, 16-bit, 30s timeout
    cmd("CONFIG_FPGA_GEN", CONFIG_FPGA_GEN, payload)
    cmd("CONFIG_PACKET_DATA", CONFIG_PACKET_DATA, struct.pack("<HHH", 1466, 25, 0))
    cmd("RECORD_START", RECORD_START)
    return sock


def send_sensor_config(cli_port, cfg_path):
    print(f"\nStarting the actual sensor over {cli_port}...")
    with serial.Serial(cli_port, 115200, timeout=1) as ser:
        ser.write(b"\n")
        time.sleep(0.2)
        ser.reset_input_buffer()
        for raw in open(cfg_path):
            line = raw.strip()
            if not line or line.startswith(("%", "#")):
                continue
            ser.write((line + "\n").encode())
            time.sleep(0.05)
            reply = ser.read(300).decode(errors="ignore") # we're not printing every line here, lesson 03 already showed that
            print(f"  > {line[:45]:45s} | {'Done' if 'Done' in reply else reply.strip()[:40]}")


def receive(seconds, out_path):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind((PC_IP, PC_DATA_PORT))
    total = 0
    t0 = time.time()
    with open(out_path, "wb") as f:
        while time.time() - t0 < seconds:
            try:
                pkt, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            if len(pkt) > 10:
                f.write(pkt[10:])  # strip the DCA1000's own 10-byte sequence/length header
                total += len(pkt) - 10
    sock.close()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--lvds-lanes", type=int, default=4, choices=[2, 4])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_bin = f"{OUT_DIR}/raw_capture.bin"

    dca_sock = arm_dca1000(args.lvds_lanes)

    result = {}
    def _go():
        result["bytes"] = receive(args.seconds + 3, out_bin)
    t = threading.Thread(target=_go, daemon=True)
    t.start()
    time.sleep(0.5)

    send_sensor_config(args.cli, args.cfg)
    print(f"Waiting {args.seconds}s for data...")
    t.join()

    dca_sock.sendto(frame(RECORD_STOP), (DCA_IP, 4096))
    dca_sock.close()

    size = result.get("bytes", 0)
    print(f"\nCaptured {size:,} bytes -> {out_bin}")
    if size < 10_000:
        print("That's too small to be real. Check: does your .cfg have")
        print("lvdsStreamCfg? Try --lvds-lanes 2 instead of 4 (or vice versa).")
        return

    raw = np.fromfile(out_bin, dtype=np.int16)
    print(f"\nThat's {raw.size:,} individual int16 numbers.")
    print(f"Value range seen: {raw.min()} to {raw.max()} "
          f"(int16 can go from -32768 to 32767)")
    print("If your adcCfg used COMPLEX mode, these numbers come in groups of")
    print("4 per pair of samples: [I0, I1, Q0, Q1] - lesson 07 handles the")
    print("reassembly into real I+jQ complex numbers automatically.")
    print(f"\nNEXT: python 07_build_and_explore_cube.py --bin {out_bin} --cfg {args.cfg}")


if __name__ == "__main__":
    main()
