"""
LESSON 01 — Is anything even there?

Before you configure a single chirp, you need to know two independent
things work: the sensor chip talks over a serial (UART) connection, and
the DCA1000 talks over Ethernet (UDP). This lesson checks both, and
explains exactly what "checking" means at this level - we are not asking
the radar to sense anything yet. We are asking two pieces of silicon
"are you powered on and listening."

Run it:
    python 01_check_hardware_alive.py --cli COM4 --dca-ip 192.168.33.180

WHAT YOU'RE ABOUT TO SEE:
For the sensor: we just open the COM port. There's no magic "ping" command
for a UART - if the port opens without error, the USB/driver layer is fine.
Whether the CHIP behind it is doing anything useful is a separate question
we can't answer until it's flashed with real firmware (see the main
RUNBOOK.md from your first setup night if you haven't flashed yet).

For the DCA1000: this box speaks a real protocol over UDP, so we can
genuinely ask it a question and get a real answer. Every command to it is
wrapped the same way:

    0xA55A  <command code, 2 bytes>  <data length, 2 bytes>  <data>  0xEEAA

0xA55A and 0xEEAA are just fixed "start" and "end" markers so the DCA1000
can find where a command begins and ends inside the stream of bytes it
receives - nothing clever, just bookkeeping. We send SYSTEM_CONNECT
(command code 0x09) with no data, and READ_FPGA_VERSION (0x0E), and print
whatever comes back. A real reply proves: power is on, Ethernet cable is
good, your PC's static IP is configured correctly, and the DCA1000's own
firmware is alive.
"""
import argparse
import os
import socket
import struct

import serial.tools.list_ports

OUT_DIR = "output/01_check_hardware_alive"


def frame(cmd, data=b""):
    """Wrap a command the way the DCA1000 expects: header, code, length, data, footer."""
    return struct.pack("<HHH", 0xA55A, cmd, len(data)) + data + struct.pack("<H", 0xEEAA)


def check_com_ports():
    print("\n--- Serial (UART) ports visible to this PC ---")
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("  None found at all. Check the USB cable and drivers.")
        return
    for p in ports:
        print(f"  {p.device:8s}  {p.description}")
    print("  (This just lists what Windows sees - it does not prove the")
    print("   chip behind any of these is running real firmware yet.)")


def check_dca1000(dca_ip, pc_ip, port=4096):
    print(f"\n--- DCA1000 at {dca_ip}, talking from {pc_ip}:{port} ---")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(3.0)
    try:
        sock.bind((pc_ip, port))
    except OSError as e:
        print(f"  Could not even bind {pc_ip}:{port} -> {e}")
        print("  Your PC's Ethernet adapter needs a STATIC IP of 192.168.33.30")
        print("  (mask 255.255.255.0, no gateway) for this to work at all.")
        return False

    ok = True
    for name, code in (("SYSTEM_CONNECT", 0x09), ("READ_FPGA_VERSION", 0x0E)):
        sock.sendto(frame(code), (dca_ip, 4096))
        try:
            resp, _ = sock.recvfrom(2048)
            print(f"  {name:20s} -> real reply: {resp.hex()}")
        except socket.timeout:
            print(f"  {name:20s} -> TIMEOUT, no reply at all")
            ok = False
    sock.close()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", help="Sensor CLI COM port, e.g. COM4 (optional, just lists ports)")
    ap.add_argument("--dca-ip", default="192.168.33.180")
    ap.add_argument("--pc-ip", default="192.168.33.30")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    check_com_ports()
    dca_ok = check_dca1000(args.dca_ip, args.pc_ip)

    report = os.path.join(OUT_DIR, "result.txt")
    with open(report, "w") as f:
        f.write(f"DCA1000 reachable: {dca_ok}\n")
    print(f"\nSaved a one-line result to {report}")
    print("\nWHAT TO DO NEXT: if the DCA1000 replied, move to lesson 02 - it")
    print("needs no hardware at all, just your .cfg file, so it's safe to")
    print("read even if the sensor side isn't flashed yet.")


if __name__ == "__main__":
    main()
