r"""
LESSON 03 - The DCA1000's binary UDP command protocol

    python 03_dca1000_udp_protocol.py --demo          (fake DCA1000, always works)
    python 03_dca1000_udp_protocol.py --real           (talks to 192.168.33.180)

This is a COMPLETELY SEPARATE protocol from lesson 02's UART CLI. The
radar chip (AWR2944P) never hears about any of this - it just streams
raw samples out over its LVDS pins. The DCA1000 card is the thing that
listens on those pins and re-packages the data as Ethernet/UDP, and
THIS is the protocol you use to tell the DCA1000 card itself what to do
(arm it, start/stop recording, ask its firmware version). It's binary,
not ASCII, and it runs over UDP to 192.168.33.180 port 4096 - the
"config port". A SEPARATE port, 4098, the "data port", is where the
actual captured samples stream out to (that's lesson 04).

------------------------------------------------------------------------
FRAME LAYOUT
------------------------------------------------------------------------
Every message in both directions shares one 8-byte skeleton (plus an
optional data payload in the middle for a few commands):

    byte:    0    1    2    3    4    5    6    7
             [0xA55A header]  [ X ]  [ Y ]  [0xEEAA footer]
             (2 bytes LE)   (2B LE)(2B LE)  (2 bytes LE)

0xA55A and 0xEEAA are fixed magic numbers marking the start/end of every
frame - not addresses or lengths, just "yes, this is a real DCA1000
frame" sentinels. Field X is always the command code. Field Y means
something DIFFERENT depending on direction:

  - In a command YOU send: Y = length in bytes of the data payload that
    follows (0 for a plain no-argument command).
  - In the DCA1000's reply to a plain command: Y = a STATUS code
    (0 = success, nonzero = error), because there's no payload to give
    a length for.
  - In the DCA1000's reply to a query like READ_FPGA_VERSION: that same
    slot instead carries the actual answer (the version number itself),
    not a status code - it's a payload-bearing reply, structured the
    same way outgoing commands are.

That "same byte position, different meaning depending on which command
and which direction" is exactly what caused a real, hard-to-see bug in
this project: code that read the WRONG two bytes as "status" (position
6-7, actually the fixed 0xEEAA footer!) instead of the right ones
(position 4-5). Every single successful command was being misreported
as "ERROR status=61098" - 61098 being 0xEEAA misread as a little-endian
uint16. The hardware was fine the entire time; only the parsing code was
wrong. Walked through with REAL bytes below.

------------------------------------------------------------------------
COMMAND CODES USED IN THIS PROJECT
------------------------------------------------------------------------
"""
import argparse
import socket
import struct
import time

from common import (DCA_IP, CONFIG_PORT, PC_IP, dca_frame, CMD_NAMES,
                     RESET_FPGA, CONFIG_FPGA_GEN, RECORD_START, RECORD_STOP,
                     SYSTEM_CONNECT, CONFIG_PACKET_DATA, READ_FPGA_VERSION,
                     FakeDCASocket)


def print_command_table():
    for code, name in sorted(CMD_NAMES.items()):
        print(f"    0x{code:02X}  {name}")


# =======================================================================
# PART 1 - decode REAL bytes captured earlier in this project, no
# network needed. This is the worked example for the status-byte bug.
# =======================================================================
def decode_frame(resp: bytes, label=""):
    header = struct.unpack("<H", resp[0:2])[0]
    cmd = struct.unpack("<H", resp[2:4])[0]
    correct_status = struct.unpack("<H", resp[4:6])[0]   # RIGHT position
    buggy_status = struct.unpack("<H", resp[6:8])[0]      # the OLD, WRONG position
    footer = struct.unpack("<H", resp[6:8])[0]
    print(f"\n{label}: {resp.hex()}")
    print(f"  bytes[0:2] header       = 0x{header:04X} "
          f"({'OK, matches 0xA55A' if header == 0xA55A else 'UNEXPECTED'})")
    print(f"  bytes[2:4] command code = 0x{cmd:04X} ({CMD_NAMES.get(cmd, '?')})")
    print(f"  bytes[4:6] STATUS (correct)  = {correct_status} "
          f"({'success' if correct_status == 0 else 'ERROR'})")
    print(f"  bytes[6:8] footer            = 0x{footer:04X} "
          f"({'OK, matches 0xEEAA' if footer == 0xEEAA else 'UNEXPECTED'})")
    if buggy_status != 0:
        print(f"  ^ if you (wrongly) read bytes[6:8] AS status instead of "
              f"the footer, you'd get {buggy_status} and think this "
              "command had FAILED - which is exactly the bug that "
              "happened in this project. Every successful reply looked "
              "like an error, for that reason alone.")
    return correct_status


def part1_real_bytes_walkthrough():
    print("\n" + "=" * 70)
    print("PART 1 - decoding REAL bytes this exact hardware sent earlier")
    print("in this project (copy-pasted from an actual terminal log)")
    print("=" * 70)
    real_replies = {
        "SYSTEM_CONNECT reply": bytes.fromhex("5aa509000000aaee"),
        "RESET_FPGA reply": bytes.fromhex("5aa501000000aaee"),
        "CONFIG_FPGA_GEN reply": bytes.fromhex("5aa503000000aaee"),
        "CONFIG_PACKET_DATA reply": bytes.fromhex("5aa50b000000aaee"),
        "RECORD_START reply": bytes.fromhex("5aa505000000aaee"),
    }
    for label, resp in real_replies.items():
        status = decode_frame(resp, label)
        assert status == 0, "these were all real successful replies"
    print("\nAll five decode to status=0 (success) once you read the RIGHT")
    print("two bytes. The DCA1000 was answering correctly every single")
    print("time - only the code reading its answers was broken.")


# =======================================================================
# PART 2 - build and send real command frames, against either a fake
# DCA1000 (--demo, default) or the real one (--real).
# =======================================================================
def parse_response_status(resp, name):
    if len(resp) < 8:
        raise RuntimeError(f"{name}: reply too short ({len(resp)} bytes)")
    return struct.unpack("<H", resp[4:6])[0]   # the FIXED, correct offset


def arm_sequence(send_recv, lvds_lanes=4):
    """The fixed sequence of commands that gets the DCA1000 ready to
    capture: identify yourself, reset the FPGA's logic, tell it HOW to
    capture (raw/LVDS/Ethernet/16-bit), tell it the UDP packet size to
    use, then start recording. Same sequence whether `send_recv` talks
    to a real card or the fake one below - that's the point of having
    built a fake that speaks the identical wire protocol.
    """
    def do(name, code, data=b""):
        frame = dca_frame(code, data)
        resp = send_recv(frame)
        status = parse_response_status(resp, name)
        print(f"  {name:20s} -> {'OK' if status == 0 else f'ERROR status={status}'}"
              f"   ({resp.hex()})")
        if status != 0:
            raise RuntimeError(f"{name} failed, status={status}")

    do("SYSTEM_CONNECT", SYSTEM_CONNECT)
    do("RESET_FPGA", RESET_FPGA)
    time.sleep(0.2)
    lvds_code = 1 if lvds_lanes == 4 else 2
    # 6 payload bytes, each a distinct setting - NOT one bitmask:
    #   [0] data logic mode   = 1  (raw mode)
    #   [1] lvds lane count   = lvds_code (1 = 4 lanes, 2 = 2 lanes)
    #   [2] data xfer mode    = 1  (LVDS capture, as opposed to other sources)
    #   [3] data capture mode = 2  (stream over Ethernet, not to onboard SD)
    #   [4] data format       = 3  (16-bit)
    #   [5] timer             = 30 (internal DCA1000 timer setting, seconds)
    do("CONFIG_FPGA_GEN", CONFIG_FPGA_GEN, bytes([1, lvds_code, 1, 2, 3, 30]))
    # 3 payload fields, each a uint16:
    #   packet payload size (bytes) - 1466 leaves room for the DCA1000's
    #     own 10-byte header plus UDP/IP headers under the standard
    #     1500-byte Ethernet MTU, so nothing fragments.
    #   inter-packet delay (in the card's own timer units) - paces
    #     packet transmission so your PC's UDP receive buffer isn't
    #     overrun.
    #   a reserved/future field, 0.
    do("CONFIG_PACKET_DATA", CONFIG_PACKET_DATA, struct.pack("<HHH", 1466, 25, 0))
    do("RECORD_START", RECORD_START)


def run_demo():
    print("\n" + "=" * 70)
    print("PART 2 - the arm sequence, against a FAKE DCA1000 over the")
    print("exact same wire protocol (see common.py's FakeDCASocket)")
    print("=" * 70)
    fake = FakeDCASocket(adc_bytes_provider=lambda: b"\x00" * 100)

    def send_recv(frame):
        fake.sendto(frame, (DCA_IP, CONFIG_PORT))
        return fake.recvfrom(2048)[0]

    arm_sequence(send_recv)
    fake.sendto(dca_frame(RECORD_STOP), (DCA_IP, CONFIG_PORT))


def run_real():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(3.0)
    sock.bind((PC_IP, CONFIG_PORT))

    def send_recv(frame):
        sock.sendto(frame, (DCA_IP, CONFIG_PORT))
        return sock.recvfrom(2048)[0]

    print("\n" + "=" * 70)
    print(f"PART 2 - the arm sequence, against the REAL DCA1000 at {DCA_IP}")
    print("=" * 70)
    try:
        arm_sequence(send_recv)
    finally:
        sock.sendto(dca_frame(RECORD_STOP), (DCA_IP, CONFIG_PORT))
        sock.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                     help="Talk to the real DCA1000 at 192.168.33.180 "
                          "instead of the fake one.")
    ap.add_argument("--demo", action="store_true",
                     help="(default when --real is omitted)")
    args = ap.parse_args()

    print("Command codes used in this project:")
    print_command_table()
    part1_real_bytes_walkthrough()
    if args.real:
        run_real()
    else:
        run_demo()

    print("\nNext: 04_capture_orchestration_and_timing.py - running lesson")
    print("02's UART send and this lesson's UDP receive AT THE SAME TIME,")
    print("and why the order/timing between them is the single biggest")
    print("source of '0 packets received' failures.")
