r"""
LESSON 11 - The OTHER acquisition path: on-chip detection over UART (TLV)

    python 11_uart_pointcloud_tlv.py --demo
    python 11_uart_pointcloud_tlv.py --cli COM5 --baud 921600

No hardware needed for --demo. Real mode needs the chip's DATA port
(often a DIFFERENT COM port and a DIFFERENT baud rate than the CLI port
lessons 02/04 use - see the note below), and only works while the
chip's own on-chip detection chain is enabled (guiMonitor / cfarCfg /
etc. in the .cfg - lesson 01 covered these as "harmless if left at
defaults"; THIS lesson is where they stop being harmless-and-ignored
and actually matter).

------------------------------------------------------------------------
TWO COMPLETELY DIFFERENT WAYS TO GET DATA OUT OF THIS RADAR
------------------------------------------------------------------------
Lessons 03/04/06/07/08 covered path A: raw ADC samples, via the
DCA1000, that YOU run range/Doppler/CFAR/angle math on yourself. This
lesson covers path B: the chip's OWN on-chip DSP does that same math
internally and reports just the RESULTS (a small list of detected
points: x, y, z, velocity) over a plain UART connection - no DCA1000
needed at all.

    Path A (raw, via DCA1000)         Path B (points, via UART)
    --------------------------        --------------------------
    Every ADC sample, full cube       Just the detected points
    You choose the DSP algorithm      Chip's built-in DSP algorithm
    Needs DCA1000 + Ethernet          Needs only the radar's own USB
    Large data volumes                Tiny data volumes
    What lessons 03-08 taught         What THIS lesson teaches

Neither is "better" - they're for different jobs. Path B is what this
project used for the very first working demo (before the DCA1000 side
was even debugged), and it's still the right choice whenever you just
need "where are the objects", not the raw signal.

------------------------------------------------------------------------
THE TWO COM PORTS, AND THE BAUD RATE GOTCHA THAT BIT THIS PROJECT
------------------------------------------------------------------------
The radar's USB connection enumerates as TWO serial ports (lesson 00):
the CLI port (config commands, lessons 01/02, 115200 baud) and the DATA
port (point-cloud/TLV output, THIS lesson). These are NOT the same
port and not the same baud rate. This project hit exactly this
confusion early on: pointing the point-cloud reader at the wrong port,
or the right port at the wrong baud, looks identical from the outside -
you just never see the magic word, forever. The data port's baud rate
for this project's firmware was 921600 - much faster than the CLI
port's 115200, which makes sense (it's streaming structured results
continuously, not waiting on typed commands). If you genuinely don't
know which of your two COM ports is which, or the right baud: try the
CLI port commands (lesson 02) against each port number until one
replies to "sensorStop" with "Done" - that one is the CLI port, so the
other is the data port.

------------------------------------------------------------------------
FRAME FORMAT - CONFIRMED AGAINST TI'S OWN DOCUMENTATION
------------------------------------------------------------------------
Every frame starts with a fixed, unmistakable magic word - 8 bytes,
always the same, regardless of chip/SDK version - which is also your
resync point if you ever start reading mid-stream (see read_frames()
below): the parser SEARCHES for this pattern rather than assuming
byte 0 of whatever you read is automatically the start of a frame.

    magic word (8 bytes): 02 01 04 03 06 05 08 07
    (this is 4 uint16 values {0x0102, 0x0304, 0x0506, 0x0708}, each
    stored little-endian - a very widely-used, stable constant across
    the whole TI mmWave product line, not something this project is
    guessing at)

Followed by a fixed-size header (9 uint32 fields including the magic
word's 8 bytes = 40 bytes by the field list TI's own documentation
gives - that same document's own summary text says "44 bytes", which
doesn't match its own field list; rather than pick one, the code below
computes the header size explicitly from the fields it actually reads,
and cross-checks against the self-describing `totalPacketLen` field in
every real frame instead of trusting either number blindly):

    magicWord          8 bytes
    version             4 bytes (uint32)
    totalPacketLen      4 bytes (uint32) - the WHOLE frame's length,
                                            including this header
    platform            4 bytes (uint32)
    frameNumber         4 bytes (uint32)
    timeCpuCycles       4 bytes (uint32)
    numDetectedObj      4 bytes (uint32)
    numTLVs             4 bytes (uint32)
    subFrameNumber      4 bytes (uint32)
                     = 40 bytes total

...then `numTLVs` TLV blocks, each:

    type    4 bytes (uint32)
    length  4 bytes (uint32) - length of the PAYLOAD that follows,
                                 not including this 8-byte TLV header
    payload `length` bytes

------------------------------------------------------------------------
TLV TYPE 1 - DETECTED POINTS (what this lesson decodes)
------------------------------------------------------------------------
TI's own "Out-Of-Box demo" documentation (matching this project's
actual firmware, confirmed from a real boot log earlier in this
project: "AWR2X44P MMW Demo") gives TLV type 1 as an array of detected
points, 16 bytes each:

    x         4 bytes (float32, meters)
    y         4 bytes (float32, meters)
    z         4 bytes (float32, meters)
    doppler   4 bytes (float32, m/s)

A DIFFERENT TI demo (the separate "People Counting" firmware, NOT what
this project flashed) uses a different scheme entirely - TLV types
6/7/8, with 16-byte records of (range, azimuth, doppler, snr) instead
of (x, y, z, doppler). If your board is running that demo instead, the
byte SIZES are the same but the FIELD MEANINGS are not - the code below
prints the raw TLV type number it sees so you can tell which one you're
actually looking at rather than assuming.
"""
import argparse
import struct
import time

import numpy as np

MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])
HEADER_FIELDS = ["version", "totalPacketLen", "platform", "frameNumber",
                  "timeCpuCycles", "numDetectedObj", "numTLVs", "subFrameNumber"]
HEADER_SIZE = len(MAGIC_WORD) + 4 * len(HEADER_FIELDS)   # 8 + 32 = 40, computed not assumed
TLV_HEADER_SIZE = 8
DETECTED_POINTS_TYPE = 1
POINT_SIZE = 16   # x, y, z, doppler - each float32


# =======================================================================
# BUILD a synthetic frame (demo mode) - the exact inverse of parsing, so
# the parser below can be proven against known input.
# =======================================================================
def build_frame(points, frame_number=1):
    """points: list of (x, y, z, doppler) tuples."""
    num_tlvs = 1
    tlv_payload = b"".join(struct.pack("<ffff", *p) for p in points)
    tlv = struct.pack("<II", DETECTED_POINTS_TYPE, len(tlv_payload)) + tlv_payload
    total_len = HEADER_SIZE + len(tlv)
    header = (MAGIC_WORD
              + struct.pack("<IIIIIIII", 1, total_len, 0, frame_number, 0,
                             len(points), num_tlvs, 0))
    assert len(header) == HEADER_SIZE
    return header + tlv


# =======================================================================
# PARSE - resyncs on the magic word, validates totalPacketLen, decodes
# TLV type 1 if present, reports (without guessing) anything else.
# =======================================================================
def find_next_frame(buf: bytes, start=0):
    idx = buf.find(MAGIC_WORD, start)
    return idx


def parse_frame(buf: bytes, offset: int):
    """Returns (frame_dict_or_None, next_offset_to_resume_scanning_from).
    None means "not enough bytes yet for a complete frame" - caller
    should read more and retry from the SAME offset, not skip forward
    (that would silently drop a real, just-not-fully-arrived frame)."""
    if len(buf) - offset < HEADER_SIZE:
        return None, offset
    header_vals = struct.unpack("<8I", buf[offset + 8: offset + HEADER_SIZE])
    header = dict(zip(HEADER_FIELDS, header_vals))
    total_len = header["totalPacketLen"]
    if total_len < HEADER_SIZE or total_len > 10_000_000:
        # Not a real frame (or we resynced on coincidental magic-word-
        # looking bytes inside other data) - skip past just the magic
        # word and let the caller re-search from there.
        return "resync", offset + len(MAGIC_WORD)
    if len(buf) - offset < total_len:
        return None, offset   # frame not fully arrived yet

    body = buf[offset + HEADER_SIZE: offset + total_len]
    points = []
    other_tlvs = []
    pos = 0
    for _ in range(header["numTLVs"]):
        if pos + TLV_HEADER_SIZE > len(body):
            break
        tlv_type, tlv_len = struct.unpack("<II", body[pos:pos + TLV_HEADER_SIZE])
        payload = body[pos + TLV_HEADER_SIZE: pos + TLV_HEADER_SIZE + tlv_len]
        if tlv_type == DETECTED_POINTS_TYPE:
            n_pts = tlv_len // POINT_SIZE
            for i in range(n_pts):
                x, y, z, dop = struct.unpack(
                    "<ffff", payload[i * POINT_SIZE:(i + 1) * POINT_SIZE])
                points.append((x, y, z, dop))
        else:
            other_tlvs.append((tlv_type, tlv_len))
        pos += TLV_HEADER_SIZE + tlv_len

    frame = {"frame_number": header["frameNumber"],
              "num_detected_obj": header["numDetectedObj"],
              "points": points, "other_tlvs": other_tlvs}
    return frame, offset + total_len


def read_frames(buf: bytes):
    """Yields every complete frame found in buf, resyncing past garbage
    as needed. Returns the leftover unparsed tail (bytes not yet a
    complete frame) so a streaming caller can prepend it to the next
    read - exactly the pattern a real serial-port reader needs, since a
    frame will routinely arrive split across two separate ser.read()
    calls."""
    offset = 0
    frames = []
    while True:
        start = find_next_frame(buf, offset)
        if start == -1:
            return frames, buf[offset:]
        result, next_offset = parse_frame(buf, start)
        if result is None:
            return frames, buf[start:]        # incomplete, wait for more bytes
        if result == "resync":
            offset = next_offset
            continue
        frames.append(result)
        offset = next_offset


# =======================================================================
# DEMO
# =======================================================================
def run_demo():
    print(f"HEADER_SIZE computed from fields = {HEADER_SIZE} bytes "
          "(8 magic + 8x uint32)")
    true_points = [(1.2, 0.5, 0.0, 0.3), (-0.8, 2.1, 0.1, -0.5)]
    frame_bytes = build_frame(true_points, frame_number=42)
    print(f"\nBuilt one synthetic frame: {len(frame_bytes)} bytes, "
          f"{len(true_points)} points")

    # Simulate a stream arriving in two ragged chunks, like a real UART
    # read would - proves the resync/incomplete-frame handling actually
    # works, not just the happy path of "one clean read = one frame".
    split = len(frame_bytes) // 3
    chunk1, chunk2 = frame_bytes[:split], frame_bytes[split:]
    print(f"Simulating it arriving in 2 ragged chunks ({len(chunk1)} + "
          f"{len(chunk2)} bytes)...")

    buf = b""
    buf += chunk1
    frames, buf = read_frames(buf)
    print(f"After chunk 1: {len(frames)} complete frame(s) "
          f"(expected 0 - frame isn't fully here yet)")
    assert len(frames) == 0

    buf += chunk2
    frames, buf = read_frames(buf)
    print(f"After chunk 2: {len(frames)} complete frame(s) (expected 1)")
    assert len(frames) == 1

    f = frames[0]
    print(f"\nDecoded frame #{f['frame_number']}: "
          f"{f['num_detected_obj']} objects reported, "
          f"{len(f['points'])} points actually decoded")
    for i, (x, y, z, dop) in enumerate(f["points"]):
        print(f"  point {i}: x={x:+.2f} y={y:+.2f} z={z:+.2f} "
              f"doppler={dop:+.2f} m/s")
    # float32 round-trip, not bit-identical to the original float64
    # literals - compare with a tolerance, not "==".
    assert len(f["points"]) == len(true_points)
    for got, want in zip(f["points"], true_points):
        assert all(abs(g - w) < 1e-5 for g, w in zip(got, want)), (got, want)
    print("\nDecoded points match the encoded ground truth (within "
          "float32 precision).")

    # A FULL (8-byte) but bogus magic word - random bytes right after it
    # decode to a nonsense totalPacketLen, which is exactly what
    # exercises the "resync" branch in parse_frame(): a real magic word
    # match that turns out not to be a real frame at all (coincidental
    # bytes, or a stream you started reading mid-frame). This is a
    # DIFFERENT case from just skipping unrelated junk (find_next_frame
    # handles that on its own via bytes.find) - it's specifically about
    # recovering after matching the magic word incorrectly.
    bogus_after_magic = MAGIC_WORD + b"\xff" * 32
    noisy = b"\x00\x11\x22" + bogus_after_magic + frame_bytes
    frames2, _ = read_frames(noisy)
    print(f"\nWith a full-but-bogus magic word match before the real "
          f"frame: still found {len(frames2)} frame(s) (the resync "
          "branch had to fire and recover).")
    assert len(frames2) == 1
    for got, want in zip(frames2[0]["points"], true_points):
        assert all(abs(g - w) < 1e-5 for g, w in zip(got, want))


def run_real(cli_port, baud):
    import serial
    print(f"Reading from {cli_port} at {baud} baud (this must be the "
          "DATA port, not the CLI port - see the module docstring)...")
    buf = b""
    with serial.Serial(cli_port, baud, timeout=0.5) as ser:
        t0 = time.time()
        while time.time() - t0 < 10:
            chunk = ser.read(4096)
            if chunk:
                buf += chunk
                frames, buf = read_frames(buf)
                for f in frames:
                    print(f"frame #{f['frame_number']}: "
                          f"{len(f['points'])} points, "
                          f"other TLVs seen: {f['other_tlvs']}")
                    for x, y, z, dop in f["points"]:
                        print(f"    x={x:+.2f} y={y:+.2f} z={z:+.2f} "
                              f"doppler={dop:+.2f}")
    if not buf and time.time() - t0 >= 10:
        print("\nNo magic word ever appeared. Check: is this really the "
              "DATA port (not the CLI port)? Is the baud rate right? Is "
              "the chip's on-chip detection chain actually enabled and "
              "sensorStart already sent (lesson 02)?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=None, help="The DATA port, e.g. COM5")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.cli:
        run_real(args.cli, args.baud)
    else:
        run_demo()

    print("\nNext: 12_saving_and_visualizing_the_cube.py - the other half "
          "of 'using the data cube': saving captures to disk and turning "
          "them into actual plots.")
