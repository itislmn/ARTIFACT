"""
tlv.py - the on-chip demo's output on the data UART: frames start with the magic word 02 01 04 03 06 05 08 07, a
40-byte header (version, length, platform, frame, CPU cycles, objects, TLV count, subframe) and TLV blocks; type 1 holds
the points (x, y, z, v as float32), type 2 the range profile (uint16, log2 magnitude in Q9).

    from radar_tutorial.tlv import FrameReader, parse_frame
"""
from __future__ import annotations

import struct

import numpy as np

MAGIC = bytes([2, 1, 4, 3, 6, 5, 8, 7])
HEADER = struct.Struct("<8I")
HEADER_BYTES = 8 + HEADER.size
DATA_BAUD = 921600
TYPES = {1: "detected points: x, y, z (m), v (m/s), float32", 2: "range profile: uint16 log2 magnitude, Q9",
         3: "noise profile: uint16", 4: "azimuth static heat map: complex int16", 5: "range-Doppler heat map: uint16",
         6: "statistics: timing and load", 7: "side info per point: SNR, noise (0.1 dB, int16)",
         8: "azimuth-elevation static heat map", 9: "temperature statistics"}


def parse_frame(buf: bytes) -> tuple[dict | None, int]:
    if len(buf) < HEADER_BYTES:
        return None, 0
    version, length, platform, number, cycles, n_obj, n_tlv, sub = HEADER.unpack(buf[8:HEADER_BYTES])
    if not HEADER_BYTES <= length <= 1 << 20:
        return {"bad": True}, 8
    if len(buf) < length:
        return None, 0
    out = {"frame": number, "n_obj": n_obj, "points": np.zeros((0, 4), np.float32), "range_profile": None,
           "snr_db": None, "types": []}
    i = HEADER_BYTES
    for _ in range(n_tlv):
        if i + 8 > length:
            break
        kind, size = struct.unpack("<2I", buf[i:i + 8])
        payload = buf[i + 8:i + 8 + size]
        out["types"].append(kind)
        if kind == 1:
            out["points"] = np.frombuffer(payload[: len(payload) // 16 * 16], np.float32).reshape(-1, 4)
        elif kind == 2:
            out["range_profile"] = np.frombuffer(payload[: len(payload) // 2 * 2], np.uint16) / 512.0
        elif kind == 7:
            out["snr_db"] = np.frombuffer(payload[: len(payload) // 4 * 4], np.int16).reshape(-1, 2)[:, 0] / 10.0
        i += 8 + size
    return out, length


class FrameReader:

    def __init__(self):
        self.buf = b""

    def feed(self, data: bytes) -> list[dict]:
        self.buf += data
        out = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                self.buf = self.buf[-7:]
                return out
            self.buf = self.buf[i:]
            frame, used = parse_frame(self.buf)
            if frame is None:
                return out
            self.buf = self.buf[used:]
            if not frame.get("bad"):
                out.append(frame)


def build_frame(number: int, points: np.ndarray, profile: np.ndarray) -> bytes:
    tlvs = [(1, np.asarray(points, np.float32).tobytes()), (2, np.round(np.asarray(profile) * 512).astype(np.uint16).tobytes())]
    body = b"".join(struct.pack("<2I", k, len(p)) + p for k, p in tlvs)
    length = HEADER_BYTES + len(body)
    return MAGIC + HEADER.pack(0x03060000, length, 0x2944, number, 0, len(points), len(tlvs), 0) + body
