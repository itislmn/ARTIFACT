"""
dca1000.py - the DCA1000EVM's binary UDP protocol: commands to 192.168.33.180:4096 framed as 0xA55A, code, length,
data, 0xEEAA (the reply carries the status in bytes 4-5), and the raw stream on port 4098 as packets of a 4-byte
sequence number, a 6-byte byte count and the ADC payload.

    from radar_tutorial.dca1000 import Dca1000, arm, receive
"""
from __future__ import annotations

import socket
import struct
import threading
import time

DCA_IP, PC_IP = "192.168.33.180", "192.168.33.30"
CONFIG_PORT, DATA_PORT = 4096, 4098
HEADER, FOOTER = 0xA55A, 0xEEAA
COMMANDS = {
    0x01: ("RESET_FPGA", "reset the FPGA logic"),
    0x03: ("CONFIG_FPGA_GEN", "6 bytes: raw mode, LVDS lanes, LVDS capture, Ethernet stream, 16 bit, timer"),
    0x05: ("RECORD_START", "forward every LVDS sample to UDP port 4098"),
    0x06: ("RECORD_STOP", "stop forwarding"),
    0x09: ("SYSTEM_CONNECT", "check that the card answers"),
    0x0B: ("CONFIG_PACKET_DATA", "3 uint16: payload bytes per packet, inter-packet delay, reserved"),
    0x0E: ("READ_FPGA_VERSION", "the FPGA version in the status field"),
}
CODE = {name: code for code, (name, _) in COMMANDS.items()}
RECORDED_REPLIES = ["5aa509000000aaee", "5aa501000000aaee", "5aa503000000aaee", "5aa50b000000aaee",
                    "5aa505000000aaee"]
PACKET_BYTES, PACKET_DELAY = 1466, 25


def command(code: int, data: bytes = b"") -> bytes:
    return struct.pack("<HHH", HEADER, code, len(data)) + data + struct.pack("<H", FOOTER)


def decode_reply(reply: bytes) -> dict:
    h, code, status, f = struct.unpack("<HHHH", reply[:8])
    return {"ok": h == HEADER and f == FOOTER and status == 0, "command": COMMANDS.get(code, ("?",))[0],
            "status": status, "footer_as_status": f}


def fpga_config(lvds_lanes=2, bits=16, timer_s=30) -> bytes:
    return bytes([1, 1 if lvds_lanes == 4 else 2, 1, 2, {12: 1, 14: 2, 16: 3}[bits], timer_s])


class Dca1000:

    def __init__(self, sock=None):
        if sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(3.0)
            sock.bind((PC_IP, CONFIG_PORT))
        self.sock = sock

    def send(self, name: str, data: bytes = b"") -> dict:
        self.sock.sendto(command(CODE[name], data), (DCA_IP, CONFIG_PORT))
        return decode_reply(self.sock.recvfrom(2048)[0])

    def close(self):
        self.sock.close()


def arm(card: Dca1000, lvds_lanes=2, log=print) -> list[dict]:
    steps = [("SYSTEM_CONNECT", b""), ("READ_FPGA_VERSION", b""), ("RESET_FPGA", b""),
             ("CONFIG_FPGA_GEN", fpga_config(lvds_lanes)),
             ("CONFIG_PACKET_DATA", struct.pack("<HHH", PACKET_BYTES, PACKET_DELAY, 0)), ("RECORD_START", b"")]
    out = []
    for name, data in steps:
        r = card.send(name, data)
        out.append(r)
        if log:
            log(f"  {name:20s} status {r['status']}")
        if name == "RESET_FPGA":
            time.sleep(0.3)
        if not r["ok"] and name != "READ_FPGA_VERSION":
            raise RuntimeError(f"{name} failed: {r}")
    return out


def stop(card: Dca1000) -> dict:
    return card.send("RECORD_STOP")


def data_socket(timeout=0.5):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 24)
    s.settimeout(timeout)
    s.bind((PC_IP, DATA_PORT))
    return s


def split_packet(packet: bytes) -> tuple[int, int, bytes]:
    seq = struct.unpack("<I", packet[:4])[0]
    offset = int.from_bytes(packet[4:10], "little")
    return seq, offset, packet[10:]


def receive(recv, seconds, sink, stop_event=None) -> dict:
    n, total, lost, last = 0, 0, 0, None
    t0 = time.time()
    while time.time() - t0 < seconds and not (stop_event and stop_event.is_set()):
        try:
            seq, offset, payload = split_packet(recv())
        except socket.timeout:
            continue
        if last is not None and seq > last + 1:
            lost += seq - last - 1
        last = seq
        n, total = n + 1, total + len(payload)
        sink(offset, payload)
    return {"packets": n, "bytes": total, "lost_packets": lost}


class FrameAssembler:

    def __init__(self, bytes_per_frame: int, on_frame):
        self.size, self.on_frame = bytes_per_frame, on_frame
        self.frame, self.index = bytearray(bytes_per_frame), 0

    def __call__(self, offset: int, payload: bytes):
        pos, chunk = offset, memoryview(payload)
        while len(chunk):
            k = pos // self.size
            while k > self.index:
                self.on_frame(self.index, bytes(self.frame))
                self.frame, self.index = bytearray(self.size), self.index + 1
            start = pos - k * self.size
            take = min(len(chunk), self.size - start)
            if k == self.index:
                self.frame[start:start + take] = chunk[:take]
            chunk, pos = chunk[take:], pos + take

    def flush(self):
        if any(self.frame):
            self.on_frame(self.index, bytes(self.frame))


class FakeDca1000:

    def __init__(self, adc_bytes=b"", start_delay_s=0.0):
        self.adc, self.delay = adc_bytes, start_delay_s
        self.replies, self.packets, self.recording = [], [], False
        self.lock = threading.Lock()

    def sendto(self, frame, addr):
        code = struct.unpack("<H", frame[2:4])[0]
        if code == CODE["RECORD_START"]:
            self.recording = True
            threading.Thread(target=self._stream, daemon=True).start()
        elif code == CODE["RECORD_STOP"]:
            self.recording = False
        with self.lock:
            self.replies.append(struct.pack("<HHHH", HEADER, code, 0, FOOTER))

    def recvfrom(self, n):
        with self.lock:
            return self.replies.pop(0), (DCA_IP, CONFIG_PORT)

    def close(self):
        self.recording = False

    def _stream(self):
        time.sleep(self.delay)
        for seq, i in enumerate(range(0, len(self.adc), PACKET_BYTES), start=1):
            if not self.recording:
                break
            with self.lock:
                self.packets.append(struct.pack("<I", seq) + i.to_bytes(6, "little") + self.adc[i:i + PACKET_BYTES])

    def recv_data(self):
        t0 = time.time()
        while time.time() - t0 < 0.05:
            with self.lock:
                if self.packets:
                    return self.packets.pop(0)
            time.sleep(0.001)
        raise socket.timeout()
