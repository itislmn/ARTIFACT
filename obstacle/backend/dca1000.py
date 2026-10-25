"""
dca1000.py - the DCA1000EVM capture board and the radar's CLI port (estimation's capture constants and commands).

    from backend.dca1000 import DCA1000
"""
from __future__ import annotations

import socket
import struct
import time

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
