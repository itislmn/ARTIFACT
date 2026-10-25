"""
uart.py - the radar's CLI over UART: every .cfg line is sent as ASCII text at 115200 baud and the firmware answers
"Done" or an error; a reply counts as complete once the port has been quiet for 0.15 s (read(n) alone waits for n bytes).

    from radar_tutorial.uart import send_cfg, quiet_read
"""
from __future__ import annotations

import threading
import time

CLI_BAUD = 115200


def quiet_read(port, quiet_s=0.15, max_s=2.0) -> bytes:
    buf, t0 = b"", time.time()
    last = t0
    while time.time() - t0 < max_s:
        chunk = port.read(256)
        if chunk:
            buf, last = buf + chunk, time.time()
        elif time.time() - last > quiet_s:
            break
    return buf


def naive_read(port, n=400) -> bytes:
    return port.read(n)


def send_cfg(port, lines, reader=quiet_read, log=print) -> list[tuple[str, bool, float]]:
    port.write(b"\n")
    time.sleep(0.05)
    port.reset_input_buffer()
    out = []
    for line in lines:
        t0 = time.time()
        port.write((line + "\n").encode())
        reply = reader(port).decode(errors="ignore")
        ok = "Done" in reply or line.startswith("sensorStop") and "Ignored" in reply
        out.append((line, ok, time.time() - t0))
        if log:
            log(f"  {line[:60]:60s} {'Done' if ok else reply.strip()[-60:] or 'no reply'}")
        if not ok and not line.startswith("sensorStop"):
            break
    return out


def open_cli(port_name):
    import serial
    return serial.Serial(port_name, CLI_BAUD, timeout=0.05)


class FakeCli:

    def __init__(self, timeout=0.05, reply_s=0.01, bad_lines=()):
        self.timeout, self.reply_s, self.bad = timeout, reply_s, set(bad_lines)
        self.pending, self.lock = b"", threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def reset_input_buffer(self):
        with self.lock:
            self.pending = b""

    def write(self, data):
        text = data.decode(errors="ignore").strip()
        if text:
            threading.Timer(self.reply_s, self._answer, args=(text,)).start()
        return len(data)

    def _answer(self, line):
        reply = "Error: command not recognized" if line in self.bad else "Done"
        with self.lock:
            self.pending += f"{line}\r\n{reply}\r\nmmwDemo:/>".encode()

    def read(self, n):
        t0 = time.time()
        while len(self.pending) < n and time.time() - t0 < self.timeout:
            time.sleep(0.002)
        with self.lock:
            chunk, self.pending = self.pending[:n], self.pending[n:]
        return chunk
