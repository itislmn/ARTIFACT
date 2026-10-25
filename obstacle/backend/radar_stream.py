"""
radar_stream.py - the AWR2944PEVM streaming through the DCA1000EVM with every chirp preloaded as a sub-frame of one
advanced frame (no blind time); a recorded take plays through the same interface.

    python -m backend.radar_stream      (self-test)
"""
from __future__ import annotations

import json
import queue
import socket
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from backend import dca1000 as cap


class Assembler:

    def __init__(self, frame_bytes: int):
        self.fb = frame_bytes
        self.buf = bytearray(frame_bytes)
        self.offset, self.cur, self.filled = 0, None, 0
        self.incomplete = 0
        self._resync = False

    def resync(self):
        self._resync = True

    def push(self, byte_count: int, payload) -> list:
        if self._resync:
            self.offset, self.cur, self.filled, self._resync = byte_count, None, 0, False
        out = []
        pos, end, src = byte_count - self.offset, byte_count - self.offset + len(payload), 0
        if pos < 0:
            return out
        while pos < end:
            fidx, off = divmod(pos, self.fb)
            n = min(end - pos, self.fb - off)
            if self.cur is None or fidx > self.cur:
                if self.cur is not None and self.filled:
                    self.incomplete += 1
                self.cur, self.filled = fidx, 0
            if fidx == self.cur:
                self.buf[off:off + n] = payload[src:src + n]
                self.filled += n
                if self.filled >= self.fb:
                    out.append((fidx, bytes(self.buf)))
                    self.cur, self.filled = fidx + 1, 0
            pos += n
            src += n
        return out


class Receiver(threading.Thread):

    def __init__(self, frame_bytes: int, raw_path: Path | None = None, raw_chirps=("search",)):
        super().__init__(daemon=True, name="radar-udp")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        self.sock.settimeout(0.2)
        self.sock.bind((cap.PC_IP, cap.DATA_PORT))
        self.asm = Assembler(frame_bytes)
        self.raw = open(raw_path, "wb") if raw_path else None
        self.index = open(Path(raw_path).with_suffix(".jsonl"), "w", encoding="utf-8") if raw_path else None
        self.raw_chirps = set(raw_chirps)
        self.raw_q = queue.SimpleQueue()
        self.writer = threading.Thread(target=self._write, name="radar-raw-writer") if raw_path else None
        self.stop_evt, self.cond = threading.Event(), threading.Condition()
        self.frames = self.packets = self.lost = self.dropped = 0
        self.names = ["search"]
        self.buffer = deque(maxlen=4)
        self.newest_t = 0.0
        self._last_seq = None

    def run(self):
        pkt = bytearray(2048)
        view = memoryview(pkt)
        if self.writer:
            self.writer.start()
        while not self.stop_evt.is_set():
            try:
                n = self.sock.recv_into(pkt)
            except socket.timeout:
                continue
            except OSError:
                break
            if n <= cap.HEADER_BYTES:
                continue
            self.packets += 1
            seq = int.from_bytes(pkt[0:4], "little")
            if self._last_seq is not None and seq > self._last_seq + 1:
                self.lost += seq - self._last_seq - 1
            self._last_seq = seq
            for fidx, frame in self.asm.push(int.from_bytes(pkt[4:10], "little"), view[cap.HEADER_BYTES:n]):
                t = time.perf_counter()
                name = self.names[fidx % len(self.names)]
                if self.writer and name in self.raw_chirps:
                    self.raw_q.put((frame, json.dumps({"frame": fidx, "t": round(t, 6), "chirp": name}) + "\n"))
                with self.cond:
                    if len(self.buffer) == self.buffer.maxlen:
                        self.dropped += 1
                    self.buffer.append((fidx, t, frame, name))
                    self.frames += 1
                    self.newest_t = t
                    self.cond.notify_all()
        self.sock.close()
        if self.writer:
            self.raw_q.put(None)
            self.writer.join()

    def _write(self):
        while (item := self.raw_q.get()) is not None:
            self.raw.write(np.frombuffer(item[0], "<i2").reshape(-1, 4)[:, :2].tobytes())
            self.index.write(item[1])
        self.raw.close()
        self.index.close()

    def take(self, after: int, timeout: float) -> list:
        with self.cond:
            self.cond.wait_for(lambda: any(f[0] > after for f in self.buffer), timeout)
            return [f for f in self.buffer if f[0] > after]


class Uart:

    def __init__(self, port: str, baud: int = 115200):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.02)

    def send(self, line: str, timeout: float = 2.0) -> str:
        self.ser.reset_input_buffer()
        self.ser.write((line + "\n").encode())
        buf, t0 = b"", time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            buf += self.ser.read(256)
            text = buf.decode(errors="ignore")
            if "Done" in text or "Error" in text or "not recognized" in text:
                time.sleep(0.005)
                buf += self.ser.read(256)
                return buf.decode(errors="ignore")
        return buf.decode(errors="ignore")

    def send_cfg(self, cfg_path: Path, log=print, before_start=None) -> float:
        t0 = time.perf_counter()
        for raw in open(cfg_path, encoding="utf-8", errors="ignore"):
            line = raw.strip()
            if not line or line.startswith(("%", "#")):
                continue
            if line.startswith("sensorStart") and before_start is not None:
                before_start()
            reply = self.send(line, timeout=5.0 if line.startswith("sensorStart") else 2.0)
            if "Done" not in reply:
                if line.startswith("sensorStop") or line.startswith("flushCfg"):
                    log(f"  [radar] '{line}' -> {reply.strip()[:60] or 'no reply'} (allowed)")
                    continue
                raise cap.CaptureError(f"the radar rejected '{line}': {reply.strip()[:80] or 'no reply'}")
        return time.perf_counter() - t0

    def close(self):
        try:
            self.ser.close()
        except Exception:                                      # noqa: BLE001
            pass


class RadarStream:
    def __init__(self, port: str, library, raw_path: Path | None = None, raw_chirps=("search",), log=print):
        self.port, self.lib, self.raw_path, self.raw_chirps, self.log = port, library, raw_path, raw_chirps, log
        self.dca = self.rx = self.uart = None
        self.mode, self.names, self.fallback_reason = None, [], ""

    def arm(self, mode: str = "subframes") -> "RadarStream":
        cap.sensor_stop(self.port)
        self.dca = cap.DCA1000()
        self.dca.start()
        self.rx = Receiver(self.lib.first.cfg.wire_bytes_per_frame(True), self.raw_path, self.raw_chirps)
        self.rx.start()
        time.sleep(0.3)
        self.uart = Uart(self.port)
        if mode == "subframes" and len(self.lib.names) > 1:
            try:
                self._arm(self.lib.subframe_path, self.lib.names)
                rate = self._rate(2.0)
                expect = len(self.lib.names) * 1000.0 / self.lib.frame_ms
                if abs(rate - expect) > 0.25 * expect:
                    raise cap.CaptureError(f"{rate:.1f} frames/s instead of {expect:.0f}: the sub-frames do not all stream")
                self.mode = "subframes"
            except cap.CaptureError as e:
                self.fallback_reason = str(e)
                self.log(f"  [radar] advanced frame NOT usable ({e}) -> the verified single chirp '{self.lib.first.name}'")
                self.uart.send("sensorStop")
                time.sleep(0.3)
        if self.mode is None:
            self._arm(self.lib.first.path, [self.lib.first.name])
            self.mode = "single"
        self.log(f"  [radar] armed: {self.mode} ({', '.join(self.names)}), {self._rate(1.0):.1f} frames/s")
        return self

    def _arm(self, cfg_path: Path, names: list[str]):
        def start():
            self.rx.names = list(names)
            self.rx.asm.resync()
        s = self.uart.send_cfg(cfg_path, self.log, before_start=start)
        self.names = list(names)
        self.log(f"  [radar] {cfg_path.name}: every line 'Done' ({s:.2f} s over UART)")

    def _rate(self, seconds: float) -> float:
        n0, t0 = self.rx.frames, time.perf_counter()
        time.sleep(seconds)
        return (self.rx.frames - n0) / (time.perf_counter() - t0)

    def take(self, after: int, timeout: float = 0.5) -> list:
        return self.rx.take(after, timeout)

    def disarm(self):
        try:
            if self.uart is not None:
                self.uart.send("sensorStop")
                self.uart.close()
            else:
                cap.sensor_stop(self.port)
        except Exception:                                      # noqa: BLE001
            pass
        if self.rx is not None:
            self.rx.stop_evt.set()
            self.rx.join(timeout=2)
        if self.dca is not None:
            self.dca.stop()


class FileStream:

    class _Rx:
        frames = packets = lost = dropped = 0

        class asm:
            incomplete = 0

    def __init__(self, folder: Path, library, log=print):
        self.folder, self.lib, self.log = Path(folder), library, log
        self.index = [json.loads(line) for line in open(self.folder / "radar_raw.jsonl")]
        self.words = np.memmap(self.folder / "radar_raw.bin", dtype="<i2", mode="r")
        full = library.first.cfg.wire_bytes_per_frame(True) // 2
        self.padded = self.words.size >= full * len(self.index) - 1
        self.per = full if self.padded else full // 2
        self.rx = self._Rx()
        self.mode, self.names, self.fallback_reason = "file", [n for n in library.names if n in {r.get("chirp", library.first.name) for r in self.index}] or [library.first.name], ""
        self.k, self.t_start = 0, None

    def arm(self, mode: str = "") -> "FileStream":
        self.t_start = time.perf_counter()
        self.log(f"FILE {self.folder.name}: {len(self.index)} frames ({'padded' if self.padded else 'unpadded'}) at their recorded pace")
        return self

    def take(self, after: int, timeout: float = 0.5) -> list:
        t_rec0 = self.index[0]["t"]
        out, t_end = [], time.perf_counter() + timeout
        while not out and time.perf_counter() < t_end and self.k < len(self.index):
            r = self.index[self.k]
            if time.perf_counter() - self.t_start >= r["t"] - t_rec0:
                w = np.asarray(self.words[self.k * self.per:(self.k + 1) * self.per])
                if not self.padded:
                    g = np.empty((w.size // 2, 4), "<i2")
                    g[:, :2], g[:, 2], g[:, 3] = w.reshape(-1, 2), -1, 0
                    w = g.ravel()
                name = r.get("chirp", r.get("waveform", self.lib.first.name))
                out.append((self.k, time.perf_counter(), w.tobytes(), name if name in self.lib.names else self.lib.first.name))
                self.k += 1
                self.rx.frames += 1
            else:
                time.sleep(0.002)
        return out

    def disarm(self):
        pass


def selftest() -> bool:
    fb, pay = 4096, 1456
    rng = np.random.default_rng(0)
    sent = [rng.integers(0, 255, fb, dtype=np.uint8).tobytes() for _ in range(8)]
    stream = b"".join(sent)
    asm, got, count = Assembler(fb), [], 0
    asm.resync()
    for k in range(0, len(stream), pay):
        piece = stream[k:k + pay]
        if not (2 * fb + 100 <= k < 2 * fb + 100 + pay):
            got += asm.push(count, piece)
        count += len(piece)
    names = ["search", "fast", "track"]
    labels = [(i, names[i % 3]) for i, _ in got]
    ok = [i for i, _ in got] == [0, 1, 3, 4, 5, 6, 7] and all(f == sent[i] for i, f in got) and \
        [n for _, n in labels] == ["search", "fast", "search", "fast", "track", "search", "fast"]
    print(f"  {'PASS' if ok else 'FAIL'}  frames delivered {[i for i, _ in got]} (frame 2 had a lost packet), labels "
          f"{[n[0] for _, n in labels]}, {asm.incomplete} incomplete")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
