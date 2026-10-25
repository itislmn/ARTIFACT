"""
radar_stream.py - the AWR2944PEVM streaming through the DCA1000EVM: cfg over UART, UDP frames reassembled, every
frame labelled with its sub-frame chirp, optional raw recording.

    python -m hardware.radar_stream      (self-test)
"""
from __future__ import annotations

import queue
import socket
import threading
import time
from collections import deque
from pathlib import Path

import paths  # noqa: F401  (import paths of ../estimation, ../computer_vision)
import estimation.capture as cap


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

    def __init__(self, frame_bytes: int, raw_path: Path | None = None):
        super().__init__(daemon=True, name="radar-udp")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        self.sock.settimeout(0.2)
        self.sock.bind((cap.PC_IP, cap.DATA_PORT))
        self.asm = Assembler(frame_bytes)
        self.raw = open(raw_path, "wb") if raw_path else None
        self.index = open(Path(raw_path).with_suffix(".jsonl"), "w", encoding="utf-8") if raw_path else None
        self.raw_q = queue.SimpleQueue()
        self.writer = threading.Thread(target=self._write, name="radar-raw-writer") if raw_path else None
        self.stop_evt, self.cond = threading.Event(), threading.Condition()
        self.frames = self.packets = self.lost = self.dropped = 0
        self.waveform = ""
        self.names = None
        self.accept = True
        self.newest, self._last_seq = None, None
        self.buffer = deque(maxlen=4)
        self.t_last_packet = 0.0

    def run(self):
        import json
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
            self.t_last_packet = time.perf_counter()
            seq = int.from_bytes(pkt[0:4], "little")
            if self._last_seq is not None and seq > self._last_seq + 1:
                self.lost += seq - self._last_seq - 1
            self._last_seq = seq
            if not self.accept:
                continue
            for fidx, frame in self.asm.push(int.from_bytes(pkt[4:10], "little"), view[cap.HEADER_BYTES:n]):
                t = time.perf_counter()
                name = self.names[fidx % len(self.names)] if self.names else self.waveform
                if self.writer:
                    self.raw_q.put((frame, json.dumps({"frame": self.frames, "t": round(t, 6), "waveform": name,
                                                       "stream_frame": fidx}) + "\n"))
                with self.cond:
                    self.newest = (self.frames, t, frame, name)
                    if len(self.buffer) == self.buffer.maxlen:
                        self.dropped += 1
                    self.buffer.append(self.newest)
                    self.frames += 1
                    self.cond.notify_all()
        self.sock.close()
        if self.writer:
            self.raw_q.put(None)
            self.writer.join()

    def _write(self):
        while (item := self.raw_q.get()) is not None:
            self.raw.write(item[0])
            self.index.write(item[1])
        self.raw.close()
        self.index.close()

    def take(self, after: int, timeout: float) -> list:
        with self.cond:
            self.cond.wait_for(lambda: any(f[0] > after for f in self.buffer), timeout)
            return [f for f in self.buffer if f[0] > after]

    def take_newest(self, after: int, timeout: float):
        with self.cond:
            self.cond.wait_for(lambda: self.newest is not None and self.newest[0] > after, timeout)
            return self.newest if self.newest is not None and self.newest[0] > after else None


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
        self.timing = []
        for raw in open(cfg_path, encoding="utf-8", errors="ignore"):
            line = raw.strip()
            if not line or line.startswith(("%", "#")):
                continue
            if line.startswith("sensorStart") and before_start is not None:
                before_start()
            t_line = time.perf_counter()
            reply = self.send(line, timeout=5.0 if line.startswith("sensorStart") else 2.0)
            self.timing.append((line.split()[0], round(time.perf_counter() - t_line, 4)))
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
    def __init__(self, port: str, waveform, raw_path: Path | None = None, log=print, library: dict | None = None):
        self.port, self.wf, self.raw_path, self.log = port, waveform, raw_path, log
        self.lib = library
        self.dca = self.rx = self.uart = None
        self.changes = []
        self.mode, self.names, self.fallback_reason = None, [], ""

    def arm(self, mode: str = "single") -> "RadarStream":
        cap.sensor_stop(self.port)
        self.dca = cap.DCA1000()
        self.dca.start()
        self.rx = Receiver(self.wf.cfg.wire_bytes_per_frame(True), self.raw_path)
        self.rx.waveform = self.wf.name
        self.rx.start()
        time.sleep(0.3)
        self.uart = Uart(self.port)
        names = sorted(self.lib, key=lambda n: self.lib[n].subframe) if self.lib else []
        if mode == "subframes" and len(names) > 1:
            path = self.lib[names[0]].path.parent / "subframes.cfg"
            try:
                def start():
                    self.rx.names = list(names)
                    self.rx.asm.resync()
                s = self.uart.send_cfg(path, self.log, before_start=start)
                self.log(f"  [radar] {path.name}: every line 'Done' ({s:.2f} s over UART)")
                rate = self._rate(2.0)
                expect = len(names) / self.lib[names[0]].frame_period_s
                if abs(rate - expect) > 0.25 * expect:
                    raise cap.CaptureError(f"{rate:.1f} frames/s instead of {expect:.0f}: the sub-frames do not all stream")
                self.mode, self.names = "subframes", list(names)
            except cap.CaptureError as e:
                self.fallback_reason = str(e)
                self.log(f"  [radar] advanced frame NOT usable ({e}) -> the verified single chirp '{self.wf.name}'")
                self.rx.names = None
                self.uart.send("sensorStop")
                time.sleep(0.3)
        if self.mode is None:
            s = self.uart.send_cfg(self.wf.path, self.log, before_start=self.rx.asm.resync)
            self.mode, self.names = "single", [self.wf.name]
            self.log(f"  [radar] armed with waveform '{self.wf.name}' ({s:.2f} s over UART)")
        return self

    def _rate(self, seconds: float) -> float:
        n0, t0 = self.rx.frames, time.perf_counter()
        time.sleep(seconds)
        return (self.rx.frames - n0) / (time.perf_counter() - t0)

    def take(self, after: int, timeout: float = 0.5) -> list:
        return self.rx.take(after, timeout)

    def newest(self, after: int, timeout: float = 0.5):
        return self.rx.take_newest(after, timeout)

    def reconfigure(self, wf) -> float:
        self.uart.send("sensorStop")
        time.sleep(max(1.5 * self.wf.frame_period_s, 0.05))
        self.rx.accept = False
        t_last = self.rx.newest[1] if self.rx.newest else time.perf_counter()
        n_before = self.rx.frames

        def start_new():
            self.rx.asm.resync()
            self.rx.waveform = wf.name
            self.rx.accept = True

        def start_old():
            self.rx.asm.resync()
            self.rx.waveform = self.wf.name
            self.rx.accept = True

        try:
            self.uart.send_cfg(wf.path, self.log, before_start=start_new)
        except Exception:
            self.log(f"  [radar] change to '{wf.name}' FAILED - restoring '{self.wf.name}'")
            self.rx.accept = False
            self.uart.send("sensorStop")
            self.uart.send_cfg(self.wf.path, self.log, before_start=start_old)
            raise
        self.wf = wf
        t0 = time.perf_counter()
        while self.rx.frames == n_before and time.perf_counter() - t0 < 5.0:
            time.sleep(0.005)
        gap = (self.rx.newest[1] if self.rx.frames > n_before else time.perf_counter()) - t_last
        lines = self.uart.timing
        start = sum(sec for cmd, sec in lines if cmd == "sensorStart")
        self.changes.append({"to": wf.name, "gap_s": round(gap, 3), "t": time.perf_counter(),
                             "uart_lines_s": round(sum(sec for _, sec in lines) - start, 3), "sensorStart_s": round(start, 3),
                             "n_lines": len(lines), "lines": lines})
        self.log(f"  [radar] waveform -> '{wf.name}': blind for {gap:.2f} s")
        return gap

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


def selftest() -> bool:
    import numpy as np
    fb, pay = 4096, 1456
    rng = np.random.default_rng(0)
    old = [rng.integers(0, 255, fb, dtype=np.uint8).tobytes() for _ in range(4)]
    new = [rng.integers(0, 255, fb, dtype=np.uint8).tobytes() for _ in range(3)]
    asm, got, count = Assembler(fb), [], 0
    stream = b"".join(old)[: 3 * fb + fb // 3]
    for k in range(0, len(stream), pay):
        got += asm.push(count, stream[k:k + pay])
        count += len(stream[k:k + pay])
    asm.resync()
    stream2 = b"".join(new)
    for k in range(0, len(stream2), pay):
        got += asm.push(count, stream2[k:k + pay])
        count += len(stream2[k:k + pay])
    ok = [f for _, f in got] == old[:3] + new and [i for i, _ in got] == [0, 1, 2, 0, 1, 2]
    print(f"  {'PASS' if ok else 'FAIL'}  {len(got)} frames delivered (expected 6: 3 old, the cut one dropped, 3 new), "
          f"{asm.incomplete} incomplete")
    asm, got, count = Assembler(fb), [], 0
    frames = [rng.integers(0, 255, fb, dtype=np.uint8).tobytes() for _ in range(6)]
    stream = b"".join(frames)
    for k in range(0, len(stream), pay):
        if fb * 2 <= k < fb * 2 + pay:
            count += len(stream[k:k + pay])
            continue
        got += asm.push(count, stream[k:k + pay])
        count += len(stream[k:k + pay])
    names = ["search", "fast", "track"]
    labels = [(names[i % 3], f) for i, f in got]
    ok2 = [i for i, _ in got] == [0, 1, 3, 4, 5] and all(f == frames[i] for i, f in got) \
        and [n for n, _ in labels] == ["search", "fast", "search", "fast", "track"]
    print(f"  {'PASS' if ok2 else 'FAIL'}  sub-frame labels survive a lost packet: "
          f"{[n for n, _ in labels]} (stream frames {[i for i, _ in got]})")
    return ok and ok2


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
