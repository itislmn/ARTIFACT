"""
reader_process.py - the text reader in its own process, so the camera loop never waits for a read (shared with
obstacle/).

    from hardware.reader_process import LiveReader
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
from collections import Counter, deque


def _reader_module():
    try:
        from methods.reader import metricsight
    except ImportError:
        from readers import metricsight
    return metricsight


def _worker(rcfg: dict, r_obj: float, qin, qout):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from methods.text_reading import read_patch
    reader = _reader_module()
    while True:
        job = qin.get()
        if job is None:
            break
        oid, range_m, patch, c, f_px, info = job
        rd = read_patch(reader, patch, c, range_m, f_px, rcfg, r_obj)
        rd.debug = {}
        qout.put((oid, rd, info))


class LiveReader:
    def __init__(self, rcfg: dict, r_obj: float):
        self.rcfg, self.r_obj, self.busy, self.n = rcfg, float(r_obj), False, 0

    def start(self) -> "LiveReader":
        ctx = mp.get_context("spawn")
        self.qin, self.qout = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_worker, args=(self.rcfg, self.r_obj, self.qin, self.qout), daemon=True, name="reader")
        self.proc.start()
        return self

    def submit(self, oid: int, range_m: float, rgb, centre_uv, f_px: float, info: dict | None = None) -> bool:
        if self.busy or centre_uv is None or range_m <= 0.05:
            return False
        from methods.text_reading import patch_around
        patch, c = patch_around(rgb, centre_uv, range_m, f_px, self.rcfg, self.r_obj)
        if patch.shape[0] < 8 or patch.shape[1] < 8:
            return False
        self.qin.put((oid, float(range_m), patch, c, f_px, info or {}))
        self.busy = True
        return True

    def poll(self) -> list:
        out = []
        while True:
            try:
                item = self.qout.get_nowait()
            except queue.Empty:
                break
            self.busy = False
            self.n += 1
            out.append(item)
        return out

    def stop(self):
        if hasattr(self, "proc"):
            self.qin.put(None)
            self.proc.join(timeout=3)
            if self.proc.is_alive():
                self.proc.terminate()


def correct_chars(text: str, truth: str) -> int:
    d = list(range(len(truth) + 1))
    for i, x in enumerate(text, 1):
        prev, d[0] = d[:], i
        for j, y in enumerate(truth, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (x != y))
    return max(len(truth) - d[-1], 0)


class Stable:

    def __init__(self, n: int = 15):
        self.last = deque(maxlen=n)

    def add(self, text: str) -> str:
        if text:
            self.last.append(text)
        return self.text

    @property
    def text(self) -> str:
        return Counter(self.last).most_common(1)[0][0] if self.last else ""
