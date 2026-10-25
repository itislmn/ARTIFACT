"""
reader_live.py - MetricSight in its own process, so the camera loop never waits for a read.

    from backend.reader_live import LiveReader
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue

import numpy as np


def _worker(cfg: dict, qin, qout):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from readers.metricsight import read
    while True:
        job = qin.get()
        if job is None:
            break
        i, obj, crop, c, f_px, (ox, oy) = job
        rd = read(crop, c, obj["range_m"], f_px, cfg)
        rd.boxes = [(a + ox, b + oy, cc + ox, d + oy) for a, b, cc, d in rd.boxes]
        if rd.line_box is not None:
            a, b, cc, d = rd.line_box
            rd.line_box = (a + ox, b + oy, cc + ox, d + oy)
        qout.put((i, obj, rd))


class LiveReader:
    def __init__(self, cfg: dict):
        self.cfg, self.busy, self.n = cfg, False, 0

    def start(self) -> "LiveReader":
        ctx = mp.get_context("spawn")
        self.qin, self.qout = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_worker, args=(self.cfg, self.qin, self.qout), daemon=True, name="reader")
        self.proc.start()
        return self

    def submit(self, i: int, obj: dict, rgb: np.ndarray, centre_uv, f_px: float) -> bool:
        if self.busy or centre_uv is None:
            return False
        rw, rh = self.cfg["text"]["region_m"]
        s = f_px / obj["range_m"]
        h, w = rgb.shape[:2]
        x0, x1 = int(max(centre_uv[0] - rw / 2 * s - 4, 0)), int(min(centre_uv[0] + rw / 2 * s + 5, w))
        y0, y1 = int(max(centre_uv[1] - rh / 2 * s - 4, 0)), int(min(centre_uv[1] + rh / 2 * s + 5, h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return False
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        self.qin.put((i, dict(obj), crop, (centre_uv[0] - x0, centre_uv[1] - y0), f_px, (x0, y0)))
        self.busy = True
        return True

    def poll(self, objects: list) -> list:
        out = []
        while True:
            try:
                i, obj, rd = self.qout.get_nowait()
            except queue.Empty:
                break
            self.busy = False
            self.n += 1
            if not objects:
                continue
            d = [abs(o["range_m"] - obj["range_m"]) / 0.15 + abs(o["azimuth_deg"] - obj["azimuth_deg"]) / 5 for o in objects]
            k = int(np.argmin(d))
            if d[k] < 1.5:
                out.append((k, rd))
        return out

    def stop(self):
        if hasattr(self, "proc"):
            self.qin.put(None)
            self.proc.join(timeout=3)
            if self.proc.is_alive():
                self.proc.terminate()
