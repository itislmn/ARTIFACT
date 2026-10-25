"""
pipeline.py - one live frame: radar bars, glare fit in light mode, MetricSight at the radar objects, the stable reading
and the drawn result view.

    from backend import pipeline
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from backend import glare as gl
from backend import overlay as ov
from readers.metricsight import read


@dataclass
class Result:
    bars: list
    clipped: np.ndarray
    glare: gl.Glare | None = None
    readings: dict = field(default_factory=dict)
    ms: dict = field(default_factory=dict)


def fast_luma(rgb: np.ndarray) -> np.ndarray:
    return ((rgb[..., 0].astype(np.uint16) * 77 + rgb[..., 1].astype(np.uint16) * 150
             + rgb[..., 2].astype(np.uint16) * 29) >> 8).astype(np.float32)


def analyse(rgb, objects, proj, cfg, mode: str, refit: bool = True, glare_prev: gl.Glare | None = None,
            do_read: bool = True, readings_prev: dict | None = None, keep_debug: bool = False,
            source: dict | None = None, read_only: set | None = None) -> Result:
    t0 = time.perf_counter()
    clip_level = cfg["camera"]["clip_level"]
    bars = [proj.bar(o["range_m"], o["azimuth_deg"], cfg["alignment"]["bar_z_m"]) for o in objects]
    res = Result(bars, None)
    t1 = time.perf_counter()
    if mode == "light":
        if refit:
            clipped = res.clipped = fast_luma(rgb) >= clip_level
            hw = [proj.f * cfg["light"]["source_size_m"] / 2 / o["range_m"] for o in objects]
            if source is not None:
                u, v = source["anchor_uv"]
                r = 3 * hw[source["object"]]
                y0, x0 = int(max(v - r, 0)), int(max(u - r, 0))
                ys, xs = np.nonzero(clipped[y0:int(v + r) + 1, x0:int(u + r) + 1])
                core = float(np.hypot(xs + x0 - u, ys + y0 - v).max()) if ys.size else 0.0
                res.glare = gl.fit_at(rgb, clipped, source["object"], (u, v), core, gl.bar_mask(clipped.shape, bars, hw))
            else:
                res.glare = gl.analyse(rgb, clipped, bars, hw)
        else:
            res.glare = glare_prev
    t2 = time.perf_counter()
    if do_read:
        t = cfg["text"]
        for i, o in enumerate(objects):
            if res.glare is not None and i == res.glare.source:
                continue
            if read_only is not None and i not in read_only:
                if readings_prev and i in readings_prev:
                    res.readings[i] = readings_prev[i]
                continue
            c = proj.point(o["range_m"], o["azimuth_deg"], t["centre_z_m"])
            if c is None:
                continue
            res.readings[i] = read(rgb, c, o["range_m"], proj.f, cfg, keep_debug=keep_debug)
    else:
        res.readings = readings_prev or {}
    t3 = time.perf_counter()
    res.ms = {"luma_bars": 1e3 * (t1 - t0), "glare": 1e3 * (t2 - t1), "read": 1e3 * (t3 - t2)}
    return res


def result_view(rgb, res: Result | None, objects, cfg, truth: str = "", scale: float = 1.0) -> np.ndarray:
    step = round(1 / scale)
    small = rgb if step == 1 else rgb[::step, ::step]
    if res is not None and res.glare is not None:
        clipped = fast_luma(small) >= cfg["camera"]["clip_level"]
        out = gl.remove(small, res.glare, clipped, scale)
    else:
        out = np.array(small)
    if res is not None:
        draw(out, res, objects, scale, truth)
    return out


def text_object(readings: dict):
    if not readings:
        return None
    return max(readings.items(), key=lambda kv: (len(kv[1].text) > 0, kv[1].margins.mean() if len(kv[1].margins) else 0))[0]


def read_plan(readings: dict, objects: list, tick: int, skip=None) -> tuple[int | None, int]:
    cand = [i for i in range(len(objects)) if i != skip]
    if not cand:
        return None, tick + 1
    best = text_object(readings)
    if best in cand and tick % 2 == 0:
        return best, tick + 1
    return cand[(tick // 2) % len(cand)], tick + 1


class Stable:

    def __init__(self, n: int = 15):
        from collections import deque
        self.last = deque(maxlen=n)

    def add(self, text: str) -> str:
        if text:
            self.last.append(text)
        return self.text

    @property
    def text(self) -> str:
        from collections import Counter
        return Counter(self.last).most_common(1)[0][0] if self.last else ""


def correct_chars(text: str, truth: str) -> int:
    d = list(range(len(truth) + 1))
    for i, x in enumerate(text, 1):
        prev, d[0] = d[:], i
        for j, y in enumerate(truth, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (x != y))
    return max(len(truth) - d[-1], 0)


def draw(img: np.ndarray, res: Result, objects: list, scale: float, truth: str = "") -> np.ndarray:
    s = max(1, round(2 * scale + 0.5))
    for i, (o, bar) in enumerate(zip(objects, res.bars)):
        if bar is None:
            continue
        pts = bar * scale
        is_light = res.glare is not None and i == res.glare.source
        for p0, p1 in zip(pts[:-1], pts[1:]):
            ov.line(img, p0, p1, ov.RADAR, t=max(2, round(4 * scale)))
        seen = pts[(pts[:, 1] >= 0) & (pts[:, 1] < img.shape[0])]
        u1, v1 = seen[np.argmin(seen[:, 1])] if len(seen) else pts[-1]
        ov.text(img, f"{o['range_m']:.2f} M {o['azimuth_deg']:+.0f}", u1 + 6, v1 + 6, ov.RADAR, s)
        if is_light:
            gu, gv = res.glare.centre_uv
            ov.cross(img, gu * scale, gv * scale, ov.LIGHT, r=round(14 * scale) + 4, t=max(2, round(4 * scale)))
            ov.text(img, f"LIGHT SOURCE  VEIL P {res.glare.fits[1][2]:.1f}", gu * scale + 12, gv * scale + 12, ov.LIGHT, s)
    for i, rd in res.readings.items():
        if rd.line_box is None:
            continue
        for (a, b, c, d) in rd.boxes:
            ov.rect(img, a * scale, b * scale, c * scale, d * scale, ov.TEXT, t=max(1, round(2 * scale)))
        x0, y0, x1, y1 = rd.line_box
        label = f"READ {rd.text}" + (f"  MARGIN {rd.margins.min():.2f}" if len(rd.margins) else "")
        if truth:
            label += f"  {correct_chars(rd.text, truth)}/{len(truth)}"
        ov.text(img, label, x0 * scale, y1 * scale + 6, ov.TEXT, s, keep=(rd.text,))
    return img
