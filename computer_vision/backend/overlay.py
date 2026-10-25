"""
overlay.py - drawing on numpy images for the live window: lines, boxes, labels in Computer Modern and the house
colours, and the radar top view.

    from backend import overlay as ov
"""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visualization import style



def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


LOSS = _rgb(style.LOSS)
RADAR, TEXT, LIGHT, BLACK, WHITE = (_rgb(style.PRIMARY), _rgb(style.SERIES[2]), _rgb(style.ACCENT), _rgb(style.NORM),
                                    (255, 255, 255))


def rect(img, x0, y0, x1, y1, colour, t: int = 2):
    h, w = img.shape[:2]
    x0, x1 = int(np.clip(min(x0, x1), 0, w - 1)), int(np.clip(max(x0, x1), 0, w - 1))
    y0, y1 = int(np.clip(min(y0, y1), 0, h - 1)), int(np.clip(max(y0, y1), 0, h - 1))
    img[y0:y0 + t, x0:x1 + 1] = colour
    img[max(y1 - t + 1, 0):y1 + 1, x0:x1 + 1] = colour
    img[y0:y1 + 1, x0:x0 + t] = colour
    img[y0:y1 + 1, max(x1 - t + 1, 0):x1 + 1] = colour


def line(img, p0, p1, colour, t: int = 3):
    h, w = img.shape[:2]
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    xs = np.linspace(p0[0], p1[0], n).round().astype(int)
    ys = np.linspace(p0[1], p1[1], n).round().astype(int)
    for dx in range(-(t // 2), t - t // 2):
        ok = (xs + dx >= 0) & (xs + dx < w) & (ys >= 0) & (ys < h)
        img[ys[ok], xs[ok] + dx] = colour


def cross(img, u, v, colour, r: int = 10, t: int = 3):
    line(img, (u - r, v), (u + r, v), colour, t)
    line(img, (u, v - r), (u, v + r), colour, t)

def _font_file() -> str | None:
    if shutil.which("kpsewhich"):
        try:
            p = subprocess.run(["kpsewhich", "cmunrm.otf"], capture_output=True, text=True, timeout=10).stdout.strip()
            if p and Path(p).exists():
                return p
        except (OSError, subprocess.SubprocessError):
            pass
    import matplotlib
    alt = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "cmr10.ttf"
    return str(alt) if alt.exists() else None


@lru_cache(maxsize=None)
def _font(size: int):
    path = _font_file()
    return ImageFont.truetype(path, size) if path else ImageFont.load_default()


UNITS = {"px", "fps", "ms", "m", "cm", "mm", "db", "hz", "s"}


def title_case(s: str, keep=()) -> str:
    s = str(s)
    for i, k in enumerate(keep):
        s = s.replace(str(k), f"§{i}§")
    words = style.caps(s.lower()).split(" ")
    words = [w.lower() if "_" in w or w.lower().strip(".,:;()") in UNITS else w for w in words]
    s = " ".join(words)
    for i, k in enumerate(keep):
        s = s.replace(f"§{i}§", str(k))
    return s


def text(img, s: str, x: int, y: int, colour=BLACK, scale: int = 2, box=WHITE, keep=()):
    s = title_case(s, keep)
    font = _font(max(8, int(round(9 * scale))))
    pad = max(1, scale)
    left, top, right, bottom = font.getbbox(s)
    tw, th = int(right - left) + 2 * pad, int(bottom - min(top, 0)) + 2 * pad
    h, w = img.shape[:2]
    x, y = int(np.clip(x, 0, max(w - tw, 0))), int(np.clip(y, 0, max(h - th, 0)))
    region = Image.fromarray(np.ascontiguousarray(img[y:y + th, x:x + tw]))
    draw = ImageDraw.Draw(region)
    if box is not None:
        draw.rectangle([0, 0, tw, th], fill=tuple(int(c) for c in box))
    draw.text((pad - left, pad - min(top, 0)), s, font=font, fill=tuple(int(c) for c in colour))
    img[y:y + th, x:x + tw] = np.asarray(region)[:img[y:y + th, x:x + tw].shape[0], :img[y:y + th, x:x + tw].shape[1]]

_TOP_VIEW = {}


def top_view(dets, objects, size=(640, 360), r_max: float = 3.0, fov_deg: float = 60.0) -> np.ndarray:
    w, h = size
    s = (h - 30) / r_max
    ox, oy = w / 2, h - 12

    def px(r, th):
        return ox + s * r * np.sin(np.radians(th)), oy - s * r * np.cos(np.radians(th))

    key = (size, r_max, fov_deg)
    if key not in _TOP_VIEW:
        bg = np.full((h, w, 3), 255, np.uint8)
        for r in np.arange(0.5, r_max + 1e-9, 0.5):
            pts = [px(r, a) for a in np.linspace(-fov_deg, fov_deg, 60)]
            for p0, p1 in zip(pts, pts[1:]):
                line(bg, p0, p1, (205, 205, 205), 1)
            text(bg, f"{r:.1f}", *map(int, px(r, fov_deg)), (120, 120, 120), 1, None)
        for a in (-fov_deg, fov_deg):
            line(bg, (ox, oy), px(r_max, a), (205, 205, 205), 1)
        _TOP_VIEW[key] = bg
    img = _TOP_VIEW[key].copy()
    for r, th in dets:
        u, v = px(r, th)
        img[int(max(v - 3, 0)):int(v + 4), int(max(u - 3, 0)):int(u + 4)] = RADAR
    for o in objects:
        u, v = px(o["range_m"], o["azimuth_deg"])
        rect(img, u - 7, v - 7, u + 7, v + 7, BLACK, 2)
        text(img, f"{o['range_m']:.2f}M {o['azimuth_deg']:+.0f}", int(u) + 10, int(v) - 8, BLACK, 1)
    cross(img, ox, oy, BLACK, 6, 2)
    return img
