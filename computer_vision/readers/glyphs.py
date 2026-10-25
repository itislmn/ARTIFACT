"""
glyphs.py - training characters from 27 fonts (10 handwriting) with distortion, and synthetic word scenes with a lamp
veil for the tests; three fonts are held out.

    from readers.glyphs import characters, word_scene, add_veil
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from readers.segment import normalise_char

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FONT_DIR = Path(r"C:\Windows\Fonts")
TRAIN_FONTS = ["arial.ttf", "arialbd.ttf", "ariali.ttf", "times.ttf", "calibri.ttf", "calibrib.ttf", "consola.ttf",
               "verdana.ttf", "tahoma.ttf", "trebuc.ttf", "cour.ttf", "corbel.ttf", "Candara.ttf", "constan.ttf",
               "bahnschrift.ttf", "segoeui.ttf", "impact.ttf",
               "Inkfree.ttf", "segoepr.ttf", "comic.ttf", "comicbd.ttf", "LHANDW.TTF", "ITCKRIST.TTF",
               "TEMPSITC.TTF", "PRISTINA.TTF", "MISTRAL.TTF", "JUICE___.TTF"]
TEST_FONTS = ["segoesc.ttf", "georgia.ttf", "BRADHITC.TTF"]
_FONTS = {}


def font(name: str, size: int):
    from PIL import ImageFont
    key = (name, size)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(str(FONT_DIR / name), size)
    return _FONTS[key]


def render_char(ch: str, font_name: str, size: int = 64) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.new("L", (2 * size, 2 * size), 0)
    ImageDraw.Draw(img).text((size // 2, size // 3), ch, fill=255, font=font(font_name, size))
    return np.asarray(img, np.float32) / 255.0


def _warp(img: np.ndarray, rng) -> np.ndarray:
    h, w = img.shape
    a, sh, sc = np.radians(rng.uniform(-12, 12)), rng.uniform(-0.3, 0.3), rng.uniform(0.75, 1.1)
    A = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([[1, sh], [0, 1]]) / sc
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2, w / 2
    u = A[0, 0] * (xx - cx) + A[0, 1] * (yy - cy) + cx
    v = A[1, 0] * (xx - cx) + A[1, 1] * (yy - cy) + cy
    amp = rng.uniform(0, 3.0)
    g = rng.normal(0, 1, (2, 5, 5))
    k = np.linspace(0, 4, w)
    gx = np.array([np.interp(k, np.arange(5), r) for r in g[0]])
    gy = np.array([np.interp(k, np.arange(5), r) for r in g[1]])
    ky = np.linspace(0, 4, h)
    dx = np.array([np.interp(ky, np.arange(5), c) for c in gx.T]).T
    dy = np.array([np.interp(ky, np.arange(5), c) for c in gy.T]).T
    u, v = u + amp * dx, v + amp * dy
    u0, v0 = np.clip(np.floor(u).astype(int), 0, w - 2), np.clip(np.floor(v).astype(int), 0, h - 2)
    fu, fv = np.clip(u - u0, 0, 1), np.clip(v - v0, 0, 1)
    return ((1 - fu) * (1 - fv) * img[v0, u0] + fu * (1 - fv) * img[v0, u0 + 1]
            + (1 - fu) * fv * img[v0 + 1, u0] + fu * fv * img[v0 + 1, u0 + 1])


def _minmax(img, k, fn):
    p = np.pad(img, k, mode="edge")
    return fn(np.stack([p[i:i + img.shape[0], j:j + img.shape[1]] for i in range(2 * k + 1) for j in range(2 * k + 1)]), axis=0)


def characters(n: int, rng, fonts=TRAIN_FONTS) -> tuple[np.ndarray, np.ndarray]:
    X, y = np.zeros((n, 32, 32), np.float32), rng.integers(0, len(ALPHABET), n)
    for i in range(n):
        img = render_char(ALPHABET[y[i]], fonts[rng.integers(len(fonts))])
        img = _warp(img, rng)
        r = rng.random()
        if r < 0.25:
            img = _minmax(img, 1, np.max)
        elif r < 0.4:
            img = _minmax(img, 1, np.min)
        c = normalise_char(img)
        if rng.random() < 0.15:
            k = rng.integers(0, 32)
            c[:, k:k + 8] *= rng.uniform(0, 0.5)
        X[i] = np.clip(c + rng.normal(0, rng.uniform(0, 0.08), c.shape) * (c > 0), 0, 1)
    return X, y


def word_scene(text: str, font_name: str, letter_px: int, size_wh=(1280, 720), centre=(640, 360),
               paper=170.0, ink=40.0, background=100.0, noise=2.0, rng=None) -> np.ndarray:
    from PIL import Image, ImageDraw
    rng = rng or np.random.default_rng()
    w, h = size_wh
    f = font(font_name, int(letter_px * 1.35))
    l, t, r, b = f.getbbox(text)
    tw, th = r - l, b - t
    sheet_w, sheet_h = int(tw * 1.25) + 20, int(th * 2.2) + 20
    img = np.full((h, w), background, np.float32)
    x0, y0 = int(centre[0] - sheet_w / 2), int(centre[1] - sheet_h / 2)
    img[max(y0, 0):y0 + sheet_h, max(x0, 0):x0 + sheet_w] = paper
    layer = Image.new("L", (w, h), 0)
    ImageDraw.Draw(layer).text((centre[0] - tw / 2 - l, centre[1] - th / 2 - t), text, fill=255, font=f)
    a = np.asarray(layer, np.float32) / 255
    img = img * (1 - a) + ink * a
    img[:, 1:-1] = 0.25 * img[:, :-2] + 0.5 * img[:, 1:-1] + 0.25 * img[:, 2:]
    img[1:-1] = 0.25 * img[:-2] + 0.5 * img[1:-1] + 0.25 * img[2:]
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255)


def add_veil(img, source_uv, level: float, at_uv, core_px: float = 25.0, p: float = 2.0) -> np.ndarray:
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rho = np.maximum(np.hypot(xx - source_uv[0], yy - source_uv[1]), 1.0)
    ref = np.hypot(at_uv[0] - source_uv[0], at_uv[1] - source_uv[1])
    out = np.clip(img + level * (ref / rho) ** p, 0, 255)
    out[rho < core_px] = 255
    return out
