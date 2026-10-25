"""
metricsight.py - MetricSight: the radar range turns the text region into a metric image (1.5 mm per pixel), censored
ink, physically plausible line, closed-form ridge classifier.

    from readers import metricsight; metricsight.read(rgb, centre_uv, range_m, f_px, cfg)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from methods.reader.segment import cut_letters, drop_edges, drop_frame_lines, despeckle, local_contrast, normalise_char, region, text_line

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MODEL = Path(__file__).resolve().parent / "metricsight_ridge.npz"
MAX_WIDTH = 1.1


@dataclass
class Reading:
    text: str
    boxes: list
    margins: np.ndarray
    ms: float
    line_box: tuple | None = None
    debug: dict = field(default_factory=dict)


def features(chars: np.ndarray) -> np.ndarray:
    c = np.asarray(chars, np.float32)
    gy, gx = np.gradient(c, axis=(1, 2))
    mag, ang = np.hypot(gx, gy), np.mod(np.arctan2(gy, gx), np.pi)
    b = np.minimum((ang / np.pi * 8).astype(int), 7)
    n = len(c)
    hog = np.zeros((n, 8, 8, 8), np.float32)
    for k in range(8):
        hog[..., k] = (mag * (b == k)).reshape(n, 8, 4, 8, 4).sum(axis=(2, 4))
    dens = c.reshape(n, 8, 4, 8, 4).mean(axis=(2, 4))
    f = np.concatenate([hog.reshape(n, -1), dens.reshape(n, -1)], axis=1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-6)


def fit_ridge(X, y, lam):
    F = np.c_[features(X), np.ones(len(X))]
    Y = -np.ones((len(y), len(ALPHABET)), np.float32)
    Y[np.arange(len(y)), y] = 1
    return np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ Y)


def classify(W, chars) -> tuple[np.ndarray, np.ndarray]:
    s = np.c_[features(chars), np.ones(len(chars))] @ W
    o = np.sort(s, axis=1)
    return s.argmax(1), o[:, -1] - o[:, -2]


def log_probs(W, chars, t: float = 8.0):
    s = t * (np.c_[features(chars), np.ones(len(chars))] @ W)
    s -= s.max(1, keepdims=True)
    lp = s - np.log(np.exp(s).sum(1, keepdims=True))
    return lp.argmax(1), lp.max(1)


_W = None


def read(rgb, centre_uv, range_m: float, f_px: float, cfg: dict, keep_debug: bool = False) -> Reading:
    global _W
    if _W is None:
        if not MODEL.exists():
            raise SystemExit(f"{MODEL} missing: python -m readers.train")
        _W = np.load(MODEL)["W"]
    t0 = time.perf_counter()
    t = cfg["text"]
    mm = t["mm_per_px"]
    rw, rh = t["region_m"]
    s = f_px / range_m
    x0, x1 = centre_uv[0] - rw / 2 * s, centre_uv[0] + rw / 2 * s
    y0, y1 = centre_uv[1] - rh / 2 * s, centre_uv[1] + rh / 2 * s
    W, H = int(rw * 1000 / mm), int(rh * 1000 / mm)
    luma, clip = region(rgb, x0, y0, x1, y1, W, H, cfg["camera"]["clip_level"])
    norm, wgt = local_contrast(luma, clip, int(t["window_mm"] / mm / 2))
    ink = drop_edges((norm < -1.0) & (wgt > 0))
    soft = np.clip(-norm / 2, 0, 1) * wgt * ink
    out = Reading("", [], np.zeros(0), 0.0)
    line = text_line(ink)
    if line is not None and 10 <= (line[1] - line[0]) * mm <= 200:
        a, b = line
        hgt = b - a
        li = despeckle(drop_frame_lines(ink, line), 0.02 * (line[1] - line[0]) ** 2)
        cols = cut_letters(li, soft[a:b] * li, lambda c: log_probs(_W, c), 0.12 * hgt, MAX_WIDTH * hgt)
        crops = np.array([normalise_char(soft[a:b, c0:c1] * li[:, c0:c1], li[:, c0:c1]) for c0, c1 in cols]) if cols else np.zeros((0, 32, 32))
        if len(crops):
            k, margin = classify(_W, crops)
            out.text = "".join(ALPHABET[i] for i in k)
            out.margins = margin
        sx, sy = (x1 - x0) / W, (y1 - y0) / H
        out.boxes = [(x0 + c0 * sx, y0 + a * sy, x0 + c1 * sx, y0 + b * sy) for c0, c1 in cols]
        out.line_box = (x0 + (cols[0][0] if cols else 0) * sx, y0 + a * sy, x0 + (cols[-1][1] if cols else W) * sx, y0 + b * sy)
        if keep_debug:
            out.debug = {"luma": luma, "norm": norm, "ink": ink, "line": line, "cols": cols, "crops": crops,
                         "features": features(crops) if len(crops) else None,
                         "scores": np.c_[features(crops), np.ones(len(crops))] @ _W if len(crops) else None}
    out.ms = 1e3 * (time.perf_counter() - t0)
    return out
