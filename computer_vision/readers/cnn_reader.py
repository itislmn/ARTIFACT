"""
cnn_reader.py - the camera-only reader: Sauvola threshold in image pixels, the shared character cutting, and a
LeNet-5-style CNN in numpy (52 k weights).

    from readers import cnn_reader; cnn_reader.read(rgb, centre_uv, range_m, f_px, cfg)
"""
from __future__ import annotations

import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from backend.settings import MODELS
from readers.metricsight import ALPHABET, MAX_WIDTH, Reading
from readers.segment import box, cut_letters, drop_edges, drop_frame_lines, despeckle, normalise_char, region, text_line

MODEL = MODELS / "cnn_chars.npz"


def conv2d(x, w, b, pad):
    F, C, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(xp, (kh, kw), axis=(2, 3))
    B, _, Ho, Wo = win.shape[:4]
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(B * Ho * Wo, C * kh * kw)
    return (cols @ w.reshape(F, -1).T + b).reshape(B, Ho, Wo, F).transpose(0, 3, 1, 2), cols


def conv2d_back(dout, cols, w, pad, need_dx=True):
    F, C, kh, kw = w.shape
    d = dout.transpose(0, 2, 3, 1).reshape(-1, F)
    dw, db = (d.T @ cols).reshape(w.shape), d.sum(0)
    if not need_dx:
        return None, dw, db
    dx, _ = conv2d(dout, w[:, :, ::-1, ::-1].transpose(1, 0, 2, 3), 0.0, kh - 1 - pad)
    return dx, dw, db


def pool(x):
    B, C, h, w = x.shape
    r = x.reshape(B, C, h // 2, 2, w // 2, 2)
    out = r.max(axis=(3, 5))
    return out, r == out[:, :, :, None, :, None]


def pool_back(dout, mask):
    B, C, h, _, w, _ = mask.shape
    return (mask * dout[:, :, :, None, :, None]).reshape(B, C, 2 * h, 2 * w)


class LeNet:
    def __init__(self, params=None, seed=0):
        rng = np.random.default_rng(seed)

        def he(*shape, fan):
            return rng.normal(0, np.sqrt(2 / fan), shape).astype(np.float32)

        self.p = params or {"c1": he(6, 1, 5, 5, fan=25), "b1": np.zeros(6, np.float32),
                            "c2": he(12, 6, 3, 3, fan=54), "b2": np.zeros(12, np.float32),
                            "d1": he(12 * 8 * 8, 64, fan=768), "e1": np.zeros(64, np.float32),
                            "d2": he(64, 36, fan=64), "e2": np.zeros(36, np.float32)}

    def forward(self, x):
        p = self.p
        a1, c1 = conv2d(x, p["c1"], p["b1"], 2)
        r1 = np.maximum(a1, 0)
        q1, m1 = pool(r1)
        a2, c2 = conv2d(q1, p["c2"], p["b2"], 1)
        r2 = np.maximum(a2, 0)
        q2, m2 = pool(r2)
        f = q2.reshape(len(x), -1)
        h = np.maximum(f @ p["d1"] + p["e1"], 0)
        z = h @ p["d2"] + p["e2"]
        z = z - z.max(1, keepdims=True)
        prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return prob, {"x": x, "a1": a1, "c1": c1, "r1": r1, "m1": m1, "q1": q1, "a2": a2, "c2": c2, "r2": r2,
                      "m2": m2, "q2": q2, "f": f, "h": h, "prob": prob}

    def backward(self, cache, y):
        p, n = self.p, len(y)
        dz = cache["prob"].copy()
        dz[np.arange(n), y] -= 1
        dz /= n
        g = {"d2": cache["h"].T @ dz, "e2": dz.sum(0)}
        dh = (dz @ p["d2"].T) * (cache["h"] > 0)
        g["d1"], g["e1"] = cache["f"].T @ dh, dh.sum(0)
        da2 = pool_back((dh @ p["d1"].T).reshape(cache["q2"].shape), cache["m2"]) * (cache["a2"] > 0)
        dq1, g["c2"], g["b2"] = conv2d_back(da2, cache["c2"], p["c2"], 1)
        da1 = pool_back(dq1, cache["m1"]) * (cache["a1"] > 0)
        _, g["c1"], g["b1"] = conv2d_back(da1, cache["c1"], p["c1"], 2, need_dx=False)
        return g


_NET = None


def n2(r, shape):
    return box(np.ones(shape), r)


def read(rgb, centre_uv, range_m: float, f_px: float, cfg: dict, keep_debug: bool = False) -> Reading:
    global _NET
    if _NET is None:
        if not MODEL.exists():
            raise SystemExit(f"{MODEL} missing: python -m readers.train")
        d = np.load(MODEL)
        _NET = LeNet({k: d[k] for k in ("c1", "b1", "c2", "b2", "d1", "e1", "d2", "e2")})
    t0 = time.perf_counter()
    rw, rh = cfg["text"]["region_m"]
    s = f_px / range_m
    x0, x1 = int(centre_uv[0] - rw / 2 * s), int(centre_uv[0] + rw / 2 * s)
    y0, y1 = int(centre_uv[1] - rh / 2 * s), int(centre_uv[1] + rh / 2 * s)
    W, H = max(x1 - x0, 8), max(y1 - y0, 8)
    luma, _ = region(rgb, x0, y0, x1, y1, W, H, cfg["camera"]["clip_level"])
    r = max(3, H // 8)
    n = box(np.ones_like(luma), r)
    m = box(luma, r) / n
    sd = np.sqrt(np.maximum(box(luma ** 2, r) / n - m ** 2, 0))
    T = m * (1 + 0.2 * (sd / 128 - 1))
    ink = drop_edges((luma < T) & (sd > 8))
    rs = max(2, H // 25)
    ink &= box(ink.astype(float), rs) / n2(rs, ink.shape) < 0.75
    soft = np.clip((T - luma) / np.maximum(sd, 8), 0, 1) * ink
    out = Reading("", [], np.zeros(0), 0.0)
    line = text_line(ink)
    if line is not None:
        a, b = line
        hgt = b - a
        def cls(c):
            pr = _NET.forward(c[:, None].astype(np.float32))[0]
            return pr.argmax(1), np.log(pr.max(1) + 1e-9)

        li = despeckle(drop_frame_lines(ink, line), 0.02 * (line[1] - line[0]) ** 2)
        cols = cut_letters(li, soft[a:b] * li, cls, 0.12 * hgt, MAX_WIDTH * hgt)
        crops = np.array([normalise_char(soft[a:b, c0:c1] * li[:, c0:c1], li[:, c0:c1]) for c0, c1 in cols]) if cols else np.zeros((0, 32, 32))
        if len(crops):
            prob, cache = _NET.forward(crops[:, None].astype(np.float32))
            o = np.sort(prob, axis=1)
            out.text = "".join(ALPHABET[i] for i in prob.argmax(1))
            out.margins = o[:, -1] - o[:, -2]
            if keep_debug:
                out.debug = {"crops": crops, "acts": [{k: cache[k][i] for k in ("x", "r1", "q1", "r2", "q2", "h", "prob")}
                                                      for i in range(len(crops))], "ink": ink, "line": line, "cols": cols}
        out.boxes = [(x0 + c0, y0 + a, x0 + c1, y0 + b) for c0, c1 in cols]
        out.line_box = (x0 + cols[0][0], y0 + a, x0 + cols[-1][1], y0 + b) if cols else None
    out.ms = 1e3 * (time.perf_counter() - t0)
    return out
