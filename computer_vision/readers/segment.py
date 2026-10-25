"""
segment.py - the character cutting both readers share: region resampling, local contrast, text line, cuts at gaps or by
recognition (dynamic programming), 32x32 normalisation.

    from readers.segment import cut_letters, normalise_char
"""
from __future__ import annotations

import numpy as np

N = 32


def box(x: np.ndarray, r: int) -> np.ndarray:
    p = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(r + 1, r), (r + 1, r)])
    c = p.cumsum(-2).cumsum(-1)
    n = 2 * r + 1
    return c[..., n:, n:] - c[..., :-n, n:] - c[..., n:, :-n] + c[..., :-n, :-n]


def resampler(pos: np.ndarray, n_src: int) -> tuple[np.ndarray, int]:
    lo = int(np.clip(np.floor(pos.min()), 0, n_src - 1))
    hi = int(np.clip(np.floor(pos.max()) + 2, lo + 1, n_src))
    i0 = np.floor(pos).astype(int)
    a = pos - i0
    M = np.zeros((pos.size, hi - lo))
    rows = np.arange(pos.size)
    for idx, wt in ((i0, 1 - a), (i0 + 1, a)):
        ok = (idx >= lo) & (idx < hi) & (pos >= 0) & (pos <= n_src - 1)
        M[rows[ok], idx[ok] - lo] += wt[ok]
    return M, lo


def region(rgb: np.ndarray, x0, y0, x1, y1, out_w: int, out_h: int, clip_level: float = 250.0):
    Mx, lx = resampler(np.linspace(x0, x1, out_w * 2).reshape(-1), rgb.shape[1])
    My, ly = resampler(np.linspace(y0, y1, out_h * 2).reshape(-1), rgb.shape[0])
    Mx = Mx.reshape(out_w, 2, -1).mean(1)
    My = My.reshape(out_h, 2, -1).mean(1)
    reg = rgb[ly:ly + My.shape[1], lx:lx + Mx.shape[1]]
    Y = reg[..., 0] * 0.299 + reg[..., 1] * 0.587 + reg[..., 2] * 0.114 if reg.ndim == 3 else reg.astype(float)
    luma = My @ Y @ Mx.T
    clip = My @ (Y >= clip_level).astype(float) @ Mx.T
    inside = (My.sum(1)[:, None] > 0.99) & (Mx.sum(1)[None, :] > 0.99)
    return luma, np.where(inside, clip, 1.0)


def local_contrast(luma: np.ndarray, clip: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    blown = box(clip, r) / box(np.ones_like(clip), r) > 0.999
    wgt = (~blown).astype(float)
    sw, s1, s2 = box(np.stack([wgt, luma * wgt, luma ** 2 * wgt]), r)
    mu = s1 / (sw + 1e-9)
    sd = np.sqrt(np.maximum(s2 / (sw + 1e-9) - mu ** 2, 1e-6))
    return (luma - mu) / sd * wgt, wgt


def otsu(x: np.ndarray) -> float:
    h, e = np.histogram(x, 256)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def drop_edges(ink: np.ndarray, frac: float = 0.6) -> np.ndarray:
    ink = ink.copy()
    ink[:, ink.mean(0) > frac] = False
    ink[ink.mean(1) > frac] = False
    return ink


def drop_frame_lines(ink: np.ndarray, line: tuple[int, int]) -> np.ndarray:
    a, b = line
    h = b - a
    k = max(2, h // 3)
    up = ink[max(a - k, 0):a].mean(0) if a > 0 else np.zeros(ink.shape[1])
    down = ink[b:b + k].mean(0) if b < len(ink) else np.zeros(ink.shape[1])
    out = ink[a:b].copy()
    out[:, (up > 0.5) & (down > 0.5)] = False
    return out


def despeckle(ink: np.ndarray, min_area: float) -> np.ndarray:
    h, w = ink.shape
    label = np.where(ink, np.arange(1, h * w + 1).reshape(h, w), 0)
    while True:
        p = np.pad(label, 1)
        m = label.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                m = np.maximum(m, p[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx])
        m = np.where(ink, m, 0)
        if np.array_equal(m, label):
            break
        label = m
    ids, area = np.unique(label[ink], return_counts=True)
    return ink & np.isin(label, ids[area >= min_area])


def text_line(ink: np.ndarray) -> tuple[int, int] | None:
    prof = ink.sum(1)
    if prof.max() <= 0:
        return None
    on = prof > 0.1 * prof.max()
    best, start = None, None
    for i, v in enumerate(np.r_[on, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def characters(ink: np.ndarray, min_gap: float, min_width: float, max_width: float, typical: float) -> list[tuple[int, int]]:
    prof = ink.sum(0)
    cols = np.flatnonzero(prof > 0)
    if not cols.size:
        return []
    runs, s, last = [], cols[0], cols[0]
    for c in cols[1:]:
        if c - last > min_gap:
            runs.append((s, last + 1))
            s = c
        last = c
    runs.append((s, last + 1))
    out = []
    for a, b in runs:
        w = b - a
        if w > max_width:
            k = max(2, int(round(w / typical)))
            cuts = [a]
            for j in range(1, k):
                c0 = a + int(j * w / k)
                lo, hi = max(c0 - int(typical / 4), a + 1), min(c0 + int(typical / 4), b - 1)
                cuts.append(lo + int(np.argmin(prof[lo:hi])) if hi > lo else c0)
            cuts.append(b)
            out += [(cuts[j], cuts[j + 1]) for j in range(k)]
        else:
            out.append((a, b))
    h, w = ink.shape
    keep = []
    for a, b in out:
        if b - a < min_width and ink[:, a:b].any(1).sum() < h / 2:
            continue
        if a == 0 or b == w:
            continue
        if ink[:, a:b].mean() > 0.7:
            continue
        keep.append((a, b))
    return keep


def _bilinear(x: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    m = x.shape[axis]
    pos = np.linspace(0, m - 1, n)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, m - 1)
    a = pos - i0
    if axis == 0:
        return x[i0] * (1 - a)[:, None] + x[i1] * a[:, None]
    return x[:, i0] * (1 - a) + x[:, i1] * a


def normalise_char(ink: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = ink > 0.5 * max(float(ink.max()), 1e-6)
    ys, xs = np.nonzero(mask)
    if not ys.size:
        return np.zeros((N, N), np.float32)
    ink = ink * mask
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = (N - 4) / max(h, w)
    oh, ow = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = _bilinear(_bilinear(crop, oh), ow, axis=1)
    out = np.zeros((N, N), np.float32)
    y0, x0 = (N - oh) // 2, (N - ow) // 2
    out[y0:y0 + oh, x0:x0 + ow] = small[:oh, :ow]
    m = out.max()
    return out / m if m > 0 else out


def cut_letters(ink: np.ndarray, soft: np.ndarray, classify, min_w: float, max_w: float) -> list[tuple[int, int]]:
    h = ink.shape[0]
    cols = characters(ink, min_gap=max(1, int(0.04 * h)), min_width=min_w, max_width=10 * max_w, typical=0.7 * h)
    if cols and all(c1 - c0 <= max_w for c0, c1 in cols):
        return cols
    return [(c0, c1) for c0, c1, _, _ in decode(ink, soft, classify, min_w, max_w)]


def decode(ink: np.ndarray, soft: np.ndarray, classify, min_w: float, max_w: float, bonus: float = 0.0):
    p = ink.sum(0).astype(float)
    W = len(p)
    if not p.any():
        return []
    nz = p[p > 0]
    low = 0.5 * np.median(nz)
    cand = [0, W]
    for c in range(1, W - 1):
        if (p[c] == 0) != (p[c - 1] == 0) or (p[c] <= low and p[c] <= p[c - 1] and p[c] <= p[c + 1]):
            cand.append(c)
    cand = sorted(set(cand))
    cuts = [cand[0]]
    for c in cand[1:]:
        if c - cuts[-1] >= max(2, 0.08 * min_w / 0.15) or c == W:
            cuts.append(c)
    segs = []
    for i in range(len(cuts) - 1):
        for j in range(i + 1, min(i + 5, len(cuts))):
            cols = np.flatnonzero(p[cuts[i]:cuts[j]] > 0)
            if not cols.size:
                continue
            a, b = cuts[i] + cols[0], cuts[i] + cols[-1] + 1
            if a == 0 or b == W or not (min_w <= b - a <= max_w) or ink[:, a:b].mean() > 0.7:
                continue
            segs.append((i, j, a, b))
    if not segs:
        return []
    crops = np.array([normalise_char(soft[:, a:b], ink[:, a:b]) for _, _, a, b in segs])
    labels, logp = classify(crops)
    best = [-np.inf] * len(cuts)
    back = [None] * len(cuts)
    best[0] = 0.0
    by_end = {}
    for k, (i, j, a, b) in enumerate(segs):
        by_end.setdefault(j, []).append(k)
    for j in range(1, len(cuts)):
        if best[j - 1] > -np.inf and not p[cuts[j - 1]:cuts[j]].any():
            best[j], back[j] = best[j - 1], ("gap", j - 1)
        for k in by_end.get(j, []):
            i = segs[k][0]
            if best[i] > -np.inf and best[i] + logp[k] + bonus > best[j]:
                best[j], back[j] = best[i] + logp[k] + bonus, ("char", k)
        if best[j] == -np.inf and best[j - 1] > -np.inf:
            best[j], back[j] = best[j - 1] - 1.0, ("gap", j - 1)
    out, j = [], len(cuts) - 1
    while j > 0 and back[j] is not None:
        kind, v = back[j]
        if kind == "char":
            i, _, a, b = segs[v]
            out.append((a, b, int(labels[v]), float(logp[v])))
            j = i
        else:
            j = v
    return out[::-1]
