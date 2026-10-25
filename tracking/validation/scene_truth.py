"""
scene_truth.py - what moves in every camera frame: pixels farther than 5 robust standard deviations from the take's
median image, cleaned of blobs smaller than a fifth of the ball; per frame the foreground pixels of every image
column (for the detection FDR) and the person's box (every large blob outside the ball's box widened by one ball
radius for the paper label; the nearest frame with the ball within one second when the label hides it).

    cols, person = motion(take, ball_rows, cache_dir)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SCALE = 4


def motion(take: Path, ball: list[dict], cache_dir: Path, force: bool = False):
    cache = Path(cache_dir) / f"{Path(take).name}_motion.npz"
    if cache.exists() and not force:
        z = np.load(cache)
        return z["cols"], z["person"]
    cap = cv2.VideoCapture(str(Path(take) / "camera.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (f.shape[1] // SCALE, f.shape[0] // SCALE), interpolation=cv2.INTER_AREA))
    F = np.array(frames, np.int16)
    bg = np.median(F[::max(len(F) // 300, 1)], axis=0)
    diff = np.abs(F - bg).max(axis=3)
    thr = 5 * max(1.4826 * np.median(diff), 1.0)
    r_ball = float(np.nanmedian([g["r_px"] for g in ball if g["found"]])) / SCALE
    a_min, a_person = np.pi * r_ball ** 2 / 5, np.pi * r_ball ** 2
    h, w = F.shape[1:3]
    found = np.array([k for k, g in enumerate(ball) if g["found"]])
    cols = np.zeros((len(F), w), np.uint16)
    person = np.full((len(F), 5), np.nan)
    for k in range(len(F)):
        m = cv2.morphologyEx((diff[k] > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        keep = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= a_min])
        cols[k] = keep.sum(axis=0)
        near = found[np.argmin(np.abs(found - k))] if len(found) else -1
        g = ball[near] if near >= 0 and abs(near - k) <= 30 else {"found": 0}
        free = keep.copy()
        if g["found"]:
            pad = SCALE * r_ball
            u0, v0 = int(max((g["u0"] - pad) // SCALE, 0)), int(max((g["v0"] - pad) // SCALE, 0))
            free[:max(int((g["v1"] + pad) // SCALE) + 1, 0), u0:max(int((g["u1"] + pad) // SCALE) + 1, 0)] = False
        n, lab, st, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8))
        big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= a_person]
        if big:
            x0 = min(st[i, 0] for i in big)
            y0 = min(st[i, 1] for i in big)
            x1 = max(st[i, 0] + st[i, 2] for i in big)
            y1 = max(st[i, 1] + st[i, 3] for i in big)
            person[k] = [SCALE * x0, SCALE * y0, SCALE * x1, SCALE * y1, float(x0 <= 1 or x1 >= w - 1)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, cols=cols, person=person, thr=thr)
    return cols, person
