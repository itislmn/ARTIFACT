"""
ball_truth.py - where the ball is in every camera frame, from colour alone (yellow, round, not in the static
background); validation only, the radar never sees it.

    rows = ball_truth(take_folder, cache_dir)
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

HUE, SAT, VAL = (20, 45), 50, 80
STATIC_SHARE = 0.7
SCALE = 2
MIN_AREA_PX = 400


def video_index(take: Path) -> list[dict]:
    rows = list(csv.DictReader(open(take / "frames.csv", encoding="utf-8")))
    idx = np.array([int(r["camera_frame"]) for r in rows])
    t = np.array([float(r["t_capture_s"]) for r in rows])
    n = int(cv2.VideoCapture(str(take / "camera.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for j in range(n):
        k = int(np.searchsorted(idx, idx[0] + j, side="right") - 1)
        out.append({"video_frame": j, "camera_frame": int(idx[k]), "t": float(t[k]), "row": k,
                    "repeat": int(idx[k] != idx[0] + j)})
    return out


def yellow(img_small: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (HUE[0], SAT, VAL), (HUE[1], 255, 255)) > 0


def ball_truth(take: Path, cache_dir: Path, force: bool = False) -> list[dict]:
    take = Path(take)
    out_csv = Path(cache_dir) / f"{take.name}_ball.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not force:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(out_csv, encoding="utf-8"))]
    vidx = video_index(take)
    cap = cv2.VideoCapture(str(take / "camera.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (f.shape[1] // SCALE, f.shape[0] // SCALE), interpolation=cv2.INTER_AREA))
    masks = np.array([yellow(f) for f in frames])
    static = masks.mean(axis=0) > STATIC_SHARE
    static = cv2.dilate(static.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blobs = []
    for m in masks:
        m = (m & ~static).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        blobs.append(cv2.connectedComponentsWithStats(m))
    big = [st[1:, cv2.CC_STAT_AREA].max() for n, _, st, _ in blobs if n > 1]
    r_ball = float(np.median(np.sqrt(np.array(big) / np.pi)))
    rows = []
    for j, ((n, lab, st, cent), vi) in enumerate(zip(blobs, vidx)):
        row = {"video_frame": j, "t": vi["t"], "found": 0, "u": np.nan, "v": np.nan, "r_px": np.nan,
               "u0": np.nan, "v0": np.nan, "u1": np.nan, "v1": np.nan}
        round_ = [i for i in range(1, n) if st[i, cv2.CC_STAT_HEIGHT] >= 0.6 * st[i, cv2.CC_STAT_WIDTH]]
        if round_:
            k = max(round_, key=lambda i: st[i, cv2.CC_STAT_AREA])
            if st[k, cv2.CC_STAT_AREA] >= MIN_AREA_PX:
                near = [i for i in range(1, n) if np.hypot(*(cent[i] - cent[k])) <= 2 * r_ball]
                x0 = min(st[i, 0] for i in near)
                y0 = min(st[i, 1] for i in near)
                x1 = max(st[i, 0] + st[i, 2] for i in near)
                y1 = max(st[i, 1] + st[i, 3] for i in near)
                row.update(found=1, u=SCALE * (x0 + x1) / 2, v=SCALE * (y0 + y1) / 2, r_px=SCALE * (x1 - x0) / 2,
                           u0=SCALE * x0, v0=SCALE * y0, u1=SCALE * x1, v1=SCALE * y1)
        rows.append(row)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return [{k: float(v) for k, v in r.items()} for r in rows]


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        r = ball_truth(Path(a), Path(a), force=True)
        print(f"{a}: ball found in {np.mean([x['found'] for x in r]):.1%} of {len(r)} frames")
