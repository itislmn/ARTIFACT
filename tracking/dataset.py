"""
dataset.py - the evaluated takes and their camera ground truth: per camera frame the ball's box (colour) and the
person's box (what moves outside the ball), written to output/dataset.csv; the radar never sees it.

    python dataset.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import paths  # noqa: E402
from validation.ball_truth import ball_truth, video_index  # noqa: E402
from validation.scene_truth import motion  # noqa: E402

TAKES = {"final_fixed_1": "Fixed Chirp", "run_1": "Cognitive"}
POLICIES = {"run_1_search": "Search Chirp Only", "run_1": "Cognitive", "run_1_all": "Every Look"}
SOURCE = {"final_fixed_1": ("TR1", "final_fixed_1", None), "run_1": ("TR2", "run_1", "cognitive"),
          "run_1_search": ("TR2", "run_1", "search"), "run_1_all": ("TR2", "run_1", "all")}
LABELS = {**TAKES, **POLICIES}
CLASSES = {"ball": {"material": "dielectric", "motion": "rigid"},
           "person": {"material": "waterlike", "motion": "articulated"}}
BALL_RANGE_M = 1.55
TEXT = "ARTIFACT"
CACHE = paths.OUTPUT / "cache"
TABLE = paths.OUTPUT / "dataset.csv"
COLUMNS = ["take", "video_frame", "t_s", "object", "found", "partial", "u", "v", "u0", "v0", "u1", "v1"]


def folder(take: str) -> Path:
    session, name, _ = SOURCE[take]
    return paths.RECORDINGS / session / name


def policy(take: str) -> str | None:
    return SOURCE[take][2]


def alignment(take: str) -> Path:
    return paths.alignment_dir(SOURCE[take][0])


def truth(take: str) -> list[dict]:
    ball = ball_truth(folder(take), CACHE)
    _, person = motion(folder(take), ball, CACHE)
    vidx = video_index(folder(take))
    out = []
    for j, (g, p, vi) in enumerate(zip(ball, person, vidx)):
        b = {"found": int(g["found"]), "partial": 0, "u": g["u"], "v": g["v"], "u0": g["u0"], "v0": g["v0"],
             "u1": g["u1"], "v1": g["v1"], "r_px": g["r_px"]}
        q = {"found": int(np.isfinite(p[0])), "partial": int(p[4] == 1) if np.isfinite(p[0]) else 0,
             "u": (p[0] + p[2]) / 2, "v": (p[1] + p[3]) / 2, "u0": p[0], "v0": p[1], "u1": p[2], "v1": p[3]}
        out.append({"video_frame": j, "t": vi["t"], "ball": b, "person": q})
    return out


def write(rows: list[dict]):
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLE, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def run() -> dict:
    out, rows = {}, []
    for take in TAKES:
        out[take] = truth(take)
    for take in POLICIES:
        out.setdefault(take, out[next(k for k in TAKES if folder(k) == folder(take))])
    for take in TAKES:
        t0 = out[take][0]["t"]
        for f in out[take]:
            for name in CLASSES:
                o = f[name]
                rows.append({"take": take, "video_frame": f["video_frame"], "t_s": round(f["t"] - t0, 4), "object": name,
                             "found": o["found"], "partial": o["partial"],
                             **{k: round(float(o[k]), 1) if o["found"] else "" for k in ("u", "v", "u0", "v0", "u1", "v1")}})
    write(rows)
    return out


if __name__ == "__main__":
    res = run()
    for take, frames in res.items():
        print(take, len(frames), "frames; ball", np.mean([f["ball"]["found"] for f in frames]).round(3),
              "person", np.mean([f["person"]["found"] for f in frames]).round(3),
              "partial", np.mean([f["person"]["partial"] for f in frames]).round(3))
