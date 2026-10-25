"""
alignment.py - radar results against the camera: loads a replayed take, moves tracks to any camera time, projects
them into the image and fits what the radar cannot know (latency, residual azimuth offset, ball height) by robust
least squares on the ball, always on the other half of the take than the one it is used on.

    from validation.alignment import load, fit_two_fold
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import paths


@dataclass
class Take:
    name: str
    folder: Path
    recs: list
    t_arr: np.ndarray
    t_ready: np.ndarray
    gt: list
    extra: dict = field(default_factory=dict)


def records(name: str) -> list[dict]:
    return [json.loads(line) for line in open(paths.RESULTS / "replay" / name / "frames.jsonl", encoding="utf-8")]


def load(take: str, gt: list, folder: Path) -> Take:
    t0 = json.loads(open(folder / "radar_raw.jsonl", encoding="utf-8").readline())["t"]
    recs = records(take)
    t_arr = np.array([r["t"] + t0 for r in recs])
    t_ready = t_arr + np.array([r["ms"]["total"] for r in recs]) / 1e3
    return Take(take, folder, recs, t_arr, t_ready, gt)


def moved(tr: dict, dt: float) -> dict:
    x, y = tr["x"] + tr["vx"] * dt, tr["y"] + tr["vy"] * dt
    return dict(tr, x=x, y=y, range_m=float(np.hypot(x, y)), theta_deg=float(np.degrees(np.arctan2(y, x))), age_s=dt)


def tracks_at(tk: Take, tq: float, d: float, live: bool = False, confirmed: bool = True) -> tuple[int, list[dict]]:
    k = int(np.searchsorted(tk.t_ready if live else tk.t_arr - d, tq, side="right") - 1)
    if k < 0:
        return k, []
    dt = tq - (tk.t_arr[k] - d)
    return k, [moved(t, dt) for t in tk.recs[k]["tracks"] if not confirmed or t["status"] == "confirmed"]


def pixel(proj, rng: float, theta: float, al: dict):
    return proj.point(rng, theta + al["dtheta_deg"], al["z_ball_m"])


def azimuth_of(proj, rng: float, u: float, al: dict) -> float:
    th = 0.0
    for _ in range(6):
        p0, p1 = pixel(proj, rng, th, al), pixel(proj, rng, th + 0.1, al)
        if p0 is None or p1 is None:
            return np.nan
        slope = (p1[0] - p0[0]) / 0.1
        th -= (p0[0] - u) / slope
    return th


def ball_pairs(tk: Take, proj, al: dict, span, max_px: float = 200.0) -> list[tuple]:
    out = []
    for g in tk.gt:
        if not g["found"] or not span[0] <= g["t"] < span[1]:
            continue
        best = None
        for tr in tracks_at(tk, g["t"], al["d_s"])[1]:
            p = pixel(proj, tr["range_m"], tr["theta_deg"], al)
            if p is not None and abs(p[0] - g["u"]) <= max_px and (best is None or abs(p[0] - g["u"]) < best[0]):
                best = (abs(p[0] - g["u"]), tr)
        if best is not None:
            out.append((best[1], g))
    return out


def fit(tk: Take, proj, z0: float, span, d_grid=np.arange(0.0, 0.205, 0.01)) -> dict:
    best = None
    for d in d_grid:
        pairs = ball_pairs(tk, proj, {"d_s": d, "dtheta_deg": 0.0, "z_ball_m": z0}, span)
        if len(pairs) < 20:
            continue

        def res(p):
            r = []
            for tr, g in pairs:
                q = proj.point(tr["range_m"], tr["theta_deg"] + p[0], p[1])
                r.extend([q[0] - g["u"], q[1] - g["v"]] if q is not None else [500.0, 500.0])
            return np.array(r)
        sol = least_squares(res, [0.0, z0], loss="soft_l1", f_scale=20.0)
        cost = float(np.median(np.abs(sol.fun[0::2])))
        if best is None or cost < best["median_du_px"]:
            best = {"d_s": float(d), "dtheta_deg": float(sol.x[0]), "z_ball_m": float(sol.x[1]),
                    "median_du_px": cost, "median_dv_px": float(np.median(np.abs(sol.fun[1::2]))), "n": len(pairs)}
    return best


def fit_two_fold(tk: Take, proj, z0: float) -> list[dict]:
    mid = float(np.median([g["t"] for g in tk.gt if g["found"]]))
    halves = [(-np.inf, mid), (mid, np.inf)]
    return [{"span": halves[k], "al": fit(tk, proj, z0, halves[1 - k])} for k in range(2)]


def alignment_at(cal: list[dict], t: float) -> dict:
    return next(c["al"] for c in cal if c["span"][0] <= t < c["span"][1])
