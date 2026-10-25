"""
track_material.py - M2-SRC on every confirmed track: one window per track every half window, its empty scene from
the reference at the same rail positions, the label by votes.

    from methods.material.track_material import TrackMaterial
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from methods.material.classifier import LabelVotes, MaterialMotionModel, classify, moved_over_ground
from methods.material.features import FeatureExtractor
from methods.material.micro_doppler import FramePreprocessor, RadarParams, load_channel_cal

MODEL = Path(__file__).resolve().parent / "material_motion_model.json"
WINDOW_S = 1.6

class MaterialSRC:
    def __init__(self, cfg_file, max_range_m: float = 3.2, extent_m: float = 0.6):
        self.p = RadarParams.from_cfg(cfg_file)
        self.pre = FramePreprocessor(self.p, max_range_m=max_range_m, channel_cal=load_channel_cal())
        self.fx = FeatureExtractor(self.p, self.pre)
        self.model = MaterialMotionModel.load(MODEL)
        self.n_window = int(round(WINDOW_S / self.p.frame_period_s))
        self.frames = deque(maxlen=self.n_window)
        self.times = deque(maxlen=self.n_window)
        self.extent = extent_m
        self.rng = self.fx.range_axis
        self.uy = self.pre.geo.uy_grid
        self.u_max = float(np.abs(self.uy[self.fx.sector]).max())

    def push(self, cube: np.ndarray, t: float, pose):
        fp = self.pre.process(cube, t)
        self.frames.append((fp, np.asarray(pose, float)))
        self.times.append(t)
        return fp

    def ready(self) -> bool:
        return len(self.frames) == self.n_window

    def classify(self, r_lo: float, r_hi: float, u_lo: float, u_hi: float, background) -> dict | None:
        if not self.ready() or background is None or max(abs(u_lo), abs(u_hi)) > self.u_max:
            return None
        self.fx.gate = (self.rng >= r_lo - self.extent / 2) & (self.rng <= r_hi + self.extent / 2)
        self.fx.sector = self.pre.geo.uy_valid & (self.uy >= u_lo - 2 / 12) & (self.uy <= u_hi + 2 / 12)
        if not self.fx.gate.any() or not self.fx.sector.any():
            return None
        w = self.fx.window([f for f, _ in self.frames], background)
        return {**classify(self.model, w.features), "detected": bool(w.detected), "range_m": w.range_m,
                "azimuth_deg": w.azimuth_deg, "features": {k: float(v) for k, v in w.features.items()}}


class TrackMaterial:

    def __init__(self, cfg_file, reference=None, extent_m: float = 0.6):
        self.src = MaterialSRC(cfg_file, extent_m=extent_m)
        self.reference = reference
        self.models = True
        self.paused = ""

    def update(self, tracker, frame, t: float, pose):
        mc = self.src
        fp = mc.push(frame, t, pose)
        if self.reference is None:
            self.paused = "no empty-path reference"
            return fp
        span = mc.n_window * mc.p.frame_period_s
        t0 = mc.times[0]
        xs = [p[0] for _, p in mc.frames]
        bg = self.reference.background_ra(min(xs), max(xs), pose) if mc.ready() else None
        offsets_ok = all(abs(p[1] - pose[1]) < 0.005 and abs(p[2] - pose[2]) < 0.005 for _, p in mc.frames)
        self.paused = "" if (bg is not None and offsets_ok) else "no empty scene from this side/height offset"
        due = []
        for tr in tracker.tracks:
            if not tr.confirmed:
                continue
            tr.mat_pos = getattr(tr, "mat_pos", deque(maxlen=400))
            r = float(np.hypot(*tr.kf.s[:2]))
            tr.mat_pos.append((t, r, float(tr.kf.s[1] / max(r, 1e-6)), float(pose[0] + tr.kf.s[0]), float(pose[1] + tr.kf.s[1])))
            if self.paused or not mc.ready() or tr.born > t0 or t - getattr(tr, "t_classified", -np.inf) < span / 2:
                continue
            if sum(1 for q in tr.mat_pos if q[0] >= t0) >= 0.8 * len(mc.frames):
                due.append(tr)
        for tr in sorted(due, key=lambda q: getattr(q, "t_classified", -np.inf)):
            rr, u, wx, wy = np.array([q[1:] for q in tr.mat_pos if q[0] >= t0]).T
            if any(rr.min() - mc.extent / 2 <= q.mat_pos[-1][1] <= rr.max() + mc.extent / 2 and
                   u.min() - 2 / 12 <= q.mat_pos[-1][2] <= u.max() + 2 / 12
                   for q in tracker.tracks if q is not tr and q.confirmed and getattr(q, "mat_pos", None)):
                continue
            moved = moved_over_ground(np.c_[wx, wy], mc.p.range_res_m)
            out = mc.classify(rr.min(), rr.max(), u.min(), u.max(), bg)
            tr.t_classified = t
            if out is not None and out["detected"] and not (moved and out["motion"] == "static"):
                tr.mat_votes = getattr(tr, "mat_votes", None) or LabelVotes()
                tr.classes = tr.mat_votes.add(out)
                tr.window = out["features"]
            break
        return fp
