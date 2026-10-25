"""
material_src.py - M2-SRC (material_classification's own features and model) on every confirmed track: a 1.6 s window
around the track's positions every 0.8 s; the label is the majority of its windows.

    from methods.material_src import for_session
"""
from __future__ import annotations

import sys
from collections import deque

import numpy as np

import paths

sys.path.insert(0, str(paths.REPO))
from material_classification.classifier import LabelVotes, MaterialMotionModel, classify, moved_over_ground  # noqa: E402
from material_classification.features import FeatureExtractor, background_map  # noqa: E402
from material_classification.utils.micro_doppler import (FramePreprocessor, RadarParams,  # noqa: E402
                                                         load_channel_cal, raw_frame_to_cube)

MODEL = paths.MATERIAL / "models" / "material_motion_model.json"
WINDOW_S = 1.6

class MaterialSRC:
    def __init__(self, background_bin, cfg_file, max_range_m: float = 3.2, extent_m: float = 0.6):
        self.p = RadarParams.from_cfg(cfg_file)
        self.pre = FramePreprocessor(self.p, max_range_m=max_range_m, channel_cal=load_channel_cal())
        self.fx = FeatureExtractor(self.p, self.pre)
        self.model = MaterialMotionModel.load(MODEL)
        words = np.memmap(background_bin, dtype="<i2", mode="r")
        per = self.p.frame_bytes // 2
        n = words.size // per
        self.background = background_map(self.pre.process(raw_frame_to_cube(np.asarray(words[i * per:(i + 1) * per]), self.p))
                                          for i in range(n))
        self.n_window = int(round(WINDOW_S / self.p.frame_period_s))
        self.frames = deque(maxlen=self.n_window)
        self.extent = extent_m
        self.rng = self.fx.range_axis
        self.uy = self.pre.geo.uy_grid

    def push(self, cube: np.ndarray, t: float):
        self.frames.append(self.pre.process(cube, t))

    def ready(self) -> bool:
        return len(self.frames) == self.n_window

    def classify(self, r_lo: float, r_hi: float, u_lo: float, u_hi: float, moved: bool = False) -> dict | None:
        if not self.ready():
            return None
        self.fx.gate = (self.rng >= r_lo - self.extent / 2) & (self.rng <= r_hi + self.extent / 2)
        self.fx.sector = self.pre.geo.uy_valid & (self.uy >= u_lo - 2 / 12) & (self.uy <= u_hi + 2 / 12)
        if not self.fx.gate.any() or not self.fx.sector.any():
            return None
        w = self.fx.window(list(self.frames), self.background)
        return {**classify(self.model, w.features, moved), "detected": bool(w.detected), "range_m": w.range_m,
                "azimuth_deg": w.azimuth_deg, "features": {k: float(v) for k, v in w.features.items()}}


class TrackMaterial:

    def __init__(self, background_bin, waveform_cfgs: dict, extent_m: float = 0.6, only: str | None = None):
        self.bg, self.cfgs, self.extent = background_bin, waveform_cfgs, extent_m
        self.src: dict[str, MaterialSRC] = {n: MaterialSRC(background_bin, c, extent_m=extent_m)
                                            for n, c in waveform_cfgs.items()}
        self.cur = None
        self.only = only
        self.models = True

    def update(self, tracker, rd, fe, frame, t, dets=None):
        name = fe.wf.name
        if self.only is not None and name != self.only:
            return
        if name != self.cur:
            if name not in self.src:
                self.src[name] = MaterialSRC(self.bg, self.cfgs[name], extent_m=self.extent)
            self.src[name].frames.clear()
            self.cur = name
        mc = self.src[name]
        if frame is None:
            return
        mc.push(frame, t)
        span = mc.n_window * mc.p.frame_period_s
        for tr in tracker.tracks:
            if not tr.confirmed:
                continue
            tr.mat_pos = getattr(tr, "mat_pos", deque(maxlen=400))
            tr.mat_pos.append((t, float(np.hypot(*tr.kf.s[:2])), float(tr.kf.s[1] / max(np.hypot(*tr.kf.s[:2]), 1e-6))))
            if not mc.ready() or t - tr.born < span or t - getattr(tr, "t_classified", -np.inf) < span / 2:
                continue
            w = [(r, u) for ti, r, u in tr.mat_pos if t - ti < span]
            if len(w) < 0.8 * mc.n_window:
                continue
            r, u = np.array(w).T
            xy = np.c_[r * np.sqrt(1 - u ** 2), r * u]
            out = mc.classify(r.min(), r.max(), u.min(), u.max(), moved_over_ground(xy, mc.p.range_res_m))
            tr.t_classified = t
            if out is None or not out["detected"]:
                continue
            tr.mat_votes = getattr(tr, "mat_votes", None) or LabelVotes()
            tr.classes = tr.mat_votes.add(out)
            tr.window = out["features"]


def for_session(cfg: dict, library: dict, only: str | None = None) -> TrackMaterial | None:
    from hardware.references import profile_key
    first = next(iter(library.values()))
    bg = paths.RECORDINGS / cfg["session"] / "radar" / f"background_{profile_key(first.cfg)}.bin"
    if not bg.exists():
        print(f"material: {bg.name} missing (python radar_setup.py background) - no material classification")
        return None
    cfgs = {k: w.path for k, w in library.items() if only is None or k == only}
    return TrackMaterial(bg, cfgs, cfg["tracking"]["object_extent_m"], only)
