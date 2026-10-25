"""
brightness.py - per-pixel running mean and variance of the luma; a light event starts when a lamp-sized area is 5 sigma
brighter than its history or newly clipped.

    from backend.brightness import BrightnessModel
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STEP = 4
S_MIN = 2.0


@dataclass
class LightEvent:
    active: bool
    onset: bool
    n_lit: int
    min_px: int
    blob_uv: tuple | None
    core_uv: tuple | None
    lit: np.ndarray | None = None


class BrightnessModel:
    def __init__(self, cfg: dict, f_px: float):
        L = cfg["light"]
        self.tau, self.k, self.off_s = L["background_s"], L["k_sigma"], L["off_s"]
        self.clip = cfg["camera"]["clip_level"]
        r_px = f_px * L["source_size_m"] / 2 / cfg["radar"]["r_max_m"] / STEP
        self.min_px = max(4, int(np.pi * r_px ** 2))
        self.mu = self.var = None
        self.t_prev, self.active, self.t_below = None, False, None

    def update(self, rgb: np.ndarray, t: float) -> LightEvent:
        s = rgb[::STEP, ::STEP].astype(np.float32)
        Y = s[..., 0] * 0.299 + s[..., 1] * 0.587 + s[..., 2] * 0.114
        if self.mu is None:
            self.mu, self.var, self.t_prev = Y.copy(), np.full_like(Y, S_MIN ** 2), t
            return LightEvent(False, False, 0, self.min_px, None, None)
        d = Y - self.mu
        new_clip = (Y >= self.clip) & (self.mu < self.clip - 5)
        lit = (d > self.k * np.sqrt(np.maximum(self.var, S_MIN ** 2))) | new_clip
        a = min(1.0, (t - self.t_prev) / self.tau)
        calm = ~lit
        self.mu[calm] += a * d[calm]
        self.var[calm] += a * (d[calm] ** 2 - self.var[calm])
        self.t_prev = t
        n = int(lit.sum())
        onset = False
        if n >= self.min_px:
            onset, self.active, self.t_below = not self.active, True, None
        elif self.active:
            self.t_below = self.t_below if self.t_below is not None else t
            if t - self.t_below >= self.off_s:
                self.active = False
        blob = core = None
        if self.active and n:
            ys, xs = np.nonzero(lit)
            w = np.maximum(d[ys, xs], 1.0)
            blob = (float((xs * w).sum() / w.sum()) * STEP + STEP / 2, float((ys * w).sum() / w.sum()) * STEP + STEP / 2)
            if new_clip.any():
                cy, cx = np.nonzero(new_clip)
                core = (float(cx.mean()) * STEP + STEP / 2, float(cy.mean()) * STEP + STEP / 2)
        return LightEvent(self.active, onset, n, self.min_px, blob, core, lit)
