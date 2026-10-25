"""
light.py - is a light on the moving camera, is it flashing: new clipped area against the frame 0.2 s earlier, onsets
and offsets with hysteresis.

    python -m methods.light      (self-test)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

SUB = 4
S_MIN = 2.0
LAG_S = 0.2


@dataclass
class LightEvent:
    active: bool
    onset: bool
    offset: bool
    flashing: bool
    rate_hz: float
    n_up: int
    n_down: int
    min_px: int
    clipped_frac: float
    blob_uv: tuple | None = None
    core_uv: tuple | None = None
    lit: np.ndarray | None = field(default=None, repr=False)


class LightMonitor:

    def __init__(self, cfg: dict, f_px: float):
        L = cfg["light"]
        self.k, self.off_s, self.window = L["k_sigma"], L["off_s"], L["flash_window_s"]
        self.clip = cfg["camera"]["clip_level"]
        r_px = f_px * L["source_size_m"] / 2 / cfg["radar"]["r_max_m"] / SUB
        self.min_px = max(4, int(np.pi * r_px ** 2))
        self.past = deque()
        self.noise = deque(maxlen=90)
        self.clip_calm = deque(maxlen=90)
        self.active, self.t_low, self.onsets = False, None, deque()
        self.events: list[dict] = []

    def update(self, rgb: np.ndarray, t: float) -> LightEvent:
        s = rgb[::SUB, ::SUB].astype(np.float32)
        Y = s[..., 0] * 0.299 + s[..., 1] * 0.587 + s[..., 2] * 0.114
        clipped = Y >= self.clip
        n_clip = int(clipped.sum())
        self.past.append((t, Y, clipped))
        while len(self.past) > 2 and t - self.past[1][0] >= LAG_S:
            self.past.popleft()
        if len(self.past) < 2:
            return LightEvent(False, False, False, False, 0.0, 0, 0, self.min_px, n_clip / clipped.size)
        sig = max(float(np.median(self.noise)) if self.noise else 0.0, S_MIN)
        best = None
        for _, Yp, cp in (self.past[-2], self.past[0]):
            d = Y - Yp
            if not self.noise:
                sig = max(float(np.median(np.abs(d))) / 0.6745, S_MIN)
            up, down = d > self.k * sig, d < -self.k * sig
            net = int(up.sum()) - int(down.sum())
            if best is None or abs(net) > abs(best[0]):
                best = (net, d, up, down, clipped & ~cp)
        net, d, up, down, new_clip = best
        n_up, n_down = int(up.sum()), int(down.sum())
        base = float(np.median(self.clip_calm)) if self.clip_calm else float(n_clip)
        onset_now = int(new_clip.sum()) >= self.min_px and net > 0 and n_clip >= base + self.min_px
        onset = offset = False
        if not self.active and onset_now:
            self.active, onset, self.t_low = True, True, None
            self.onsets.append(t)
            self.events.append({"onset_s": t, "lit_px": n_up * SUB * SUB})
        elif self.active:
            if -net >= self.min_px and n_clip <= base + self.min_px / 2:
                self.active, offset = False, True
            elif n_clip <= base + self.min_px / 2:
                self.t_low = t if self.t_low is None else self.t_low
                if t - self.t_low >= self.off_s:
                    self.active, offset = False, True
            else:
                self.t_low = None
            if offset and self.events:
                self.events[-1]["off_s"] = t
        if not self.active and not onset_now and not offset and abs(net) < self.min_px:
            self.noise.append(float(np.median(np.abs(Y - self.past[-2][1]))) / 0.6745)
            self.clip_calm.append(n_clip)
        while self.onsets and t - self.onsets[0] > self.window:
            self.onsets.popleft()
        blob = core = None
        if self.active and n_up:
            ys, xs = np.nonzero(up)
            w = np.maximum(d[ys, xs], 1.0)
            blob = (float((xs * w).sum() / w.sum()) * SUB + SUB / 2, float((ys * w).sum() / w.sum()) * SUB + SUB / 2)
            if new_clip.any():
                cy, cx = np.nonzero(new_clip)
                core = (float(cx.mean()) * SUB + SUB / 2, float(cy.mean()) * SUB + SUB / 2)
        return LightEvent(self.active, onset, offset, len(self.onsets) >= 2, len(self.onsets) / self.window,
                          n_up, n_down, self.min_px, n_clip / clipped.size, blob, core, up if self.active else None)


def selftest() -> bool:
    cfg = {"light": {"k_sigma": 5.0, "off_s": 0.5, "flash_window_s": 2.0, "source_size_m": 0.15},
           "camera": {"clip_level": 250}, "radar": {"r_max_m": 2.6}}
    rng = np.random.default_rng(3)
    H, W = 360, 640
    tex = np.clip(120 + 60 * np.sin(np.arange(W + 400)[None, :] / 9.0) * np.cos(np.arange(H)[:, None] / 13.0), 0, 255)
    yy, xx = np.mgrid[0:H, 0:W]
    lamp = np.clip(420 * np.exp(-((xx - 500) ** 2 + (yy - 120) ** 2) / (2 * 40.0 ** 2)), 0, 255)

    def frame(i, on):
        img = tex[:, i:i + W] + rng.normal(0, 1.5, (H, W)) + (lamp if on else 0)
        if 20 <= i < 50:
            img[:, : max(0, 300 - 10 * (i - 20))] *= 0.3
        return np.repeat(np.clip(img, 0, 255)[..., None], 3, axis=2).astype(np.uint8)

    m = LightMonitor(cfg, 950.0)
    seq = [False] * 60 + [True] * 30 + [False] * 30 + [True, True, True, False, False, False] * 6 + [False] * 20
    on_n = off_n = false_n = 0
    flashing_seen = False
    timeline = []
    for i, on in enumerate(seq):
        ev = m.update(frame(i, on), i / 30.0)
        if ev.onset or ev.offset:
            timeline.append(f"{i}{'+' if ev.onset else '-'}")
        on_n += ev.onset
        off_n += ev.offset
        false_n += ev.onset and i < 60
        flashing_seen |= ev.flashing
    ok = false_n == 0 and on_n == 1 + 6 and off_n >= 6 and flashing_seen
    print(f"  {'PASS' if ok else 'FAIL'}  moving texture: {false_n} false onsets; lamp: {on_n} onsets, {off_n} offsets; "
          f"flashing detected: {flashing_seen} (A_min {m.min_px} px on the 4x grid); frames on+/off-: {' '.join(timeline)}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
