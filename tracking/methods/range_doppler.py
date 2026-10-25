"""
range_doppler.py - one radar frame to range-Doppler cells of the 16 virtual channels (the background subtracted only
if it removes energy) and the gates for detection: OS-CFAR search gates and beams at the predicted regions (shared
with obstacle/).

    from methods.range_doppler import FrontEnd
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from methods.array_model import ArrayModel
    from methods.cfar import os_factor
except ImportError:
    import paths  # noqa: F401
    from estimation.methods.cfar import os_factor
    from estimation.core.array import ArrayModel

GUARD, TRAIN = 2, 8


@dataclass
class RD:
    D: np.ndarray
    P: np.ndarray
    noise_ch: np.ndarray
    bins: np.ndarray
    ms: dict = field(default_factory=dict)
    bg_used: bool = False


@dataclass
class Gate:
    k: int
    d: int
    range_m: float
    v_mps: float
    power: float
    excess: float
    source: str
    track: int = -1

    @property
    def cell(self):
        return (self.k, self.d)


class FrontEnd:
    def __init__(self, wf, r_min: float, r_max: float, channel_cal=None, background=None, az_sign: int = 1):
        c = wf.cfg
        self.wf, self.cfg = wf, c
        self.array = ArrayModel(c, az_sign=az_sign)
        self.az = self.array.channels
        k_lo = max(int(np.ceil(r_min / c.range_res_m)), 1)
        k_hi = min(int(np.floor(r_max / c.range_res_m)), c.n_range_bins - 1)
        self.bins = np.arange(max(k_lo - GUARD - TRAIN, 1), min(k_hi + GUARD + TRAIN, c.n_range_bins - 1) + 1)
        self.roi = (self.bins >= k_lo) & (self.bins <= k_hi)
        self.range_m = self.bins * c.range_res_m
        n = np.arange(c.n_samples)
        win = np.hanning(c.n_samples)
        W = win[None, :] * np.exp(-2j * np.pi * self.bins[:, None] * n[None, :] / c.n_samples)
        self.Wr, self.Wi = W.real.astype(np.float32), W.imag.astype(np.float32)
        K = c.n_loops
        self.K = K
        self.dwin = np.hanning(K).astype(np.float32)[None, :, None]
        self.v_axis = (np.arange(K) - K // 2) * wf.dv
        tx_of_v = np.arange(c.n_virt) // c.n_rx
        slot = np.asarray(c.tx_reorder)[tx_of_v]
        frac = (np.arange(K) - K // 2)[:, None] / K
        self.tdm = np.exp(-2j * np.pi * frac * slot[None, :] / c.n_tx).astype(np.complex64)
        self.cal = (np.ones(c.n_virt, np.complex64) if channel_cal is None else np.asarray(channel_cal, np.complex64))
        self.bg = None
        if background is not None:
            self.bg = (np.asarray(background["rfft"])[self.bins] / self.cal[None, :]).astype(np.complex64)
        self.offs = np.r_[-np.arange(GUARD + 1, GUARD + TRAIN + 1), np.arange(GUARD + 1, GUARD + TRAIN + 1)]
        self._factor = {}

    def process(self, frame: np.ndarray) -> RD:
        import time
        t0 = time.perf_counter()
        c = self.cfg
        x = np.asarray(frame, np.float32)
        x = x - x.mean(axis=0, keepdims=True)
        x2 = x.reshape(c.n_samples, -1)
        X = (self.Wr @ x2 + 1j * (self.Wi @ x2)).astype(np.complex64).reshape(len(self.bins), self.K, c.n_virt)
        X /= self.cal[None, None, :]
        bg_used = False
        if self.bg is not None:
            M = X[:, :, self.az].mean(axis=1)
            g = np.exp(1j * np.angle(np.sum(self.bg[:, self.az].conj() * M, axis=1))).astype(np.complex64)
            B = g[:, None] * self.bg
            if float(np.sum(np.abs(M[self.roi] - B[self.roi][:, self.az]) ** 2)) < float(np.sum(np.abs(M[self.roi]) ** 2)):
                X -= B[:, None, :]
                bg_used = True
        t1 = time.perf_counter()
        D = np.fft.fftshift(np.fft.fft(X * self.dwin, axis=1), axes=1).astype(np.complex64)
        D *= self.tdm[None, :, :]
        P = np.sum(np.abs(D[:, :, self.az]) ** 2, axis=2)
        sub = D.reshape(-1, c.n_virt)[::3]
        noise_ch = np.median(np.abs(sub) ** 2, axis=0) / np.log(2.0)
        t2 = time.perf_counter()
        return RD(D, P, noise_ch, self.bins, {"range_ms": 1e3 * (t1 - t0), "doppler_ms": 1e3 * (t2 - t1)}, bg_used)

    def change_db(self, k: int, u: float, power: float) -> float:
        if self.bg is None:
            return float("inf")
        a = self.array.steering([u])[:, 0].conj()
        bg = np.abs(self.bg[k, self.az] @ a) ** 2 * float(np.sum(self.dwin)) ** 2 / self.array.n
        return float(10 * np.log10(max(power, 1e-30) / max(bg, 1e-30)))

    def factor(self, pfa: float) -> float:
        if pfa not in self._factor:
            self._factor[pfa] = os_factor(2 * TRAIN, TRAIN, float(pfa), self.array.n)
        return self._factor[pfa]

    def search_gates(self, rd: RD, pfa: float, static_line: bool | None = None) -> list[Gate]:
        P = rd.P
        n = P.shape[0]
        idx = np.clip(np.arange(n)[:, None] + self.offs[None, :], 0, n - 1)
        ref = np.partition(P[idx], TRAIN - 1, axis=1)[:, TRAIN - 1]
        thr = self.factor(pfa) * ref
        peak = np.ones(P.shape, bool)
        Q = np.pad(P, ((1, 1), (0, 0)), constant_values=-np.inf)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    peak &= P >= np.roll(Q, -dj, axis=1)[1 + di:n + 1 + di]
        ok = (P > thr) & peak & self.roi[:, None]
        if static_line is None:
            static_line = self.bg is not None
        if not static_line:
            ok[:, self.K // 2 - 1:self.K // 2 + 2] = False
        ks, ds = np.nonzero(ok)
        return [Gate(int(k), int(d), float(self.range_m[k]), float(self.v_axis[d]), float(P[k, d]),
                     float(P[k, d] / thr[k, d]), "search") for k, d in zip(ks, ds)]

    def beam_power(self, rd: RD, u: float) -> np.ndarray:
        a = self.array.steering([u])[:, 0]
        return np.abs(rd.D[:, :, self.az] @ a.conj()) ** 2 / self.array.n

    def beam_gates(self, rd: RD, priors: list[dict], pfa: float, existing: list[Gate]) -> list[Gate]:
        out, taken = [], {g.cell for g in existing}
        for pr in priors:
            kk = np.flatnonzero((self.range_m >= pr["r_lo"]) & (self.range_m <= pr["r_hi"]) & self.roi)
            dd = self.doppler_bins(pr["v_lo"], pr["v_hi"])
            if kk.size == 0 or dd.size == 0:
                continue
            B = self.beam_power(rd, pr["u"])
            sigma2 = float(np.median(B)) / np.log(2.0)
            win = B[np.ix_(kk, dd)]
            i, j = np.unravel_index(int(np.argmax(win)), win.shape)
            t = np.log(win.size / pfa)
            k, d = int(kk[i]), int(dd[j])
            if win[i, j] > t * sigma2 and (k, d) not in taken:
                taken.add((k, d))
                out.append(Gate(k, d, float(self.range_m[k]), float(self.v_axis[d]), float(rd.P[k, d]),
                                float(win[i, j] / (t * sigma2)), "beam", int(pr["track"])))
        return out

    def doppler_bins(self, v_lo: float, v_hi: float) -> np.ndarray:
        span = 2 * self.wf.v_max
        if v_hi - v_lo >= span:
            return np.arange(self.K)
        lo = (v_lo + self.wf.v_max) % span - self.wf.v_max
        hi = lo + (v_hi - v_lo)
        d_lo = int(np.floor(lo / self.wf.dv)) + self.K // 2
        d_hi = int(np.ceil(hi / self.wf.dv)) + self.K // 2
        return np.arange(d_lo, d_hi + 1) % self.K

    def refine(self, rd: RD, k: int, d: int, u: float) -> tuple[float, float, float]:
        a = self.array.steering([u])[:, 0].conj()
        n, K = rd.D.shape[0], self.K

        def interp(vals):
            lv = np.log(np.maximum(vals, 1e-30))
            den = lv[0] - 2 * lv[1] + lv[2]
            return 0.0 if den >= 0 else float(np.clip(0.5 * (lv[0] - lv[2]) / den, -0.5, 0.5))

        kk = np.clip([k - 1, k, k + 1], 0, n - 1)
        br = np.abs(rd.D[kk, d][:, self.az] @ a) ** 2
        dd = np.array([d - 1, d, d + 1]) % K
        bd = np.abs(rd.D[k, dd][:, self.az] @ a) ** 2
        r = (self.bins[k] + (interp(br) if 0 < k < n - 1 else 0.0)) * self.cfg.range_res_m
        v = (d - K // 2 + interp(bd)) * self.wf.dv
        return float(r), float(v), float(br[1] / self.array.n)
