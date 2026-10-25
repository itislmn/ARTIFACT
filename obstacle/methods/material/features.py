"""
features.py - finds the object in a 1.6 s window (largest change against the empty room, 0.3-1.6 m, +-25 deg) and
measures 14 physical features: 7 of its motion (echo phase over time) and 7 of its material (echo strength and shape,
measured in every frame at the object's own range cell and direction, so a moving object shows the same echo as a still one).

    from material_classification.features import FeatureExtractor    (byte-identical copy in obstacle/methods/material/)
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal

MOTION_FEATURES = ["speed_mps", "speed_change_mps", "width_mps", "pattern_change", "flicker",
                   "range_motion_m", "azimuth_motion_m"]
MATERIAL_FEATURES = ["change_db", "reflectivity_db", "back_echo_db", "shadow_db", "ripple_db",
                     "slope_db_per_ghz", "depth_m"]
FEATURES = MOTION_FEATURES + MATERIAL_FEATURES
HEAD_FEATURES = {"motion": FEATURES, "material": MATERIAL_FEATURES}

DETECT_DB = 8.0


@dataclass
class Window:
    detected: bool
    range_m: float
    azimuth_deg: float
    features: dict
    change_map_db: np.ndarray
    spectrogram: tuple

    def vector(self, names=FEATURES):
        return np.array([self.features[n] for n in names], float)


def background_map(frames):
    total, n = 0.0, 0
    for f in frames:
        total, n = total + f.ra, n + 1
    return total / n


class FeatureExtractor:
    def __init__(self, params, preproc, min_range_m=0.3, max_range_m=1.6, max_azimuth_deg=25.0):
        self.p, self.pre, self.geo = params, preproc, preproc.geo
        rng = params.range_axis[:preproc.n_bins]
        self.range_axis = rng
        self.gate = (rng >= min_range_m) & (rng <= max_range_m)
        self.sector = self.geo.uy_valid & (np.abs(self.geo.uy_grid) <= np.sin(np.radians(max_azimuth_deg)))

    def window(self, frames, background):
        p, geo = self.p, self.geo
        RA = np.mean([f.ra for f in frames], axis=0)
        noise = float(np.median(RA[:, geo.uy_valid]))
        change = 10 * np.log10((RA + noise) / (background + noise))

        search = np.where(self.gate[:, None] & self.sector[None, :], change, -np.inf)
        b0, a0 = np.unravel_index(np.argmax(search), search.shape)
        change_db = float(change[b0, a0])
        r0 = float(self.range_axis[b0])
        az = float(np.degrees(np.arcsin(np.clip(geo.uy_grid[a0], -1, 1))))

        bins = np.arange(max(b0 - 2, 0), min(b0 + 3, len(self.range_axis)))
        beam = np.stack([geo.az_spectrum(f.rfft[bins])[..., a0] for f in frames])
        best = np.argmax(np.mean(np.abs(beam) ** 2, axis=2), axis=1)
        x = beam[np.arange(len(frames)), best]
        b_f = bins[best]
        seen = np.array([10 * np.log10((f.ra[b, a0] + noise) / (background[b, a0] + noise))
                         for f, b in zip(frames, b_f)]) >= DETECT_DB
        if seen.sum() < 3:
            seen[:] = True
        xs, fr = x[seen], [f for f, s in zip(frames, seen) if s]

        lag1 = np.mean(xs[:, 1:] * np.conj(xs[:, :-1]), axis=1)
        power = np.mean(np.abs(xs) ** 2, axis=1)
        k = p.wavelength / (4 * np.pi * p.chirp_period_s)
        v_f = p.doppler_sign * k * np.angle(lag1)
        width = p.wavelength / (2 * np.sqrt(2) * np.pi * p.chirp_period_s) \
            * np.sqrt(np.maximum(np.log(power / np.maximum(np.abs(lag1), 1e-20)), 0))
        a, b = xs[1:], xs[:-1]
        coh = np.abs(np.sum(a * np.conj(b), axis=1)) / np.sqrt(
            np.sum(np.abs(a) ** 2, axis=1) * np.sum(np.abs(b) ** 2, axis=1) + 1e-30)
        amp = np.sqrt(power)
        motion = {
            "speed_mps": float(np.median(np.abs(v_f))),
            "speed_change_mps": float(np.std(v_f)),
            "width_mps": float(np.median(width)),
            "pattern_change": float(np.log10(1 - np.mean(coh) + 1e-3)) if coh.size else -3.0,
            "flicker": float(np.std(amp) / np.mean(amp)),
            "range_motion_m": float(np.std(self.range_axis[b_f[seen]])),
            "azimuth_motion_m": self._cross_range_spread(fr, b_f[seen], a0),
        }

        echo = np.array([self._echo(f.ra[:, a0], background[:, a0], bb, noise) for f, bb in zip(fr, b_f[seen])])
        reflectivity, back_echo, depth = np.median(echo, axis=0)
        far = slice(b0 + 4, None)
        cols = slice(max(a0 - 3, 0), a0 + 4)
        bg_far = background[far, cols].sum()
        shadow = np.nan
        if bg_far > 10 * noise * max(background[far, cols].size, 1):
            shadow = float(np.clip(10 * np.log10(RA[far, cols].sum() / bg_far), -40, 10))
        sub = np.mean([f.subband[:, bb, a0] for f, bb in zip(fr, b_f[seen])], axis=0) + 1e-12
        sub_db = 10 * np.log10(sub)
        fg = self.pre.subband_freq_ghz
        slope, icpt = np.polyfit(fg, sub_db, 1)
        material = {
            "change_db": change_db,
            "reflectivity_db": float(reflectivity),
            "back_echo_db": float(back_echo),
            "shadow_db": shadow,
            "ripple_db": float(np.std(sub_db - (slope * fg + icpt))),
            "slope_db_per_ghz": float(slope),
            "depth_m": float(depth),
        }

        return Window(change_db >= DETECT_DB, r0, az, {**motion, **material}, change,
                      self._spectrogram(x))

    def _echo(self, prof, bg, b, noise):
        prof_c = np.maximum(prof - bg, 1e-3 * noise)
        behind = prof_c[b + 2:b + 9]
        lo, hi = max(b - 6, 0), min(b + 7, len(prof_c))
        w = prof_c[lo:hi] / prof_c[lo:hi].sum()
        return (10 * np.log10(prof[b]) + 40 * np.log10(max(self.range_axis[b], 0.1)),
                max(10 * np.log10(behind.max() / prof_c[b]), -40.0) if behind.size else -40.0,
                np.sqrt(np.sum(w * (np.arange(lo, hi) - b) ** 2)) * self.p.range_res_m)

    def _cross_range_spread(self, frames, bins, a0) -> float:
        u = self.geo.uy_grid
        near = self.sector & (np.abs(u - u[a0]) <= 2 / 12)
        idx = np.flatnonzero(near)
        y = []
        for f, b in zip(frames, bins):
            row = f.ra[b]
            j = int(idx[np.argmax(row[idx])])
            uj = u[j]
            if 0 < j < len(u) - 1:
                lo, mid, hi = np.log(row[j - 1:j + 2] + 1e-30)
                den = lo - 2 * mid + hi
                if den < 0:
                    uj = u[j] + 0.5 * (lo - hi) / den * (u[j + 1] - u[j])
            y.append(float(self.range_axis[b]) * uj)
        return float(np.std(y)) if len(y) > 1 else 0.0

    def _spectrogram(self, xs):
        p = self.p
        f, t, Z = signal.stft(xs.ravel(), fs=p.prf_hz, window="hann", nperseg=128, noverlap=96,
                              nfft=256, return_onesided=False, boundary=None, padded=False)
        vel = p.doppler_sign * np.fft.fftshift(f) * p.wavelength / 2
        P = np.fft.fftshift(np.abs(Z) ** 2, axes=0)
        return vel, t * p.time_scale, 10 * np.log10(P + 1e-20)
