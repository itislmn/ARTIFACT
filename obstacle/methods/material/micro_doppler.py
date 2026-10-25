"""
micro_doppler.py - radar parameters from the TI cfg, ADC frames to range FFT, range-azimuth map and sub-band responses.

    from material_classification.utils.micro_doppler import FramePreprocessor, RadarParams    (copy in obstacle/methods/material/)
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

C0 = 299_792_458.0
_HERE = Path(__file__).resolve()
_CONFIG = next((d for d in (_HERE.parents[1] / "config", _HERE.parent) if (d / "channel_cal.npy").exists()),
               _HERE.parents[1] / "config")
CHANNEL_CAL_PATH = _CONFIG / "channel_cal.npy"
CONVENTIONS_PATH = _CONFIG / "array_conventions.json"

_AWR2944_ROWS = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
_AWR2944_COLS = [0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 10, 11]


@dataclass
class RadarParams:
    n_samples: int = 256
    n_chirps: int = 128
    n_virt: int = 16
    fs_hz: float = 10e6
    slope_hz_per_s: float = 70e12
    f0_hz: float = 77e9
    adc_start_s: float = 7e-6
    chirp_period_s: float = 250e-6
    frame_period_s: float = 40e-3
    real_adc: bool = True
    n_tx: int = 4
    n_rx: int = 4
    tx_reorder: list = field(default_factory=lambda: [0, 3, 1, 2])
    virt_rows: list = field(default_factory=lambda: list(_AWR2944_ROWS))
    virt_cols: list = field(default_factory=lambda: list(_AWR2944_COLS))
    d_az: float = 0.5
    d_el: float = 0.8
    az_sign: int = 1
    el_sign: int = 1
    doppler_sign: int = 1
    adc_padded: bool = True

    @property
    def wavelength(self):
        return C0 / (self.f0_hz + self.slope_hz_per_s * self.adc_start_s + 0.5 * self.bandwidth_hz)

    @property
    def bandwidth_hz(self):
        return self.slope_hz_per_s * self.n_samples / self.fs_hz

    @property
    def range_res_m(self):
        return C0 * self.fs_hz / (2 * self.slope_hz_per_s * self.n_samples)

    @property
    def n_range_bins(self):
        return self.n_samples // 2 if self.real_adc else self.n_samples

    @property
    def range_axis(self):
        return np.arange(self.n_range_bins) * self.range_res_m

    @property
    def prf_hz(self):
        return 1.0 / self.chirp_period_s

    @property
    def single_chirp_s(self):
        return self.chirp_period_s / self.n_tx

    @property
    def v_max(self):
        return self.wavelength / (4 * self.chirp_period_s)

    @property
    def dv_frame(self):
        return self.wavelength / (2 * self.n_chirps * self.chirp_period_s)

    @property
    def time_scale(self):
        return self.frame_period_s / (self.n_chirps * self.chirp_period_s)

    @property
    def payload_bytes(self):
        per_sample = 2 if self.real_adc else 4
        return self.n_samples * self.n_rx * self.n_chirps * self.n_tx * per_sample

    @property
    def frame_bytes(self):
        return self.payload_bytes * (2 if (self.real_adc and self.adc_padded) else 1)

    @classmethod
    def from_cfg(cls, path, **overrides):
        chirp_tx, p = {}, {}
        for raw in open(path):
            line = raw.strip()
            if not line or line.startswith(("%", "#")):
                continue
            tok = re.split(r"\s+", line)
            cmd, v = tok[0], tok[1:]
            if cmd == "profileCfg":
                p["f0"] = float(v[1]) * 1e9
                p["idle"] = float(v[2]) * 1e-6
                p["adc_start"] = float(v[3]) * 1e-6
                p["ramp_end"] = float(v[4]) * 1e-6
                p["slope"] = float(v[7]) * 1e12
                p["n_samples"] = int(v[9])
                p["fs"] = float(v[10]) * 1e3
            elif cmd == "channelCfg":
                p["n_rx"] = bin(int(v[0])).count("1")
            elif cmd == "adcCfg":
                p["real"] = int(v[1]) == 0
            elif cmd == "chirpCfg":
                chirp_tx[int(v[0])] = int(v[7])
            elif cmd == "frameCfg":
                p["chirp_start"], p["chirp_end"] = int(v[0]), int(v[1])
                p["loops"] = int(v[2])
                p["frame_period"] = float(v[5] if len(v) >= 8 else v[4]) * 1e-3
            elif cmd == "antGeometryCfg":
                nums = [float(t) for t in v]
                pairs, spacing = nums[:-2], nums[-2:]
                p["rows"] = [int(a) for a in pairs[0::2]]
                p["cols"] = [int(a) for a in pairs[1::2]]
                p["d_az"], p["d_el"] = spacing

        chirps_per_loop = p["chirp_end"] - p["chirp_start"] + 1
        n_tx = len({m for m in chirp_tx.values() if m}) or 1
        perm = list(range(n_tx))
        for slot, mask in chirp_tx.items():
            if mask and (mask.bit_length() - 1) < n_tx:
                perm[mask.bit_length() - 1] = slot
        n_virt = n_tx * p["n_rx"]
        kw = dict(
            n_samples=p["n_samples"], n_chirps=p["loops"], n_virt=n_virt, fs_hz=p["fs"],
            slope_hz_per_s=p["slope"], f0_hz=p["f0"], adc_start_s=p["adc_start"],
            chirp_period_s=chirps_per_loop * (p["idle"] + p["ramp_end"]),
            frame_period_s=p["frame_period"], real_adc=p.get("real", True),
            n_tx=n_tx, n_rx=p["n_rx"], tx_reorder=perm)
        if len(p.get("rows", [])) == n_virt:
            kw.update(virt_rows=p["rows"], virt_cols=p["cols"], d_az=p["d_az"], d_el=p["d_el"])
        else:
            kw.update(virt_rows=[0] * n_virt, virt_cols=list(range(n_virt)))
        if CONVENTIONS_PATH.exists():
            import json
            conv = json.load(open(CONVENTIONS_PATH))
            kw.update({k: int(v) for k, v in conv.items() if k in ("az_sign", "el_sign", "doppler_sign")})
            if "adc_padded" in conv:
                kw["adc_padded"] = bool(conv["adc_padded"])
        kw.update(overrides)
        return cls(**kw)

    def describe(self):
        return (f"{self.n_samples} samples x {self.n_chirps} chirps x {self.n_virt} virt | "
                f"dR={self.range_res_m * 100:.1f} cm, Rmax={self.range_axis[-1]:.1f} m, "
                f"vmax=+-{self.v_max:.2f} m/s, dv(frame)={self.dv_frame * 100:.1f} cm/s, "
                f"{1 / self.frame_period_s:.1f} fps")


def raw_frame_to_cube(frame, params):
    raw = (np.frombuffer(frame, dtype=np.int16)
           if isinstance(frame, (bytes, bytearray, memoryview)) else np.asarray(frame))
    if params.real_adc:
        if params.adc_padded:
            raw = raw.reshape(-1, 4)[:, :2].ravel()
        data = raw.astype(np.float32)
    else:
        quads = raw.reshape(-1, 4)
        data = (quads[:, 0:2] + 1j * quads[:, 2:4]).astype(np.complex64).reshape(-1)
    cube = data.reshape(params.n_chirps, params.n_tx, params.n_rx, params.n_samples)
    cube = cube[:, params.tx_reorder]
    cube = cube.reshape(params.n_chirps, params.n_virt, params.n_samples)
    return np.ascontiguousarray(cube.transpose(2, 0, 1))


def cube_to_raw_frame(cube, params):
    x = np.clip(np.round(np.real(cube)), -32768, 32767).astype("<i2")
    x = x.transpose(1, 2, 0).reshape(params.n_chirps, params.n_tx, params.n_rx, params.n_samples)
    out = np.empty_like(x)
    out[:, params.tx_reorder] = x
    out = out.ravel()
    if params.real_adc and params.adc_padded:
        g = np.empty((out.size // 2, 4), "<i2")
        g[:, :2] = out.reshape(-1, 2)
        g[:, 2], g[:, 3] = -1, 0
        out = g.ravel()
    return out.tobytes()


class ArrayGeometry:

    def __init__(self, params, n_az_fft=64, channel_cal=None):
        p = params
        rows, cols = np.asarray(p.virt_rows), np.asarray(p.virt_cols)
        self.p = p
        main = np.bincount(rows - rows.min()).argmax() + rows.min()
        self.main_row = int(main)
        az = np.flatnonzero(rows == main)
        self.az_idx = az[np.argsort(cols[az])]
        self.az_pos = cols[self.az_idx] - cols[self.az_idx].min()
        self.n_az_fft = int(max(n_az_fft, 2 * (self.az_pos.max() + 1)))
        self.az_taper = np.hanning(self.az_pos.max() + 3)[1:-1][self.az_pos].astype(np.float32)
        k = np.arange(self.n_az_fft) - self.n_az_fft // 2
        self.uy_grid = p.az_sign * k / (self.n_az_fft * p.d_az)
        self.uy_valid = np.abs(self.uy_grid) <= 1.0
        pairs, drow = [], []
        for ch in np.flatnonzero(rows != main):
            mate = np.flatnonzero((rows == main) & (cols == cols[ch]))
            if mate.size:
                pairs.append((ch, mate[0]))
                drow.append(rows[ch] - main)
        self.el_pairs = np.array(pairs, dtype=int).reshape(-1, 2)
        self.el_drow = float(np.mean(drow)) if drow else 0.0
        tx = np.arange(p.n_virt) // p.n_rx
        self.slot = np.asarray(p.tx_reorder)[tx]
        self.t_offset = self.slot * p.single_chirp_s
        self.cal = (np.ones(p.n_virt, np.complex64) if channel_cal is None
                    else np.asarray(channel_cal, np.complex64))
        self.inv_cal = (1.0 / self.cal).astype(np.complex64)
        self.rows, self.cols = rows, cols

    def az_spectrum(self, X):
        A = np.zeros(X.shape[:-1] + (self.n_az_fft,), np.complex64)
        A[..., self.az_pos] = X[..., self.az_idx] * self.az_taper
        return np.fft.fftshift(np.fft.fft(A, axis=-1), axes=-1)

    def uz_from(self, X):
        if not len(self.el_pairs) or self.el_drow == 0:
            return np.full(X.shape[:-1], np.nan)
        z = np.sum(X[..., self.el_pairs[:, 0]] * np.conj(X[..., self.el_pairs[:, 1]]), axis=-1)
        s = np.angle(z) / (2 * np.pi * self.p.d_el * self.el_drow * self.p.el_sign)
        return np.clip(s, -1.0, 1.0)

    def steering(self, uy, uz):
        p = self.p
        ph = 2 * np.pi * (p.d_az * self.cols * uy * p.az_sign + p.d_el * self.rows * uz * p.el_sign)
        return np.exp(1j * ph).astype(np.complex64)

    def tdm_compensate(self, X, v_mps):
        p = self.p
        fd = 2.0 * np.asarray(v_mps)[..., None] / p.wavelength * p.doppler_sign
        return X * np.exp(-2j * np.pi * fd * self.t_offset)


def padding_matches(frame, params, n_groups=4096):
    w = np.frombuffer(frame, dtype=np.int16, count=min(n_groups * 4, len(frame) // 2))
    g = w[:w.size // 4 * 4].reshape(-1, 4)
    return bool(np.mean((g[:, 2] == -1) & (g[:, 3] == 0)) > 0.99), bool(params.adc_padded)


def load_channel_cal(path=None):
    p = Path(path) if path else CHANNEL_CAL_PATH
    return np.load(p) if p.exists() else None


def direction(uy, uz):
    uy, uz = np.asarray(uy, float), np.nan_to_num(np.asarray(uz, float))
    ux = np.sqrt(np.clip(1.0 - uy ** 2 - uz ** 2, 0.0, 1.0))
    return np.stack([ux, uy, uz], axis=-1)


@dataclass
class FrameProducts:
    rfft: np.ndarray
    subband: np.ndarray
    profile: np.ndarray
    ra: np.ndarray
    cells: dict
    t: float = 0.0


class FramePreprocessor:

    def __init__(self, params, max_range_m=6.0, n_subbands=4, subband_chirp_step=8,
                 cell_threshold_db=12.0, max_cells=200, channel_cal=None):
        p = self.p = params
        self.geo = ArrayGeometry(params, channel_cal=channel_cal)
        self.n_bins = int(min(p.n_range_bins, max_range_m / p.range_res_m + 1))
        self.win = np.hanning(p.n_samples).astype(np.float32)[:, None, None]
        self.dwin = np.hanning(p.n_chirps).astype(np.float32)[None, :, None]
        self.n_sub = n_subbands
        self.seg = p.n_samples // n_subbands
        self.chirp_step = subband_chirp_step
        n = np.arange(self.seg)
        k = np.arange(self.n_bins)[:, None]
        self.sub_dft = (np.exp(-2j * np.pi * k * n / p.n_samples)
                        * np.hanning(self.seg)).astype(np.complex64)
        t_seg = (np.arange(n_subbands) + 0.5) * self.seg / p.fs_hz
        self.subband_freq_ghz = (p.f0_hz + p.slope_hz_per_s * (p.adc_start_s + t_seg)) / 1e9
        self.v_axis = ((np.arange(p.n_chirps) - p.n_chirps // 2) * p.dv_frame * p.doppler_sign)
        self.cell_thr = 10 ** (cell_threshold_db / 10)
        self.ra_step = max(1, p.n_chirps // 32)
        self.max_cells = max_cells

    def process(self, cube, t=0.0):
        p, geo = self.p, self.geo
        x = np.asarray(cube, dtype=np.float32 if p.real_adc else np.complex64)
        x = x - x.mean(axis=0, keepdims=True)
        if p.real_adc:
            X = np.fft.rfft(np.real(x) * self.win, axis=0)[:self.n_bins]
        else:
            X = np.fft.fft(x * self.win, axis=0)[:self.n_bins]
        X = (X * geo.inv_cal).astype(np.complex64)
        profile = np.mean(np.abs(X) ** 2, axis=(1, 2))

        xs = x[:self.n_sub * self.seg, ::self.chirp_step]
        Y = np.matmul(self.sub_dft[None], xs.reshape(self.n_sub, self.seg, -1).astype(np.complex64))
        Y = Y.reshape(self.n_sub, self.n_bins, xs.shape[1], -1) * geo.inv_cal
        subband = np.mean(np.abs(geo.az_spectrum(Y)) ** 2, axis=2)

        ra = np.mean(np.abs(geo.az_spectrum(X[:, ::self.ra_step])) ** 2, axis=1)

        D = np.fft.fftshift(np.fft.fft(X * self.dwin, axis=1), axes=1)
        Pd = np.sum(np.abs(D) ** 2, axis=2)
        floor = np.median(Pd)
        b, d = np.nonzero(Pd > floor * self.cell_thr)
        if b.size > self.max_cells:
            keep = np.argsort(Pd[b, d])[-self.max_cells:]
            b, d = b[keep], d[keep]
        v = self.v_axis[d]
        Xc = geo.tdm_compensate(D[b, d], v)
        S = np.abs(geo.az_spectrum(Xc)) ** 2
        S[:, ~geo.uy_valid] = 0
        uy = geo.uy_grid[np.argmax(S, axis=1)]
        cells = {"bin": b.astype(np.int16), "v": v.astype(np.float32),
                 "uy": uy.astype(np.float32), "uz": geo.uz_from(Xc).astype(np.float32),
                 "power": Pd[b, d].astype(np.float32), "floor": float(floor)}
        return FrameProducts(X, subband.astype(np.float32), profile.astype(np.float32),
                             ra.astype(np.float32), cells, float(t))


def ca_cfar_1d(power, guard=2, train=8, scale_db=10.0):
    n = power.size
    kernel = np.ones(2 * (guard + train) + 1)
    kernel[train:train + 2 * guard + 1] = 0
    noise = np.convolve(power, kernel, mode="same") / kernel.sum()
    hits = np.flatnonzero(power > noise * 10 ** (scale_db / 10))
    return np.array([i for i in hits
                     if power[i] >= power[max(i - 1, 0)] and power[i] >= power[min(i + 1, n - 1)]],
                    dtype=int)


def pulse_pair(x, params):
    R1 = np.sum(x[:, 1:] * np.conj(x[:, :-1]))
    R0 = 0.5 * (np.sum(np.abs(x[:, 1:]) ** 2) + np.sum(np.abs(x[:, :-1]) ** 2))
    v = params.doppler_sign * params.wavelength * np.angle(R1) / (4 * np.pi * params.chirp_period_s)
    return float(v), float(np.abs(R1) / max(R0, 1e-20))


def stft_spectrogram(x, prf_hz, nperseg=128, noverlap=96, nfft=256):
    nperseg = min(nperseg, x.size)
    noverlap = min(noverlap, nperseg - 1)
    f, t, Z = signal.stft(x, fs=prf_hz, window="hann", nperseg=nperseg, noverlap=noverlap,
                          nfft=max(nfft, nperseg), return_onesided=False,
                          boundary=None, padded=False)
    return np.fft.fftshift(f), t, np.fft.fftshift(np.abs(Z) ** 2, axes=0)


def micro_doppler_features(P, vel, t_real, v_static=0.12, cadence_band=(0.3, 5.0)):
    floor = np.percentile(P, 20, axis=0, keepdims=True)
    Pc = np.clip(P - 2 * floor, 0, None) + 1e-20
    w = Pc / Pc.sum(axis=0, keepdims=True)
    centroid = (vel[:, None] * w).sum(axis=0)
    bandwidth = np.sqrt(((vel[:, None] - centroid) ** 2 * w).sum(axis=0))
    cdf = np.cumsum(w, axis=0)
    v_lo = vel[np.argmax(cdf >= 0.05, axis=0)]
    v_hi = vel[np.argmax(cdf >= 0.95, axis=0)]
    total = Pc.sum()
    w_all = Pc.sum(axis=1) / total
    mu = (vel * w_all).sum()
    doppler_var = ((vel - mu) ** 2 * w_all).sum()
    dynamic_ratio = Pc[np.abs(vel) > v_static].sum() / total
    entropy = -(w * np.log(w)).sum(axis=0).mean() / np.log(len(vel))
    cadence_hz, cadence_strength = 0.0, 0.0
    if t_real.size >= 8:
        dt = np.median(np.diff(t_real))
        s = signal.detrend(bandwidth)
        S = np.abs(np.fft.rfft(s * np.hanning(s.size), n=4 * s.size)) ** 2
        fc = np.fft.rfftfreq(4 * s.size, dt)
        band = (fc >= cadence_band[0]) & (fc <= cadence_band[1])
        if band.any() and S[1:].sum() > 0:
            i = np.flatnonzero(band)[np.argmax(S[band])]
            cadence_hz, cadence_strength = fc[i], S[i] / S[1:].sum()
    return {
        "md_bandwidth_mps": float(bandwidth.mean()),
        "md_envelope_width_mps": float((v_hi - v_lo).mean()),
        "md_log_doppler_var": float(np.log10(doppler_var + 1e-5)),
        "md_centroid_std": float(centroid.std()),
        "md_dynamic_ratio": float(dynamic_ratio),
        "md_cadence_hz": float(cadence_hz),
        "md_cadence_strength": float(cadence_strength),
        "md_entropy": float(entropy),
    }
