"""
radar.py - the AWR2944P radar parameters from the .cfg, the virtual array, the range FFT and access to the raw
DCA1000 recordings.

    from estimation.core.radar import RadarConfig, open_recording
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

C0 = 299_792_458.0

_AWR2944_ROWS = (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)
_AWR2944_COLS = (0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 10, 11)


@dataclass(frozen=True)
class RadarConfig:
    f0_hz: float = 77e9
    slope_hz_per_s: float = 70e12
    fs_hz: float = 10e6
    n_samples: int = 256
    adc_start_s: float = 7e-6
    idle_s: float = 26.5e-6
    ramp_end_s: float = 36e-6
    n_rx: int = 4
    n_tx: int = 4
    n_loops: int = 128
    n_frames: int = 50
    frame_period_s: float = 0.1
    real_adc: bool = True
    tx_reorder: tuple = (0, 3, 1, 2)
    virt_rows: tuple = _AWR2944_ROWS
    virt_cols: tuple = _AWR2944_COLS
    d_az: float = 0.5
    d_el: float = 0.8
    source: str = "defaults"
    virt_x: tuple = ()
    virt_z: tuple = ()

    @property
    def x_lam(self) -> np.ndarray:
        return np.asarray(self.virt_x, float) if self.virt_x else np.asarray(self.virt_cols, float) * self.d_az

    @property
    def z_lam(self) -> np.ndarray:
        if self.virt_z:
            return np.asarray(self.virt_z, float)
        rows = np.asarray(self.virt_rows)
        return np.where(rows != rows.max(), self.d_el, 0.0)

    @property
    def bandwidth_hz(self) -> float:
        return self.slope_hz_per_s * self.n_samples / self.fs_hz

    @property
    def range_res_m(self) -> float:
        return C0 / (2.0 * self.bandwidth_hz)

    @property
    def n_range_bins(self) -> int:
        return self.n_samples // 2 if self.real_adc else self.n_samples

    @property
    def range_axis_m(self) -> np.ndarray:
        return np.arange(self.n_range_bins) * self.range_res_m

    @property
    def max_range_m(self) -> float:
        return float(self.range_axis_m[-1])

    @property
    def wavelength_m(self) -> float:
        fc = self.f0_hz + self.slope_hz_per_s * (self.adc_start_s + 0.5 * self.n_samples / self.fs_hz)
        return C0 / fc

    @property
    def chirp_period_s(self) -> float:
        return self.n_tx * (self.idle_s + self.ramp_end_s)

    @property
    def n_virt(self) -> int:
        return self.n_tx * self.n_rx

    @property
    def azimuth_channels(self) -> np.ndarray:
        rows, cols = np.asarray(self.virt_rows), np.asarray(self.virt_cols)
        main = np.bincount(rows - rows.min()).argmax() + rows.min()
        idx = np.flatnonzero(rows == main)
        return idx[np.argsort(cols[idx], kind="stable")]

    @property
    def azimuth_columns(self) -> np.ndarray:
        cols = np.asarray(self.virt_cols)[self.azimuth_channels]
        return cols - cols.min()

    @property
    def payload_bytes_per_frame(self) -> int:
        per_sample = 2 if self.real_adc else 4
        return self.n_samples * self.n_rx * self.n_tx * self.n_loops * per_sample

    def wire_bytes_per_frame(self, padded: bool = True) -> int:
        return self.payload_bytes_per_frame * (2 if (self.real_adc and padded) else 1)

    @classmethod
    def from_cfg(cls, path, **overrides) -> "RadarConfig":
        chirp_tx, p = {}, {}
        for raw in open(path, encoding="utf-8", errors="ignore"):
            line = raw.strip()
            if not line or line.startswith(("%", "#")):
                continue
            tok = re.split(r"\s+", line)
            cmd, v = tok[0], tok[1:]
            if cmd == "profileCfg":
                p["f0_hz"] = float(v[1]) * 1e9
                p["idle_s"] = float(v[2]) * 1e-6
                p["adc_start_s"] = float(v[3]) * 1e-6
                p["ramp_end_s"] = float(v[4]) * 1e-6
                p["slope_hz_per_s"] = float(v[7]) * 1e12
                p["n_samples"] = int(v[9])
                p["fs_hz"] = float(v[10]) * 1e3
            elif cmd == "channelCfg":
                p["n_rx"] = bin(int(v[0])).count("1")
            elif cmd == "adcCfg":
                p["real_adc"] = int(v[1]) == 0
            elif cmd == "chirpCfg":
                chirp_tx[int(v[0])] = int(v[7])
            elif cmd == "frameCfg":
                start, end = int(v[0]), int(v[1])
                p["_chirps_per_loop"] = end - start + 1
                p["n_loops"] = int(v[2])
                p["n_frames"] = int(v[3])
                p["frame_period_s"] = float(v[5] if len(v) >= 8 else v[4]) * 1e-3
            elif cmd == "antGeometryCfg":
                nums = [float(t) for t in v]
                pairs = nums[:-2]
                p["virt_rows"] = tuple(int(a) for a in pairs[0::2])
                p["virt_cols"] = tuple(int(a) for a in pairs[1::2])
                p["d_az"], p["d_el"] = nums[-2], nums[-1]

        n_tx = len({m for m in chirp_tx.values() if m}) or 1
        perm = list(range(n_tx))
        for slot, mask in chirp_tx.items():
            if mask and (mask.bit_length() - 1) < n_tx:
                perm[mask.bit_length() - 1] = slot
        p.pop("_chirps_per_loop", None)
        n_virt = n_tx * p.get("n_rx", 4)
        if len(p.get("virt_rows", ())) != n_virt:
            p["virt_rows"], p["virt_cols"] = (0,) * n_virt, tuple(range(n_virt))
        p.update(n_tx=n_tx, tx_reorder=tuple(perm), source=str(path))
        p.update(overrides)
        return cls(**p)

    def with_loops(self, n_loops: int) -> "RadarConfig":
        return replace(self, n_loops=int(n_loops))

    def describe(self) -> str:
        return (rf"$f_0 = {self.f0_hz / 1e9:.1f}\,\mathrm{{GHz}}$, "
                rf"$S = {self.slope_hz_per_s / 1e12:.0f}\,\mathrm{{MHz}}/\mu\mathrm{{s}}$, "
                rf"$B = {self.bandwidth_hz / 1e9:.3f}\,\mathrm{{GHz}}$, "
                rf"$\Delta R = {self.range_res_m * 100:.2f}\,\mathrm{{cm}}$, "
                rf"$R_\mathrm{{max}} = {self.max_range_m:.2f}\,\mathrm{{m}}$, "
                rf"$N_s = {self.n_samples}$, $K = {self.n_loops}$, "
                rf"$N_v = {self.n_virt}$ ({len(self.azimuth_channels)} azimuth), "
                rf"$\lambda = {self.wavelength_m * 1e3:.3f}\,\mathrm{{mm}}$, "
                rf"ADC {'real' if self.real_adc else 'complex'}")


def detect_padding(words: np.ndarray, n_groups: int = 4096) -> bool:
    w = np.asarray(words[: min(4 * n_groups, len(words) // 4 * 4)])
    if w.size < 4:
        return False
    g = w.reshape(-1, 4)
    return bool(np.mean((g[:, 2] == -1) & (g[:, 3] == 0)) > 0.99)


def frame_from_words(words: np.ndarray, cfg: RadarConfig, padded: bool) -> np.ndarray:
    w = np.asarray(words)
    if cfg.real_adc:
        if padded:
            w = w.reshape(-1, 4)[:, :2].reshape(-1)
        data = w.astype(np.float32)
    else:
        q = w.reshape(-1, 4).astype(np.float32)
        data = (q[:, 0:2] + 1j * q[:, 2:4]).astype(np.complex64).reshape(-1)
    cube = data.reshape(cfg.n_loops, cfg.n_tx, cfg.n_rx, cfg.n_samples)
    cube = cube[:, list(cfg.tx_reorder)]
    cube = cube.reshape(cfg.n_loops, cfg.n_virt, cfg.n_samples)
    return np.ascontiguousarray(cube.transpose(2, 0, 1))


def _npy_first_frame(path: Path, n_frames: int) -> int:
    meta = path.with_name(path.stem + ".meta.json")
    if not meta.exists():
        return 0
    t = np.asarray(json.load(open(meta)).get("frame_t") or [], float)
    if t.size != n_frames:
        return 0
    gaps = np.flatnonzero(np.diff(t) > 1.0)
    return int(gaps[-1]) + 1 if gaps.size else 0


def open_recording(path, cfg: RadarConfig):
    path = Path(path)
    if path.suffix == ".npy":
        stack = np.load(path, mmap_mode="r")
        if stack.ndim != 4 or stack.shape[1:] != (cfg.n_samples, cfg.n_loops, cfg.n_virt):
            raise ValueError(f"{path.name}: shape {stack.shape} does not match the cfg "
                             f"(frames, {cfg.n_samples}, {cfg.n_loops}, {cfg.n_virt}); "
                             "pass the cfg the take was recorded with")
        first = _npy_first_frame(path, stack.shape[0])
        return stack.shape[0] - first, lambda i: np.asarray(stack[first + i], dtype=np.float32)

    words = np.memmap(path, dtype="<i2", mode="r")
    padded = detect_padding(words) if cfg.real_adc else False
    per_frame = cfg.wire_bytes_per_frame(padded) // 2
    n_frames = words.size // per_frame
    if n_frames == 0:
        raise ValueError(f"{path.name}: {words.size} int16 words, one frame needs {per_frame} "
                         f"(padding {'on' if padded else 'off'}); wrong cfg or empty capture")
    return n_frames, lambda i: frame_from_words(words[i * per_frame:(i + 1) * per_frame], cfg, padded)


def load_truth(recording_path):
    p = Path(recording_path)
    j = p.with_suffix(".json")
    return json.load(open(j, encoding="utf-8")) if j.exists() else None


def range_window(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def range_fft(frame: np.ndarray, cfg: RadarConfig, channel_cal=None) -> np.ndarray:
    x = np.asarray(frame)
    x = x - x.mean(axis=0, keepdims=True)
    w = range_window(cfg.n_samples)[:, None, None]
    if cfg.real_adc:
        X = np.fft.rfft(np.real(x).astype(np.float32) * w, axis=0)[: cfg.n_range_bins]
    else:
        X = np.fft.fft(x.astype(np.complex64) * w, axis=0)
    X = X.astype(np.complex64)
    if channel_cal is not None:
        X = X / np.asarray(channel_cal, np.complex64)[None, None, :]
    return X


def with_layout(cfg: RadarConfig, path) -> RadarConfig:
    lay = json.loads(Path(path).read_text(encoding="utf-8"))
    lam_mm = cfg.wavelength_m * 1e3
    x = [(lay["tx_x_mm"][t] + lay["rx_x_mm"][r]) / lam_mm for t in range(cfg.n_tx) for r in range(cfg.n_rx)]
    z = [lay["tx_z_mm"][t] / lam_mm for t in range(cfg.n_tx) for r in range(cfg.n_rx)]
    return replace(cfg, virt_x=tuple(x), virt_z=tuple(z), source=f"{cfg.source} + {Path(path).name}")
