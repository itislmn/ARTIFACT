"""
cfgfile.py - reads a TI mmWave .cfg (the CLI commands sent over UART) into the radar's parameters and derives range,
velocity and array numbers from them; frameCfg has 8 fields on this firmware (the 5th is the ADC sample count).

    from radar_tutorial.cfgfile import Radar, read_lines
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

C0 = 299_792_458.0

MEANING = {
    "sensorStop": "stop a running frame sequence",
    "flushCfg": "forget the previous configuration",
    "dfeDataOutputMode": "1 = frame-based chirps (one profile)",
    "channelCfg": "RX mask, TX mask, cascading (15 15 0: all 4 RX and 4 TX)",
    "adcCfg": "ADC bits (2 = 16 bit) and output (0 = real, the only raw format of the AWR2944 in TDM)",
    "adcbufCfg": "ADC buffer: subframe, real/complex, I/Q swap, channel interleave, chirps per interrupt",
    "lowPower": "low-power ADC mode off",
    "profileCfg": "chirp: start frequency, idle, ADC start, ramp end, power, phase, slope, TX start, samples, rate, HPF, gain",
    "chirpCfg": "one chirp of the loop: its profile and the TX that fires (bit mask)",
    "frameCfg": "chirps start/end, loops, frames (0 = endless), ADC samples, period in ms, trigger, delay",
    "guiMonitor": "which results the on-chip demo sends over the data UART (points, range profile, ...)",
    "cfarCfg": "on-chip CFAR in range (direction 0) and Doppler (direction 1)",
    "multiObjBeamForming": "on-chip second object in the same range-Doppler cell",
    "calibDcRangeSig": "on-chip removal of the near-range DC signal",
    "clutterRemoval": "on-chip static clutter removal (off)",
    "antGeometryCfg": "row and column of every virtual channel (physical TX order), then the spacings in wavelengths",
    "compRangeBiasAndRxChanPhase": "on-chip range bias and channel phase correction",
    "measureRangeBiasAndRxChanPhase": "on-chip calibration measurement (off)",
    "aoaFovCfg": "on-chip angle field of view: azimuth min/max, elevation min/max",
    "cfarFovCfg": "on-chip detection window in range (direction 0) or velocity (direction 1)",
    "extendedMaxVelocity": "on-chip velocity disambiguation (off)",
    "calibData": "on-chip calibration data in flash (off)",
    "lvdsStreamCfg": "LVDS stream to the DCA1000: subframe, HSI header, data (1 = raw ADC), software data",
    "sensorStart": "start chirping with everything above",
}


def read_lines(path) -> list[str]:
    out = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line and not line.startswith(("%", "#")):
            out.append(line)
    return out


@dataclass(frozen=True)
class Radar:
    f0_hz: float = 77e9
    idle_s: float = 7e-6
    adc_start_s: float = 7e-6
    ramp_end_s: float = 57.14e-6
    slope_hz_per_s: float = 70e12
    n_samples: int = 256
    fs_hz: float = 10e6
    real_adc: bool = True
    rx_mask: int = 15
    tx_firing: tuple = (1, 4, 8, 2)
    n_loops: int = 16
    n_frames: int = 0
    frame_period_s: float = 0.1
    geometry: tuple = ()
    d_az: float = 0.5
    d_el: float = 0.8
    lines: tuple = field(default=(), repr=False)

    @classmethod
    def from_cfg(cls, path) -> "Radar":
        lines = read_lines(path)
        kw = {"lines": tuple(lines)}
        firing = {}
        for line in lines:
            cmd, *v = line.split()
            if cmd == "profileCfg":
                kw.update(f0_hz=float(v[1]) * 1e9, idle_s=float(v[2]) * 1e-6, adc_start_s=float(v[3]) * 1e-6,
                          ramp_end_s=float(v[4]) * 1e-6, slope_hz_per_s=float(v[7]) * 1e12, n_samples=int(v[9]),
                          fs_hz=float(v[10]) * 1e3)
            elif cmd == "adcCfg":
                kw["real_adc"] = int(v[1]) == 0
            elif cmd == "channelCfg":
                kw["rx_mask"] = int(v[0])
            elif cmd == "chirpCfg":
                firing[int(v[0])] = int(v[7])
            elif cmd == "frameCfg":
                kw.update(n_loops=int(v[2]), n_frames=int(v[3]),
                          frame_period_s=float(v[5] if len(v) >= 8 else v[4]) * 1e-3)
            elif cmd == "antGeometryCfg":
                nums = [float(x) for x in v]
                kw.update(geometry=tuple((int(a), int(b)) for a, b in zip(nums[:-2:2], nums[1:-2:2])),
                          d_az=nums[-2], d_el=nums[-1])
        kw["tx_firing"] = tuple(firing[k] for k in sorted(firing))
        return cls(**kw)

    @property
    def n_rx(self) -> int:
        return bin(self.rx_mask).count("1")

    @property
    def n_tx(self) -> int:
        return len(self.tx_firing)

    @property
    def tx_reorder(self) -> list[int]:
        slot = {mask.bit_length() - 1: k for k, mask in enumerate(self.tx_firing)}
        return [slot[t] for t in range(self.n_tx)]

    @property
    def bandwidth_hz(self) -> float:
        return self.slope_hz_per_s * self.n_samples / self.fs_hz

    @property
    def range_res_m(self) -> float:
        return C0 / (2 * self.bandwidth_hz)

    @property
    def n_range_bins(self) -> int:
        return self.n_samples // 2 if self.real_adc else self.n_samples

    @property
    def range_axis_m(self) -> np.ndarray:
        return np.arange(self.n_range_bins) * self.range_res_m

    @property
    def wavelength_m(self) -> float:
        fc = self.f0_hz + self.slope_hz_per_s * (self.adc_start_s + 0.5 * self.n_samples / self.fs_hz)
        return C0 / fc

    @property
    def tx_period_s(self) -> float:
        return self.n_tx * (self.idle_s + self.ramp_end_s)

    @property
    def v_max_mps(self) -> float:
        return self.wavelength_m / (4 * self.tx_period_s)

    @property
    def v_res_mps(self) -> float:
        return self.wavelength_m / (2 * self.n_loops * self.tx_period_s)

    @property
    def velocity_axis_mps(self) -> np.ndarray:
        return (np.arange(self.n_loops) - self.n_loops // 2) * self.v_res_mps

    @property
    def virtual_positions(self) -> np.ndarray:
        return np.array(self.geometry, int)[:, ::-1] if self.geometry else np.c_[np.arange(self.n_tx * self.n_rx),
                                                                                 np.zeros(self.n_tx * self.n_rx, int)]

    @property
    def azimuth_channels(self) -> np.ndarray:
        col, row = self.virtual_positions.T
        main = np.bincount(row).argmax()
        idx = np.flatnonzero(row == main)
        return idx[np.argsort(col[idx], kind="stable")]

    @property
    def bytes_per_frame(self) -> int:
        return self.n_samples * self.n_rx * self.n_tx * self.n_loops * (2 if self.real_adc else 4)

    def summary(self) -> dict:
        return {"bandwidth_ghz": self.bandwidth_hz / 1e9, "range_res_cm": 100 * self.range_res_m,
                "max_range_m": float(self.range_axis_m[-1]), "wavelength_mm": 1e3 * self.wavelength_m,
                "tx_period_us": 1e6 * self.tx_period_s, "v_max_mps": self.v_max_mps, "v_res_mps": self.v_res_mps,
                "frame_rate_hz": 1 / self.frame_period_s, "n_virtual": self.n_tx * self.n_rx,
                "n_azimuth": int(self.azimuth_channels.size), "kib_per_frame": self.bytes_per_frame / 1024}
