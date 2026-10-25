"""
simulate.py - synthetic raw ADC bytes in the exact wire order of the AWR2944P (loop, firing slot, RX, sample; real
int16) for point targets at known range, velocity and azimuth, so every step can be checked against the truth.

    from radar_tutorial.simulate import raw_bytes
"""
from __future__ import annotations

import numpy as np

from radar_tutorial.cfgfile import C0


def raw_bytes(radar, targets, frames=1, snr_db=20.0, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    n = np.arange(radar.n_samples)
    lam = radar.wavelength_m
    cols = radar.virtual_positions[:, 0]
    t_chirp = radar.idle_s + radar.ramp_end_s
    x = np.zeros((frames, radar.n_loops, radar.n_tx, radar.n_rx, radar.n_samples))
    for f in range(frames):
        for l in range(radar.n_loops):
            for s, mask in enumerate(radar.tx_firing):
                tx = mask.bit_length() - 1
                t = f * radar.frame_period_s + (l * radar.n_tx + s) * t_chirp
                for r in range(radar.n_rx):
                    c = cols[tx * radar.n_rx + r]
                    sig = np.zeros(radar.n_samples, complex)
                    for rng_m, vel, az, amp in targets:
                        fb = 2 * radar.slope_hz_per_s * rng_m / C0
                        ph = 4 * np.pi * (rng_m + vel * t) / lam + 2 * np.pi * radar.d_az * c * np.sin(np.radians(az))
                        sig += amp * np.exp(1j * (2 * np.pi * fb * n / radar.fs_hz + ph))
                    x[f, l, s, r] = sig.real
    sigma = np.sqrt(0.5 * 10 ** (-snr_db / 10))
    x = x + rng.normal(0.0, sigma, x.shape)
    scale = 4000.0 / max(np.abs(x).max(), 1e-12)
    return np.round(x * scale).astype("<i2").tobytes()
