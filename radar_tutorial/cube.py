"""
cube.py - raw DCA1000 bytes to the radar cube (frame, loop, virtual channel, sample): int16 samples in the order
loop, firing slot, RX, sample, put back into physical TX order (the chirps fire TX0, TX2, TX3, TX1).

    from radar_tutorial.cube import to_cube, read_capture
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def is_padded(words: np.ndarray) -> bool:
    g = np.asarray(words[: len(words) // 4 * 4][:16384]).reshape(-1, 4)
    return bool(g.size and np.mean((g[:, 2] == -1) & (g[:, 3] == 0)) > 0.99)


def to_samples(words: np.ndarray, real_adc: bool, padded: bool) -> np.ndarray:
    w = np.asarray(words)
    if real_adc:
        return (w.reshape(-1, 4)[:, :2].reshape(-1) if padded else w).astype(np.float32)
    q = w.reshape(-1, 4).astype(np.float32)
    return (q[:, :2] + 1j * q[:, 2:]).reshape(-1).astype(np.complex64)


def to_cube(samples: np.ndarray, radar) -> np.ndarray:
    n_tx, n_rx, n = radar.n_tx, radar.n_rx, radar.n_samples
    per_frame = radar.n_loops * n_tx * n_rx * n
    frames = samples.size // per_frame
    c = samples[: frames * per_frame].reshape(frames, radar.n_loops, n_tx, n_rx, n)
    c = c[:, :, radar.tx_reorder]
    return c.reshape(frames, radar.n_loops, n_tx * n_rx, n)


def read_capture(path, radar) -> np.ndarray:
    words = np.fromfile(Path(path), dtype="<i2")
    return to_cube(to_samples(words, radar.real_adc, radar.real_adc and is_padded(words)), radar)


def frame_from_bytes(data: bytes, radar) -> np.ndarray:
    return to_cube(to_samples(np.frombuffer(data, dtype="<i2"), radar.real_adc, False), radar)[0]
