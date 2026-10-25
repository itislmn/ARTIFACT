"""
sync.py - pairs radar frames with camera frames by their common QueryPerformanceCounter time.

    from backend import sync
"""
from __future__ import annotations

import numpy as np


def match(radar_times, camera_times) -> list[tuple[int, int, float]]:
    rt, ct = np.asarray(radar_times, float), np.asarray(camera_times, float)
    if not rt.size or not ct.size:
        return []
    j = np.clip(np.searchsorted(ct, rt), 1, len(ct) - 1)
    j = np.where(np.abs(ct[j - 1] - rt) <= np.abs(ct[j] - rt), j - 1, j)
    return [(int(i), int(k), round(1e3 * float(ct[k] - r), 2)) for i, (k, r) in enumerate(zip(j, rt))]
