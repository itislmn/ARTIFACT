"""
complexity.py - run time of both readers against the board range on synthetic frames: MetricSight resamples the region
to a fixed metric image (constant work), the CNN works in camera pixels (work grows with 1/R^2).

    python complexity.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from backend.settings import load_config  # noqa: E402
from readers import cnn_reader, metricsight  # noqa: E402
from readers.glyphs import TEST_FONTS, word_scene  # noqa: E402

RANGES = np.round(np.arange(0.5, 2.01, 0.25), 2)
REPEATS = 3
F_PX = 1112.0


def measure() -> list[dict]:
    cfg = load_config()
    rng = np.random.default_rng(0)
    rw, rh = cfg["text"]["region_m"]
    rows = []
    for R in RANGES:
        scenes = [np.repeat(word_scene("ARTIFACT", font, int(F_PX * cfg["text"]["letter_mm"] / 1000 / R), rng=rng)
                            [..., None], 3, axis=2).astype(np.uint8) for font in TEST_FONTS]
        for name, mod in (("MetricSight", metricsight), ("CNN", cnn_reader)):
            mod.read(scenes[0], (640, 360), R, F_PX, cfg)
            ms = []
            for rgb in scenes:
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    mod.read(rgb, (640, 360), R, F_PX, cfg)
                    ms.append(1e3 * (time.perf_counter() - t0))
            pixels = (rw * 1000 / cfg["text"]["mm_per_px"]) * (rh * 1000 / cfg["text"]["mm_per_px"]) \
                if name == "MetricSight" else (rw * F_PX / R) * (rh * F_PX / R)
            rows.append({"reader": name, "range_m": float(R), "ms": float(np.median(ms)), "ms_q25": float(np.percentile(ms, 25)),
                         "ms_q75": float(np.percentile(ms, 75)), "pixels": float(pixels)})
    return rows


if __name__ == "__main__":
    for r in measure():
        print(f"  {r['reader']:<12} {r['range_m']:.2f} m  {r['ms']:6.1f} ms  {r['pixels']:9.0f} px")
