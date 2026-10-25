"""
model_order.py - why the classical methods over-count on measured data: per range gate the MDL order and the
eigenvalues of the smoothed covariance against the number of reflectors in the gate, measured and simulated, at K = 1
and K = 128 chirps; writes output/cache/model_order.csv.

    python model_order.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from estimation import dataset  # noqa: E402
from estimation.core.array import classical_subspace  # noqa: E402
from estimation.core.frame import propose_gates, range_fft  # noqa: E402
from estimation.core.interfaces import Settings  # noqa: E402
from estimation.core.radar import open_recording  # noqa: E402
from estimation.evaluate import SNAPSHOT_SNR_PER_CHIRP  # noqa: E402
from estimation.simulation import draw_scene, render_frame  # noqa: E402

CACHE = dataset.OUTPUT / "cache"
K_VALUES = (1, 128)
FRAME_STEP = 4
SIM_SCENES = 40


def gate_rows(source, unit, frame, truth, cfg, array, settings, tol, cal=None, bg=None):
    X = range_fft(frame, cfg, cal)
    x_avg = np.asarray(frame, float).mean(axis=1)
    if bg is not None:
        X = X - bg["rfft"][:, None, :].astype(X.dtype)
        x_avg = x_avg - bg["adc"]
    rows = []
    for g in propose_gates(X, x_avg, cfg, array, settings):
        n_true = int(np.sum(np.abs(truth[:, 0] - g.range_m) <= tol.range_m)) if truth.size else 0
        for K in K_VALUES:
            sub = classical_subspace(g.snapshots[:, :K])
            lam_db = 10 * np.log10(sub["eigvals"] / sub["eigvals"][-1])
            rows.append({"source": source, "unit": unit, "range_m": g.range_m, "K": K, "n_true": n_true,
                         "k_hat": sub["k_hat"], **{f"lam{i + 1}_db": float(v) for i, v in enumerate(lam_db)}})
    return rows


def main():
    s = dataset.load(dataset.MEASURED)
    rows = []
    for t in s.evaluated:
        n, get = open_recording(t["path"], s.cfg)
        for i in range(0, n, FRAME_STEP):
            rows += gate_rows("measured", t["name"], get(i), t["truth"], s.cfg, s.array, s.settings, s.tolerance,
                              s.calibration, s.background)
    rng = np.random.default_rng(20261025)
    for k in range(SIM_SCENES):
        scene = draw_scene("multi", SNAPSHOT_SNR_PER_CHIRP + 10 * np.log10(s.cfg.n_loops), rng, s.cfg)
        rows += gate_rows("simulated", f"multi_{k}", render_frame(scene, s.cfg, rng), scene.truth, s.cfg, s.array,
                          Settings(), s.tolerance)
    dataset.write_csv(CACHE / "model_order.csv", rows)
    print(f"output/cache/model_order.csv: {len(rows)} gates x K")


if __name__ == "__main__":
    main()
