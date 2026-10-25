"""
guarantee.py - when does FDR control hold? One range gate, one snapshot, two reflectors (0 and -3 dB, at least two
Rayleigh cells apart) under three models: exactly on the angle grid, between grid angles, and between grid angles with
5 deg phase errors per channel; the FDR of every method vs. SNR at alpha = 0.1; writes output/cache/guarantee.csv.

    python guarantee.py --workers 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from estimation import dataset  # noqa: E402
from estimation.core.array import ArrayModel  # noqa: E402
from estimation.core.interfaces import Gate, Settings  # noqa: E402
from estimation.core.metrics import Tolerance, frame_record  # noqa: E402
from estimation.core.radar import RadarConfig  # noqa: E402
from estimation.methods import doa_methods  # noqa: E402

CACHE = dataset.OUTPUT / "cache"
MODELS = ("on_grid", "off_grid", "phase_error")
SNR_GRID = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
TRIALS = 300
PHASE_ERROR_DEG = 5.0
METHODS = ["os_cfar", "bh", "ctrex_tlars", "ctrex_tomp", "trex_tlars", "infinirex", "infinirex_radar"]
SEED = 20261019
_W = {}


def _init():
    cfg = RadarConfig.from_cfg(dataset.CFG)
    array = ArrayModel(cfg)
    _W.update(array=array, tol=Tolerance.from_hardware(cfg, array), methods=doa_methods(METHODS))


def gate(model, snr, trial):
    array = _W["array"]
    rng = np.random.default_rng([SEED, zlib.crc32(model.encode()), int(snr), trial])
    while True:
        u = np.sort(rng.uniform(-0.75, 0.75, 2))
        if np.diff(u)[0] >= 2 * array.rayleigh_u:
            break
    if model == "on_grid":
        u = array.u_grid[np.argmin(np.abs(array.u_grid[None, :] - u[:, None]), axis=1)]
    s = np.sqrt(10 ** ((snr - np.array([0.0, 3.0])) / 10)) * np.exp(2j * np.pi * rng.uniform(size=2))
    y = array.steering(u) @ s
    if model == "phase_error":
        y = y * np.exp(1j * np.radians(PHASE_ERROR_DEG) * rng.standard_normal(array.n))
    y = y + (rng.standard_normal(array.n) + 1j * rng.standard_normal(array.n)) / np.sqrt(2)
    return Gate(0, 1.0, y[:, None], 1.0), np.c_[np.ones(2), array.u_to_deg(u)]


def task(t):
    model, snr, trial = t
    g, truth = gate(model, snr, trial)
    out = []
    for m in _W["methods"]:
        d = np.atleast_1d(m.fn(g, _W["array"], Settings(), np.random.default_rng([SEED, trial, zlib.crc32(m.name.encode())])).angles_deg)
        r = frame_record(np.c_[np.ones(d.size), d], truth, _W["tol"])
        out.append({"model": model, "snr_db": snr, "trial": trial, "method": m.name, "R": r["R"], "V": r["V"],
                    "FDP": r["FDP"], "TPP": r["TPP"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    tasks = [(m, s, k) for m in MODELS for s in SNR_GRID for k in range(TRIALS)]
    t0, rows = time.time(), []
    with Pool(a.workers, initializer=_init) as pool:
        for n, r in enumerate(pool.imap_unordered(task, tasks, chunksize=8), 1):
            rows += r
            if n % (len(tasks) // 10) == 0:
                print(f"  guarantee: {n}/{len(tasks)}, {(time.time() - t0) / 60:.1f} min", flush=True)
    dataset.write_csv(CACHE / "guarantee.csv", rows)


if __name__ == "__main__":
    main()
