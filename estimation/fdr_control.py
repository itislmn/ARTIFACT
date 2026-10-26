"""
fdr_control.py - does each FDR method keep its promise: the achieved FDR and the TPR of every FDR-controlled method at
nominal levels alpha from 0.02 to 0.5 and at every chirp count K, on the measured session and in simulation (one and
three reflectors); writes output/cache/sweep_measured.csv and sweep_simulation.csv.

    python fdr_control.py --workers 3       (--methods m1 m2 / --levels 0.15 0.25 / --only measured: the rest is kept)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from estimation import dataset  # noqa: E402
from estimation.core.array import ArrayModel  # noqa: E402
from estimation.core.frame import process_frame  # noqa: E402
from estimation.core.interfaces import Settings  # noqa: E402
from estimation.core.metrics import Tolerance, frame_record  # noqa: E402
from estimation.core.radar import RadarConfig, open_recording  # noqa: E402
from estimation.evaluate import SNAPSHOT_K, SNAPSHOT_SNR_PER_CHIRP  # noqa: E402
from estimation.methods import WITH_ALPHA, doa_methods  # noqa: E402
from estimation.simulation import draw_scene, render_frame  # noqa: E402

CACHE = dataset.OUTPUT / "cache"
METHODS = ["bh", "by", "knockoffs", "knockoffs_offset0", "knockoffs_lowdim", "knockoffs_lowdim_offset0", "ctrex_tlars",
           "ctrex_tomp", "trex_tlars", "infinirex", "infinirex_radar"]
ALPHAS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
SESSION_K = [1, 4, 16, 128]
SIM_TRIALS = 40
_W = {}


def _init(session, settings, tol, methods, levels):
    _W.update(session=session, settings=settings, tol=tol, methods=doa_methods(methods), levels=levels)
    if session is None:
        cfg = RadarConfig.from_cfg(dataset.CFG)
        _W.update(cfg=cfg, array=ArrayModel(cfg))
    else:
        _W.update(cfg=session.cfg, array=session.array)


def _rows(study, unit, frame, K, frame_data, truth, cal=None, bg=None):
    out = []
    scenario = unit.rsplit("_", 1)[0] if study == "simulation" else ""
    for m in _W["methods"]:
        for level in _W["levels"]:
            st = replace(_W["settings"], alpha=level)
            res = process_frame(frame_data[:, :K], _W["cfg"], _W["array"], [m], st, [frame, K], cal, bg)
            r = frame_record(res.detections[m.name], truth, _W["tol"])
            out.append({"study": study, "unit": unit, "scenario": scenario, "frame": frame, "K": K, "method": m.name,
                        "param": "alpha", "level": level, "R": r["R"], "TP": r["TP"], "V": r["V"], "n_true": r["n_true"],
                        "FDP": r["FDP"], "TPP": r["TPP"]})
    return out


def session_task(task):
    t_idx, frame = task
    s = _W["session"]
    t = s.evaluated[t_idx]
    f = open_recording(t["path"], s.cfg)[1](frame)
    return [row for K in SESSION_K for row in _rows(s.name, t["name"], frame, K, f, t["truth"], s.calibration, s.background)]


def sim_task(task):
    scenario, trial = task
    cfg = _W["cfg"]
    rng = np.random.default_rng([20261016, len(scenario), trial])
    scene = draw_scene(scenario, SNAPSHOT_SNR_PER_CHIRP + 10 * np.log10(cfg.n_loops), rng, cfg)
    frame = render_frame(scene, cfg, rng)
    return [row for K in SNAPSHOT_K for row in _rows("simulation", f"{scenario}_{trial}", trial, K, frame, scene.truth)]


def _pool(tasks, fn, initargs, workers, label):
    t0, rows = time.time(), []
    with Pool(max(1, min(workers, len(tasks))), initializer=_init, initargs=initargs) as pool:
        for n, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=1), 1):
            rows += r
            if n % max(1, len(tasks) // 10) == 0:
                print(f"  {label}: {n}/{len(tasks)}, {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def kept(name, methods, levels):
    path = CACHE / name
    if not path.exists():
        return []
    return [r for r in dataset.read_csv(path) if not (r["method"] in methods and float(r["level"]) in levels)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--levels", nargs="+", type=float, default=ALPHAS)
    ap.add_argument("--only", nargs="+", default=["measured", "simulation"], choices=["measured", "simulation"])
    a = ap.parse_args()
    assert set(a.methods) <= set(WITH_ALPHA)
    if "measured" in a.only:
        s = dataset.load(dataset.MEASURED)
        tasks = [(i, f) for i, t in enumerate(s.evaluated) for f in range(open_recording(t["path"], s.cfg)[0])]
        rows = _pool(tasks, session_task, (s, s.settings, s.tolerance, a.methods, a.levels), a.workers, "sweep measured")
        dataset.write_csv(CACHE / "sweep_measured.csv", kept("sweep_measured.csv", a.methods, a.levels) + rows)
    if "simulation" not in a.only:
        return
    cfg = RadarConfig.from_cfg(dataset.CFG)
    tol = Tolerance.from_hardware(cfg, ArrayModel(cfg))
    tasks = [(sc, k) for sc in ("single", "multi") for k in range(SIM_TRIALS)]
    rows = _pool(tasks, sim_task, (None, Settings(), tol, a.methods, a.levels), a.workers, "sweep simulation")
    dataset.write_csv(CACHE / "sweep_simulation.csv", kept("sweep_simulation.csv", a.methods, a.levels) + rows)


if __name__ == "__main__":
    main()
