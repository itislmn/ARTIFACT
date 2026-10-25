"""
evaluate.py - every method on every frame of three studies: the measured session S2 (first K = 1, 4, 16, 128 chirps of
each frame) and two Monte Carlo studies (vs. SNR, vs. K at a fixed SNR per chirp); writes output/predictions.csv and one
record per frame, K and method to output/cache/.

    python evaluate.py --only measured snr snapshots --workers 3            (--methods m1 m2: only these, the rest is kept)
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
from estimation.core.frame import process_frame  # noqa: E402
from estimation.core.interfaces import Settings  # noqa: E402
from estimation.core.metrics import Tolerance, frame_record  # noqa: E402
from estimation.core.radar import RadarConfig, open_recording  # noqa: E402
from estimation.methods import doa_methods  # noqa: E402
from estimation.simulation import draw_scene, render_frame  # noqa: E402

CACHE = dataset.OUTPUT / "cache"
SESSION_K = [1, 4, 16, 128]
SNR_GRID = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
SNR_TRIALS = 50
SNAPSHOT_K = [1, 2, 4, 8, 16, 32, 64, 128]
SNAPSHOT_SNR_PER_CHIRP = 15.0
SNAPSHOT_TRIALS = 30
SEED = 20261009
STUDIES = ("measured", "snr", "snapshots")
_W = {}


def record_rows(study, unit, scenario, frame, snr, K, res, truth, tol, methods):
    recs, preds = [], []
    for m in methods:
        det = res.detections[m.name]
        r = frame_record(det, truth, tol, n_cells=len(res.gates) * _W["array"].p)
        st = res.stats.get(m.name, [])
        recs.append({"study": study, "unit": unit, "scenario": scenario, "frame": frame, "snr_db": snr, "K": K,
                     "method": m.name, "applicable": int(res.applicable[m.name]), "n_true": r["n_true"], "R": r["R"],
                     "TP": r["TP"], "V": r["V"], "FDP": r["FDP"], "TPP": r["TPP"], "n_gates": len(res.gates),
                     "seconds": res.seconds[m.name], "peak_mb": res.peak_mb.get(m.name, float("nan")),
                     "steps": int(sum(s.get("steps", 0) for s in st)),
                     "found": ";".join(str(int(x)) for x in r["found"]),
                     "err_angle_deg": ";".join(f"{e:.4f}" for e in r["err_angle"]),
                     "err_range_m": ";".join(f"{e:.4f}" for e in r["err_range"])})
        matched = dict(r["pairs"])
        for i, (rg, az) in enumerate(det):
            j = matched.get(i, -1)
            preds.append({"study": study, "unit": unit, "scenario": scenario, "frame": frame, "snr_db": snr, "K": K,
                          "method": m.name, "range_m": rg, "azimuth_deg": az, "truth": j,
                          "truth_range_m": truth[j, 0] if j >= 0 else float("nan"),
                          "truth_azimuth_deg": truth[j, 1] if j >= 0 else float("nan")})
    return recs, preds


def _init(session_or_cfg, methods, settings, tol):
    _W.update(methods=doa_methods(methods), settings=settings, tol=tol)
    if isinstance(session_or_cfg, dataset.Session):
        _W.update(session=session_or_cfg, cfg=session_or_cfg.cfg, array=session_or_cfg.array)
    else:
        cfg = RadarConfig.from_cfg(session_or_cfg)
        _W.update(cfg=cfg, array=ArrayModel(cfg))


def session_task(task):
    study, t_idx, frame = task
    s = _W["session"]
    t = s.evaluated[t_idx]
    f = open_recording(t["path"], s.cfg)[1](frame)
    recs, preds = [], []
    for K in SESSION_K:
        res = process_frame(f[:, :K], s.cfg, s.array, _W["methods"], _W["settings"], [zlib.crc32(t["name"].encode()), frame, K],
                            s.calibration, s.background, profile_memory=(frame == 0))
        a, b = record_rows(study, t["name"], t["scenario"], frame, float("nan"), K, res, t["truth"], _W["tol"], _W["methods"])
        recs += a
        preds += b
    return recs, preds


def mc_task(task):
    study, scenario, snr, trial, Ks = task
    cfg, array = _W["cfg"], _W["array"]
    rng = np.random.default_rng([SEED, zlib.crc32(study.encode()), zlib.crc32(scenario.encode()), int(10 * snr) + 1000, trial])
    scene_snr = snr + (10 * np.log10(cfg.n_loops) if study == "snapshots" else 0.0)
    scene = draw_scene(scenario, scene_snr, rng, cfg)
    frame = render_frame(scene, cfg, rng, az_sign=array.az_sign)
    recs, preds = [], []
    for K in Ks:
        res = process_frame(frame[:, :K], cfg, array, _W["methods"], _W["settings"], [trial, int(10 * snr) + 1000, K],
                            profile_memory=(trial == 0))
        a, b = record_rows(study, f"{scenario}_{trial}", scenario, trial, snr, K, res, scene.truth, _W["tol"], _W["methods"])
        recs += a
        preds += b
    return recs, preds


def keep_others(path, methods):
    return [r for r in dataset.read_csv(path) if r["method"] not in methods] if path.exists() else []


def run(study, workers=3, frames=None, methods=None):
    t0 = time.time()
    if study == "measured":
        s = dataset.load(dataset.MEASURED)
        tasks = [(study, i, f) for i, t in enumerate(s.evaluated)
                 for f in range(min(open_recording(t["path"], s.cfg)[0], frames or 10 ** 6))]
        init, fn = (s, methods, s.settings, s.tolerance), session_task
        if frames is None:
            dataset.write_csv(CACHE / f"checks_{study}.csv", dataset.checks(s))
    else:
        cfg = RadarConfig.from_cfg(dataset.CFG)
        tol = Tolerance.from_hardware(cfg, ArrayModel(cfg))
        grid = SNR_GRID if study == "snr" else [SNAPSHOT_SNR_PER_CHIRP]
        Ks = [cfg.n_loops] if study == "snr" else SNAPSHOT_K
        trials = frames or (SNR_TRIALS if study == "snr" else SNAPSHOT_TRIALS)
        tasks = [(study, sc, snr, k, Ks) for sc in ("single", "multi") for snr in grid for k in range(trials)]
        init, fn = (str(dataset.CFG), methods, Settings(), tol), mc_task
    recs, preds = [], []
    with Pool(max(1, min(workers, len(tasks))), initializer=_init, initargs=init) as pool:
        for n, (a, b) in enumerate(pool.imap_unordered(fn, tasks, chunksize=1), 1):
            recs += a
            preds += b
            if n % max(1, len(tasks) // 10) == 0:
                print(f"  {study}: {n}/{len(tasks)} frames, {(time.time() - t0) / 60:.1f} min", flush=True)
    if methods and frames is None:
        recs, preds = (keep_others(CACHE / f"{kind}_{study}.csv", methods) + rows for kind, rows in (("records", recs), ("predictions", preds)))
    dataset.write_csv(CACHE / f"records_{study}.csv", recs)
    dataset.write_csv(CACHE / f"predictions_{study}.csv", preds)
    print(f"{study}: {len(recs)} records, {len(preds)} detections, {(time.time() - t0) / 60:.1f} min")


def merge_predictions():
    rows = []
    for study in STUDIES:
        p = CACHE / f"predictions_{study}.csv"
        if p.exists():
            rows += dataset.read_csv(p)
    dataset.write_csv(dataset.OUTPUT / "predictions.csv", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=list(STUDIES), choices=STUDIES)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    a = ap.parse_args()
    for study in a.only:
        run(study, a.workers, a.frames, a.methods)
    merge_predictions()


if __name__ == "__main__":
    main()
