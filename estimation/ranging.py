"""
ranging.py - the three range estimators (FFT bin centre, FFT with a parabola through the peak, CZT zoom) on the measured
session (every reflector in every frame against the tape, the calibration take for the offset, every pair of reflectors
closer than 3.5 range cells resolved or not, judged by the spacing of the two peaks) and in simulation (precision vs. SNR against the Cramer-Rao bound, resolution
of two equal reflectors vs. their spacing); writes output/cache/range_*.csv.

    python ranging.py --workers 3       (--only resolution: the rest is kept)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from estimation import dataset  # noqa: E402
from estimation.core.array import refine_peak  # noqa: E402
from estimation.core.radar import C0, RadarConfig, open_recording  # noqa: E402
from estimation.methods import range_czt, range_fft  # noqa: E402
from estimation.simulation import Reflector, Scene, amplitude_for_snr, render_frame  # noqa: E402

CACHE = dataset.OUTPUT / "cache"
ESTIMATORS = ("fft", "parabolic", "czt")
SNR_GRID = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
PRECISION_TRIALS = 200
SEPARATIONS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
RESOLUTION_SNR = 30.0
RESOLUTION_TRIALS = 200
SESSION_K = [1, 128]
PAIR_CELLS = 3.5
SEED = 20261018


def estimate(x, cfg, k):
    return {"fft": float(range_fft.estimate(x, cfg, [k])[0]),
            "parabolic": float(range_fft.estimate(x, cfg, [k], "parabolic")[0]),
            "czt": float(range_czt.estimate(x, cfg, [k])[0])}


def peak_near(x, cfg, r, half=2):
    r_axis, P = range_fft.range_profile(x, cfg)
    k0 = int(round(r / cfg.range_res_m))
    peaks = [k for k in range(max(1, k0 - half), min(P.size - 1, k0 + half + 1)) if P[k] >= P[k - 1] and P[k] >= P[k + 1]]
    if not peaks:
        return k0
    return min(peaks, key=lambda k: (abs(k - r / cfg.range_res_m), -P[k]))


def crb_range_m(snr_db, cfg, n_channels):
    a = np.array([amplitude_for_snr(float(s), cfg) for s in np.atleast_1d(snr_db)])
    N = cfg.n_samples
    var_w = 12.0 / (n_channels * cfg.n_loops * a ** 2 / 2.0 * N * (N ** 2 - 1))
    return np.sqrt(var_w) * cfg.fs_hz / (2 * np.pi) * C0 / (2 * cfg.slope_hz_per_s)


def measured_rows(session):
    cfg, bg = session.cfg, session.background
    rows = []
    cals = [t for t in session.takes if t["scenario"] == "calibration" and t["truth"].size]
    for t in session.evaluated + cals:
        n, get = open_recording(t["path"], cfg)
        for i in range(n):
            f = np.asarray(get(i), float)
            for K in SESSION_K:
                x = f[:, :K].mean(axis=1) - (bg["adc"] if bg is not None else 0.0)
                for j, (r, _) in enumerate(t["truth"]):
                    est = estimate(x, cfg, peak_near(x, cfg, r))
                    for m in ESTIMATORS:
                        rows.append({"take": t["name"], "role": "calibration" if t in cals else "evaluated", "reflector": j, "label": t["doc"]["targets"][j].get("label", ""),
                                     "frame": i, "K": K, "method": m, "range_true_m": r, "range_m": est[m],
                                     "error_cm": 100 * (est[m] - r)})
    return rows


def measured_resolution_rows(session):
    cfg, bg = session.cfg, session.background
    rows = []
    for t in session.evaluated:
        r = t["truth"][:, 0]
        pairs = [(i, j) for i in range(r.size) for j in range(i + 1, r.size) if abs(r[j] - r[i]) < PAIR_CELLS * cfg.range_res_m]
        if not pairs:
            continue
        n, get = open_recording(t["path"], cfg)
        for f in range(n):
            frame = np.asarray(get(f), float)
            for K in SESSION_K:
                x = frame[:, :K].mean(axis=1) - (bg["adc"] if bg is not None else 0.0)
                for i, j in pairs:
                    truths = np.sort(r[[i, j]])
                    for m, ok in resolutions(x, cfg, truths, spacing=True).items():
                        rows.append({"take": t["name"], "pair": f"{i}-{j}", "frame": f, "K": K, "method": m,
                                     "separation_cells": (truths[1] - truths[0]) / cfg.range_res_m, "resolved": int(ok)})
    return rows


def precision_task(task):
    snr, trial = task
    cfg = RadarConfig.from_cfg(dataset.CFG)
    rng = np.random.default_rng([SEED, int(10 * snr) + 1000, trial])
    r = float(rng.uniform(1.0, 4.0))
    scene = Scene("single", snr, [Reflector(r, float(rng.uniform(-30, 30)), snr, float(rng.uniform(0, 2 * np.pi)))])
    x = render_frame(scene, cfg, rng).astype(float).mean(axis=1)
    est = estimate(x, cfg, peak_near(x, cfg, r))
    return [{"snr_db": snr, "trial": trial, "method": m, "range_true_m": r, "error_cm": 100 * (est[m] - r)} for m in ESTIMATORS]


def local_maxima(P):
    return [i for i in range(1, P.size - 1) if P[i] > P[i - 1] and P[i] >= P[i + 1]]


def resolved(r_axis, P, truths, sep_m, parabola=False, spacing=False):
    w = (r_axis >= truths.min() - 1.5 * sep_m) & (r_axis <= truths.max() + 1.5 * sep_m)
    ra, Pw = r_axis[w], P[w]
    peaks = sorted(local_maxima(Pw), key=lambda i: -Pw[i])[:2]
    if len(peaks) < 2:
        return False
    est = np.sort([refine_peak(ra, 10 * np.log10(Pw + 1e-30), i) if parabola else ra[i] for i in peaks])
    if spacing:
        return bool(abs(est[1] - est[0] - sep_m) < 0.5 * sep_m)
    return bool(np.all(np.abs(est - np.sort(truths)) < 0.5 * sep_m))


def resolution_task(task):
    sep, trial = task
    cfg = RadarConfig.from_cfg(dataset.CFG)
    rng = np.random.default_rng([SEED + 1, int(100 * sep), trial])
    r1 = float(rng.uniform(1.5, 3.5))
    sep_m = sep * cfg.range_res_m
    refl = [Reflector(r1, 0.0, RESOLUTION_SNR, float(rng.uniform(0, 2 * np.pi))),
            Reflector(r1 + sep_m, 0.0, RESOLUTION_SNR, float(rng.uniform(0, 2 * np.pi)))]
    x = render_frame(Scene("multi", RESOLUTION_SNR, refl), cfg, rng).astype(float).mean(axis=1)
    return [{"separation_cells": sep, "trial": trial, "method": m, "resolved": int(ok)}
            for m, ok in resolutions(x, cfg, np.array([r1, r1 + sep_m])).items()]


def resolutions(x, cfg, truths, spacing=False):
    sep_m = truths[1] - truths[0]
    r_fft, P_fft = range_fft.range_profile(x, cfg)
    r_czt, P_czt = range_czt.zoom_profile(x, cfg, truths[0] - 2 * cfg.range_res_m - sep_m,
                                          truths[1] + 2 * cfg.range_res_m + sep_m, n_points=1025)
    return {m: resolved(ra, P, truths, sep_m, m == "parabolic", spacing)
            for m, (ra, P) in (("fft", (r_fft, P_fft)), ("parabolic", (r_fft, P_fft)), ("czt", (r_czt, P_czt)))}


def pool_rows(tasks, fn, workers, label):
    t0, rows = time.time(), []
    with Pool(workers) as pool:
        for n, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=4), 1):
            rows += r
            if n % max(1, len(tasks) // 5) == 0:
                print(f"  {label}: {n}/{len(tasks)}, {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="+", default=["measured", "precision", "resolution"],
                    choices=["measured", "precision", "resolution"])
    a = ap.parse_args()
    if "measured" in a.only:
        session = dataset.load(dataset.MEASURED)
        dataset.write_csv(CACHE / "range_measured.csv", measured_rows(session))
        dataset.write_csv(CACHE / "range_resolution_measured.csv", measured_resolution_rows(session))
        print("  range measured: done", flush=True)
    if "precision" in a.only:
        dataset.write_csv(CACHE / "range_precision.csv",
                          pool_rows([(s, k) for s in SNR_GRID for k in range(PRECISION_TRIALS)], precision_task, a.workers, "precision"))
    if "resolution" in a.only:
        dataset.write_csv(CACHE / "range_resolution.csv",
                          pool_rows([(s, k) for s in SEPARATIONS for k in range(RESOLUTION_TRIALS)], resolution_task, a.workers,
                                    "resolution"))


if __name__ == "__main__":
    main()
