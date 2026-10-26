"""
model_x_check.py - Model-X knockoffs where their assumption holds: the rows of the design are drawn independently from
a Gaussian with the covariance Model-X uses for the steering matrix, and the knockoffs are drawn from that true
covariance; 1-3 reflectors per scene, knockoff+ and offset 0 over the alpha sweep, with the time per gate; writes
output/cache/model_x_check.csv.

    python model_x_check.py --workers 3
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
from estimation.fdr_control import ALPHAS  # noqa: E402
from estimation.methods.model_x_knockoffs import (cov_shrink, glmnet_coefdiff, is_posdef, real_split, solve_sdp,  # noqa: E402
                                                  threshold)

CACHE = dataset.OUTPUT / "cache"
SNR_DB = 20.0
TRIALS = 300
SEED = 20261026
_W = {}


def design():
    s = dataset.load(dataset.MEASURED)
    Xr = real_split(s.array.dictionary(1), np.zeros(s.array.n, complex))[0]
    Sigma = np.cov(Xr, rowvar=False)
    if not is_posdef(Sigma):
        Sigma = cov_shrink(Xr)
    t0 = time.perf_counter()
    D = np.diag(solve_sdp(Sigma))
    sdp_s = time.perf_counter() - t0
    Sinv_D = np.linalg.solve(Sigma, D)
    V = 2 * D - D @ Sinv_D
    return {"Sigma": Sigma, "L": np.linalg.cholesky(Sigma), "Sinv_D": Sinv_D,
            "C": np.linalg.cholesky(0.5 * (V + V.T)).T, "n": Xr.shape[0], "p": Xr.shape[1] // 2, "sdp_s": sdp_s}


def _init(d):
    _W.update(d)


def trial(k):
    rng = np.random.default_rng([SEED, k])
    n, p = _W["n"], _W["p"]
    X = rng.standard_normal((n, 2 * p)) @ _W["L"].T
    angles = rng.choice(p, int(rng.integers(1, 4)), replace=False)
    beta = np.zeros(2 * p)
    phase = rng.uniform(0, 2 * np.pi, angles.size)
    amp = 10 ** (SNR_DB / 20) / np.sqrt(np.diag(_W["Sigma"])[angles])
    beta[angles], beta[angles + p] = amp * np.cos(phase), amp * np.sin(phase)
    y = X @ beta + rng.standard_normal(n)
    t0 = time.perf_counter()
    Xk = X - X @ _W["Sinv_D"] + rng.standard_normal(X.shape) @ _W["C"]
    W, _ = glmnet_coefdiff(X, Xk, y, rng)
    ms = 1e3 * (time.perf_counter() - t0)
    rows = []
    for method, offset in (("knockoffs", 1), ("knockoffs_offset0", 0)):
        for alpha in ALPHAS:
            sel = np.unique(np.flatnonzero(W >= threshold(W, alpha, offset)) % p)
            V = int(np.sum(~np.isin(sel, angles)))
            rows.append({"trial": k, "method": method, "level": alpha, "snr_db": SNR_DB, "n_true": angles.size,
                         "R": sel.size, "V": V, "TP": sel.size - V, "FDP": V / max(sel.size, 1),
                         "TPP": (sel.size - V) / angles.size, "ms": ms, "sdp_s": _W["sdp_s"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--trials", type=int, default=TRIALS)
    a = ap.parse_args()
    t0 = time.time()
    d = design()
    print(f"  design: n = {d['n']}, {d['p']} angles, SDP {d['sdp_s']:.1f} s", flush=True)
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(d,)) as pool:
        for n, r in enumerate(pool.imap_unordered(trial, range(a.trials), chunksize=2), 1):
            rows += r
            if n % max(1, a.trials // 5) == 0:
                print(f"  {n}/{a.trials} scenes, {(time.time() - t0) / 60:.1f} min", flush=True)
    dataset.write_csv(CACHE / "model_x_check.csv", rows)
    print(f"output/cache/model_x_check.csv: {len(rows)} rows, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
