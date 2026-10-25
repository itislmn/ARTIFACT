"""
complexity.py - processing time (median per range gate, one process, one BLAS thread, after a warm-up) and peak memory
(traced in a separate run) of every method vs. grid size p, regression rows n and chirps K, with the fitted exponent
of time ~ p^a; writes output/cache/complexity.csv.

    python complexity.py            (--methods knockoffs: only these methods, --sweep K: only this sweep; the rest is kept)
"""
from __future__ import annotations

import os
import sys
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from estimation import dataset  # noqa: E402
from estimation.core.array import ArrayModel  # noqa: E402
from estimation.core.frame import method_rng, propose_gates  # noqa: E402
from estimation.core.interfaces import Settings  # noqa: E402
from estimation.core.radar import RadarConfig, range_fft  # noqa: E402
from estimation.methods import doa_methods  # noqa: E402
from estimation.simulation import draw_scene, render_frame  # noqa: E402

Q_SWEEP = [2, 4, 8, 16]
G_SWEEP = [1, 2, 4, 8]
K_SWEEP = [1, 4, 16, 128]
REGRESSION = ("ctrex", "trex", "custom")
SNR_DB = 20.0
FRAMES = 2


def gates(cfg, array, settings, K=None):
    snr = SNR_DB if K is None else SNR_DB + 10 * np.log10(cfg.n_loops)
    out = []
    for scen in ("single", "multi"):
        for i in range(FRAMES):
            rng = np.random.default_rng([7, i, len(scen)])
            frame = render_frame(draw_scene(scen, snr, rng, cfg), cfg, rng)
            frame = frame if K is None else frame[:, :K]
            out += propose_gates(range_fft(frame, cfg), frame.mean(axis=1), cfg, array, settings)
    return out


def fresh(m):
    for name in (m.fn.__module__, "estimation.methods.model_x_knockoffs"):
        getattr(sys.modules.get(name), "_STATS", {}).clear()


def measure(m, gs, array, settings):
    if not m.fn(gs[0], array, settings, method_rng((7, 0), m.name)).info.get("applicable", True):
        return None
    times = []
    for i, g in enumerate(gs):
        rng = method_rng((7, i), m.name)
        fresh(m)
        t0 = time.perf_counter()
        m.fn(g, array, settings, rng)
        times.append(time.perf_counter() - t0)
    peaks = []
    for i, g in enumerate(gs[:2]):
        fresh(m)
        tracemalloc.start()
        m.fn(g, array, settings, method_rng((7, i), m.name))
        peaks.append(tracemalloc.get_traced_memory()[1] / 2 ** 20)
        tracemalloc.stop()
    return {"seconds_per_gate": float(np.median(times)), "peak_mb_per_gate": float(np.max(peaks))}


def main():
    only = sys.argv[sys.argv.index("--methods") + 1:] if "--methods" in sys.argv else None
    sweeps = [sys.argv[sys.argv.index("--sweep") + 1]] if "--sweep" in sys.argv else ["p", "n", "K"]
    cfg = RadarConfig.from_cfg(dataset.CFG)
    methods = doa_methods(only)
    settings = Settings()
    rows = []
    t0 = time.time()
    for Q in Q_SWEEP if "p" in sweeps else []:
        array = ArrayModel(cfg, oversampling=Q)
        gs = gates(cfg, array, settings)
        for m in methods:
            r = measure(m, gs, array, settings)
            if r:
                rows.append(dict(r, sweep="p", method=m.name, p=array.p, n=array.n, K=cfg.n_loops))
        print(f"  p = {array.p}: done, {(time.time() - t0) / 60:.1f} min", flush=True)
    array = ArrayModel(cfg)
    gs = gates(cfg, array, settings)
    for G in G_SWEEP if "n" in sweeps else []:
        st = replace(settings, n_groups=G)
        for m in methods:
            r = measure(m, gs, array, st) if m.family in REGRESSION or m.name in ("knockoffs", "infinirex") else None
            if r:
                rows.append(dict(r, sweep="n", method=m.name, p=array.p, n=G * array.n, K=cfg.n_loops))
        print(f"  n = {G * array.n}: done, {(time.time() - t0) / 60:.1f} min", flush=True)
    for K in K_SWEEP if "K" in sweeps else []:
        gk = gates(cfg, array, settings, K)
        for m in methods:
            r = measure(m, gk, array, settings)
            if r:
                rows.append(dict(r, sweep="K", method=m.name, p=array.p, n=array.n, K=K))
        print(f"  K = {K}: done, {(time.time() - t0) / 60:.1f} min", flush=True)
    path = dataset.OUTPUT / "cache" / "complexity.csv"
    if (only or len(sweeps) < 3) and path.exists():
        rows = [r for r in dataset.read_csv(path)
                if not ((not only or r["method"] in only) and r["sweep"] in sweeps)] + rows
    dataset.write_csv(path, rows)


def exponent(rows, method, sweep, key):
    pts = sorted((float(r[key]), float(r["seconds_per_gate"])) for r in rows if r["method"] == method and r["sweep"] == sweep)
    if len(pts) < 2:
        return float("nan")
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])


if __name__ == "__main__":
    main()
