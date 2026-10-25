"""
doa_study.py - which direction-of-arrival method finds which object from the moving platform: the beam peak, spatial
OS-CFAR, OMP stopped at the noise level, CT-Rex t-OMP (estimation/) and Infini-Rex as the method uses it, all on the
method's candidate cells of the recorded runs, every detection matched to the measured scene -> output/doa/<run>.csv.

    python doa_study.py            (computes only the runs without a result in output/doa/)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.stats import gamma  # noqa: E402

import dataset  # noqa: E402
import paths  # noqa: E402
from backend.radar_process import channel_cal  # noqa: E402
from estimation.core.interfaces import Gate as CellGate, Settings  # noqa: E402
from estimation.methods import doa_methods  # noqa: E402
from methods.beam_search import BeamSearch  # noqa: E402
from methods.infinirex import FrameSelector  # noqa: E402
from methods.path_reference import PathReference  # noqa: E402
from methods.radar_config import frame_from_words  # noqa: E402
from methods.range_doppler import FrontEnd  # noqa: E402
from methods.waveforms import Library  # noqa: E402
from methods.world import World  # noqa: E402

OUT = paths.OUTPUT / "doa"
STRIDE = 3
METHODS = ("beam_peak", "os_cfar", "omp_noise", "ctrex_tomp", "infinirex_look")
LABELS = {"beam_peak": "Beam peak", "os_cfar": "Spatial OS-CFAR", "omp_noise": "OMP, noise stop",
          "ctrex_tomp": "CT-Rex t-OMP", "infinirex_look": "Infini-Rex"}
CLASSICAL = ("beam_peak", "os_cfar", "omp_noise")
RANGE_CELL_M = 0.0837


def raw_runs() -> list[str]:
    return [r for r in dataset.RUNS if (paths.TAKES / r / "radar_raw.bin").exists()]


def candidates(fe, rd, ref, pose, pfa: float, cap: int, beam):
    gates = fe.search_gates(rd, pfa, static_line=True)
    gates += beam(fe, rd, gates)
    L = ref.level_map(pose) if ref is not None else None
    A = fe.array.steering(fe.array.u_grid).conj()
    explained = set()
    for g in gates:
        k = ref.row(g.range_m) if L is not None else -1
        if L is not None and abs(g.d - fe.K // 2) <= 1 and k >= 0 and \
                not np.any(np.abs(rd.D[g.k, g.d, fe.az] @ A) ** 2 / fe.array.n > L[k]):
            explained.add(g.cell)
    return sorted(gates, key=lambda g: (g.cell in explained, -g.excess))[:cap]


def beam_peak(y, array):
    p = np.abs(array.steering(array.u_grid).conj().T @ y) ** 2
    i = int(np.argmax(p))
    if 0 < i < p.size - 1:
        den = p[i - 1] - 2 * p[i] + p[i + 1]
        du = array.u_grid[1] - array.u_grid[0]
        return [float(array.u_grid[i] + (0.5 * (p[i - 1] - p[i + 1]) / den * du if den < 0 else 0.0))]
    return [float(array.u_grid[i])]


def omp_noise(y, noise_ch, array, pfa: float):
    W = np.diag(1 / np.sqrt(noise_ch))
    X = W @ array.steering(array.u_grid)
    X = X / np.linalg.norm(X, axis=0, keepdims=True)
    r, Q, picked = W @ y, np.zeros((len(y), 0), complex), []
    stop = gamma.ppf(1 - pfa, len(y))
    while np.vdot(r, r).real > stop and len(picked) < len(y) // 2:
        j = int(np.argmax(np.abs(X.conj().T @ r)))
        if j in picked:
            break
        q = X[:, j] - Q @ (Q.conj().T @ X[:, j])
        q /= np.linalg.norm(q)
        Q = np.column_stack([Q, q])
        r = r - q * np.vdot(q, r)
        picked.append(j)
    return [float(array.u_grid[j]) for j in picked]


def angles(method, gates, rd, fe, sel, specs, settings, rng, pfa):
    az = fe.az
    if method == "infinirex_look":
        ev = sel.evalues(rd, gates)
        picked = sel.weighted_ebh(ev["E"], np.ones_like(ev["E"]))
        out = [[] for _ in gates]
        for d in sel.detections(rd, gates, picked, fe, ev, np.ones_like(ev["E"]), fe.cfg):
            out[d["gate"]].append(d["u"])
        return out
    out = []
    for g in gates:
        y = rd.D[g.k, g.d, az].astype(np.complex128)
        if method == "beam_peak":
            out.append(beam_peak(y, fe.array))
        elif method == "omp_noise":
            out.append(omp_noise(y, rd.noise_ch[az], fe.array, pfa))
        else:
            cell = CellGate(int(g.k), g.range_m, y[:, None], float(np.mean(rd.noise_ch[az])),
                            channel_noise=rd.noise_ch[az].astype(float))
            deg = specs[method].fn(cell, fe.array, settings, rng).angles_deg
            out.append(list(np.sin(np.radians(np.asarray(deg, float)))))
    return out


def study_run(run: str) -> list[dict]:
    cfg = paths.load_config()
    tk = dataset.load(run)
    folder = paths.TAKES / run
    lib = Library(cfg)
    main = lib.names[0]
    cal, _ = channel_cal(cfg)
    ref_path = folder.parent / "reference" / "path_reference.npz"
    ref = PathReference.load(ref_path) if ref_path.exists() else None
    world = World(cfg)
    if ref is not None:
        world.set_known(ref)
    known = [(so.lo[:2], so.hi[:2]) for so in world.known]
    rc, dc = cfg["radar"], cfg["detection"]
    fe = FrontEnd(lib[main], rc["r_min_m"], rc["r_max_m"], cal, None)
    fe_still = FrontEnd(lib[main], rc["r_min_m"], rc["r_max_m"], cal, {"rfft": ref.bg_rfft}) \
        if ref is not None and ref.bg_rfft is not None else None
    sel = FrameSelector(fe.array, dc["alpha"])
    beam = BeamSearch(fe.array, dc["gate_pfa"])
    specs = {s.name: s for s in doa_methods(["os_cfar", "ctrex_tomp"])}
    settings = Settings(alpha=dc["alpha"], pfa=dc["gate_pfa"], gate_pfa=dc["gate_pfa"], r_min_m=rc["r_min_m"])
    rng = np.random.default_rng(0)
    index = [json.loads(line) for line in open(folder / "radar_raw.jsonl")]
    words = np.memmap(folder / "radar_raw.bin", dtype="<i2", mode="r")
    full = lib[main].cfg.wire_bytes_per_frame(True) // 2
    padded = words.size >= full * len(index) - 1
    per = full if padded else full // 2
    R = np.genfromtxt(folder / "rail.csv", delimiter=",", names=True)
    ts, P = R["t"], np.c_[R["x"], R["y"], R["z"]]
    pose_at = lambda t: np.array([np.interp(t, ts, P[:, a]) for a in range(3)])
    looks = [i for i, r in enumerate(index) if r.get("chirp", r.get("waveform", main)) == main][::STRIDE]
    rows = []
    for n, i in enumerate(looks):
        t = index[i]["t"] - lib[main].active_s / 2
        pose = pose_at(t)
        v = (pose_at(t + 0.1) - pose) / 0.1
        still = fe_still is not None and ref.still_background(pose, v) is not None
        front = fe_still if still else fe
        rd = front.process(frame_from_words(np.asarray(words[i * per:(i + 1) * per]), lib[main].cfg, padded))
        gates = candidates(front, rd, ref, pose, dc["gate_pfa"], dc["gates_per_frame"], beam)
        truth = [o for o in tk.truth if o.active(t)]
        view = [o.id for o in truth if dataset.in_view(o, pose, cfg)]
        for m in METHODS:
            t0 = time.perf_counter()
            us = angles(m, gates, rd, front, sel, specs, settings, rng, dc["gate_pfa"])
            ms = 1e3 * (time.perf_counter() - t0)
            found, n_det, n_room, n_false = set(), 0, 0, 0
            for g, ug in zip(gates, us):
                for u in ug:
                    r, _, _ = front.refine(rd, g.k, g.d, float(u))
                    p = pose[:2] + r * np.array([np.sqrt(max(1 - u * u, 0.0)), u])
                    tol = max(RANGE_CELL_M, r / 12.0)
                    n_det += 1
                    hit = [o.id for o in truth if np.all(p >= o.lo[:2] - tol) and np.all(p <= o.hi[:2] + tol)]
                    if hit:
                        found.update(hit)
                    elif any(np.all(p >= lo - tol) and np.all(p <= hi + tol) for lo, hi in known):
                        n_room += 1
                    else:
                        n_false += 1
            rows.append({"run": run, "t_s": round(t - tk.t0, 3), "rail_x_m": round(float(pose[0]), 4), "method": m,
                         "cells": len(gates), "detections": n_det, "room": n_room, "false": n_false,
                         "found": ";".join(sorted(o for o in found if o in view)), "in_view": ";".join(sorted(view)),
                         "ms": round(ms, 2)})
        if n % 50 == 0:
            print(f"  {run}: look {n}/{len(looks)}", flush=True)
    return rows


def run(again: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    for r in raw_runs():
        path = OUT / f"{r.replace('/', '_')}.csv"
        if again or not path.exists():
            rows = study_run(r)
            with open(path, "w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
    return summary()


def summary() -> dict:
    rows = [r for p in sorted(OUT.glob("*.csv")) for r in csv.DictReader(open(p, encoding="utf-8"))]
    if not rows:
        return {}
    kind = {}
    for r in {x["run"] for x in rows}:
        tk = dataset.load(r)
        kind.update({(r, o.id): o.kind for o in tk.truth})
    out = {}
    for m in METHODS:
        mr = [r for r in rows if r["method"] == m]
        fdp = [int(r["false"]) / int(r["detections"]) if int(r["detections"]) else 0.0 for r in mr]
        per_kind = {}
        for r in mr:
            found = set(filter(None, r["found"].split(";")))
            for o in filter(None, r["in_view"].split(";")):
                k = kind.get((r["run"], o), "other")
                a = per_kind.setdefault(k, [0, 0])
                a[0] += o in found
                a[1] += 1
        out[m] = {"label": LABELS[m], "classical": m in CLASSICAL, "looks": len(mr),
                  "detections": float(np.mean([int(r["detections"]) for r in mr])),
                  "fdr": float(np.mean(fdp)), "recall": {k: v[0] / v[1] for k, v in per_kind.items() if v[1]},
                  "n_view": {k: v[1] for k, v in per_kind.items()}, "ms": float(np.median([float(r["ms"]) for r in mr]))}
    return out


if __name__ == "__main__":
    res = run(again="--again" in sys.argv)
    for m, v in res.items():
        rec = ", ".join(f"{k} {100 * x:.0f} %" for k, x in sorted(v["recall"].items()))
        print(f"{v['label']:26s} {v['detections']:5.1f} det/look  FDR {v['fdr']:.3f}  found: {rec}  {v['ms']:.1f} ms")
