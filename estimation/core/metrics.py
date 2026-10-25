"""
metrics.py - one-to-one matching of detections to the ground truth (Hungarian, within the tolerance) and the error
rates FDR, TPR, FWER, precision, RMSE and the Cramer-Rao bound of the angle.

    from estimation.core.metrics import Tolerance, frame_record, summarize
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Tolerance:
    range_m: float
    angle_deg: float

    @classmethod
    def from_hardware(cls, cfg, array) -> "Tolerance":
        return cls(float(cfg.range_res_m), float(np.degrees(np.arcsin(0.5 * array.rayleigh_u))))

    def describe(self) -> str:
        return (rf"$\Delta R = {self.range_m * 100:.1f}\,\mathrm{{cm}}$, "
                rf"$\Delta\theta = {self.angle_deg:.2f}^\circ$")


def match(detections: np.ndarray, truth: np.ndarray, tol: Tolerance):
    det = np.asarray(detections, float).reshape(-1, 2)
    tru = np.asarray(truth, float).reshape(-1, 2)
    if det.size == 0 or tru.size == 0:
        return []
    dR = (det[:, None, 0] - tru[None, :, 0]) / tol.range_m
    dA = (det[:, None, 1] - tru[None, :, 1]) / tol.angle_deg
    allowed = (np.abs(dR) <= 1.0) & (np.abs(dA) <= 1.0)
    if not allowed.any():
        return []
    cost = np.where(allowed, dR ** 2 + dA ** 2, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if allowed[i, j]]


def frame_record(detections: np.ndarray, truth: np.ndarray, tol: Tolerance, n_cells: int = 0) -> dict:
    det = np.asarray(detections, float).reshape(-1, 2)
    tru = np.asarray(truth, float).reshape(-1, 2)
    pairs = match(det, tru, tol)
    R, TP = det.shape[0], len(pairs)
    V = R - TP
    found = np.zeros(tru.shape[0], bool)
    err_r, err_a = [], []
    for i, j in pairs:
        found[j] = True
        err_r.append(det[i, 0] - tru[j, 0])
        err_a.append(det[i, 1] - tru[j, 1])
    return {"R": R, "TP": TP, "V": V, "n_true": int(tru.shape[0]),
            "FDP": V / max(R, 1), "TPP": TP / max(tru.shape[0], 1),
            "found": found.tolist(), "pairs": pairs, "err_range": err_r, "err_angle": err_a, "n_cells": int(n_cells)}


def summarize(records: list) -> dict:
    M = len(records)
    if M == 0:
        return {}
    fdp = np.array([r["FDP"] for r in records])
    tpp = np.array([r["TPP"] for r in records])
    V = np.array([r["V"] for r in records])
    R = np.array([r["R"] for r in records])
    TP = np.array([r["TP"] for r in records])
    ea = np.concatenate([np.asarray(r["err_angle"], float) for r in records])
    er = np.concatenate([np.asarray(r["err_range"], float) for r in records])
    cells = sum(r["n_cells"] for r in records)
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(M)) if M > 1 else float("nan")
    return {
        "n_frames": M,
        "FDR": float(fdp.mean()), "FDR_se": se(fdp),
        "TPR": float(tpp.mean()), "TPR_se": se(tpp),
        "FWER": float(np.mean(V >= 1)),
        "precision": float(TP.sum() / R.sum()) if R.sum() else float("nan"),
        "rmse_angle_deg": float(np.sqrt(np.mean(ea ** 2))) if ea.size else float("nan"),
        "rmse_range_cm": float(100 * np.sqrt(np.mean(er ** 2))) if er.size else float("nan"),
        "mean_detections": float(R.mean()),
        "pfa_emp": float(V.sum() / cells) if cells else float("nan"),
    }


SUMMARY_COLUMNS = ["scenario", "method", "snapshot", "snr_db", "K", "n_frames", "FDR", "FDR_se", "TPR",
                   "TPR_se", "FWER", "precision", "rmse_angle_deg", "rmse_range_cm", "mean_detections",
                   "pfa_emp", "seconds_per_frame", "seconds_per_gate", "peak_mb_per_gate", "solver_steps",
                   "dummies_L", "gates_per_frame"]


def summary_table(records: list, keys=("scenario", "method", "snr_db", "K")) -> list:
    groups = {}
    for r in records:
        groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    rows = []
    for key, recs in groups.items():
        s = summarize(recs)
        s.update(dict(zip(keys, key)))
        s["snapshot"] = recs[0].get("snapshot", "")
        gates = np.array([r.get("n_gates", 0) for r in recs], float)
        secs = np.array([r.get("seconds", np.nan) for r in recs], float)
        s["seconds_per_frame"] = float(np.nanmedian(secs)) if np.isfinite(secs).any() else float("nan")
        s["gates_per_frame"] = float(gates.mean())
        per_gate = secs[gates > 0] / gates[gates > 0]
        s["seconds_per_gate"] = float(np.nanmedian(per_gate)) if per_gate.size else float("nan")
        pk = np.array([r.get("peak_mb", np.nan) for r in recs], float)
        s["peak_mb_per_gate"] = float(np.nanmean(pk)) if np.isfinite(pk).any() else float("nan")
        s["solver_steps"] = float(np.mean([r.get("solver_steps", 0) for r in recs]))
        s["dummies_L"] = float(np.mean([r.get("dummies_L", 0) for r in recs]))
        rows.append(s)
    return sorted(rows, key=lambda r: tuple(str(r.get(k)) if isinstance(r.get(k), str) else (r.get(k) or 0)
                                            for k in keys))


def records_from_trials(path, n_cells_per_gate: int, drop_methods=()) -> list:
    recs = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["method"] in drop_methods:
                continue
            rec = dict(r)
            for k in ("n_true", "R", "TP", "V", "n_gates", "solver_steps", "dummies_L", "K", "trial", "frame"):
                if rec.get(k) not in (None, ""):
                    rec[k] = int(float(rec[k]))
            for k in ("FDP", "TPP", "seconds", "peak_mb", "snr_db"):
                if rec.get(k) not in (None, ""):
                    rec[k] = float(rec[k])
            rec["err_angle"] = [float(x) for x in (r.get("err_angle_deg") or "").split(";") if x]
            rec["err_range"] = [float(x) for x in (r.get("err_range_m") or "").split(";") if x]
            rec["n_cells"] = int(float(r["n_cells"])) if r.get("n_cells") else rec["n_gates"] * n_cells_per_gate
            rec["found"] = [x == "1" for x in (r.get("found") or "").split(";") if x != ""]
            recs.append(rec)
    return recs


def crb_theta_deg(snr_db, n: int = 12, d: float = 0.5, theta_max_deg: float = 45.0):
    rho = 10.0 ** (np.asarray(snr_db, float) / 10.0)
    var_u = 6.0 / ((2 * np.pi * d) ** 2 * rho * n * (n ** 2 - 1))
    tm = np.radians(theta_max_deg)
    return np.degrees(np.sqrt(var_u * np.tan(tm) / tm))


def write_csv(path, rows: list, columns: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in rows:
            w.writerow([(f"{row.get(c):.6g}" if isinstance(row.get(c), float) else row.get(c, ""))
                        for c in columns])
