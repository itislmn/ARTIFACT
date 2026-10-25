"""
baseline.py - the classical chain to compare with: OS-CFAR cells, one beam peak per cell, the known room masked,
DBSCAN in the world plane, global nearest neighbour association with constant-velocity Kalman filters and M-of-N
confirmation, static or moving by the points' Doppler; no FDR control, no prediction regions, no classification.
Same front end, same settings.

    from methods.baseline import BaselineScene
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from methods.range_doppler import FrontEnd
from methods.world import World

GATE_CHI2 = 9.21
MIN_POINTS = 2


def dbscan(P: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    n = len(P)
    lab = np.full(n, -1)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2) <= eps
    core = D.sum(axis=1) >= min_points
    c = 0
    for i in range(n):
        if lab[i] != -1 or not core[i]:
            continue
        lab[i], todo = c, [i]
        while todo:
            j = todo.pop()
            for k in np.flatnonzero(D[j]):
                if lab[k] == -1:
                    lab[k] = c
                    if core[k]:
                        todo.append(k)
        c += 1
    return lab


class Track:
    def __init__(self, tid: int, z: np.ndarray, t: float, sigma: float, v0: float):
        self.id, self.t, self.born = tid, t, t
        self.s = np.r_[z, 0.0, 0.0]
        self.P = np.diag([sigma ** 2, sigma ** 2, v0 ** 2, v0 ** 2])
        self.hits = deque([1], maxlen=8)
        self.confirmed, self.last_update, self.n_updates = False, t, 1
        self.lo = self.hi = None
        self.n_points = 0
        self.still = 1.0


class GNNTracker:
    def __init__(self, cfg: dict):
        tc = cfg["tracking"]
        self.q, self.v0 = tc["accel_sigma_mps2"], tc["max_speed_mps"]
        self.m, self.n, self.lost = tc["confirm_hits"], tc["confirm_window"], tc["lost_after_s"]
        self.tracks: list[Track] = []
        self.doppler_sign = 1
        self.next_id = 1

    def step(self, Z: np.ndarray, R: np.ndarray, t: float) -> tuple[list[int], int, int]:
        for tr in self.tracks:
            dt = max(t - tr.t, 0.0)
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
            tr.s, tr.P, tr.t = F @ tr.s, F @ tr.P @ F.T + self.q ** 2 * G @ G.T, t
        H = np.eye(2, 4)
        owner = [-1] * len(Z)
        if self.tracks and len(Z):
            C = np.full((len(self.tracks), len(Z)), 1e6)
            for i, tr in enumerate(self.tracks):
                for j, z in enumerate(Z):
                    S = H @ tr.P @ H.T + R[j]
                    e = z - tr.s[:2]
                    d2 = float(e @ np.linalg.solve(S, e))
                    if d2 <= GATE_CHI2:
                        C[i, j] = d2
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < 1e6:
                    owner[j] = i
        updated = set()
        for j, i in enumerate(owner):
            if i < 0:
                continue
            tr = self.tracks[i]
            S = H @ tr.P @ H.T + R[j]
            K = tr.P @ H.T @ np.linalg.inv(S)
            tr.s = tr.s + K @ (Z[j] - tr.s[:2])
            tr.P = (np.eye(4) - K @ H) @ tr.P
            tr.last_update, tr.n_updates = t, tr.n_updates + 1
            updated.add(i)
        for i, tr in enumerate(self.tracks):
            tr.hits.append(int(i in updated))
            if not tr.confirmed and sum(list(tr.hits)[-self.n:]) >= self.m:
                tr.confirmed = True
        ids = [self.tracks[i].id if i >= 0 else -1 for i in owner]
        births = 0
        for j, i in enumerate(owner):
            if i < 0:
                tr = Track(self.next_id, Z[j], t, float(np.sqrt(R[j][0, 0])), self.v0)
                self.tracks.append(tr)
                ids[j] = tr.id
                self.next_id += 1
                births += 1
        n0 = len(self.tracks)
        self.tracks = [tr for tr in self.tracks if t - tr.last_update <= self.lost]
        return ids, births, n0 - len(self.tracks)


class BaselineScene:
    def __init__(self, cfg: dict, lib, channel_cal=None, reference=None):
        self.cfg, self.lib = cfg, lib
        rc = cfg["radar"]
        self.r_min, self.r_max, self.x_max, self.y_lim = rc["r_min_m"], rc["r_max_m"], rc["x_max_m"], (rc["y_min_m"], rc["y_max_m"])
        self.pfa = cfg["detection"]["gate_pfa"]
        self.eps = cfg["tracking"]["object_extent_m"] / 2
        self.v_max, self.horizons = cfg["tracking"]["max_speed_mps"], cfg["tracking"]["horizons_s"]
        self.fe = {n: FrontEnd(lib[n], self.r_min, self.r_max, channel_cal, None) for n in lib.names}
        self.fe_still = {}
        if reference is not None and reference.bg_rfft is not None:
            self.fe_still = {n: FrontEnd(lib[n], self.r_min, self.r_max, channel_cal, {"rfft": reference.bg_rfft})
                             for n in lib.names}
        w = World(cfg)
        self.n_known = w.set_known(reference) if reference is not None else 0
        self.known = [(so.lo[:2], so.hi[:2]) for so in w.known]
        self.reference = reference
        self.tracker = GNNTracker(cfg)
        self.t_last = None

    def step(self, frame: np.ndarray, t: float, name: str, pose, v_rail, keep_rd: bool = False) -> dict:
        t0 = time.perf_counter()
        pose, v_rail = np.asarray(pose, float), np.asarray(v_rail, float)
        revisit = float(t - self.t_last) if self.t_last is not None else self.lib.frame_ms / 1e3
        wf = self.lib.look(name, round(min(max(revisit, self.lib.min_period_ms / 1e3), 3 * self.lib.frame_ms / 1e3), 2))
        still = self.reference is not None and self.fe_still and self.reference.still_background(pose, v_rail) is not None
        fe = self.fe_still[name] if still else self.fe[name]
        rd = fe.process(frame)
        still = bool(still and rd.bg_used)
        t1 = time.perf_counter()
        gates = fe.search_gates(rd, self.pfa, static_line=True)
        A = fe.array.steering(fe.array.u_grid).conj()
        dets = []
        for g in gates:
            p = np.abs(rd.D[g.k, g.d, fe.az] @ A) ** 2
            i = int(np.argmax(p))
            du = fe.array.u_grid[1] - fe.array.u_grid[0]
            den = p[i - 1] - 2 * p[i] + p[i + 1] if 0 < i < p.size - 1 else 0.0
            u = float(fe.array.u_grid[i] + (0.5 * (p[i - 1] - p[i + 1]) / den * du if den < 0 else 0.0))
            r, v, bp = fe.refine(rd, g.k, g.d, u)
            th = float(np.degrees(np.arcsin(np.clip(u, -1, 1))))
            xy = pose[:2] + r * np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
            tol = max(fe.cfg.range_res_m, r / 12.0)
            if xy[0] > self.x_max or not (self.y_lim[0] <= xy[1] <= self.y_lim[1]) or \
                    any(np.all(xy >= lo - tol) and np.all(xy <= hi + tol) for lo, hi in self.known):
                continue
            s = abs(v * (self.tracker.doppler_sign or 1) + v_rail[0] * np.cos(np.radians(th)) +
                    v_rail[1] * np.sin(np.radians(th))) <= 2 * wf.dv
            noise = float(np.mean(rd.noise_ch[fe.az]))
            d = {"range_m": r, "theta_deg": th, "u": u, "v_mps": v, "snr_db": float(10 * np.log10(max(p[i] / noise, 1e-12))),
                 "power_db": float(10 * np.log10(max(bp, 1e-30))), "world": [round(float(xy[0]), 4), round(float(xy[1]), 4), 0.0],
                 "z_ok": False, "still": bool(s), "cell": [int(g.k), int(g.d)], "source": "search", "elev_deg": float("nan")}
            if still:
                d["change_db"] = fe.change_db(g.k, u, bp)
            else:
                d["change_db"] = (self.reference.change_db(r, bp, np.r_[xy, 0.0], pose)
                                  if (s and self.reference is not None) else float("inf"))
            dets.append(d)
        t2 = time.perf_counter()
        P = np.array([d["world"][:2] for d in dets]).reshape(-1, 2)
        lab = dbscan(P, self.eps, MIN_POINTS) if len(P) else np.zeros(0, int)
        clusters = [np.flatnonzero(lab == c) for c in range(int(lab.max()) + 1 if lab.size else 0)]
        Z = np.array([P[c].mean(axis=0) for c in clusters]).reshape(-1, 2)
        R = []
        for c, z in zip(clusters, Z):
            rng = float(np.linalg.norm(z - pose[:2]))
            sig2 = fe.cfg.range_res_m ** 2 + (rng * fe.array.rayleigh_u / 2) ** 2 + float(np.var(P[c], axis=0).sum()) / max(len(c), 1)
            R.append(np.eye(2) * sig2)
        t3 = time.perf_counter()
        ids, births, deleted = self.tracker.step(Z, R, t)
        for c, tid in zip(clusters, ids):
            tr = next(x for x in self.tracker.tracks if x.id == tid)
            half = fe.cfg.range_res_m / 2
            tr.lo, tr.hi, tr.n_points = P[c].min(axis=0) - half, P[c].max(axis=0) + half, len(c)
            tr.still = float(np.mean([dets[k]["still"] for k in c]))
        for d, l in zip(dets, lab):
            d["label"] = "core" if l >= 0 else "noise"
            d["track"] = int(ids[l]) if l >= 0 else -1
        t4 = time.perf_counter()
        objects, snap = [], []
        for tr in self.tracker.tracks:
            rel = tr.s[:2] - pose[:2]
            speed = float(np.hypot(tr.s[2], tr.s[3]))
            snap.append({"id": tr.id, "status": "confirmed" if tr.confirmed else "tentative", "x": float(rel[0]),
                         "y": float(rel[1]), "vx": float(tr.s[2]), "vy": float(tr.s[3]),
                         "range_m": float(np.hypot(*rel)), "theta_deg": float(np.degrees(np.arctan2(rel[1], rel[0]))),
                         "wx": float(tr.s[0]), "wy": float(tr.s[1]), "classes": {},
                         "region": {"lo": None, "hi": None, "calibrated": False}, "n_updates": tr.n_updates,
                         "updated": tr.last_update == t, "entropy_bits": 0.0, "info_bits": 0.0})
            if not tr.confirmed or tr.lo is None:
                continue
            lo = np.r_[np.minimum(tr.lo, tr.s[:2]), -np.inf]
            hi = np.r_[np.maximum(tr.hi, tr.s[:2]), np.inf]
            kind = "static" if tr.still >= 0.5 else "moving"
            regions = {str(h): [[*(lo[:2] - self.v_max * h), lo[2]], [*(hi[:2] + self.v_max * h), hi[2]]] for h in self.horizons}
            objects.append({"id": tr.id, "kind": kind, "still": kind == "static", "lo": lo.tolist(), "hi": hi.tolist(),
                            "z_known": False, "person": False, "material": None, "motion": None, "speed": speed,
                            "v": [float(tr.s[2]), float(tr.s[3])], "regions": regions, "calibrated": False,
                            "footprint": [lo.tolist(), hi.tolist()], "live": tr.last_update == t, "n_points": tr.n_points,
                            "centre": [float(tr.s[0]), float(tr.s[1]), 0.0], "last_seen": tr.last_update})
        t5 = time.perf_counter()
        self.t_last = t
        ms = {"range_doppler": 1e3 * (t1 - t0), "detection": 1e3 * (t2 - t1), "cluster": 1e3 * (t3 - t2),
              "track": 1e3 * (t4 - t3), "world": 1e3 * (t5 - t4), "total": 1e3 * (t5 - t0)}
        return {"t": t, "chirp": name, "revisit_s": round(revisit, 4), "pose": [round(float(c), 5) for c in pose],
                "v_rail": [round(float(c), 5) for c in v_rail], "gates": len(gates), "beam_gates": 0,
                "search_gates": len(gates), "gates_explained": 0, "explained_left_out": 0, "subtracted": still,
                "lambda": 1.0, "alpha": None, "dets": dets, "tracks": snap, "objects": objects, "info_bits": 0.0,
                "births": births, "deleted": deleted, "doppler_sign": self.tracker.doppler_sign, "coverage": {},
                "misses": {}, "alpha_t": {}, "material_paused": "classical chain: no classification",
                "noise_db": float(10 * np.log10(np.mean(rd.noise_ch[fe.az]))), "ms": ms}
