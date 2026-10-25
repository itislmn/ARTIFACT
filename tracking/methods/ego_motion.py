"""
ego_motion.py - the radar's own motion from the Doppler of static scatterers, so that tracks stay on the ground when
the radar is moved by hand.

    from methods.ego_motion import EgoMotion
"""
from __future__ import annotations

import itertools

import numpy as np


class EgoMotion:
    def __init__(self, wf, enabled: bool = True, v_max: float = 1.0):
        self.enabled, self.v_max = enabled, v_max
        self.prev_static = None
        self.wf = wf

    def estimate(self, dets: list[dict], exclude: set, rayleigh_u: float, doppler_sign: int, dt: float) -> dict:
        out = {"v": (0.0, 0.0), "dpsi": 0.0, "moving": False, "inliers": 0, "inliers_still": 0, "n": 0}
        if not self.enabled:
            return out
        dv = self.wf.dv
        pts = [d for i, d in enumerate(dets) if i not in exclude and d.get("label") != "ghost"]
        out["n"] = len(pts)
        if len(pts) < 3:
            self.prev_static = None
            return out
        th = np.radians([d["theta_deg"] for d in pts])
        v = np.array([d["v_mps"] for d in pts]) * (doppler_sign or 1)
        A = -np.c_[np.cos(th), np.sin(th)]
        still = int(np.sum(np.abs(v) <= dv))
        out["inliers_still"] = still
        best, best_in = None, still
        pairs = list(itertools.combinations(range(len(pts)), 2))
        if len(pairs) > 300:
            rng = np.random.default_rng(len(pts))
            pairs = [pairs[k] for k in rng.choice(len(pairs), 300, replace=False)]
        for i, j in pairs:
            M = A[[i, j]]
            if abs(np.linalg.det(M)) < 0.05:
                continue
            ve = np.linalg.solve(M, v[[i, j]])
            inl = np.abs(A @ ve - v) <= dv
            if inl.sum() > best_in:
                best, best_in = inl, int(inl.sum())
        if best is not None and best_in >= max(still + 3, 2 + 3):
            u = np.sin(th[best])
            if np.ptp(u) >= 2 * rayleigh_u:
                ve = np.linalg.lstsq(A[best], v[best], rcond=None)[0]
                if np.hypot(*ve) <= self.v_max:
                    out.update(v=(float(ve[0]), float(ve[1])), moving=True, inliers=best_in)
        ve = np.asarray(out["v"])
        static = [(d["range_m"], d["theta_deg"]) for d, a, vv in zip(pts, A, v) if abs(a @ ve - vv) <= dv]
        if self.prev_static and static and dt > 0:
            diffs = []
            for r, t in static:
                c = [(abs(t - t0), t - t0) for r0, t0 in self.prev_static if abs(r - r0) <= self.wf.range_res_m]
                c = [x for x in c if x[0] <= 5.0]
                if c:
                    diffs.append(min(c)[1])
            if len(diffs) >= 3:
                med = float(np.median(diffs))
                mad = float(np.median(np.abs(np.array(diffs) - med)))
                if abs(med) > mad and abs(med) > 0.2:
                    out["dpsi"] = float(-np.radians(med))
                    out["moving"] = True
        self.prev_static = static
        return out
