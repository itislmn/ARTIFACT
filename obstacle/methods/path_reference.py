"""
path_reference.py - the empty path: what the radar sees of the static world from every rail position (change test of
a detection, the still empty scene at home, the static objects of the reference).

    python -m methods.path_reference      (self-test)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def zero_doppler_map(rd, fe) -> np.ndarray:
    K = rd.D.shape[1]
    A = fe.array.steering(fe.array.u_grid).conj()
    Y = rd.D[:, K // 2 - 1:K // 2 + 2][:, :, fe.az]
    return (np.abs(Y @ A) ** 2).max(axis=1).astype(np.float32) / fe.array.n


class PathReference:
    STILL_TOL_M = 5e-5
    STILL_V_MPS = 1e-4

    def __init__(self, x_m, y_m, z_m, Z, range_m, u_grid, ra=None, meta=None, bg_rfft=None, bg_pose=None, static_pts=None):
        self.x, self.y, self.z = (np.asarray(a, float) for a in (x_m, y_m, z_m))
        self.Z = np.asarray(Z, np.float32)
        self.range_m = np.asarray(range_m, float)
        self.u = np.asarray(u_grid, float)
        self.ra = None if ra is None else np.asarray(ra, np.float32)
        self.meta = meta or {}
        self.bg_rfft = None if bg_rfft is None else np.asarray(bg_rfft, np.complex64)
        self.bg_pose = np.zeros(3) if bg_pose is None else np.asarray(bg_pose, float)
        self.static_pts = np.zeros((0, 8)) if static_pts is None else np.asarray(static_pts, float).reshape(-1, 8)
        self.res = float(self.range_m[1] - self.range_m[0])
        self.du = float(self.u[1] - self.u[0])
        self.half_cell = int(round(1.0 / 12.0 / self.du))
        n, d = 12, 0.5
        x = np.pi * d * self.du / 2
        self.straddle_db = float(-20 * np.log10(abs(np.sin(n * x) / (n * np.sin(x)))))

    def save(self, path: Path):
        import json
        np.savez_compressed(path, x=self.x, y=self.y, z=self.z, Z=self.Z, range_m=self.range_m, u=self.u,
                            ra=self.ra if self.ra is not None else np.zeros(0), meta=json.dumps(self.meta),
                            bg_rfft=self.bg_rfft if self.bg_rfft is not None else np.zeros(0, np.complex64),
                            bg_pose=self.bg_pose, static_pts=self.static_pts)

    @classmethod
    def load(cls, path: Path) -> "PathReference":
        import json
        d = np.load(path)
        ra = d["ra"] if d["ra"].size else None
        bg = d["bg_rfft"] if "bg_rfft" in d.files and d["bg_rfft"].size else None
        return cls(d["x"], d["y"], d["z"], d["Z"], d["range_m"], d["u"], ra, json.loads(str(d["meta"])), bg,
                   d["bg_pose"] if "bg_pose" in d.files else None, d["static_pts"] if "static_pts" in d.files else None)

    def still_background(self, pose, v_rail) -> dict | None:
        if self.bg_rfft is None or np.max(np.abs(np.asarray(pose, float) - self.bg_pose)) > self.STILL_TOL_M \
                or np.linalg.norm(v_rail) > self.STILL_V_MPS:
            return None
        return {"rfft": self.bg_rfft}

    @property
    def x_span(self) -> tuple[float, float]:
        return float(self.x.min()), float(self.x.max())

    def change_db(self, range_m: float, power: float, p_world: np.ndarray, pose: np.ndarray) -> float:
        near = np.flatnonzero(np.abs(self.x - pose[0]) <= self.res / 2)
        if near.size == 0:
            return float("inf")
        rel = np.asarray(p_world, float)[None, :] - np.c_[self.x[near], self.y[near], self.z[near]]
        rh = np.hypot(rel[:, 0], rel[:, 1])
        r_ref = np.sqrt(rh ** 2 + rel[:, 2] ** 2)
        u_ref = rel[:, 1] / np.maximum(rh, 1e-6)
        k = np.rint((r_ref - self.range_m[0]) / self.res).astype(int)
        j = np.rint((u_ref - self.u[0]) / self.du).astype(int)
        best = 0.0
        for f, kk, jj, rr in zip(near, k, j, r_ref):
            k0, k1 = max(kk - 1, 0), min(kk + 2, self.Z.shape[1])
            j0, j1 = max(jj - self.half_cell, 0), min(jj + self.half_cell + 1, self.Z.shape[2])
            if k0 >= k1 or j0 >= j1:
                continue
            best = max(best, float(self.Z[f, k0:k1, j0:j1].max()) * rr ** 4)
        if best <= 0:
            return float("inf")
        return float(10 * np.log10(max(power, 1e-30) * range_m ** 4 / best) - self.straddle_db)

    def level_map(self, pose, tol_offset_m: float = 0.01):
        from scipy.ndimage import maximum_filter
        if abs(pose[1]) > tol_offset_m or abs(pose[2]) > tol_offset_m:
            return None
        near = np.flatnonzero(np.abs(self.x - pose[0]) <= self.res / 2)
        if near.size == 0:
            return None
        return maximum_filter(self.Z[near].max(axis=0), size=(3, 2 * self.half_cell + 1), mode="nearest")

    def row(self, range_m: float) -> int:
        k = int(np.rint((range_m - self.range_m[0]) / self.res))
        return k if 0 <= k < self.Z.shape[1] else -1

    def background_ra(self, x_lo: float, x_hi: float, pose: np.ndarray, tol_offset_m: float = 0.01):
        if self.ra is None or abs(pose[1]) > tol_offset_m or abs(pose[2]) > tol_offset_m:
            return None
        sel = (self.x >= x_lo - self.res / 2) & (self.x <= x_hi + self.res / 2)
        return self.ra[sel].mean(axis=0) if sel.any() else None


class ReferenceRecorder:

    def __init__(self, fe, meta: dict, detector=None):
        self.fe, self.meta, self.detector = fe, meta, detector
        self.rows = []
        self.points = []
        self.bg_sum, self.bg_n, self.bg_pose = None, 0, None

    def add_still(self, rfft_mean, pose):
        self.bg_sum = rfft_mean if self.bg_sum is None else self.bg_sum + rfft_mean
        self.bg_n += 1
        self.bg_pose = np.asarray(pose, float)

    def add(self, rd, pose, ra=None):
        if self.detector is not None:
            self.points += [(*p, len(self.rows)) for p in self.detector(rd, np.asarray(pose, float))]
        self.rows.append((np.asarray(pose, float), zero_doppler_map(rd, self.fe), ra))

    def result(self) -> PathReference:
        P = np.array([r[0] for r in self.rows])
        ra = [r[2] for r in self.rows]
        ra = np.array(ra, np.float32) if all(a is not None for a in ra) and ra else None
        bg = self.bg_sum / self.bg_n if self.bg_n else None
        return PathReference(P[:, 0], P[:, 1], P[:, 2], np.array([r[1] for r in self.rows]), self.fe.range_m,
                             self.fe.array.u_grid, ra, {**self.meta, "still_frames": self.bg_n}, bg, self.bg_pose,
                             np.array(self.points, float) if self.points else None)


def selftest() -> bool:
    rng_m = np.arange(3, 40) * 0.0837
    u = np.arange(-41, 42) / 48
    xs = np.arange(0.0, 1.0, 0.0025)
    Z = np.full((xs.size, rng_m.size, u.size), 1e-3, np.float32)
    for i, x in enumerate(xs):
        k = int(np.rint((1.6 - x - rng_m[0]) / 0.0837))
        Z[i, k, 41] = 1.0 / (1.6 - x) ** 4
    ref = PathReference(xs, 0 * xs, 0 * xs, Z, rng_m, u)
    pose = np.array([0.40, 0.0, 0.0])
    R = 1.20
    same = ref.change_db(R, 1.0 / R ** 4, np.array([1.6, 0.0, 0.0]), pose)
    louder = ref.change_db(R, 100.0 / R ** 4, np.array([1.6, 0.0, 0.0]), pose)
    fresh = ref.change_db(0.8, 1e-1, np.array([1.2, 0.3, 0.0]), pose)
    far = ref.change_db(R, 1.0, np.array([3.0, 0.0, 0.0]), np.array([1.5, 0.0, 0.0]))
    L = ref.level_map(pose)
    k = ref.row(R)
    agree = L is not None and L[k, 41] >= 1.0 / R ** 4 * 0.999 and L[ref.row(0.8), 60] < 1e-1
    side = ref.level_map(np.array([0.40, 0.05, 0.0]))
    ok = same <= 0.0 and louder > 15.0 and fresh > 0.0 and far == float("inf") and agree and side is None
    print(f"  {'PASS' if ok else 'FAIL'}  end plate {same:+.1f} dB (empty path), +20 dB object {louder:+.1f} dB, "
          f"echo in an empty cell {fresh:+.1f} dB, no reference {far}; level map agrees: {agree}, "
          f"none after a sidestep: {side is None}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
