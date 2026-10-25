"""
alignment.py - radar to camera: the pixel of a radar point and the arc a radar object covers in the image
(computer_vision's camera model and Projector).

    from backend.alignment import Projector
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import paths

CALIB = paths.CALIB
R0 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
NAMES = ("rot_x", "rot_y", "rot_z", "c_x", "c_y_left", "c_z", "f", "k1")


def right_handed(xyz) -> np.ndarray:
    return np.asarray(xyz, float).reshape(-1, 3) * [1.0, -1.0, 1.0]


def rotation(w) -> np.ndarray:
    a = float(np.linalg.norm(w))
    if a < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def project(p: np.ndarray, xyz_rh: np.ndarray, centre) -> tuple[np.ndarray, np.ndarray]:
    Xc = (xyz_rh - p[3:6]) @ (rotation(p[:3]) @ R0).T
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    d = 1 + p[7] * (x ** 2 + y ** 2)
    return np.c_[centre[0] + p[6] * x * d, centre[1] + p[6] * y * d], Xc[:, 2]


class Projector:

    def __init__(self, calib: Path = CALIB):
        path = calib / "alignment.json"
        if not path.exists():
            raise SystemExit("calibration/alignment.json missing: python align.py nominal (rough) or points + solve")
        a = json.loads(path.read_text())
        self.p = np.array([a["params"][n] for n in NAMES])
        self.size_wh = tuple(a["image_size_wh"])
        self.centre = (self.size_wh[0] / 2, self.size_wh[1] / 2)
        self.f = self.p[6]
        self.loo_rms_px = a["loo_rms_px"]

    def point(self, range_m: float, azimuth_deg: float, z: float) -> np.ndarray | None:
        y = range_m * np.sin(np.radians(azimuth_deg))
        xyz = [np.sqrt(max(range_m ** 2 - y ** 2 - z ** 2, 0.0)), y, z]
        uv, depth = project(self.p, right_handed(xyz), self.centre)
        return uv[0] if depth[0] > 0 else None

    def bar(self, range_m, azimuth_deg, z_pair, n: int = 17):
        pts = [self.point(range_m, azimuth_deg, z) for z in np.linspace(z_pair[0], z_pair[1], n)]
        return None if any(e is None for e in pts) else np.array(pts)

    def u_at_row(self, range_m, azimuth_deg, v, z_pair) -> float:
        lo, hi = z_pair
        for _ in range(40):
            z = (lo + hi) / 2
            lo, hi = (lo, z) if self.point(range_m, azimuth_deg, z)[1] < v else (z, hi)
        return float(self.point(range_m, azimuth_deg, z)[0])
