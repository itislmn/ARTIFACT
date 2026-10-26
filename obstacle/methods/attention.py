"""
attention.py - what the camera looks at: the radar object in its picture that matters most (the collider, a person,
the ball, the nearest), kept while it stays there; its image region and its crop.

    from methods.attention import choose, rect
"""
from __future__ import annotations

import numpy as np

ROLES = ("collider", "person", "ball", "object")


def choose(objects: list[dict], cause_id, pose, visible=lambda o: True, keep=None) -> tuple[dict | None, str]:
    pose = np.asarray(pose, float)
    seen = [o for o in objects if not o.get("known") and np.asarray(o["centre"])[0] > pose[0] and visible(o)
            and (o.get("live", True) or o["id"] == cause_id)]

    def dist(o):
        return float(np.linalg.norm(np.asarray(o["centre"])[:2] - pose[:2]))
    ranks = (("collider", [o for o in seen if o["id"] == cause_id]), ("person", [o for o in seen if o.get("person")]),
             ("ball", [o for o in seen if not o.get("person") and (o["kind"] == "moving" and o.get("motion") == "rigid" and
                       o.get("material") in ("dielectric", "waterlike") or o["kind"] == "static" and o.get("material") == "dielectric")]),
             ("object", seen))
    for role, pick in ranks:
        if pick:
            same = [o for o in pick if o["id"] == keep]
            return (same[0] if same else min(pick, key=dist)), role
    return None, ""


def crop_rect(proj, o: dict, pose, z_default: float, size_m: float) -> tuple | None:
    uv, R = centre_uv(proj, o, pose, z_default)
    if uv is None:
        return None
    half = proj.f * size_m / max(R, 0.1) / 2
    return uv[0] - half, uv[1] - half, uv[0] + half, uv[1] + half


def in_picture(proj, o: dict, pose, z_default: float, size_wh) -> bool:
    if o["centre"][0] <= pose[0]:
        return False
    uv, _ = centre_uv(proj, o, pose, z_default)
    return uv is not None and 0 <= uv[0] < size_wh[0] and 0 <= uv[1] < size_wh[1]


def moved(o: dict, dt: float) -> dict:
    if o["kind"] != "moving" or "v" not in o:
        return o
    shift = np.r_[np.asarray(o["v"]) * dt, 0.0]
    return dict(o, lo=(np.asarray(o["lo"]) + shift).tolist(), hi=(np.asarray(o["hi"]) + shift).tolist(),
                centre=(np.asarray(o["centre"]) + shift).tolist())


def _rel(p, pose):
    q = np.asarray(p, float) - np.asarray(pose, float)
    R = float(np.linalg.norm(q))
    return R, float(np.degrees(np.arcsin(np.clip(q[1] / max(R, 1e-6), -1, 1)))), float(q[2])


def rect(proj, o: dict, pose, bar_z) -> tuple | None:
    lo, hi = np.asarray(o["lo"], float), np.asarray(o["hi"], float)
    z_lo = lo[2] if np.isfinite(lo[2]) else pose[2] + bar_z[0]
    z_hi = hi[2] if np.isfinite(hi[2]) else pose[2] + bar_z[1]
    pts = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (z_lo, z_hi):
                R, az, zr = _rel((x, y, z), pose)
                if R > 0.05:
                    p = proj.point(R, az, zr)
                    if p is not None:
                        pts.append(p)
    if not pts:
        return None
    P = np.array(pts)
    m = proj.loo_rms_px or 0.0
    return float(P[:, 0].min() - m), float(P[:, 1].min() - m), float(P[:, 0].max() + m), float(P[:, 1].max() + m)


def region_rect(proj, tr: dict, z_lo: float, z_hi: float, dt: float = 0.0):
    lo, hi = tr["region"]["lo"], tr["region"]["hi"]
    if lo is None or hi is None or not (np.all(np.isfinite(lo[:2])) and np.all(np.isfinite(hi[:2]))):
        lo = [tr["range_m"] - 0.1, np.radians(tr["theta_deg"]) - 0.05]
        hi = [tr["range_m"] + 0.1, np.radians(tr["theta_deg"]) + 0.05]
    x = tr["range_m"] * np.cos(np.radians(tr["theta_deg"])) + tr.get("vx", 0.0) * dt
    y = tr["range_m"] * np.sin(np.radians(tr["theta_deg"])) + tr.get("vy", 0.0) * dt
    dr, dth = float(np.hypot(x, y)) - tr["range_m"], float(np.arctan2(y, x)) - np.radians(tr["theta_deg"])
    pts = [proj.point(max(r + dr, 0.05), np.degrees(t + dth), z) for r in (lo[0], hi[0]) for t in (lo[1], hi[1])
           for z in (z_lo, z_hi)]
    pts = [p for p in pts if p is not None]
    if not pts:
        return None
    P = np.array(pts)
    m = proj.loo_rms_px or 0.0
    return float(P[:, 0].min() - m), float(P[:, 1].min() - m), float(P[:, 0].max() + m), float(P[:, 1].max() + m)


def centre_uv(proj, o: dict, pose, z_default: float):
    c = np.asarray(o["centre"], float).copy()
    if not o.get("z_known"):
        c[2] = pose[2] + z_default
    R, az, zr = _rel(c, pose)
    return proj.point(R, az, zr), R
