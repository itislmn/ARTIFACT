"""
camera_view.py - the tracks in the camera image: every track moved to the camera time, its prediction region
projected, the ball (the confirmed track M2-SRC does not call waterlike) and its crop.

    from hardware import camera_view
"""
from __future__ import annotations

import numpy as np

import paths  # noqa: F401  (import paths of ../estimation, ../computer_vision)
from backend import overlay as ov

PALETTE = [(0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167), (230, 159, 0), (86, 180, 233), (240, 228, 66)]
FLOOR_Z, HEAD_Z = -0.38, 1.52


def colour(track_id: int):
    return PALETTE[track_id % len(PALETTE)]


def moved(tr: dict, dt: float) -> dict:
    x, y = tr["x"] + tr["vx"] * dt, tr["y"] + tr["vy"] * dt
    r, th = float(np.hypot(x, y)), float(np.degrees(np.arctan2(y, x)))
    out = dict(tr, x=x, y=y, range_m=r, theta_deg=th)
    lo, hi = tr["region"]["lo"], tr["region"]["hi"]
    if np.all(np.isfinite(lo[:2])) and np.all(np.isfinite(hi[:2])):
        dr, dth = r - tr["range_m"], np.radians(th - tr["theta_deg"])
        out["region"] = dict(tr["region"], lo=[lo[0] + dr, lo[1] + dth, lo[2]], hi=[hi[0] + dr, hi[1] + dth, hi[2]])
    return out


def ball_of(tracks: list[dict]) -> dict | None:
    for head, ok in (("material", lambda lab: lab != "waterlike"), ("motion", lambda lab: lab in ("rigid", "static"))):
        cand = []
        for t in tracks:
            c = (t.get("classes") or {}).get(head)
            if t["status"] == "confirmed" and c and c["label"] is not None and ok(c["label"]):
                cand.append((c.get("windows", 1) * c["confidence"], t["n_updates"], t))
        if cand:
            return max(cand, key=lambda c: c[:2])[2]
    return None


def heights(tr: dict, cfg: dict, is_ball: bool):
    b = cfg["ball"]
    if is_ball:
        r = b["diameter_m"] / 2
        return b["centre_z_m"] - r, b["centre_z_m"] + b["swing_z_m"] + r
    return FLOOR_Z, HEAD_Z


def region_rect(proj, tr: dict, z_lo: float, z_hi: float, pad_m: float = 0.0):
    lo, hi = tr["region"]["lo"], tr["region"]["hi"]
    if not (np.all(np.isfinite(lo[:2])) and np.all(np.isfinite(hi[:2]))):
        lo = [tr["range_m"] - 0.1, np.radians(tr["theta_deg"]) - 0.05]
        hi = [tr["range_m"] + 0.1, np.radians(tr["theta_deg"]) + 0.05]
    r0, r1 = max(lo[0] - pad_m, 0.05), hi[0] + pad_m
    dth = pad_m / max(tr["range_m"], 0.1)
    t0, t1 = np.degrees(lo[1] - dth), np.degrees(hi[1] + dth)
    pts = [proj.point(r, t, z) for r in (r0, r1) for t in (t0, t1) for z in (z_lo, z_hi)]
    pts = [p for p in pts if p is not None]
    if not pts:
        return None
    P = np.array(pts)
    m = proj.loo_rms_px or 0.0
    return P[:, 0].min() - m, P[:, 1].min() - m, P[:, 0].max() + m, P[:, 1].max() + m


def crop(rgb: np.ndarray, rect, size: int) -> np.ndarray | None:
    if rect is None:
        return None
    h, w = rgb.shape[:2]
    u0, v0, u1, v1 = rect
    side = max(u1 - u0, v1 - v0, 8)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    x0, y0 = int(round(cu - side / 2)), int(round(cv - side / 2))
    xs = np.clip(x0 + (np.arange(size) * side / size).astype(int), 0, w - 1)
    ys = np.clip(y0 + (np.arange(size) * side / size).astype(int), 0, h - 1)
    return rgb[ys][:, xs]


def label(tr: dict) -> str:
    cls = tr.get("classes") or {}
    mat = cls.get("material", {}).get("label") or ""
    mot = cls.get("motion", {}).get("label") or ""
    kind = f" {mat}/{mot}".upper() if mat or mot else ""
    return f"{tr['id']}{kind} {tr['range_m']:.2f}M {tr['theta_deg']:+.0f}"


def draw_tracks(view, proj, tracks, cfg, scale: float, ball_id: int | None):
    for tr in tracks:
        is_ball = tr["id"] == ball_id
        z_lo, z_hi = heights(tr, cfg, is_ball)
        rect = region_rect(proj, tr, z_lo, z_hi)
        if rect is None:
            continue
        c = colour(tr["id"])
        u0, v0, u1, v1 = (np.array(rect) * scale)
        ov.rect(view, u0, v0, u1, v1, c, t=4 if is_ball else 2)
        ov.text(view, ("BALL " if is_ball else "") + label(tr), u0, v0 - 18, c, 2)


def top_view(tracks, dets, size=(360, 360), r_max: float = 4.0, ball_id=None) -> np.ndarray:
    w, h = size
    s = (h - 20) / r_max
    ox, oy = w / 2, h - 10
    img = np.full((h, w, 3), 255, np.uint8)

    def px(x, y):
        return ox + s * y, oy - s * x

    for r in np.arange(0.5, r_max + 1e-9, 0.5):
        pts = [px(r * np.cos(np.radians(a)), r * np.sin(np.radians(a))) for a in np.linspace(-60, 60, 40)]
        for p0, p1 in zip(pts, pts[1:]):
            ov.line(img, p0, p1, (215, 215, 215), 1)
    for d in dets:
        th = np.radians(d["theta_deg"])
        u, v = px(d["range_m"] * np.cos(th), d["range_m"] * np.sin(th))
        c = colour(d["track"]) if d.get("track", -1) >= 0 and d.get("label") in ("core", "part") else (170, 170, 170)
        img[int(max(v - 2, 0)):int(v + 3), int(max(u - 2, 0)):int(u + 3)] = c
    for tr in tracks:
        c = colour(tr["id"])
        lo, hi = tr["region"]["lo"], tr["region"]["hi"]
        if np.all(np.isfinite(lo[:2])) and np.all(np.isfinite(hi[:2])):
            corners = [px(r * np.cos(t), r * np.sin(t)) for r, t in ((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], hi[1]), (hi[0], lo[1]))]
            for p0, p1 in zip(corners, corners[1:] + corners[:1]):
                ov.line(img, p0, p1, c, 2 if tr["id"] == ball_id else 1)
        u, v = px(tr["x"], tr["y"])
        u2, v2 = px(tr["x"] + 0.5 * tr["vx"], tr["y"] + 0.5 * tr["vy"])
        ov.line(img, (u, v), (u2, v2), c, 2)
        ov.text(img, str(tr["id"]), int(u) + 6, int(v) - 6, c, 1)
    ov.cross(img, ox, oy, ov.BLACK, 6, 2)
    return img
