"""
truth.py - the ground truth of a reflector from tape measurements: R = sqrt(x^2 + y^2 + z^2), theta = arcsin(y / R),
phi = arcsin(z / R), with apex heights and depths from the session plan.

    from estimation.core.truth import complete
"""
from __future__ import annotations

import numpy as np

EL_SIGN = -1


def apex_depths(plan: dict | None) -> dict:
    return {k: float(v["apex_depth_m"]) for k, v in (plan or {}).get("reflectors", {}).items()
            if isinstance(v, dict) and v.get("apex_depth_m")}


def heights(plan: dict | None) -> tuple[float | None, dict]:
    plan = plan or {}
    apex = {k: float(v["apex_height_m"]) for k, v in plan.get("reflectors", {}).items()
            if isinstance(v, dict) and "apex_height_m" in v}
    h0 = plan.get("antenna_height_m")
    return (None if h0 is None else float(h0)), apex


def complete(target: dict, plan: dict | None = None) -> dict:
    t = dict(target)
    if "x_m" not in t:
        a = np.radians(t["azimuth_deg"])
        t["x_m"], t["y_m"] = round(float(t["range_m"] * np.cos(a)), 4), round(float(t["range_m"] * np.sin(a)), 4)
    h0, apex = heights(plan)
    z = apex[t["label"]] - h0 if h0 is not None and t.get("label") in apex else 0.0
    d = float(t["apex_depth_m"]) if "apex_depth_m" in t else apex_depths(plan).get(t.get("label"), 0.0)
    rh = float(np.hypot(t["x_m"], t["y_m"]))
    x, y = (t["x_m"], t["y_m"]) if rh == 0 else (t["x_m"] * (rh + d) / rh, t["y_m"] * (rh + d) / rh)
    R = float(np.sqrt(x ** 2 + y ** 2 + z ** 2))
    t.update(z_m=round(z, 4), apex_depth_m=d, range_m=round(R, 4),
             azimuth_deg=round(float(np.degrees(np.arcsin(y / R))), 2),
             elevation_deg=round(float(np.degrees(np.arcsin(z / R))), 2))
    return t
