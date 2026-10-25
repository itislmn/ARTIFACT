"""
references.py - the radar references of a session: the empty-room background of every chirp profile and the channel
calibration.

    from hardware.references import load_references
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import paths
from estimation.core.frame import background_from_recording
from estimation.core.radar import RadarConfig, load_truth, open_recording, with_layout
from estimation.core.steering import all_channel_steering, take_snapshot


def profile_key(rc: RadarConfig) -> str:
    return f"S{rc.slope_hz_per_s / 1e12:g}_f{rc.f0_hz / 1e9:.3f}"


def _takes(folder: Path):
    if not folder.exists():
        return []
    return [(p, load_truth(p)) for p in sorted(folder.iterdir())
            if p.suffix in (".bin", ".npy") and p.with_suffix(".json").exists()]


def load_references(cfg: dict, library: dict, log=print):
    session = cfg["session"]
    folders = [paths.radar_dir(session)]
    cv_session = paths.load_config(paths.CV / "config.toml")["session"]
    folders.append(paths.CV / "recordings" / cv_session / "radar")
    takes = [t for f in folders for t in _takes(f)]
    bgs, cal = {}, None
    for p, doc in takes:
        try:
            rc = with_layout(RadarConfig.from_cfg(doc["cfg_file"]), paths.LAYOUT) \
                if doc.get("cfg_file") and Path(doc["cfg_file"]).exists() else None
        except Exception:                                      # noqa: BLE001
            rc = None
        if rc is None:
            continue
        key = profile_key(rc)
        if doc.get("scenario") == "background" and key not in bgs:
            n, get = open_recording(p, rc)
            bgs[key] = (background_from_recording(n, get, rc), p)
        elif doc.get("scenario") == "calibration" and cal is None and doc.get("targets"):
            t = doc["targets"][0]
            bg = bgs.get(key, (None, None))[0]
            y16, _ = take_snapshot(p, rc, background=bg, range_m=t["range_m"])
            c = y16 / all_channel_steering(rc, t["azimuth_deg"], t.get("elevation_deg", 0.0))
            cal = ((c / c[0]).astype(np.complex64), p)
    refs, unusable = {}, []
    for name, wf in library.items():
        key = profile_key(wf.cfg)
        if key in bgs:
            refs[name] = {"bg": bgs[key][0], "cal": cal[0] if cal else None}
        else:
            unusable.append(name)
    log("radar references: " + ", ".join(f"{k}: {Path(v[1]).name}" for k, v in bgs.items())
        + f"; calibration: {Path(cal[1]).name if cal else 'NONE (angles uncalibrated)'}"
        + (f"; no background for {unusable} (not offered to the controller)" if unusable else ""))
    return refs, unusable
