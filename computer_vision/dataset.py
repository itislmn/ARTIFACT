"""
dataset.py - the evaluated frames of the recorded takes as output/dataset.csv: every 2nd camera frame with the lamp
state and the radar objects as the live program logged them, the text object's image position and the glare fit.

    python dataset.py            (python dataset.py --again rebuilds every take)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import csv  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from backend import glare as gl  # noqa: E402
from backend import pipeline  # noqa: E402
from backend.imgio import read_video  # noqa: E402
from backend.settings import CACHE, CALIB, CONFIG, OUTPUT, take_dir, load_config  # noqa: E402

SESSION = "CV1"
TAKES = {"text_085b": "text", "text_085c": "text", "light_1": "light", "light_2": "light", "light_3": "light"}
TRUTH = "ARTIFACT"
STEP, FPS, REFIT_S = 2, 30.0, 0.5
TABLE = OUTPUT / "dataset.csv"
COLUMNS = ["take", "scene", "k", "t_s", "lamp", "text_found", "text_range_m", "text_azimuth_deg", "centre_u", "centre_v",
           "letter_px", "lamp_radar", "source_range_m", "source_azimuth_deg", "glare_u", "glare_v", "glare_core_px",
           "glare_fit", "radar_ms", "objects", "dets"]


def fresh(target: Path, deps: list[Path]) -> bool:
    return target.exists() and all(target.stat().st_mtime >= d.stat().st_mtime for d in deps if d.exists())


def live_log(take: Path):
    rows = [r for r in csv.DictReader(open(take / "frames.csv")) if r.get("recording") == "1"]
    radar = {}
    for line in open(take / "radar_live.jsonl", encoding="utf-8"):
        m = json.loads(line)
        if m.get("type") == "frame":
            radar[m["frame"]] = m
    summary = json.loads((take / "summary.json").read_text())
    events = sorted((e for e in summary.get("events", []) if e.get("core_uv")), key=lambda e: e["camera_frame"])
    ranges = [float(r["text_range_m"]) for r in rows if r.get("text_range_m")]
    return rows, radar, events, float(np.median(ranges))


def video_frames(take: Path, ks: set):
    last = max(ks)
    for k, rgb in enumerate(read_video(take / "camera.mp4")):
        if k in ks:
            yield k, rgb
        if k >= last:
            return


def glare_fit(rgb, clipped, core_uv, objects, proj, cfg):
    u, v = core_uv
    r = 120
    u0, v0 = int(max(u - r, 0)), int(max(v - r, 0))
    ys, xs = np.nonzero(clipped[v0:int(v + r), u0:int(u + r)])
    if not ys.size:
        return None
    cu, cv = xs.mean() + u0, ys.mean() + v0
    core = float(np.percentile(np.hypot(xs + u0 - cu, ys + v0 - cv), 95))
    bars = [proj.bar(o["range_m"], o["azimuth_deg"], cfg["alignment"]["bar_z_m"]) for o in objects]
    hw = [proj.f * cfg["light"]["source_size_m"] / 2 / o["range_m"] for o in objects]
    return gl.fit_at(rgb, clipped, -1, (cu, cv), core, gl.bar_mask(clipped.shape, bars, hw))


def take_rows(name: str, scene: str, proj, cfg) -> list[dict]:
    take = take_dir(SESSION, name)
    rows, radar, events, text_range = live_log(take)
    t_rows = np.array([float(r["t_capture_s"]) for r in rows])
    n_video = int(round((t_rows[-1] - t_rows[0]) * FPS)) + 1
    out, fit, t_fit = [], None, -1e9
    for k, rgb in video_frames(take, set(range(0, n_video, STEP))):
        t = t_rows[0] + k / FPS
        row = rows[min(int(np.searchsorted(t_rows, t)), len(rows) - 1)]
        m = radar.get(int(row["radar_frame"])) if row.get("radar_frame") else None
        objects = m["objects"] if m else []
        lamp = row.get("light") == "1"
        rec = {"take": name, "scene": scene, "k": k, "t_s": round(k / FPS, 3), "lamp": int(lamp), "text_found": 0,
               "lamp_radar": int(bool(row.get("source_range_m"))), "radar_ms": m["ms"] if m else "",
               "objects": json.dumps([{key: o[key] for key in ("range_m", "azimuth_deg")} for o in objects]),
               "dets": json.dumps(np.round(m["dets"], 4).tolist() if m else [])}
        if row.get("source_range_m"):
            rec.update(source_range_m=float(row["source_range_m"]), source_azimuth_deg=float(row["source_azimuth_deg"]))
        if objects:
            o = min(objects, key=lambda o: abs(o["range_m"] - text_range))
            c = proj.point(o["range_m"], o["azimuth_deg"], cfg["text"]["centre_z_m"])
            if c is not None:
                rec.update(text_found=1, text_range_m=o["range_m"], text_azimuth_deg=o["azimuth_deg"],
                           centre_u=round(float(c[0]), 1), centre_v=round(float(c[1]), 1),
                           letter_px=round(proj.f * cfg["text"]["letter_mm"] / 1000 / o["range_m"], 1))
        if lamp:
            prev = [e for e in events if e["camera_frame"] <= int(row["camera_frame"])]
            if prev and t - t_fit >= REFIT_S:
                clipped = pipeline.fast_luma(rgb) >= cfg["camera"]["clip_level"]
                fit, t_fit = glare_fit(rgb, clipped, prev[-1]["core_uv"], objects, proj, cfg), t
        else:
            fit, t_fit = None, -1e9
        if fit is not None:
            rec.update(glare_u=round(fit.centre_uv[0], 1), glare_v=round(fit.centre_uv[1], 1),
                       glare_core_px=round(fit.core_px, 1), glare_fit=json.dumps(np.round(fit.fits, 5).tolist()))
        out.append(rec)
    return out


def write(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read(path: Path) -> list[dict]:
    def value(x):
        try:
            return float(x)
        except ValueError:
            return x
    return [{k: value(v) for k, v in r.items()} for r in csv.DictReader(open(path))]


def build(again: bool = False) -> list[dict]:
    from backend.alignment import Projector
    cfg, proj = load_config(), Projector()
    rows = []
    for name, scene in TAKES.items():
        take = take_dir(SESSION, name)
        cache = CACHE / f"dataset_{name}.csv"
        deps = [take / "camera.mp4", take / "frames.csv", Path(__file__), CALIB / "alignment.json", CONFIG,
                HERE / "backend" / "glare.py", HERE / "backend" / "alignment.py"]
        if again or not fresh(cache, deps):
            print(f"  dataset: {name} ...", flush=True)
            write(cache, take_rows(name, scene, proj, cfg), COLUMNS)
        rows += read(cache)
    write(TABLE, rows, COLUMNS)
    print(f"  dataset: {len(rows)} frames -> {TABLE.relative_to(HERE)}")
    return rows


if __name__ == "__main__":
    build(again="--again" in sys.argv)
