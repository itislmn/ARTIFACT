"""
light_experiment.py - the live light experiment: a lamp switched at random moments is found by the camera, located by
the radar and its veil fitted, while MetricSight keeps reading.

    python light_experiment.py --take light_3 --truth ARTIFACT --exposure -7 --record
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import argparse
import time
from collections import deque

import numpy as np

from backend import overlay as ov
from backend import pipeline
from backend.brightness import STEP, BrightnessModel
from backend.glare import locate_source
from backend.radar_link import cluster
from backend.reader_live import LiveReader
from backend.session import Live, add_common_args
from backend.settings import load_config


WEAK_FRAMES, WEAK_MIN = 30, 2


def surveyed_source() -> dict | None:
    import json
    from backend.settings import CALIB
    p = CALIB / "light_source.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"range_m": d["range_m"], "azimuth_deg": d["azimuth_deg"], "rate": 0.0, "surveyed": True}


def weak_objects(objects: list, recent) -> list:
    if not recent:
        return []
    out = []
    for c in cluster(list(recent), 2 * 0.0836, 4.78, WEAK_MIN / len(recent)):
        if all(abs(c["range_m"] - o["range_m"]) > 0.17 or abs(c["azimuth_deg"] - o["azimuth_deg"]) > 4.78 for o in objects):
            out.append({**c, "weak": True})
    return out


def follow(source: dict, objects: list) -> dict | None:
    if not objects:
        return None
    d = [abs(o["range_m"] - source["range_m"]) / 0.1 + abs(o["azimuth_deg"] - source["azimuth_deg"]) / 5 for o in objects]
    i = int(np.argmin(d))
    return {**source, "object": i} if d[i] < 2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--truth", default="", help="the written text, e.g. ARTIFACT: every reading is scored")
    ap.add_argument("--record", action="store_true", help="camera video from the start (for make_video.py)")
    args = ap.parse_args()
    cfg = load_config()
    if str(args.exposure if args.exposure is not None else cfg["camera"]["exposure"]) == "auto":
        print("WARNING: automatic exposure - give --exposure (e.g. -7): auto exposure reacts to the light itself")
    truth = args.truth.upper()
    from backend.alignment import Projector
    proj = Projector()
    live = Live("LIGHT + TEXT", cfg, args.session or cfg["session"], args.take, args.exposure, args.camera)
    events, source, glare_t, read_t, pending, tick = [], None, -1e9, -1e9, None, 0
    score = {"before": [0, 0], "light": [0, 0], "after": [0, 0]}
    model = BrightnessModel(cfg, proj.f)
    reader = None
    try:
        radar = live.live_radar(args.cli)
        live.video(args.record)
        reader = LiveReader(cfg).start()
        res, readings, t0 = None, {}, time.perf_counter()
        surveyed = surveyed_source()
        if surveyed:
            print(f"light source surveyed at {surveyed['range_m']:.2f} m, {surveyed['azimuth_deg']:+.1f} deg (calibration/light_source.json)")
        recent = deque(maxlen=WEAK_FRAMES)
        while live.alive and (args.seconds <= 0 or time.perf_counter() - t0 < args.seconds):
            f = live.frame()
            for msg in radar.poll():
                if msg["type"] == "frame":
                    recent.append(np.array(msg["dets"], float).reshape(-1, 2))
            m = radar.last
            objects = (m["objects"] if m else []) + weak_objects(m["objects"] if m else [], recent)
            if surveyed and all(abs(o["range_m"] - surveyed["range_m"]) > 0.17 or abs(o["azimuth_deg"] - surveyed["azimuth_deg"]) > 4.78
                                for o in objects):
                objects = objects + [surveyed]
            rgb, t = f.image, f.t_capture
            ev = model.update(rgb, t)
            if ev.onset:
                bars = [proj.bar(o["range_m"], o["azimuth_deg"], cfg["alignment"]["bar_z_m"]) for o in objects]
                source = locate_source(ev.core_uv, objects, bars, proj, cfg) if ev.core_uv else None
                events.append({"onset_perf_s": round(t, 4), "camera_frame": f.index, "radar_frame": m["frame"] if m else None,
                               "lit_px": ev.n_lit * STEP ** 2, "blob_uv": ev.blob_uv, "core_uv": ev.core_uv, "source": source})
                glare_t, pending = -1e9, t + 0.5
                print(f"LIGHT ON  (camera frame {f.index}): blob at {np.round(ev.core_uv or ev.blob_uv, 0)} px -> "
                      + (f"radar object at {source['range_m']:.2f} m, {source['azimuth_deg']:+.1f} deg, height "
                         f"{source['z_m']:+.2f} m (blob {source['du_px']:.0f} px from its bar, tolerance {source['tolerance_px']:.0f} px)"
                         if source else ("no clipped core: the scene got brighter, no source in view" if not ev.core_uv
                                         else "no radar object at the clipped core: not a radar-visible source")))
            if source is not None:
                source = follow(source, objects) if ev.active else None
                if source is None and events and "off_perf_s" not in events[-1] and not ev.active:
                    events[-1]["off_perf_s"] = round(t, 4)
                    print(f"LIGHT OFF (camera frame {f.index})")
            refit = t - glare_t >= cfg["light"]["refit_s"]
            readings = {i: r for i, r in readings.items() if i < len(objects)}
            fresh = dict(reader.poll(objects))
            readings.update(fresh)
            res = pipeline.analyse(rgb, objects, proj, cfg, "light", refit=refit, glare_prev=res.glare if res else None,
                                   do_read=False, readings_prev=readings, source=source)
            if not reader.busy and t - read_t >= 1 / cfg["text"]["read_hz"]:
                i, tick = pipeline.read_plan(readings, objects, tick, skip=res.glare.source if res.glare else None)
                if i is not None:
                    o = objects[i]
                    if reader.submit(i, o, rgb, proj.point(o["range_m"], o["azimuth_deg"], cfg["text"]["centre_z_m"]), proj.f):
                        read_t = t
            glare_t = t if refit else glare_t
            light_on = ev.active or res.glare is not None
            phase = "light" if light_on else ("after" if events else "before")
            row = {"camera_frame": f.index, "t_capture_s": round(t, 6), "light": int(light_on), "event": int(ev.active),
                   "onset": int(ev.onset), "lit_px": ev.n_lit * STEP ** 2, "radar_frame": m["frame"] if m else "",
                   "source_range_m": source["range_m"] if source else "", "source_azimuth_deg": source["azimuth_deg"] if source else "",
                   "veil_p": round(res.glare.fits[1][2], 3) if res.glare else "", "new_read": int(bool(fresh)),
                   "recording": int(live.writer is not None)}
            i = pipeline.text_object(readings)
            if i in fresh:
                rd = readings[i]
                row.update({"read": rd.text, "min_margin": round(float(rd.margins.min()), 3) if len(rd.margins) else "",
                            "text_range_m": objects[i]["range_m"]})
                if truth:
                    k = pipeline.correct_chars(rd.text, truth)
                    row["correct"] = k
                    score[phase][0] += k
                    score[phase][1] += len(truth)
            live.log(row)
            if pending and t >= pending:
                g = res.glare
                live.snapshot({"raw": rgb, "result": pipeline.result_view(rgb, res, objects, cfg, truth)},
                              {"kind": "light", "event": events[-1], "objects": objects, "truth": truth,
                               "light": {"object": g.source, "centre_uv": list(np.round(g.centre_uv, 1)),
                                         "veil_b_a_p_rgb": [list(np.round(x, 4)) for x in g.fits],
                                         "profile_radii_px": g.profile["radii"].round(1).tolist(),
                                         "profile_levels_rgb": [lv.round(2).tolist() for lv in g.profile["levels"]]} if g else None,
                               "readings": {str(i): {"text": r.text, "margins": r.margins.round(3).tolist(),
                                                     "boxes": np.round(r.boxes, 1).tolist()} for i, r in res.readings.items()}})
                pending = None
            keys = []
            if live.due():
                left = np.array(rgb[::2, ::2])
                if ev.lit is not None and ev.active:
                    lit = np.kron(ev.lit, np.ones((STEP // 2, STEP // 2), bool))[:left.shape[0], :left.shape[1]]
                    left[lit] = (0.5 * left[lit] + 0.5 * np.array(ov.LIGHT)).astype(np.uint8)
                right = pipeline.result_view(rgb, res, objects, cfg, truth, 0.5)
                view = np.concatenate([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right], axis=1)
                ov.text(view, ("LIGHT ON" if light_on else "NO LIGHT") + ("  REC" if live.writer else ""), 6, 6,
                        ov.LIGHT if light_on else ov.BLACK, 2)
                ov.text(view, ("LIGHT REMOVED + READING" if light_on else "READING") + f"  RADAR FRAME {m['frame'] if m else '-'}",
                        left.shape[1] + 12, 6, ov.BLACK, 2)
                keys = live.show(view)
            if "s" in keys:
                live.snapshot({"raw": rgb, "result": pipeline.result_view(rgb, res, objects, cfg, truth)},
                              {"kind": "light", "light_on": bool(light_on), "objects": objects})
            if "r" in keys:
                live.video(live.writer is None)
            if "q" in keys or "escape" in keys:
                break
    finally:
        if reader is not None:
            reader.stop()
        acc = {k: (round(100 * a / n, 1) if n else None) for k, (a, n) in score.items()}
        live.close({"kind": "light", "truth": truth, "events": events, "characters_correct_pct": acc,
                    "brightness_min_px": model.min_px * STEP ** 2})
    print(f"{len(events)} light events; MetricSight reader, characters correct before / during / after the light: {acc}")


if __name__ == "__main__":
    main()
