"""
read_text.py - live text reading: the radar finds the board, MetricSight reads at the radar range in its own process,
the window shows camera and radar together.

    python read_text.py --take text_085c --truth ARTIFACT --exposure -6 --record
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import argparse
import time

import numpy as np

from backend import overlay as ov
from backend import pipeline
from backend.reader_live import LiveReader
from backend.session import Live, add_common_args
from backend.settings import load_config

MIN_LETTER_PX = 16


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--truth", default="", help="the written text, e.g. ARTIFACT: every reading is scored")
    ap.add_argument("--record", action="store_true", help="camera video from the start")
    args = ap.parse_args()
    cfg = load_config()
    truth = args.truth.upper()
    from backend.alignment import Projector
    proj = Projector()
    cap = cfg["text"]["letter_mm"] / 1000
    print(f"resolution: letters of {1000 * cap:.0f} mm stay >= {MIN_LETTER_PX} px high up to "
          f"{proj.f * cap / MIN_LETTER_PX:.2f} m")
    live = Live("TEXT READING", cfg, args.session or cfg["session"], args.take, args.exposure, args.camera)
    n_ok = n_all = s_ok = 0
    stable = pipeline.Stable()
    reader = None
    try:
        radar = live.live_radar(args.cli)
        live.video(args.record)
        reader = LiveReader(cfg).start()
        readings, t_read, tick, t0 = {}, -1e9, 0, time.perf_counter()
        while live.alive and (args.seconds <= 0 or time.perf_counter() - t0 < args.seconds):
            f = live.frame()
            radar.poll()
            m = radar.last
            objects = m["objects"] if m else []
            rgb = f.image
            readings = {i: r for i, r in readings.items() if i < len(objects)}
            fresh = dict(reader.poll(objects))
            readings.update(fresh)
            if not reader.busy and f.t_capture - t_read >= 1 / cfg["text"]["read_hz"]:
                i, tick = pipeline.read_plan(readings, objects, tick)
                if i is not None:
                    o = objects[i]
                    if reader.submit(i, o, rgb, proj.point(o["range_m"], o["azimuth_deg"], cfg["text"]["centre_z_m"]), proj.f):
                        t_read = f.t_capture
            res = pipeline.analyse(rgb, objects, proj, cfg, "text", do_read=False, readings_prev=readings)
            row = {"camera_frame": f.index, "t_capture_s": round(f.t_capture, 6), "new_read": int(bool(fresh)),
                   "radar_frame": m["frame"] if m else "", "radar_age_ms": round(1e3 * (f.t_capture - m["t"]), 1) if m else "",
                   "objects": ";".join(f"{o['range_m']:.3f}@{o['azimuth_deg']:+.2f}" for o in objects),
                   "recording": int(live.writer is not None)}
            i = pipeline.text_object(readings)
            if i in fresh:
                rd = readings[i]
                row.update({"text_object": i, "text_range_m": objects[i]["range_m"], "read": rd.text,
                            "min_margin": round(float(rd.margins.min()), 3) if len(rd.margins) else "",
                            "read_ms": round(rd.ms, 1)})
                row["stable"] = stable.add(rd.text)
                if truth:
                    k = pipeline.correct_chars(rd.text, truth)
                    ks = pipeline.correct_chars(row["stable"], truth)
                    row["correct"], row["correct_stable"] = k, ks
                    n_ok, n_all, s_ok = n_ok + k, n_all + len(truth), s_ok + ks
            live.log(row)
            keys = []
            if live.due():
                left = pipeline.result_view(rgb, res, objects, cfg, truth, 0.5)
                for o in objects:
                    hpx = proj.f * cap / o["range_m"]
                    uv = proj.point(o["range_m"], o["azimuth_deg"], cfg["text"]["centre_z_m"])
                    if uv is not None and hpx < MIN_LETTER_PX:
                        ov.text(left, f"TOO FAR: LETTERS {hpx:.0f} PX", uv[0] / 2, uv[1] / 2 + 30, ov.LOSS, 2)
                right = ov.top_view(np.array(m["dets"]).reshape(-1, 2) if m else [], objects)
                view = np.concatenate([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right], axis=1)
                ov.text(view, "CAMERA + RADAR: READING" + ("  REC" if live.writer else "")
                        + (f"  STABLE {stable.text}" if stable.text else "")
                        + (f"  {100 * n_ok / n_all:.0f}% / STABLE {100 * s_ok / n_all:.0f}% CORRECT" if n_all else ""), 6, 6,
                        (200, 0, 0) if live.writer else ov.BLACK, 2)
                ov.text(view, f"RADAR LIVE  FRAME {m['frame'] if m else '-'}  INFINI-REX {m['ms'] if m else 0:.0f} MS",
                        left.shape[1] + 12, 6, ov.BLACK, 2)
                keys = live.show(view)
            if "s" in keys:
                live.snapshot({"raw": rgb, "result": pipeline.result_view(rgb, res, objects, cfg, truth)},
                              {"kind": "text", "objects": objects, "truth": truth, "radar_frame": m["frame"] if m else None,
                               "readings": {str(i): {"text": r.text, "margins": r.margins.round(3).tolist(),
                                                     "boxes": np.round(r.boxes, 1).tolist()}
                                            for i, r in res.readings.items()}})
            if "r" in keys:
                live.video(live.writer is None)
            if "q" in keys or "escape" in keys:
                break
    finally:
        if reader is not None:
            reader.stop()
        live.close({"kind": "text", "truth": truth, "characters_correct_pct": round(100 * n_ok / n_all, 1) if n_all else None,
                    "stable_characters_correct_pct": round(100 * s_ok / n_all, 1) if n_all else None, "stable_final": stable.text})
    if n_all:
        print(f"MetricSight reader: {100 * n_ok / n_all:.1f} % of the characters correct per reading, "
              f"{100 * s_ok / n_all:.1f} % for the stable reading (majority of the last 15), over {n_all // len(truth)} readings; "
              f"final stable reading: {stable.text}")


if __name__ == "__main__":
    main()
