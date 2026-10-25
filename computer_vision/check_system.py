"""
check_system.py - pass/fail check of the camera alone, the radar alone and both together (frame rates, lost packets,
Infini-Rex time).

    python check_system.py --exposure -6            (camera | radar | together for one part)
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
from backend.session import Live, stamp
from backend.settings import CALIB, load_config


def fps_between(ts, t0, t1):
    t = [x for x in ts if t0 <= x <= t1]
    return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", nargs="?", default="all", choices=["all", "camera", "radar", "together"])
    ap.add_argument("--session", default=None)
    ap.add_argument("--cli", default=None)
    ap.add_argument("--exposure", default=None)
    ap.add_argument("--camera", default=None, help="part of another camera's name (tests only)")
    ap.add_argument("--seconds", type=float, default=0.0, help="duration of each part (default 10 / 15 / until q)")
    args = ap.parse_args()
    cfg = load_config()
    session = args.session or cfg["session"]
    live = Live("SYSTEM CHECK", cfg, session, f"check_{stamp()[:15]}", args.exposure, args.camera)
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    try:
        if args.part in ("all", "camera"):
            print("\nCAMERA ALONE")
            t0 = time.perf_counter()
            while live.alive and time.perf_counter() - t0 < (args.seconds or 10):
                f = live.frame()
                view = np.array(f.image[::2, ::2])
                ov.text(view, f"CAMERA ALONE  {live.cam.recent_fps():.1f} FPS", 6, 6, ov.BLACK, 2)
                if "q" in live.show(view):
                    break
            c = live.camera
            fps = fps_between(live.cam.t_capture, t0 + 1, time.perf_counter())
            check("camera found by name", True, f"{c['name']} (DirectShow index {c['device_index']})")
            check("camera format", (c["width"], c["height"]) == (cfg["camera"]["width"], cfg["camera"]["height"]),
                  f"{c['format']} {c['width']}x{c['height']}")
            check("camera frame rate", fps >= 25, f"{fps:.1f} fps (exposure {c['exposure']['value']}, {c['exposure']['mode']})")
            check("camera frames skipped", live.cam.stats()["frames_skipped"] == 0, str(live.cam.stats()["frames_skipped"]))

        if args.part in ("all", "radar", "together"):
            print("\nRADAR (arming)")
            radar = live.live_radar(args.cli)
            m0 = None
            while live.alive and (radar.last is None or radar.last["received"] < 5):
                radar.poll()
                live.idle("RADAR: WAITING FOR FRAMES")
            m0 = radar.last
            check("radar armed", True, radar.status)

        if args.part in ("all", "radar"):
            print("\nRADAR ALONE")
            t0 = time.perf_counter()
            while live.alive and time.perf_counter() - t0 < (args.seconds or 15):
                live.frame()
                radar.poll()
                m = radar.last
                view = ov.top_view(np.array(m["dets"]).reshape(-1, 2), m["objects"])
                ov.text(view, f"RADAR ALONE  FRAME {m['frame']}  INFINI-REX {m['ms']:.0f} MS  LOST {m['lost']}", 6, 6, ov.BLACK, 2)
                if "q" in live.show(view):
                    break
            m = radar.last
            el = time.perf_counter() - t0
            period = radar.frame_period_s
            rate = (m["received"] - m0["received"]) / el
            check("radar frame rate", abs(rate * period - 1) <= 0.1, f"{rate:.2f} frames/s (cfg {1 / period:.1f})")
            check("radar UDP packets", m["lost"] == 0, f"{m['packets']} received, {m['lost']} lost")
            check("radar frames complete", m["incomplete"] == 0, f"{m['incomplete']} incomplete frames dropped")
            check("Infini-Rex in real time", m["ms"] < 1000 * period, f"{m['ms']:.0f} ms per frame (period {1000 * period:.0f} ms), "
                  f"{m['skipped']} frames not processed")
            check("radar objects", True, "; ".join(f"{o['range_m']:.2f} m {o['azimuth_deg']:+.1f} deg" for o in m["objects"]) or "none")

        if args.part in ("all", "together"):
            from backend.alignment import Projector
            proj = Projector() if (CALIB / "alignment.json").exists() else None
            print("\nCAMERA + RADAR TOGETHER (q = stop)" + ("" if proj else " - no alignment yet: run  python align.py nominal"))
            t0, m_start, ages = time.perf_counter(), radar.last, []
            while live.alive and (args.seconds <= 0 or time.perf_counter() - t0 < args.seconds):
                f = live.frame()
                radar.poll()
                m = radar.last
                ages.append(1e3 * (f.t_capture - m["t"]))
                if proj is not None:
                    res = pipeline.analyse(np.ascontiguousarray(f.image), m["objects"], proj, cfg, "text", do_read=False)
                    left = pipeline.result_view(f.image, res, m["objects"], cfg, "", 0.5)
                else:
                    left = np.array(f.image[::2, ::2])
                right = ov.top_view(np.array(m["dets"]).reshape(-1, 2), m["objects"])
                view = np.concatenate([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right], axis=1)
                ov.text(view, f"CAMERA {live.cam.recent_fps():.0f} FPS + RADAR OBJECTS", 6, 6, ov.BLACK, 2)
                ov.text(view, f"RADAR FRAME {m['frame']}  AGE {ages[-1]:.0f} MS", left.shape[1] + 12, 6, ov.BLACK, 2)
                live.log({"camera_frame": f.index, "t_capture_s": round(f.t_capture, 6), "radar_frame": m["frame"],
                          "radar_age_ms": round(ages[-1], 1), "objects": len(m["objects"])})
                if "q" in live.show(view):
                    break
            fps = fps_between(live.cam.t_capture, t0 + 1, time.perf_counter())
            m = radar.last
            check("camera while the radar streams", fps >= 25, f"{fps:.1f} fps")
            check("radar keeps streaming", m["received"] > m_start["received"],
                  f"{m['received'] - m_start['received']} frames, {m['lost'] - m_start['lost']} packets lost")
            check("radar -> camera age", True, f"median {np.median(ages):.0f} ms (newest radar frame vs. camera frame; "
                  f"the radar frame period is {1000 * radar.frame_period_s:.0f} ms)")
    finally:
        live.close({"checks": checks})
    n_fail = sum(not c["pass"] for c in checks)
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed" + ("" if not n_fail else " - fix the FAILs before continuing"))


if __name__ == "__main__":
    main()
