"""
radar_capture.py - a finite radar take (estimation capture) with every radar frame time on the camera clock.

    python -m backend.radar_capture --cli COM4 --session CV1 --name align_1 --scenario single --target-xy ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from backend.settings import RECORDINGS, ROOT, load_config, radar_dir

EST = (ROOT / load_config()["radar"]["estimation_dir"]).resolve()
sys.path.insert(0, str(EST))
sys.path.insert(0, str(EST.parent))
import estimation.capture as cap  # noqa: E402


class TimedReceiver(cap.Receiver):
    last = None

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.t_frames = []
        push = self.asm.push

        def timed_push(byte_count, payload):
            frames = push(byte_count, payload)
            if frames:
                self.t_frames.extend([time.perf_counter()] * len(frames))
            return frames

        self.asm.push = timed_push
        TimedReceiver.last = self


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--session", default="")
    ap.add_argument("--name", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--dry-run", action="store_true")
    known, _ = ap.parse_known_args(argv)
    if known.plan is None:
        argv += ["--plan", str(EST / "config" / "session_plan.json")]
    cap.Receiver = TimedReceiver
    cap.RAW = RECORDINGS
    argv = [a if i == 0 or argv[i - 1] != "--session" else f"{a}/radar" for i, a in enumerate(argv)]
    sys.argv = ["core.capture"] + argv
    t_start = time.perf_counter()
    cap.main()
    rx = TimedReceiver.last
    if known.dry_run or rx is None:
        return
    path = radar_dir(known.session) / f"{known.name}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    from estimation.core.radar import RadarConfig
    cfg = RadarConfig.from_cfg(doc.get("cfg_file") or EST / "config" / "awr2944_doa_perfect.cfg")
    doc.update(frame_time_perf_s=[round(t, 6) for t in rx.t_frames[:rx.frames_written]],
               capture_started_perf_s=round(t_start, 6), frame_period_s=cfg.frame_period_s,
               clock="time.perf_counter = QueryPerformanceCounter, shared with camera.py (computer_vision)")
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    t = rx.t_frames[:rx.frames_written]
    if len(t) > 1:
        d = [1e3 * (b - a) for a, b in zip(t, t[1:])]
        print(f"radar frame times on the camera clock: {len(t)} frames, period {sorted(d)[len(d) // 2]:.1f} ms "
              f"(cfg {1e3 * cfg.frame_period_s:.0f} ms)")


if __name__ == "__main__":
    main()
