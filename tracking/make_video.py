"""
make_video.py - the video of a take as the live program showed it: every camera frame goes through track_live.py's
own camera_step and live_frame with the newest radar result that existed at that moment (replay, causal), so the
labels and colours are the classifier's output, with the reader's readings of the ball crop (reading.py).

    python make_video.py final_fixed_1           (--still 12.5 writes one frame at 12.5 s as a PNG)
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import dataset  # noqa: E402
import paths  # noqa: E402
from hardware.scene_view import SceneView  # noqa: E402
from methods.radar_model import frame_ms_of  # noqa: E402
from reading import read_take  # noqa: E402
from track_live import camera_step, live_frame, update_trails  # noqa: E402
from validation import alignment as va  # noqa: E402
from validation.ball_truth import video_index  # noqa: E402

VIDEOS = paths.OUTPUT / "videos"


def frames(take: str, only: float | None = None):
    from backend.alignment import Projector
    cfg, proj = paths.load_config(), Projector(dataset.alignment(take))
    W, H = proj.size_wh
    tk = va.load(take, [], dataset.folder(take))
    take_cfg = paths.load_config(dataset.folder(take) / "config.toml")
    sub = take_cfg["radar"].get("mode") == "subframes"
    period = (frame_ms_of(take_cfg) if sub else take_cfg["waveform"][0]["frame_ms"]) / 1e3
    view = SceneView(note="colour of the dot: material head, ring: motion head")
    live = list(csv.DictReader(open(dataset.folder(take) / "frames.csv", encoding="utf-8")))
    cap = cv2.VideoCapture(str(dataset.folder(take) / "camera.mp4"))
    trails, last, t0, read = {}, None, tk.t_arr[0], ("", -1e9)
    reads = list(read_take(take))
    for j, vi in enumerate(video_index(dataset.folder(take))):
        ok, bgr = cap.read()
        if not ok:
            break
        tc = vi["t"]
        k = int(np.searchsorted(tk.t_ready, tc, side="right") - 1)
        m = dict(tk.recs[k], t=float(tk.t_arr[k])) if k >= 0 else None
        trails, last = update_trails(trails, m, last)
        while reads and reads[0]["t_s"] <= tc - t0:
            read = (reads.pop(0)["text"], tc)
        if only is not None and tc - t0 < only:
            continue
        f = SimpleNamespace(image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_capture=tc, index=j)
        tracks, ball, row, rect, _, _ = camera_step(f, m, by_hand(live[vi["row"]], m), proj, cfg, W, H, period)
        change = k >= 0 and k + 1 < len(tk.recs) and tk.recs[k + 1]["waveform"] != tk.recs[k]["waveform"]
        radar = SimpleNamespace(armed_info={"frame_period_s": period}, status="waveform change" if change else "")
        yield tc - t0, live_frame(view, f, m, tracks, ball, row, proj, cfg, trails, rect,
                                    read[0] if tc - read[1] < 1.5 else None, False, radar, t0)


def by_hand(row: dict, m: dict | None) -> int | None:
    if not m or not row.get("ball_track"):
        return None
    live = {i: (float(r), np.radians(float(a))) for i, r, a in
            (re.match(r"(\d+)@([\d.]+)/([+-][\d.]+)", s).groups() for s in row["tracks"].split(";") if s)}
    if row["ball_track"] not in live:
        return None
    r, a = live[row["ball_track"]]
    near = [(np.hypot(t["x"] - r * np.cos(a), t["y"] - r * np.sin(a)), t["id"]) for t in m["tracks"] if t["status"] == "confirmed"]
    return min(near)[1] if near and min(near)[0] <= 0.3 else None


def still(take: str, t: float) -> np.ndarray:
    return next(img for _, img in frames(take, only=t))


def render(take: str):
    VIDEOS.mkdir(parents=True, exist_ok=True)
    tmp, mp4 = VIDEOS / f"_{take}.mp4", VIDEOS / f"{take}.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
    for _, img in frames(take):
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", str(mp4)])
    tmp.unlink() if r.returncode == 0 else tmp.replace(mp4)
    print(f"  videos/{mp4.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="*", default=list(dataset.TAKES))
    ap.add_argument("--still", type=float, default=None)
    args = ap.parse_args()
    for take in args.takes:
        if args.still is None:
            render(take)
        else:
            path = VIDEOS / f"{take}_{args.still:g}s.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(still(take, args.still), cv2.COLOR_RGB2BGR))
            print(f"  {path}")


if __name__ == "__main__":
    main()
