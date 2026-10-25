"""
reading.py - the text on the ball read during the replay as the live program reads it: on every sure ball crop, at most
read_hz times per second, MetricSight reads a label-sized region at the radar's ball centre (the reader of
../computer_vision, the same call as live and in ../obstacle).

    from reading import read_take
"""
from __future__ import annotations

import csv
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
from hardware.reader_process import correct_chars  # noqa: E402
from methods.radar_model import frame_ms_of  # noqa: E402
from methods.text_reading import read_object  # noqa: E402
from track_live import camera_step  # noqa: E402
from validation import alignment as va  # noqa: E402
from validation.ball_truth import video_index  # noqa: E402

COLUMNS = ["t_s", "text", "correct", "min_margin"]


def read_take(take: str) -> list[dict]:
    cache = dataset.CACHE / f"{take}_reads.csv"
    log = paths.RESULTS / "replay" / take / "frames.jsonl"
    if cache.exists() and cache.stat().st_mtime > log.stat().st_mtime:
        return [{k: (r[k] if k == "text" else float(r[k])) for k in COLUMNS} for r in csv.DictReader(open(cache, encoding="utf-8"))]
    from backend.alignment import Projector
    from backend.settings import load_config as cv_config
    from make_video import by_hand
    from readers import metricsight
    cfg, cvcfg, proj = paths.load_config(), cv_config(), Projector(dataset.alignment(take))
    rcfg = {**cvcfg, "text": {**cvcfg["text"], "region_m": list(cfg["ball"]["label_m"])}}
    take_cfg = paths.load_config(dataset.folder(take) / "config.toml")
    period = (frame_ms_of(take_cfg) if take_cfg["radar"].get("mode") == "subframes"
              else take_cfg["waveform"][0]["frame_ms"]) / 1e3
    W, H = proj.size_wh
    tk = va.load(take, [], dataset.folder(take))
    live = list(csv.DictReader(open(dataset.folder(take) / "frames.csv", encoding="utf-8")))
    cap = cv2.VideoCapture(str(dataset.folder(take) / "camera.mp4"))
    rows, t_read = [], -1e9
    for j, vi in enumerate(video_index(dataset.folder(take))):
        ok, bgr = cap.read()
        if not ok:
            break
        tc = vi["t"]
        k = int(np.searchsorted(tk.t_ready, tc, side="right") - 1)
        if k < 0 or tc - t_read < 1 / cvcfg["text"]["read_hz"]:
            continue
        m = dict(tk.recs[k], t=float(tk.t_arr[k]))
        f = SimpleNamespace(image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_capture=tc, index=j)
        _, _, row, _, _, bt = camera_step(f, m, by_hand(live[vi["row"]], m), proj, cfg, W, H, period)
        if not row.get("sure"):
            continue
        uv = proj.point(bt["range_m"], bt["theta_deg"], cfg["ball"]["centre_z_m"])
        rd = read_object(metricsight, f.image, uv, bt["range_m"], proj.f, rcfg, cfg["ball"]["diameter_m"] / 2)
        t_read = tc
        rows.append({"t_s": round(tc - tk.t_arr[0], 3), "text": rd.text, "correct": correct_chars(rd.text, dataset.TEXT),
                     "min_margin": round(float(np.min(rd.margins)), 4) if len(rd.margins) else np.nan})
    with open(cache, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows
