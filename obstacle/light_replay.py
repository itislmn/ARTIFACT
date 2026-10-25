"""
light_replay.py - the final light monitor on every recorded camera video (it needs no radar): onsets, offsets and
flashing against the annotated light events -> output/light.csv.

    python light_replay.py         (computes only the runs without a result in output/light.csv)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import dataset  # noqa: E402
import paths  # noqa: E402
from backend.alignment import Projector  # noqa: E402
from methods.light import LightMonitor  # noqa: E402

OUT = paths.OUTPUT / "light.csv"
FIELDS = ["run", "t_s", "event", "flashing"]


def replay_run(run: str, cfg: dict, f_px: float) -> list[dict]:
    folder = paths.TAKES / run
    rows = list(csv.DictReader(open(folder / "camera.csv", encoding="utf-8")))
    idx = np.array([int(r["camera_frame"]) for r in rows])
    tt = np.array([float(r["t"]) for r in rows])
    cap = cv2.VideoCapture(str(folder / "camera.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    times = np.interp(np.arange(n) + idx[0], idx, tt)
    tk = dataset.load(run)
    mon = LightMonitor(cfg, f_px)
    out = []
    for t in times:
        ok, img = cap.read()
        if not ok:
            break
        ev = mon.update(img[..., ::-1], float(t))
        if ev.onset or ev.offset:
            out.append({"run": run, "t_s": round(float(t) - tk.t0, 3), "event": "onset" if ev.onset else "offset",
                        "flashing": int(ev.flashing)})
    return out


def run() -> list[dict]:
    done = [x for x in csv.DictReader(open(OUT, encoding="utf-8")) if x["run"] in dataset.RUNS] if OUT.exists() else []
    todo = [r for r in dataset.RUNS if (paths.TAKES / r / "camera.mp4").exists() and r not in {x["run"] for x in done}]
    if todo:
        cfg, f_px = paths.load_config(), Projector().f
        for r in todo:
            done += replay_run(r, cfg, f_px) or [{"run": r, "t_s": "", "event": "", "flashing": ""}]
        with open(OUT, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(done)
    return done


def score(rows: list[dict]) -> dict:
    tp = n_on = n_true = found = flash_true = flash_found = 0
    latency = []
    for r in sorted({x["run"] for x in rows}):
        tk = dataset.load(r)
        lit = [iv for k in ("light", "flashing") for iv in tk.events.get(k, [])]
        on = [tk.t0 + float(x["t_s"]) for x in rows if x["run"] == r and x["event"] == "onset"]
        fl = [tk.t0 + float(x["t_s"]) for x in rows if x["run"] == r and x["event"] == "onset" and x["flashing"] == "1"]
        n_on += len(on)
        tp += sum(any(a - 0.2 <= t <= b + 1.0 for a, b in lit) for t in on)
        for a, _ in tk.events.get("light", []):
            n_true += 1
            hit = [t - a for t in on if -0.2 <= t - a <= 1.0]
            found += bool(hit)
            latency += hit[:1]
        for a, b in tk.events.get("flashing", []):
            flash_true += 1
            flash_found += any(a <= t <= b + 1.0 for t in fl)
    return {"onsets": n_on, "real": tp, "truth": n_true, "found": found, "latency_s": np.array(latency),
            "flashing_truth": flash_true, "flashing_found": flash_found,
            "runs": sorted({x["run"] for x in rows})}


if __name__ == "__main__":
    s = score(run())
    print(f"light on all recorded videos: {s['real']}/{s['onsets']} onsets real, {s['found']}/{s['truth']} lights found, "
          f"flashing {s['flashing_found']}/{s['flashing_truth']}")
