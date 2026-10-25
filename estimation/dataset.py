"""
dataset.py - a recorded (or simulated) session as evaluation input: takes with ground truth, the empty-room background,
the channel calibration from the calibration take and the evaluation region from the room geometry; writes
output/dataset.csv (every reflector in every frame: truth, SNR and its 16 channel values).

    python dataset.py
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from estimation.core.array import ArrayModel  # noqa: E402
from estimation.core.frame import background_from_recording  # noqa: E402
from estimation.core.interfaces import Settings  # noqa: E402
from estimation.core.metrics import Tolerance  # noqa: E402
from estimation.core.radar import RadarConfig, load_truth, open_recording, range_fft, with_layout  # noqa: E402
from estimation.core.steering import all_channel_steering, check, elevation_deg, take_snapshot  # noqa: E402

CFG = HERE / "config" / "awr2944_doa_perfect.cfg"
PLAN = HERE / "config" / "session_plan.json"
LAYOUT = HERE / "config" / "awr2944pevm_layout.json"
MEASURED = HERE / "data" / "raw" / "S2"
OUTPUT = HERE / "output"
EVAL_ROLES = ("single", "multi", "null")
WALL_MARGIN_M = 0.15
RANGE_MARGIN_M = 0.5
MATCH_CELLS = 2


@dataclass
class Session:
    name: str
    simulated: bool
    cfg: RadarConfig
    array: ArrayModel
    takes: list
    background: dict | None
    calibration: np.ndarray | None
    settings: Settings
    tolerance: Tolerance

    @property
    def evaluated(self):
        return [t for t in self.takes if t["scenario"] in EVAL_ROLES]


def plan():
    return json.loads(PLAN.read_text(encoding="utf-8"))


def region(p):
    far = max(t["range_m"] for take in p["takes"] for t in take.get("targets", []))
    wall = p.get("geometry", {}).get("right_wall_y_m", float("inf"))
    return far + RANGE_MARGIN_M, wall - WALL_MARGIN_M


def takes(folder):
    out = []
    for path in sorted(folder.iterdir()):
        if path.suffix in (".bin", ".npy") and path.with_suffix(".json").exists():
            doc = load_truth(path)
            targets = doc.get("targets", [])
            out.append({"path": path, "name": path.stem, "scenario": doc.get("scenario", ""), "doc": doc,
                        "truth": np.array([[t["range_m"], t["azimuth_deg"]] for t in targets], float).reshape(-1, 2),
                        "elev": np.array([t.get("elevation_deg", 0.0) for t in targets], float)})
    return out


def load(folder=MEASURED) -> Session:
    cfg = with_layout(RadarConfig.from_cfg(CFG), LAYOUT)
    array = ArrayModel(cfg)
    r_max, y_max = region(plan())
    settings = Settings(r_max_m=r_max, y_max_m=y_max)
    tk = takes(folder)
    simulated = any(t["doc"].get("simulated") for t in tk)
    bgs = [t for t in tk if t["scenario"] == "background"]
    bg = None
    if bgs:
        parts = [background_from_recording(*open_recording(t["path"], cfg), cfg) for t in bgs]
        bg = {k: np.mean([p[k] for p in parts], axis=0) for k in ("rfft", "adc")}
    cal = None
    cals = [t for t in tk if t["scenario"] == "calibration" and t["truth"].size]
    if cals:
        t = cals[0]
        y16, _ = take_snapshot(t["path"], cfg, background=bg, range_m=t["truth"][0, 0])
        cal = y16 / all_channel_steering(cfg, t["truth"][0, 1], t["elev"][0])
        cal = (cal / cal[0]).astype(np.complex64)
        if bg is not None:
            bg = {"rfft": bg["rfft"] / cal[None, :], "adc": bg["adc"]}
    tol = Tolerance(MATCH_CELLS * cfg.range_res_m, Tolerance.from_hardware(cfg, array).angle_deg)
    return Session(folder.name, simulated, cfg, array, tk, bg, cal, settings, tol)


def frames(session, take, K=None):
    n, get = open_recording(take["path"], session.cfg)
    for i in range(n):
        f = get(i)
        yield i, (f if K is None or K >= f.shape[1] else f[:, :K, :])


def checks(session):
    cfg, array, cal, bg = session.cfg, session.array, session.calibration, session.background
    rows = []
    for t in session.takes:
        for j, ((r, th), el) in enumerate(zip(t["truth"], t["elev"])):
            near = np.abs(t["truth"][:, 0] - r) < 2 * cfg.range_res_m
            y16, r_found = take_snapshot(t["path"], cfg, cal=cal, background=bg, range_m=r, others_m=t["truth"][~near, 0])
            phi, ratio = elevation_deg(y16, cfg, t["truth"][near, 1], array.az_sign)
            i = int(np.argmin(np.abs(t["truth"][near, 1] - th)))
            fit = check(y16, cfg, array, th)
            rows.append({"take": t["name"], "scenario": t["scenario"], "reflector": j,
                         "label": t["doc"]["targets"][j].get("label", ""), "range_m": r, "range_found_m": r_found,
                         "azimuth_deg": th, "azimuth_fit_deg": fit["theta_hat"], "plane_wave_fit": fit["fit"],
                         "elevation_deg": float(el), "elevation_measured_deg": float(phi[i]),
                         "tx1_ratio": float(ratio[i])})
    return rows


def write_dataset(session, path):
    cfg = session.cfg
    rows = []
    for t in session.evaluated:
        for i, f in frames(session, t):
            X = range_fft(f, cfg, session.calibration)
            if session.background is not None:
                X = X - session.background["rfft"][:, None, :].astype(X.dtype)
            for j, tg in enumerate(t["doc"].get("targets", [])):
                k = int(round(tg["range_m"] / cfg.range_res_m))
                Y = X[k]
                ybar = Y.mean(axis=0)
                noise = float(np.mean(np.var(Y, axis=0, ddof=1)))
                snr = float(10 * np.log10(np.mean(np.abs(ybar) ** 2) / (noise / Y.shape[0])))
                row = {"take": t["name"], "scenario": t["scenario"], "frame": i, "reflector": j, "label": tg.get("label", ""),
                       "x_m": tg.get("x_m"), "y_m": tg.get("y_m"), "z_m": tg.get("z_m", 0.0), "range_m": tg["range_m"],
                       "azimuth_deg": tg["azimuth_deg"], "elevation_deg": tg.get("elevation_deg", 0.0), "snr_db": snr}
                for v in range(cfg.n_virt):
                    row[f"ch{v + 1}_re"], row[f"ch{v + 1}_im"] = float(ybar[v].real), float(ybar[v].imag)
                rows.append(row)
    write_csv(path, rows)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def with_settings(session, **kw):
    return replace(session, settings=replace(session.settings, **kw))


if __name__ == "__main__":
    s = load()
    rows = write_dataset(s, OUTPUT / "dataset.csv")
    write_csv(OUTPUT / "cache" / "checks_measured.csv", checks(s))
    print(f"output/dataset.csv: {len(rows)} rows, region R <= {s.settings.r_max_m:.2f} m, y <= {s.settings.y_max_m:.2f} m")
