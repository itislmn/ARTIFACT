"""
dataset.py - the measured sweeps: every reading (angle, TX, RX, gain, noise, range) of the data sets in TAGS from
data/<cut>_<tag>_measurements.json; writes output/dataset.csv with one row per data set.

    python dataset.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

DATA = HERE / "data"
OUTPUT = HERE / "output"
CUT = "azimuth"
TAGS = {"air": "No Radome (Air)", "pla": "PLA Radome", "petg": "PETG Radome"}
REFERENCE = "air"


def load(tag, cut=CUT) -> dict:
    log = json.loads((DATA / f"{cut}_{tag}_measurements.json").read_text())
    a = {k: np.array([r.get(k, np.nan) if r.get(k) is not None else np.nan for r in log], float)
         for k in ("angle", "tx", "rx", "gain_linear", "gain_raw_linear", "noise_linear", "range_m")}
    ok = np.isfinite(a["gain_linear"]) & (a["gain_linear"] > 0)
    out = {k: v[ok] for k, v in a.items()}
    out["channel"] = (4 * out["tx"] + out["rx"]).astype(int)
    out["background"] = float(np.mean([bool(r.get("bg_subtracted")) for r in log]))
    out["runs"] = len({r.get("run") for r in log})
    return out


def rows():
    out = []
    for tag, label in TAGS.items():
        d = load(tag)
        out.append({"tag": tag, "label": label, "cut": CUT, "readings": d["angle"].size,
                    "frames": d["angle"].size // 16, "angle_min_deg": round(float(d["angle"].min()), 1),
                    "angle_max_deg": round(float(d["angle"].max()), 1),
                    "target_range_m": round(float(np.median(d["range_m"])), 3),
                    "background_subtracted": d["background"], "runs": d["runs"]})
    return out


def write(path=OUTPUT / "dataset.csv"):
    path.parent.mkdir(parents=True, exist_ok=True)
    r = rows()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(r[0]))
        w.writeheader()
        w.writerows(r)
    return r


if __name__ == "__main__":
    for row in write():
        print(row)
