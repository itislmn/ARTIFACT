"""
dataset.py - the inputs of the tutorial: the .cfg, the two recordings (raw DCA1000 frames, on-chip UART point cloud)
and the synthetic scenes with known targets; writes output/dataset.csv.

    python dataset.py
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from radar_tutorial.cfgfile import Radar  # noqa: E402

CFG = HERE / "config" / "awr2944p.cfg"
RAW = HERE / "recordings" / "raw_capture.bin"
POINTS = HERE / "recordings" / "pointcloud.npz"
OUTPUT = HERE / "output"
SCENES = {
    "one_static": [(2.0, 0.0, 10.0, 1.0)],
    "two_moving": [(1.5, 0.3, 20.0, 1.0), (3.2, -0.5, -35.0, 0.7)],
    "fast_wide": [(4.0, 0.6, 45.0, 1.0)],
    "close_pair": [(2.5, 0.0, -5.0, 1.0), (2.5, 0.0, 8.0, 1.0)],
}
SCENE_SNR_DB = 0.0


def rows():
    radar = Radar.from_cfg(CFG)
    out = [{"name": RAW.stem, "kind": "raw DCA1000 frames", "file": RAW.name, "bytes": RAW.stat().st_size,
            "frames": RAW.stat().st_size // radar.bytes_per_frame, "truth": ""},
           {"name": POINTS.stem, "kind": "on-chip UART point cloud", "file": POINTS.name, "bytes": POINTS.stat().st_size,
            "frames": "", "truth": ""}]
    for name, targets in SCENES.items():
        out.append({"name": name, "kind": f"synthetic, SNR {SCENE_SNR_DB:g} dB per sample", "file": "", "bytes":
                    radar.bytes_per_frame, "frames": 1, "truth": json.dumps(targets)})
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
