"""
dataset.py - turns every recorded session into one feature table, output/dataset.csv (one row per 1.6 s window); a
session is recomputed only when its recordings changed. The folder name of a take is its label.

    python dataset.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from material_classification.features import FEATURES, FeatureExtractor, background_map  # noqa: E402
from material_classification.utils.micro_doppler import FramePreprocessor, RadarParams, load_channel_cal  # noqa: E402
from material_classification.utils.recording import load_take  # noqa: E402

CFG = HERE / "config" / "awr2944_material_spec.cfg"
RAW = HERE / "data" / "raw"
TABLE = HERE / "output" / "dataset.csv"
WINDOW, HOP = 40, 10
MATERIALS = ("metal", "dielectric", "waterlike")
MOTIONS = ("static", "rigid", "articulated", "distributed")
TEXT = ("session", "take", "material", "motion")
NUMBERS = ("t_start_s", "detected", "range_m", "azimuth_deg", *FEATURES)


def label_of(folder):
    if folder == "none":
        return "none", "none"
    material, _, motion = folder.partition("-")
    return (material, motion) if material in MATERIALS and motion in MOTIONS else None


def sessions():
    return sorted(p for p in RAW.glob("session*") if (p / "background").is_dir())


def takes(session):
    return sorted(p for p in session.glob("*/*.npy") if label_of(p.parent.name))


class Radar:
    def __init__(self):
        self.params = RadarParams.from_cfg(CFG)
        self.pre = FramePreprocessor(self.params, max_range_m=2.6, channel_cal=load_channel_cal())
        self.fx = FeatureExtractor(self.params, self.pre)
        self._backgrounds = {}

    def background(self, session):
        if session.name not in self._backgrounds:
            frames = (self.pre.process(c) for f in sorted((session / "background").glob("*.npy")) for c in load_take(f))
            self._backgrounds[session.name] = background_map(frames)
        return self._backgrounds[session.name]

    def frames(self, path, start=0, count=None):
        stack = load_take(path)
        stop = len(stack) if count is None else start + count
        return [self.pre.process(c, i * self.params.frame_period_s) for i, c in enumerate(stack[start:stop])]

    def windows(self, path, session):
        frames, bg = self.frames(path), self.background(session)
        return [(s, self.fx.window(frames[s:s + WINDOW], bg)) for s in range(0, len(frames) - WINDOW + 1, HOP)]

    def window(self, session, take, t_start_s):
        start = int(round(t_start_s / self.params.frame_period_s))
        path = RAW / session / f"{take}.npy"
        return self.fx.window(self.frames(path, start, WINDOW), self.background(RAW / session))


def session_rows(radar, session):
    rows = []
    for path in takes(session):
        material, motion = label_of(path.parent.name)
        for s, w in radar.windows(path, session):
            rows.append({"session": session.name, "take": f"{path.parent.name}/{path.stem}", "material": material,
                         "motion": motion, "t_start_s": round(s * radar.params.frame_period_s, 2),
                         "detected": int(w.detected), "range_m": round(w.range_m, 3),
                         "azimuth_deg": round(w.azimuth_deg, 1),
                         **{n: round(v, 5) if np.isfinite(v) else "" for n, v in w.features.items()}})
        print(f"  {session.name}/{path.parent.name}/{path.stem}")
    return rows


def unchanged(session, rows):
    if not TABLE.exists():
        return False
    newest = max(p.stat().st_mtime for p in session.rglob("*.npy"))
    on_disk = {f"{p.parent.name}/{p.stem}" for p in takes(session)}
    return {r["take"] for r in rows} == on_disk and TABLE.stat().st_mtime > newest


def build(again=False):
    old = list(csv.DictReader(open(TABLE, encoding="utf-8"))) if TABLE.exists() else []
    radar, rows = Radar(), []
    for session in sessions():
        kept = [r for r in old if r["session"] == session.name]
        if kept and not again and unchanged(session, kept):
            rows += kept
        else:
            print(f"[{session.name}] computing features")
            rows += session_rows(radar, session)
    TABLE.parent.mkdir(exist_ok=True)
    with open(TABLE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[*TEXT, *NUMBERS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} windows -> {TABLE.relative_to(HERE)}")
    return TABLE


def read_table(*paths, sessions=None):
    rows = [r for p in paths for r in csv.DictReader(open(p, encoding="utf-8"))
            if sessions is None or r["session"] in sessions]
    t = {k: np.array([r[k] for r in rows], object) for k in TEXT}
    for k in NUMBERS:
        t[k] = np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
    t["detected"] = t["detected"].astype(bool)
    t["take"] = t["session"] + "/" + t["take"]
    t["joint"] = np.where(t["motion"] == "none", "none", t["material"] + "-" + t["motion"])
    return t


if __name__ == "__main__":
    build(again="--again" in sys.argv)
