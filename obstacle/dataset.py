"""
dataset.py - the evaluated runs and their ground truth: the tape-measured objects of scene.json and the events marked
with annotate.py (light on, flashing, person in view, obstacle placed), per radar look -> output/dataset.csv.

    python dataset.py
"""
from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import paths

TEST = {"replay/tr1_fixed_test"}
RUNS = sorted(f"{p.parent.name}/{p.name}" for p in paths.TAKES.glob("OB*/*") if (p / "scene.json").exists())
REPLAYS = sorted(f"replay/{p.name}" for p in paths.TAKES.glob("replay/OB*_*") if (p / "summary.json").exists())
TAKES = (RUNS + REPLAYS) or sorted(TEST)
PERSON_HEIGHT_M = 1.95
EVENTS = ("light", "flashing", "person")


@dataclass
class Truth:
    id: str
    kind: str
    label: str
    lo: np.ndarray
    hi: np.ndarray
    material: str | None
    motion: str | None
    times: list | None = None

    def active(self, t: float) -> bool:
        return self.times is None or any(a <= t <= b for a, b in self.times)

    @property
    def centre(self) -> np.ndarray:
        return (self.lo[:2] + self.hi[:2]) / 2


@dataclass
class Take:
    name: str
    folder: Path
    cfg: dict
    looks: list
    decisions: list
    rail: np.ndarray
    rail_info: dict
    camera: list
    readings: list
    summary: dict
    scene: dict
    events: dict
    truth: list = field(default_factory=list)
    t0: float = 0.0
    test: bool = False

    def during(self, name: str, t: float) -> bool:
        return any(a <= t <= b for a, b in self.events.get(name, []))


def _lines(path: Path) -> list:
    return [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")] if path.exists() else []


def _rows(path: Path) -> list:
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


def events_of(folder: Path, scene: dict, t0: float) -> dict:
    ann = folder / "annotations.json"
    if ann.exists():
        return {k: [list(map(float, iv)) for iv in v] for k, v in json.loads(ann.read_text())["events"].items()}
    rel = lambda ivs: [[t0 + a, t0 + b] for a, b in ivs or []]
    ev = {"person": [iv for p in scene.get("persons", []) for iv in rel(p.get("in_scene_s"))],
          "light": rel((scene.get("light") or {}).get("on_s")), "flashing": rel((scene.get("light") or {}).get("flashing_s"))}
    for i, o in enumerate(scene.get("obstacles", []), 1):
        if o.get("in_scene_s"):
            ev[f"obstacle:{i}"] = rel(o["in_scene_s"])
    return ev


def truth_objects(scene: dict, cfg: dict, events: dict) -> list[Truth]:
    floor = -float(cfg["platform"]["antenna_height_m"])
    out = []
    for i, o in enumerate(scene.get("obstacles", []), 1):
        out.append(Truth(f"obstacle:{i}", "obstacle", o.get("label", f"obstacle {i}"),
                         np.array([o["x_m"][0], o["y_m"][0], o["z_m"][0]], float),
                         np.array([o["x_m"][1], o["y_m"][1], o["z_m"][1]], float),
                         o.get("material"), o.get("motion", "static"), events.get(f"obstacle:{i}")))
    b = scene.get("ball")
    if b:
        c, r = np.asarray(b["centre_xyz_m"], float), b.get("diameter_m", 0.2) / 2
        s = b.get("swing_m", 0.0)
        out.append(Truth("ball", "ball", b.get("label", "ball"), c - [r + s, r + s, r], c + [r + s, r + s, r],
                         b.get("material"), b.get("motion"), None))
    for i, p in enumerate(scene.get("persons", []), 1):
        out.append(Truth(f"person:{i}", "person", p.get("label", "person"),
                         np.array([p["walk_zone_x_m"][0], p["walk_zone_y_m"][0], floor], float),
                         np.array([p["walk_zone_x_m"][1], p["walk_zone_y_m"][1], floor + PERSON_HEIGHT_M], float),
                         p.get("material", "waterlike"), p.get("motion", "articulated"), events.get("person") or None))
    lamp = scene.get("light")
    if lamp and "x_m" in lamp:
        h = cfg["light"]["source_size_m"]
        c = np.array([lamp["x_m"], lamp["y_m"], lamp.get("z_m", 0.0)], float)
        out.append(Truth("lamp", "lamp", lamp.get("label", "lamp"), c - h, c + h, lamp.get("material"), "static", None))
    return out


def in_view(t: Truth, pose, cfg: dict) -> bool:
    c = t.centre - np.asarray(pose, float)[:2]
    r = float(np.hypot(*c))
    return bool(c[0] > 0 and cfg["radar"]["r_min_m"] < r < cfg["radar"]["r_max_m"]
                and abs(np.degrees(np.arctan2(c[1], c[0]))) <= 60.0)


def recorded(folder: Path) -> Path:
    s = folder / "summary.json"
    src = json.loads(s.read_text()).get("replay_of") if s.exists() else None
    return Path(src) if src and (Path(src) / "scene.json").exists() else folder


def load(name: str) -> Take | None:
    folder = paths.TAKES / name
    truth = recorded(folder)
    looks = _lines(folder / "radar.jsonl.gz")
    scene = json.loads((truth / "scene.json").read_text()) if (truth / "scene.json").exists() else {}
    if not looks or not scene.get("measured"):
        return None
    cfg = paths.load_config(folder / "config.toml") if (folder / "config.toml").exists() else paths.load_config()
    rail = np.zeros((0, 5))
    if (truth / "rail.csv").exists():
        R = np.genfromtxt(truth / "rail.csv", delimiter=",", names=True)
        if R.size:
            rail = np.c_[R["t"], R["x"], R["y"], R["z"], R["moving"]].reshape(-1, 5)
    t0 = looks[0]["t"]
    events = events_of(truth, scene, t0)
    js = lambda f, d=folder: json.loads((d / f).read_text()) if (d / f).exists() else {}
    tk = Take(name, folder, cfg, looks, _lines(folder / "decisions.jsonl.gz"), rail, js("rail.json", truth),
              _rows(folder / "camera.csv"), _rows(folder / "readings.csv"), js("summary.json"), scene, events,
              t0=t0, test=name in TEST)
    tk.truth = truth_objects(scene, cfg, events)
    return tk


def load_all(names=None) -> list[Take]:
    return [tk for tk in (load(n) for n in (names or TAKES)) if tk is not None]


def rows(takes: list[Take]) -> list[dict]:
    out = []
    for tk in takes:
        for k, lk in enumerate(tk.looks):
            t = lk["t"]
            base = {"take": tk.name, "look": k, "t_s": round(t - tk.t0, 4), "rail_x_m": lk["pose"][0],
                    "rail_y_m": lk["pose"][1], "rail_z_m": lk["pose"][2],
                    **{e: int(tk.during(e, t)) for e in EVENTS}}
            for o in tk.truth:
                out.append({**base, "object": o.id, "kind": o.kind, "label": o.label, "material": o.material,
                            "motion": o.motion, "present": int(o.active(t)), "in_view": int(in_view(o, lk["pose"], tk.cfg)),
                            "x_lo": o.lo[0], "x_hi": o.hi[0], "y_lo": o.lo[1], "y_hi": o.hi[1], "z_lo": o.lo[2],
                            "z_hi": o.hi[2]})
    return out


def write_csv(path: Path, rows_: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows_ for k in r)) or ["empty"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows_:
            wr.writerow({k: (round(v, 5) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


if __name__ == "__main__":
    takes = load_all()
    write_csv(paths.OUTPUT / "dataset.csv", rows(takes))
    print(f"{len(takes)} runs -> output/dataset.csv")
