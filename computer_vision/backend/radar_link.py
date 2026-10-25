"""
radar_link.py - the estimation pipeline (background, calibration, Infini-Rex) on finite takes and the clustering of
detections into stable objects.

    python -m backend.radar_link recordings/CV1/radar align_1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from backend.settings import ROOT, load_config


def cluster(dets_per_frame: list, tol_r: float, tol_a: float, min_rate: float) -> list[dict]:
    clusters = []
    for i, r, th in sorted(((i, r, th) for i, d in enumerate(dets_per_frame) for r, th in d), key=lambda x: x[1]):
        for c in clusters:
            cr, cth = np.median(np.array(c)[:, 1:], axis=0)
            if abs(r - cr) <= tol_r and abs(th - cth) <= tol_a:
                c.append((i, r, th))
                break
        else:
            clusters.append([(i, r, th)])
    out = []
    for c in clusters:
        rate = len({p[0] for p in c}) / max(len(dets_per_frame), 1)
        if rate >= min_rate:
            r, th = np.median(np.array(c)[:, 1:], axis=0)
            out.append({"range_m": round(float(r), 4), "azimuth_deg": round(float(th), 3), "rate": round(rate, 2)})
    return sorted(out, key=lambda o: o["azimuth_deg"])


class RadarTakes:
    def __init__(self, session: Path, rc: dict):
        sys.path.insert(0, str((ROOT / rc["estimation_dir"]).resolve()))
        sys.path.insert(0, str((ROOT / rc["estimation_dir"]).resolve().parent))
        from estimation.core.array import ArrayModel
        from estimation.core.interfaces import Settings
        from estimation.core.frame import background_from_recording, process_frame
        from estimation.methods import doa_methods
        from estimation.core.radar import RadarConfig, load_truth, open_recording, with_layout
        from estimation.core.metrics import Tolerance
        from estimation.core.steering import all_channel_steering, take_snapshot

        self.session, self.rc = Path(session), rc
        self._process, self._open = process_frame, open_recording
        self.cfg = cfg = with_layout(RadarConfig.from_cfg((ROOT / rc["cfg"]).resolve()),
                                     (ROOT / rc["estimation_dir"]).resolve() / "config" / "awr2944pevm_layout.json")
        self.array = ArrayModel(cfg)
        (self.method,) = doa_methods([rc["method"]])
        if rc.get("gates", "range") == "range_angle":
            from backend.radar_gates import install
            install(process_frame)
        self.settings = Settings(r_min_m=rc.get("r_min_m", 0.5), r_max_m=rc["r_max_m"], y_max_m=rc["y_max_m"])
        self.tol_r = rc["range_tol_cells"] * cfg.range_res_m
        self.tol_a = Tolerance.from_hardware(cfg, self.array).angle_deg
        self.takes = {p.stem: (p, load_truth(p)) for p in (sorted(self.session.iterdir()) if self.session.exists() else [])
                      if p.suffix in (".bin", ".npy") and p.with_suffix(".json").exists()}

        bgs = [p for p, d in self.takes.values() if d.get("scenario") == "background"]
        self.bg = None
        if bgs:
            parts = [background_from_recording(*open_recording(p, cfg), cfg) for p in bgs]
            self.bg = {k: np.mean([q[k] for q in parts], axis=0) for k in ("rfft", "adc")}
        cals = [(p, d) for p, d in self.takes.values() if d.get("scenario") == "calibration" and d.get("targets")]
        self.cal = None
        if cals:
            p, d = cals[0]
            t = d["targets"][0]
            y16, _ = take_snapshot(p, cfg, background=self.bg, range_m=t["range_m"])
            cal = y16 / all_channel_steering(cfg, t["azimuth_deg"], t.get("elevation_deg", 0.0))
            self.cal = (cal / cal[0]).astype(np.complex64)
            if self.bg is not None:
                self.bg = {"rfft": self.bg["rfft"] / self.cal[None, :], "adc": self.bg["adc"]}
        print(f"radar session {self.session.name}: background {[p.stem for p in bgs] or 'none'}, "
              f"calibration {cals[0][0].stem if cals else 'none'}, {self.method.name}, {rc['chirps']} chirps")

    def detections(self, take: str) -> tuple[list, dict, float]:
        if take not in self.takes:
            raise SystemExit(f"take {take} not in {self.session} (takes: {sorted(self.takes)})")
        path, doc = self.takes[take]
        n, get = self._open(path, self.cfg)
        n, K = min(n, self.rc["frames"]), self.rc["chirps"]
        dets, ms = [], []
        for i in range(n):
            t0 = time.perf_counter()
            dets.append(self._process(get(i)[:, :K], self.cfg, self.array, [self.method], self.settings, [i],
                                      self.cal, self.bg).detections[self.method.name])
            ms.append(1e3 * (time.perf_counter() - t0))
        return dets, doc, float(np.median(ms))

    def objects(self, take: str) -> tuple[list[dict], float]:
        dets, _, ms = self.detections(take)
        return cluster(dets, self.tol_r, self.tol_a, self.rc["min_rate"]), ms

    def points(self, take: str) -> list[dict]:
        dets, doc, ms = self.detections(take)
        out = []
        for t in doc.get("targets", []):
            found = []
            for d in dets:
                dr, da = np.abs(d[:, 0] - t["range_m"]), np.abs(d[:, 1] - t["azimuth_deg"])
                ok = (dr <= self.tol_r) & (da <= self.tol_a)
                if ok.any():
                    k = np.flatnonzero(ok)[np.argmin((dr[ok] / self.tol_r) ** 2 + (da[ok] / self.tol_a) ** 2)]
                    found.append(d[k])
            if not found:
                print(f"  {take} / {t['label']}: not detected in {len(dets)} frames - no point")
                continue
            f = np.array(found)
            r, th = np.median(f, axis=0)
            z = float(t["z_m"])
            y = r * np.sin(np.radians(th))
            out.append({"take": take, "label": t["label"], "range_m": round(float(r), 4),
                        "azimuth_deg": round(float(th), 3), "xyz_m": [round(float(np.sqrt(max(r**2 - y**2 - z**2, 0))), 4),
                                                                  round(float(y), 4), z],
                        "frames_detected": len(f), "frames": len(dets), "method": self.method.name,
                        "tape_xyz_m": [t["x_m"], t["y_m"], t["z_m"]]})
            print(f"  {take} / {t['label']}: R {r:.3f} m, theta {th:+.2f} deg (tape {t['range_m']:.3f} m, "
                  f"{t['azimuth_deg']:+.2f} deg), {len(f)}/{len(dets)} frames, {ms:.0f} ms/frame")
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("take")
    args = ap.parse_args()
    objs, ms = RadarTakes(args.session, load_config()["radar"]).objects(args.take)
    print(f"{len(objs)} objects, {ms:.1f} ms per radar frame ({1e3 / ms:.0f} Hz possible):")
    for o in objs:
        print(f"  R {o['range_m']:.3f} m   theta {o['azimuth_deg']:+6.2f} deg   detected in {100 * o['rate']:.0f} % of the frames")


if __name__ == "__main__":
    main()


def live_log(take_folder: Path, rc: dict, cfg_period_s: float = 0.1):
    import json
    rows = [json.loads(line) for line in open(Path(take_folder) / "radar_live.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("type") == "frame"]
    dets = [np.array(r["dets"], float).reshape(-1, 2) for r in rows]
    tol_r = rc["range_tol_cells"] * 0.0836
    tol_a = 4.78
    return cluster(dets, tol_r, tol_a, rc["min_rate"]), dets, [r["t"] for r in rows]
