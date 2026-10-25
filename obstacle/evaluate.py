"""
evaluate.py - every metric of the method on every recorded run and of its baselines on the same recordings: tracking
(HOTA, CLEAR, IDF1 per look and in real time), scene understanding, separation, detection, regions, obstacle boxes,
motion, safety, light and reading, classification, cognition and latency -> output/predictions.csv, detections.csv,
decisions.csv, dataset.csv, tables.tex and output/cache/results.pkl.

    python evaluate.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import dataset  # noqa: E402
import doa_study  # noqa: E402
import light_replay  # noqa: E402
import paths  # noqa: E402
import scene_metrics as sm  # noqa: E402
import tables  # noqa: E402

CACHE = paths.OUTPUT / "cache" / "results.pkl"


def predictions(takes) -> list[dict]:
    out = []
    for tk in takes:
        for lk in tk.looks:
            truth, objs, S, pairs = sm.match_look(tk, lk)
            of = {j: (truth[i].id, s) for i, j, s in pairs}
            state = next((d["state"] for d in tk.decisions if d.get("frame") == lk.get("frame")), "")
            for o in lk["objects"]:
                j = next((k for k, q in enumerate(objs) if q is o), None)
                m = of.get(j, ("", np.nan))
                out.append({"take": tk.name, "t_s": lk["t"] - tk.t0, "chirp": lk["chirp"], "state": state, "object": o["id"],
                            "kind": o["kind"], "known": int(bool(o.get("known"))), "person": int(bool(o.get("person"))),
                            "material": o.get("material") or "", "motion": o.get("motion") or "",
                            "x_lo": o["lo"][0], "x_hi": o["hi"][0], "y_lo": o["lo"][1], "y_hi": o["hi"][1],
                            "z_lo": o["lo"][2], "z_hi": o["hi"][2], "speed_mps": o.get("speed", 0.0),
                            "truth": m[0], "similarity": m[1]})
    return out


def decision_rows(takes) -> list[dict]:
    out = []
    for tk in takes:
        for d in tk.decisions:
            out.append({"take": tk.name, "t_s": d["t"] - tk.t0, "state": d["state"], "command": d["kind"],
                        "reason": d["reason"], "x_m": d["pose"][0], "y_m": d["pose"][1], "z_m": d["pose"][2],
                        "target_x_m": (d.get("target") or [np.nan])[0], "x_free_m": d.get("x_free"),
                        "response_s": d.get("t_resp"), "cause": d.get("cause", ""), "chirps": "+".join(d.get("looks") or []),
                        **{e: int(tk.during(e, d["t"])) for e in dataset.EVENTS}})
    return out


def per_run(tk) -> dict:
    u = sm.understanding([tk])
    return {"name": tk.name, "test": tk.test, "duration_s": tk.looks[-1]["t"] - tk.t0, "looks": len(tk.looks),
            "camera_frames": len(tk.camera), "objects": [t.kind for t in tk.truth],
            "person_entries": len(tk.events.get("person", [])),
            "light_events": len(tk.events.get("light", [])) + len(tk.events.get("flashing", [])),
            "travel_m": float(np.sum(np.linalg.norm(np.diff(tk.rail[:, 1:4], axis=0), axis=1))) if len(tk.rail) > 1 else 0.0,
            "PQ": u["PQ"], "PQ_full": u["PQ_full"], "RQ": u["RQ"], "SQ": u["SQ"],
            "fdr": sm.detection([tk])["fdr"], "series": u["series"]}


def group(takes) -> dict:
    res = {"runs": [per_run(tk) for tk in takes], "understanding": sm.understanding(takes),
           "separation": sm.separation(takes), "detection": sm.detection(takes), "regions": sm.regions(takes),
           "boxes": sm.obstacle_boxes(takes), "decisions": sm.decisions(takes), "waits": sm.waits(takes),
           "light": sm.light(takes), "classification": sm.classification(takes), "cognition": sm.cognition(takes),
           "latency": sm.latency(takes), "tracking": sm.tracking(takes), "motion": sm.motion(takes),
           "camera": sm.camera(takes)}
    res["tracks"] = {tk.name: [(lk["t"] - tk.t0, lk["pose"], lk["objects"], lk["tracks"],
                                [d["world"][:2] for d in lk["dets"]]) for lk in tk.looks] for tk in takes}
    res["truth"] = {tk.name: tk.truth for tk in takes}
    res["events"] = {tk.name: {k: [[a - tk.t0, b - tk.t0] for a, b in v] for k, v in tk.events.items()} for tk in takes}
    res["states"] = {tk.name: [(d["t"] - tk.t0, d["state"], d["pose"][0]) for d in tk.decisions] for tk in takes}
    return res


def run() -> dict:
    takes = dataset.load_all()
    if not takes:
        raise SystemExit("no evaluated run: add takes with a measured scene.json to dataset.TAKES")
    takes = [tk for tk in takes if tk.name in dataset.REPLAYS]
    method = lambda tk: tk.summary.get("method", "final")
    sets = {k: [tk for tk in takes if method(tk) == k] for k in ("final", "ctrex", "cfar", "baseline")}
    res = {k: group(v) for k, v in sets.items() if v}
    out = paths.OUTPUT
    dataset.write_csv(out / "dataset.csv", dataset.rows(takes))
    dataset.write_csv(out / "predictions.csv", predictions(takes))
    dataset.write_csv(out / "detections.csv", [r for k in ("final", "ctrex", "cfar", "baseline") if k in res
                                               for r in res[k]["detection"].pop("rows")])
    dataset.write_csv(out / "decisions.csv", decision_rows(takes))
    for v in res.values():
        v["detection"].pop("rows", None)
    res["doa"] = doa_study.run()
    res["light_videos"] = light_replay.score(light_replay.run())
    (out / "tables.tex").write_text(tables.render_all(res), encoding="utf-8")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(res, fh)
    return res


if __name__ == "__main__":
    r = run()
    for k, v in r.items():
        if k in ("doa", "light_videos"):
            continue
        u = v["understanding"]
        print(f"{k}: {len(v['runs'])} runs, Scene PQ {u['PQ']:.3f}, FDR {v['detection']['fdr']:.3f}")
    print("-> output/tables.tex, predictions.csv, detections.csv, decisions.csv, dataset.csv")
