"""
evaluate.py - the method replayed on the recorded radar data of every take and checked against the camera: tracks
matched to the ball and the person per look and on a 10 Hz live grid (HOTA, CLEAR MOT, IDF1), state estimation,
prediction regions, detections (FDR), classification, information and run time, as predictions.csv,
detections.csv and tables.tex.

    python evaluate.py           (python evaluate.py --again replays every take with the current method)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

import dataset  # noqa: E402
import metrics as mt  # noqa: E402
from hardware.reader_process import correct_chars  # noqa: E402
import paths  # noqa: E402
from hardware.camera_view import region_rect  # noqa: E402
from validation import alignment as va  # noqa: E402
from validation.scene_truth import SCALE, motion  # noqa: E402

OUT = paths.OUTPUT
PREDICTIONS, DETECTIONS, TABLES = OUT / "predictions.csv", OUT / "detections.csv", OUT / "tables.tex"
GRID_HZ = 10.0
WINDOW_LOOKS = 16
FRAME_S = 1 / 30
RANGE_GATE_M = 0.3
GOSPA_C_DEG = 5.0
COLUMN_MARGIN_PX = 25.0
STAGES = ("predict", "range_doppler", "infinirex_fdr", "track", "material", "cognitive")
HEADS = ("material", "motion")
DEPS = [HERE / "methods", HERE / "hardware" / "recording.py", HERE / "replay.py", HERE / "config.toml",
        paths.ESTIMATION / "methods" / "infinirex.py", paths.MATERIAL / "classifier.py", paths.MATERIAL / "features.py",
        paths.MATERIAL / "utils" / "micro_doppler.py",
        paths.MATERIAL / "models" / "material_motion_model.json"]


class Aligned:
    def __init__(self, proj, al):
        self.p, self.al, self.f, self.loo_rms_px, self.size_wh = proj, al, proj.f, 0.0, proj.size_wh

    def point(self, r, th, z):
        return self.p.point(r, th + self.al["dtheta_deg"], z)


def newest(path: Path) -> float:
    return max((q.stat().st_mtime for q in path.rglob("*.py")), default=0.0) if path.is_dir() else path.stat().st_mtime


def stale(log: Path, again: bool, extra=()) -> bool:
    deps = [newest(d) for d in DEPS] + [q.stat().st_mtime for q in extra]
    return again or not log.exists() or log.stat().st_mtime < max(deps)


def run_replay(folder: Path, out: str, args: list):
    print(f"  replaying {out} ...", flush=True)
    subprocess.run([sys.executable, str(HERE / "replay.py"), str(folder), "--quiet", "--out", out] + args, check=True,
                   cwd=HERE)


def costs(take: str, again: bool) -> Path:
    folder = dataset.folder(take)
    name = f"{folder.parent.name}_{folder.name}_calibration"
    log, path = paths.RESULTS / "replay" / name / "frames.jsonl", paths.RESULTS / "replay" / name / "costs.json"
    if stale(log, again) or not path.exists():
        run_replay(folder, name, ["--policy", "all"])
        per = {}
        for line in open(log, encoding="utf-8"):
            r = json.loads(line)
            per.setdefault(r["waveform"], []).append(r["ms"]["total"] / 1e3)
        path.write_text(json.dumps({k: float(np.median(v)) for k, v in per.items()}, indent=1))
    return path


def replay(take: str, again: bool, done: set):
    log = paths.RESULTS / "replay" / take / "frames.jsonl"
    if dataset.policy(take) is None:
        if stale(log, again):
            run_replay(dataset.folder(take), take, [])
        return
    cost = costs(take, again and str(dataset.folder(take)) not in done)
    done.add(str(dataset.folder(take)))
    if stale(log, again, [cost]):
        run_replay(dataset.folder(take), take, ["--policy", dataset.policy(take), "--costs", str(cost)])


def roi_area() -> float:
    rc = paths.load_config()["radar"]
    x, y = np.meshgrid(np.linspace(0, rc["r_max_m"], 501), np.linspace(rc["y_min_m"], rc["y_max_m"], 501))
    r = np.hypot(x, y)
    inside = (r >= rc["r_min_m"]) & (r <= rc["r_max_m"]) & (np.abs(np.degrees(np.arctan2(y, x))) <= 60)
    return float(inside.mean() * rc["r_max_m"] * (rc["y_max_m"] - rc["y_min_m"]))


def nearest_found(truth, j, name, reach=30):
    for d in range(reach + 1):
        for q in (j - d, j + d):
            if 0 <= q < len(truth) and truth[q][name]["found"]:
                return truth[q][name]
    return None


def scene_at(truth, j, wb):
    gts, ignore = [], []
    b = truth[j]["ball"]
    if b["found"]:
        gts.append(("ball", b["u"], wb))
    else:
        q = nearest_found(truth, j, "ball")
        if q is not None:
            ignore.append((q["u"] - wb, q["u"] + wb))
    p = truth[j]["person"]
    if p["found"] and not p["partial"]:
        gts.append(("person", p["u"], p["u1"] - p["u0"]))
    else:
        q = p if p["found"] else nearest_found(truth, j, "person")
        if q is not None:
            ignore.append((q["u0"], q["u1"]))
    return gts, ignore


def similarity(u, r, gt) -> float:
    name, ug, w = gt
    if name == "ball" and abs(r - dataset.BALL_RANGE_M) > RANGE_GATE_M:
        return 0.0
    if name == "person" and r < dataset.BALL_RANGE_M - RANGE_GATE_M:
        return 0.0
    return max(0.0, 1.0 - abs(u - ug) / w)


def match(tracks, gts, ignore, proj, al, W):
    seen = []
    for tr in tracks:
        p = va.pixel(proj, tr["range_m"], tr["theta_deg"], al)
        if p is not None and 0 <= p[0] < W:
            seen.append((tr, float(p[0])))
    S = np.array([[similarity(u, tr["range_m"], g) for tr, u in seen] for g in gts]).reshape(len(gts), len(seen))
    keep = [i for i, (tr, u) in enumerate(seen)
            if (S[:, i].max() if len(gts) else 0) >= 0.5 or not any(a <= u <= b for a, b in ignore)]
    seen, S = [seen[i] for i in keep], S[:, keep]
    who = {}
    if S.size:
        r, c = linear_sum_assignment(-S)
        who = {seen[ci][0]["id"]: (gts[ri][0], S[ri, ci]) for ri, ci in zip(r, c) if S[ri, ci] >= 0.5}
    return seen, S, who


def lateral_sigma_deg(tr) -> float:
    th = np.radians(tr["theta_deg"])
    e = np.array([-np.sin(th), np.cos(th)])
    return float(np.degrees(np.sqrt(max(e @ np.array(tr["P_pos"]) @ e, 0.0)) / max(tr["range_m"], 0.1)))


def iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def iou_1d(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(max(a1, b1) - min(a0, b0), 1e-9)


def region_score(det, region) -> float:
    lo, hi = np.array(region["lo"], float), np.array(region["hi"], float)
    z = np.array([det["range_m"], np.radians(det["theta_deg"]), det["v_mps"]])
    rel = (z - lo) / np.maximum(hi - lo, 1e-9)
    return float(np.max(np.abs(2 * rel - 1)))


class Votes:
    def __init__(self):
        self.w = {h: {} for h in HEADS}

    def add(self, labels, confs):
        for h in HEADS:
            self.w[h][labels[h]] = self.w[h].get(labels[h], 0.0) + 1.0

    def bits(self, h, k):
        v = np.array(list(self.w[h].values()))
        if not len(v):
            return 0.0
        q = v / v.sum()
        return float(np.log2(k) + np.sum(q * np.log2(q)))


def classifier():
    from material_classification.classifier import MaterialMotionModel
    return MaterialMotionModel.load(paths.MATERIAL / "models" / "material_motion_model.json")


def head_scores(model, h, feats):
    clf = getattr(model, h)
    X = np.array([[f.get(n, np.nan) for n in model.features[h]] for f in feats], float)
    labels, _, res = clf.predict(X)
    A = clf.code(clf._scale(X).T)
    share = np.stack([A[clf.labels == c].sum(axis=0) for c in clf.classes], axis=1)
    return list(clf.classes), labels, share / np.maximum(share.sum(axis=1, keepdims=True), 1e-12), -res


def mot_frame(t, gts, seen, S, who, proj, al) -> dict:
    D = np.full((len(gts), len(seen)), np.inf)
    for i, g in enumerate(gts):
        for j, (tr, _) in enumerate(seen):
            if similarity(g[1], tr["range_m"], (g[0], g[1], 1e9)) > 0:
                D[i, j] = abs(tr["theta_deg"] - va.azimuth_of(proj, tr["range_m"], g[1], al))
    gs = mt.gospa(D, GOSPA_C_DEG)
    gs["assign"] = {gts[i][0]: seen[j][0]["id"] for i, j in gs.pop("pairs")}
    return {"t": t, "gt": [g[0] for g in gts], "tr": [tr["id"] for tr, _ in seen], "sim": S,
            "hit": {o: i for i, (o, _) in who.items()}, "gospa": gs}


def scores(frames: list[dict]) -> dict:
    out = {"hota": mt.hota(frames), "clear": mt.clear(frames), "idf1": mt.idf1(frames),
           "segments": [{"HOTA": mt.hota(s)["HOTA"], "MOTA": mt.clear(s)["MOTA"], "IDF1": mt.idf1(s)["IDF1"],
                         "MOTP": mt.clear(s)["MOTP"], "IDSW": mt.clear(s)["IDSW"],
                         "GOSPA": float(np.mean([f["gospa"]["gospa"] for f in s]))}
                        for s in mt.segments(frames)], "objects": {}}
    g = [f["gospa"] for f in frames]
    pairs = sum(len(x["assign"]) for x in g)
    out["gospa"] = {"gospa": float(np.mean([x["gospa"] for x in g])), "ospa": float(np.mean([x["ospa"] for x in g])),
                    "loc_rms": float(np.sqrt(sum(x["loc"] for x in g) / max(pairs, 1))),
                    "missed": float(np.mean([x["n_missed"] for x in g])), "false": float(np.mean([x["n_false"] for x in g])),
                    "switches": mt.switches(frames),
                    "t_gospa": float(np.sqrt((sum(x["loc"] + x["missed"] + x["false"] for x in g)
                                              + GOSPA_C_DEG ** 2 * mt.switches(frames)) / len(g)))}
    for o in dataset.CLASSES:
        ids = [f["hit"].get(o) for f in frames if o in f["gt"]]
        got = [i for i in ids if i is not None]
        counts = {i: got.count(i) for i in set(got)}
        out["objects"][o] = {"frames": len(ids), "tracked": len(got) / max(len(ids), 1), "ids": len(counts),
                             "longest": max(counts.values()) / max(len(ids), 1) if counts else 0.0,
                             "ci": mt.wilson(len(got), len(ids))}
    return out


def reference(take: str) -> str:
    same = [k for k in dataset.SOURCE if dataset.folder(k) == dataset.folder(take)]
    return next((k for k in same if dataset.policy(k) == "search"), take)


def calibration(take: str, truth: list[dict], proj, cfg) -> list[dict]:
    gt_rows = [{"t": f["t"], **f["ball"]} for f in truth]
    tk = va.load(reference(take), gt_rows, dataset.folder(take))
    return va.fit_two_fold(tk, proj, cfg["ball"]["centre_z_m"])


def evaluate_take(take: str, truth: list[dict], proj, cfg, area: float, cal: list[dict]) -> dict:
    gt_rows = [{"t": f["t"], **f["ball"]} for f in truth]
    tk = va.load(take, gt_rows, dataset.folder(take))
    W = proj.size_wh[0]
    tg = np.array([f["t"] for f in truth])
    wb = 2 * float(np.nanmedian([f["ball"]["r_px"] for f in truth if f["ball"]["found"]]))
    cols, _ = motion(dataset.folder(take), [f["ball"] for f in truth], dataset.CACHE)
    r_ball = cfg["ball"]["diameter_m"] / 2
    t0 = tk.t_arr[0]
    preds, dets, windows, looks, regions_seq, frames = [], [], [], [], {}, {"look": [], "live": []}
    seen_windows, votes = {}, {}
    for k, rec in enumerate(tk.recs):
        al = va.alignment_at(cal, tk.t_arr[k])
        tm = tk.t_arr[k] - al["d_s"]
        j = int(np.argmin(np.abs(tg - tm)))
        has_truth = abs(tg[j] - tm) <= FRAME_S
        conf = [t for t in rec["tracks"] if t["status"] == "confirmed"]
        who = {}
        if has_truth:
            gts, ignore = scene_at(truth, j, wb)
            seen, S, who = match(conf, gts, ignore, proj, al, W)
            frames["look"].append(mot_frame(tk.t_arr[k] - t0, gts, seen, S, who, proj, al))
        look = {"k": k, "t": tk.t_arr[k] - t0, "waveform": rec["waveform"], "lambda": rec["lambda"],
                "alpha_t": rec.get("alpha_t", {}), "coverage": rec.get("coverage", {}), "ms": rec["ms"], "info_bits": rec["info_bits"],
                "n_dets": len(rec["dets"]), "truth": has_truth, "objects": {}}
        for tr in rec["tracks"]:
            c = tr.get("classes") or {}
            if tr["status"] == "confirmed" and "material" in c and seen_windows.get(tr["id"]) != c["material"]["classified"]:
                seen_windows[tr["id"]] = c["material"]["classified"]
                votes.setdefault(tr["id"], Votes()).add({h: c[h]["last"] for h in HEADS}, {h: c[h]["last_conf"] for h in HEADS})
                if tr.get("window"):
                    windows.append({"take": take, "t": look["t"], "look": k, "track": tr["id"], "n": c["material"]["classified"],
                                    "truth": who.get(tr["id"], (None, 0))[0] if tr["status"] == "confirmed" else None,
                                    "features": tr["window"],
                                    **{f"{h}_window": c[h]["last"] for h in HEADS}, **{f"{h}_vote": c[h]["label"] for h in HEADS}})
        for tr in conf:
            c = tr.get("classes") or {}
            obj, sim = who.get(tr["id"], ("", np.nan))
            row = {"take": take, "look": k, "t_s": round(look["t"], 4), "track": tr["id"],
                   "range_m": round(tr["range_m"], 4), "azimuth_deg": round(tr["theta_deg"], 3),
                   "vx_mps": round(tr["vx"], 4), "vy_mps": round(tr["vy"], 4),
                   "P_xx": tr["P_pos"][0][0], "P_xy": tr["P_pos"][0][1], "P_yy": tr["P_pos"][1][1],
                   "entropy_bits": round(tr["entropy_bits"], 4), "sigma_azimuth_deg": round(lateral_sigma_deg(tr), 4),
                   "region_r_lo": tr["region"]["lo"][0], "region_r_hi": tr["region"]["hi"][0],
                   "region_az_lo_deg": np.degrees(tr["region"]["lo"][1]), "region_az_hi_deg": np.degrees(tr["region"]["hi"][1]),
                   "region_calibrated": int(tr["region"]["calibrated"]), "updated": int(tr["updated"]),
                   "material": (c.get("material") or {}).get("label") or "", "material_conf": (c.get("material") or {}).get("confidence", ""),
                   "motion": (c.get("motion") or {}).get("label") or "", "motion_conf": (c.get("motion") or {}).get("confidence", ""),
                   "truth": obj, "similarity": round(sim, 4) if obj else "", "azimuth_truth_deg": "", "azimuth_error_deg": "",
                   "z_azimuth": "", "region_holds_truth": "", "iou": ""}
            if obj:
                g = truth[j][obj]
                th_gt = va.azimuth_of(proj, tr["range_m"], g["u"], al)
                err = tr["theta_deg"] - th_gt
                row.update(azimuth_truth_deg=round(th_gt, 3), azimuth_error_deg=round(err, 3),
                           z_azimuth=round(err / max(lateral_sigma_deg(tr), 1e-6), 3))
                lo, hi = tr["region"]["lo"], tr["region"]["hi"]
                if tr["region"]["calibrated"] and np.all(np.isfinite(lo[:2])):
                    row["region_holds_truth"] = int(np.degrees(lo[1]) <= th_gt <= np.degrees(hi[1]))
                    ap = Aligned(proj, al)
                    if obj == "ball" and g["u1"] - g["u0"] >= 0.8 * wb:
                        rr = region_rect(ap, tr, al["z_ball_m"] - r_ball, al["z_ball_m"] + r_ball)
                        row["iou"] = round(iou(rr, (g["u0"], g["v0"], g["u1"], g["v1"])), 4) if rr else ""
                    elif obj == "person":
                        rr = region_rect(ap, tr, 0.0, 0.0)
                        row["iou"] = round(iou_1d(rr[0], rr[2], g["u0"], g["u1"]), 4) if rr else ""
                v = votes.get(tr["id"])
                look["objects"][obj] = {"track": tr["id"], "entropy_bits": tr["entropy_bits"],
                                        "position_bits": float(np.log2(area) - tr["entropy_bits"]),
                                        "class_bits": sum(v.bits(h, n) for h, n in (("material", 3), ("motion", 4))) if v else 0.0}
            preds.append(row)
        for d in rec["dets"]:
            c = None
            p = va.pixel(proj, d["range_m"], d["theta_deg"], al)
            if p is not None and 0 <= p[0] < W:
                c = int(p[0] // SCALE)
            row = {"take": take, "look": k, "t_s": round(look["t"], 4), "range_m": round(d["range_m"], 4),
                   "azimuth_deg": round(d["theta_deg"], 3), "elevation_deg": round(d.get("elev_deg", np.nan), 3),
                   "v_mps": round(d["v_mps"], 4), "snr_db": round(d["snr_db"], 2), "e_value": d["e"], "weight": d["w"],
                   "role": d.get("label", ""), "track": d.get("track", ""), "inside_region": d.get("inside_region", ""),
                   "in_view": int(c is not None and has_truth), "moving_there": "", "on_ball": "", "on_person": ""}
            if c is not None and has_truth:
                m = int(np.ceil(COLUMN_MARGIN_PX / SCALE))
                occ = cols[j] >= 3
                b, q = truth[j]["ball"], truth[j]["person"]
                if b["found"]:
                    occ[max(int(b["u0"] // SCALE), 0):max(int(b["u1"] // SCALE) + 1, 0)] = True
                row["moving_there"] = int(occ[max(c - m, 0):c + m + 1].any())
                if b["found"]:
                    row["on_ball"] = int(b["u0"] <= p[0] <= b["u1"] and abs(d["range_m"] - dataset.BALL_RANGE_M) <= RANGE_GATE_M)
                if q["found"] and not q["partial"]:
                    row["on_person"] = int(q["u0"] <= p[0] <= q["u1"] and d["range_m"] >= dataset.BALL_RANGE_M + RANGE_GATE_M)
            dets.append(row)
            if d.get("label") == "core" and d.get("track") is not None:
                tr = next((t for t in rec["tracks"] if t["id"] == d["track"]), None)
                if tr is not None and tr["region"]["calibrated"] and np.all(np.isfinite(tr["region"]["lo"])):
                    regions_seq.setdefault(d["track"], []).append(
                        (region_score(d, tr["region"]), int(not d.get("inside_region", True))))
        looks.append(look)
    live_err = []
    for tc in np.arange(tk.t_ready[0], tg[-1], 1 / GRID_HZ):
        j = int(np.argmin(np.abs(tg - tc)))
        if abs(tg[j] - tc) > FRAME_S:
            continue
        al = va.alignment_at(cal, tc)
        k, tracks = va.tracks_at(tk, tc, al["d_s"], live=True)
        gts, ignore = scene_at(truth, j, wb)
        seen, S, who = match(tracks, gts, ignore, proj, al, W)
        frames["live"].append(dict(mot_frame(tc - t0, gts, seen, S, who, proj, al), blind=bool(k < 0 or tc - tk.t_ready[k] > 0.5)))
        for tr, u in seen:
            if tr["id"] in who:
                o = who[tr["id"]][0]
                live_err.append((o, tr["theta_deg"] - va.azimuth_of(proj, tr["range_m"], truth[j][o]["u"], al)))
    mot = {view: scores(f) for view, f in frames.items()}
    on = {}
    for p in preds:
        on.setdefault(p["track"], {})[p["look"]] = p["truth"]
    for w in windows:
        span = [on[w["track"]].get(q, "") for q in range(w["look"] - WINDOW_LOOKS + 1, w["look"] + 1)]
        named = [x for x in span if x]
        top = max(set(named), key=named.count) if named else None
        w["truth"] = top if top and named.count(top) >= WINDOW_LOOKS / 2 else None
    truth_az = {"ball": [], "person": []}
    r_person = float(np.median([p["range_m"] for p in preds if p["truth"] == "person"] or [2.3]))
    for f in truth[::3]:
        al = va.alignment_at(cal, f["t"])
        if f["ball"]["found"]:
            truth_az["ball"].append((f["t"] - t0, va.azimuth_of(proj, dataset.BALL_RANGE_M, f["ball"]["u"], al)))
        q = f["person"]
        if q["found"] and not q["partial"]:
            truth_az["person"].append((f["t"] - t0, va.azimuth_of(proj, r_person, q["u0"], al),
                                       va.azimuth_of(proj, r_person, q["u1"], al), r_person))
    return {"take": take, "label": dataset.LABELS[take], "duration_s": float(tk.t_arr[-1] - t0), "looks": looks,
            "preds": preds, "dets": dets, "windows": windows, "frames": frames, "mot": mot,
            "live_err": live_err, "truth_az": truth_az, "regions_seq": regions_seq, "calibration": cal, "camera_frames": len(truth),
            "truth": truth, "t0": t0, "wb": wb, "recs": tk.recs, "t_arr": tk.t_arr - t0}


def classification(res: dict) -> dict:
    model = classifier()
    out = {}
    wins = [w for r in res.values() for w in r["windows"]]
    for h in HEADS:
        known = [w for w in wins if w["truth"]]
        true = np.array([dataset.CLASSES[w["truth"]][h] for w in known], object)
        pred = np.array([w[f"{h}_window"] for w in known], object)
        classes, _, share, score = head_scores(model, h, [w["features"] for w in known])
        rows = mt.per_class(true, pred, score, classes)
        vote = np.array([w[f"{h}_vote"] or "undecided" for w in known], object)
        out[h] = {"true": true, "pred": pred, "vote": vote, "n": np.array([w["n"] for w in known]),
                  "take": np.array([w["take"] for w in known]), "classes": classes, "per_class": rows,
                  "macro_f1": float(np.mean([r["f1"] for r in rows])) if rows else np.nan,
                  "accuracy": float(np.mean(true == pred)) if len(true) else np.nan,
                  "vote_accuracy": float(np.mean(true == vote)) if len(true) else np.nan,
                  "cross_entropy_bits": mt.cross_entropy_bits(share, true, classes) if len(true) else np.nan,
                  "chance_bits": float(np.log2(len(classes))), "unmatched": len(wins) - len(known)}
    return out


def detection(r: dict) -> dict:
    seen = [d for d in r["dets"] if d["in_view"] == 1]
    fdp = {lk["k"]: [] for lk in r["looks"] if lk["truth"]}
    for d in seen:
        fdp.setdefault(d["look"], []).append(1 - d["moving_there"])
    per_look = np.array([np.mean(v) if v else 0.0 for _, v in sorted(fdp.items())])
    ball = {}
    person = {}
    for d in seen:
        if d["on_ball"] != "":
            ball[d["look"]] = max(ball.get(d["look"], 0), d["on_ball"])
        if d["on_person"] != "":
            person[d["look"]] = max(person.get(d["look"], 0), d["on_person"])
    return {"fdr": float(per_look.mean()) if len(per_look) else np.nan, "fdr_ci": mt.block_bootstrap(per_look),
            "fdp": per_look, "fdp_t": np.array(sorted(fdp)), "precision": 1 - float(per_look.mean()) if len(per_look) else np.nan,
            "recall_ball": float(np.mean(list(ball.values()))) if ball else np.nan, "n_ball": len(ball),
            "recall_person": float(np.mean(list(person.values()))) if person else np.nan, "n_person": len(person),
            "dets_per_look": len(r["dets"]) / max(len(r["looks"]), 1)}


def estimation(r: dict) -> dict:
    out = {}
    for o in dataset.CLASSES:
        rows = [p for p in r["preds"] if p["truth"] == o and p["azimuth_error_deg"] != ""]
        e = np.array([p["azimuth_error_deg"] for p in rows], float)
        z = np.array([p["z_azimuth"] for p in rows], float)
        s = np.array([p["sigma_azimuth_deg"] for p in rows], float)
        live = np.array([v for oo, v in r["live_err"] if oo == o], float)
        cov = [p["region_holds_truth"] for p in rows if p["region_holds_truth"] != ""]
        ious = np.array([p["iou"] for p in rows if p["iou"] != ""], float)
        out[o] = {"err": e, "z": z, "sigma": s, "live": live, "abs_median": float(np.median(np.abs(e))) if len(e) else np.nan,
                  "abs_q90": float(np.quantile(np.abs(e), 0.9)) if len(e) else np.nan,
                  "live_abs_median": float(np.median(np.abs(live))) if len(live) else np.nan,
                  "bias": float(np.median(e)) if len(e) else np.nan,
                  "z_rms": float(np.sqrt(np.mean(z[np.isfinite(z)] ** 2))) if len(z) else np.nan,
                  "coverage_camera": float(np.mean(cov)) if cov else np.nan, "coverage_n": len(cov),
                  "coverage_ci": mt.wilson(sum(cov), len(cov)), "iou": ious}
        if o == "ball":
            rng = np.array([p["range_m"] for p in rows], float)
            out[o]["range"] = rng
            out[o]["range_bias"] = float(np.median(rng) - dataset.BALL_RANGE_M) if len(rng) else np.nan
            out[o]["range_std"] = float(np.std(rng)) if len(rng) else np.nan
    return out


def regions(r: dict) -> dict:
    seqs = [np.array([s for s, _ in v]) for v in r["regions_seq"].values()]
    miss = [np.array([m for _, m in v]) for v in r["regions_seq"].values()]
    allm = np.concatenate(miss) if miss else np.array([])
    ex = mt.exchangeability(seqs)
    runs = [mt.runs_test(m) for m in miss if len(m) > 20]
    final = r["recs"][-1]["coverage"]
    return {"coverage_radar": final, "misses": allm, "exchange": ex,
            "runs_p_median": float(np.nanmedian(runs)) if runs else np.nan, "runs_n": len(runs),
            "miss_sequences": miss}


def reading_scores(take: str) -> dict:
    from reading import read_take
    reads, stable, last = read_take(take), [], []
    for r in reads:
        if r["text"]:
            last = (last + [r["text"]])[-15:]
        stable.append((r["t_s"], correct_chars(max(set(last), key=last.count), dataset.TEXT) if last else 0))
    share = np.array([r["correct"] for r in reads], float) / len(dataset.TEXT)
    margin = np.array([r["min_margin"] for r in reads], float)
    ok = np.isfinite(margin)
    order = np.argsort(-margin[ok], kind="stable")
    accuracy = np.cumsum(share[ok][order]) / np.arange(1, ok.sum() + 1) if ok.any() else np.zeros(0)
    good = share[ok] >= 3 / len(dataset.TEXT)
    auc = mt.roc(margin[ok], good)[2] if good.any() and (~good).any() else np.nan
    return {"reads": reads, "stable": stable, "share": share, "letters": len(dataset.TEXT),
            "mean_share": float(share.mean()) if len(share) else np.nan,
            "three": float(np.mean(share >= 3 / len(dataset.TEXT))) if len(share) else np.nan,
            "final": stable[-1][1] if stable else 0, "auc": auc,
            "coverage": np.arange(1, ok.sum() + 1) / max(ok.sum(), 1), "accuracy": accuracy}


def timing(r: dict) -> dict:
    search = r["looks"][0]["waveform"]
    ms = {s: np.array([lk["ms"].get(s, np.nan) for lk in r["looks"] if s != "material" or lk["waveform"] == search],
                      float) for s in STAGES + ("total",)}
    lat = np.median([c["al"]["d_s"] for c in r["calibration"]]) * 1e3
    return {"ms": ms, "radar_latency_ms": float(lat)}


def write_csv(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def run(again: bool = False) -> dict:
    from backend.alignment import Projector
    cfg, area = paths.load_config(), roi_area()
    truth = dataset.run()
    res, pol, cals = {}, {}, {}
    takes = list(dict.fromkeys([*dataset.TAKES, *dataset.POLICIES]))
    done = set()
    for take in takes:
        replay(take, again, done)
    for take in takes:
        print(f"  evaluating {take} ...", flush=True)
        proj = Projector(dataset.alignment(take))
        key = str(dataset.folder(take))
        if key not in cals:
            cals[key] = calibration(take, truth[take], proj, cfg)
        r = evaluate_take(take, truth[take], proj, cfg, area, cals[key])
        r.update(detection=detection(r), timing=timing(r), reading=reading_scores(take))
        if take in dataset.TAKES:
            r.update(estimation=estimation(r), regions=regions(r))
            res[take] = r
        if take in dataset.POLICIES:
            pol[take] = r
    out = {"takes": res, "policies": pol, "classification": classification(res), "area_m2": area}
    import pickle
    (paths.RESULTS / "results.pkl").write_bytes(pickle.dumps(out))
    write_csv(PREDICTIONS, [p for r in res.values() for p in r["preds"]])
    write_csv(DETECTIONS, [d for r in res.values() for d in r["dets"]])
    write_tables(out)
    return out


def write_tables(res: dict):
    import tables
    TABLES.write_text(tables.render(res), encoding="utf-8")


if __name__ == "__main__":
    out = run(again="--again" in sys.argv)
    for take, r in out["takes"].items():
        for view, m in r["mot"].items():
            h, c, i = m["hota"], m["clear"], m["idf1"]
            print(f"{take} {view}: HOTA {h['HOTA']:.3f} (DetA {h['DetA']:.3f}, AssA {h['AssA']:.3f}, LocA {h['LocA']:.3f}), "
                  f"MOTA {c['MOTA']:.3f}, IDF1 {i['IDF1']:.3f}, IDSW {c['IDSW']}, FP {c['FP']}, FN {c['FN']}; "
                  f"objects {[(o, round(v['tracked'], 3), v['ids'], round(v['longest'], 3)) for o, v in m['objects'].items()]}")
        print(f"   FDR {r['detection']['fdr']:.3f} {r['detection']['fdr_ci']}, recall ball {r['detection']['recall_ball']:.3f} "
              f"person {r['detection']['recall_person']:.3f}")
        for o, e in r["estimation"].items():
            print(f"   {o}: az err median {e['abs_median']:.2f} q90 {e['abs_q90']:.2f} live {e['live_abs_median']:.2f}, z_rms "
                  f"{e['z_rms']:.2f}, camera coverage {e['coverage_camera']:.3f} (n {e['coverage_n']}), IoU median "
                  f"{np.median(e['iou']) if len(e['iou']) else float('nan'):.3f}" + (f", range bias {e['range_bias']:.3f} std {e['range_std']:.3f}" if o == "ball" else ""))
        g = r["regions"]
        print(f"   radar coverage {g['coverage_radar']}, KS p {g['exchange']['ks_p']:.3f}, lag1 {g['exchange']['lag1']:.3f}, runs p {g['runs_p_median']:.3f}")
        print(f"   timing total median {np.nanmedian(r['timing']['ms']['total']):.1f} ms, latency {r['timing']['radar_latency_ms']:.0f} ms")
    for h, c in out["classification"].items():
        print(h, f"acc {c['accuracy']:.3f} vote {c['vote_accuracy']:.3f} macroF1 {c['macro_f1']:.3f} CE {c['cross_entropy_bits']:.2f} "
              f"(chance {c['chance_bits']:.2f}) n {len(c['true'])} unmatched {c['unmatched']}", [(x['class'], x['n'], round(x['f1'], 3), round(x['auc'], 3)) for x in c['per_class']])
