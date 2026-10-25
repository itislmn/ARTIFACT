"""
scene_metrics.py - what the platform understood of the scene, per radar look against the ground truth: Scene Panoptic
Quality (found x located x kind x material), separation of close objects, detections (FDR), decisions against the
oracle, safety, light and reading, cognition (chirps, blind time) and latency.

    from scene_metrics import understanding, separation, decisions
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import metrics as mt
import paths
from dataset import Take, Truth, in_view
from metrics import block_bootstrap, wilson

SCENE = paths.load_config()["radar"]

CUTOFF_M = 0.5
GOSPA_C_M = 0.5
MATCH = 0.5
RANGE_CELL_M = 0.0837
EXTENT_M = 0.6


def gap_xy(lo_a, hi_a, lo_b, hi_b) -> float:
    g = np.maximum(np.maximum(np.asarray(lo_a)[:2] - np.asarray(hi_b)[:2], np.asarray(lo_b)[:2] - np.asarray(hi_a)[:2]), 0)
    return float(np.hypot(*g))


def similarity(o: dict, t: Truth) -> float:
    ok = (o["kind"] == "moving" or bool(o.get("person"))) if t.kind == "person" else o["kind"] == expected_kind(t)
    return max(0.0, 1.0 - gap_xy(o["lo"], o["hi"], t.lo, t.hi) / CUTOFF_M) if ok else 0.0


def in_scene(p) -> bool:
    return bool(p[0] <= SCENE["x_max_m"] and SCENE["y_min_m"] <= p[1] <= SCENE["y_max_m"])


def obj_in_view(o: dict, pose, cfg) -> bool:
    m = (np.asarray(o["lo"], float)[:2] + np.asarray(o["hi"], float)[:2]) / 2
    c = m - np.asarray(pose, float)[:2]
    r = float(np.hypot(*c))
    return bool(c[0] > 0 and cfg["radar"]["r_min_m"] < r < cfg["radar"]["r_max_m"]
                and abs(np.degrees(np.arctan2(c[1], c[0]))) <= 60.0 and in_scene(m))


def expected_kind(t: Truth) -> str:
    return "static" if t.kind in ("obstacle", "lamp") else "moving"


def attributes(o: dict, t: Truth) -> dict:
    return {"kind": o["kind"] == expected_kind(t),
            "material": None if not t.material or not o.get("material") else o["material"] == t.material,
            "person": bool(o.get("person")) == (t.kind == "person")}


def match_look(tk: Take, lk: dict, t: float | None = None):
    t = lk["t"] if t is None else t
    truth = [q for q in tk.truth if q.active(t) and in_view(q, lk["pose"], tk.cfg)]
    objs = [o for o in lk["objects"] if not o.get("known") and obj_in_view(o, lk["pose"], tk.cfg)]
    S = np.array([[similarity(o, t) for o in objs] for t in truth]).reshape(len(truth), len(objs))
    pairs = []
    if S.size:
        r, c = linear_sum_assignment(-S)
        pairs = [(i, j, S[i, j]) for i, j in zip(r, c) if S[i, j] >= MATCH]
    return truth, objs, S, pairs


def understanding(takes: list[Take]) -> dict:
    per_kind, rows, series = {}, [], []
    for tk in takes:
        for lk in tk.looks:
            truth, objs, S, pairs = match_look(tk, lk)
            matched_t, matched_o = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
            look = {"tp": 0, "fp": len(objs) - len(matched_o), "fn": len(truth) - len(matched_t), "s": 0.0, "s_full": 0.0}
            for i, j, s in pairs:
                t, o = truth[i], objs[j]
                a = attributes(o, t)
                full = a["kind"] and a["material"] is not False and a["person"]
                k = per_kind.setdefault(t.kind, {"tp": 0, "fn": 0, "s": 0.0, "s_full": 0.0, "kind": 0, "material": [0, 0],
                                                 "person": 0})
                k["tp"] += 1
                k["s"] += s
                k["s_full"] += s * full
                k["kind"] += a["kind"]
                k["person"] += a["person"]
                if a["material"] is not None:
                    k["material"][0] += a["material"]
                    k["material"][1] += 1
                look["tp"] += 1
                look["s"] += s
                look["s_full"] += s * full
                rows.append({"take": tk.name, "t_s": lk["t"] - tk.t0, "truth": t.id, "kind": t.kind, "object": o["id"],
                             "similarity": s, **{f"{n}_ok": v for n, v in a.items()}})
            for i in set(range(len(truth))) - matched_t:
                per_kind.setdefault(truth[i].kind, {"tp": 0, "fn": 0, "s": 0.0, "s_full": 0.0, "kind": 0,
                                                    "material": [0, 0], "person": 0})["fn"] += 1
            den = look["tp"] + 0.5 * (look["fp"] + look["fn"])
            series.append({"take": tk.name, "t_s": lk["t"] - tk.t0, **look,
                           "pq": look["s"] / den if den else np.nan, "pq_full": look["s_full"] / den if den else np.nan})
    tp = sum(s["tp"] for s in series)
    fp, fn = sum(s["fp"] for s in series), sum(s["fn"] for s in series)
    s_sum, s_full = sum(s["s"] for s in series), sum(s["s_full"] for s in series)
    den = tp + 0.5 * (fp + fn)
    pq = np.array([s["pq"] for s in series])
    pq_full = np.array([s["pq_full"] for s in series])
    kinds = {}
    for name, k in per_kind.items():
        n = k["tp"] + k["fn"]
        kinds[name] = {"tp": k["tp"], "fn": k["fn"], "recall": k["tp"] / n if n else np.nan, "recall_ci": wilson(k["tp"], n),
                       "sq": k["s"] / k["tp"] if k["tp"] else np.nan,
                       "kind_acc": k["kind"] / k["tp"] if k["tp"] else np.nan,
                       "material_acc": k["material"][0] / k["material"][1] if k["material"][1] else np.nan,
                       "material_n": k["material"][1], "person_acc": k["person"] / k["tp"] if k["tp"] else np.nan}
    return {"PQ": s_sum / den if den else np.nan, "SQ": s_sum / tp if tp else np.nan, "RQ": tp / den if den else np.nan,
            "PQ_full": s_full / den if den else np.nan, "PQ_ci": block_bootstrap(pq), "PQ_full_ci": block_bootstrap(pq_full),
            "TP": tp, "FP": fp, "FN": fn, "kinds": kinds, "rows": rows, "series": series}


def separation(takes: list[Take]) -> dict:
    sep = merged = lost = 0
    rows = []
    for tk in takes:
        for lk in tk.looks:
            truth, objs, S, pairs = match_look(tk, lk)
            of = {i: j for i, j, _ in pairs}
            for a in range(len(truth)):
                for b in range(a + 1, len(truth)):
                    ta, tb = truth[a], truth[b]
                    if "person" in (ta.kind, tb.kind) or np.hypot(*(ta.centre - tb.centre)) > EXTENT_M:
                        continue
                    if a in of and b in of:
                        sep += 1
                        verdict = "separated"
                    elif S.size and any(S[a, j] >= MATCH and S[b, j] >= MATCH for j in range(len(objs))):
                        merged += 1
                        verdict = "merged"
                    else:
                        lost += 1
                        verdict = "missed"
                    rows.append({"take": tk.name, "t_s": lk["t"] - tk.t0, "a": ta.id, "b": tb.id,
                                 "distance_m": float(np.hypot(*(ta.centre - tb.centre))), "verdict": verdict})
    n = sep + merged + lost
    return {"pairs": n, "separated": sep, "merged": merged, "missed": lost,
            "rate": sep / n if n else np.nan, "rate_ci": wilson(sep, n), "rows": rows}


def detection(takes: list[Take]) -> dict:
    fdp, rows, series = [], [], []
    for tk in takes:
        for lk in tk.looks:
            truth = [t for t in tk.truth if t.active(lk["t"])]
            known = [(np.asarray(o["lo"][:2], float), np.asarray(o["hi"][:2], float)) for o in lk["objects"] if o.get("known")]
            false = 0
            dets = [d for d in lk["dets"] if d.get("world") and in_scene(d["world"])]
            for d in dets:
                p = np.asarray(d["world"], float)
                tol = max(RANGE_CELL_M, d["range_m"] / 12.0)
                room = (bool(d.get("still")) and d.get("change_db") is not None and d["change_db"] <= 0) or \
                    any(np.all(p[:2] >= lo - tol) and np.all(p[:2] <= hi + tol) for lo, hi in known)
                hit = next((t.id for t in truth if np.all(p[:2] >= t.lo[:2] - tol) and np.all(p[:2] <= t.hi[:2] + tol)), "")
                verdict = "room" if room else ("object" if hit else "false")
                false += verdict == "false"
                rows.append({"take": tk.name, "t_s": lk["t"] - tk.t0, "x_m": p[0], "y_m": p[1], "z_m": p[2],
                             "range_m": d["range_m"], "theta_deg": d["theta_deg"], "role": d.get("label"),
                             "track": d.get("track"), "change_db": d.get("change_db"), "truth": hit, "verdict": verdict})
            fdp.append(false / len(dets) if dets else 0.0)
            series.append((tk.name, lk["t"] - tk.t0, fdp[-1]))
    fdp = np.asarray(fdp)
    n_looks = max(sum(len(tk.looks) for tk in takes), 1)
    return {"fdr": float(fdp.mean()) if len(fdp) else np.nan, "fdr_ci": block_bootstrap(fdp),
            "alpha": takes[0].cfg["detection"]["alpha"] if takes else np.nan, "rows": rows, "series": series,
            "per_look": len(rows) / n_looks, "true_per_look": sum(r["verdict"] == "object" for r in rows) / n_looks,
            "false_per_look": sum(r["verdict"] == "false" for r in rows) / n_looks}


def oracle(tk: Take) -> dict:
    from methods.planner import Planner
    obs = [t for t in tk.truth if t.kind == "obstacle"]
    if not obs:
        return {"decision": "pass"}
    pl = Planner(tk.cfg)
    stop = tk.rail_info.get("stop_m", tk.cfg["rail"]["stop_mm"] / 1e3)
    blk = [(t.lo, t.hi, {"id": i, "z_known": True}) for i, t in enumerate(obs)]
    xh, oh = pl.x_head(0.0, (0.0, 0.0), blk)
    xc, oc = pl.x_carriage(0.0, blk)
    if oc is not None and xc <= xh:
        return {"decision": "stop", "x_stop": xc - stop}
    if oh is None:
        return {"decision": "pass"}
    best = pl.best_offset(np.array([xh - stop, 0.0, 0.0]), blk, min(pl.goal, xc))
    return {"decision": "stop" if best is None else "avoid", "x_stop": xh - stop}


def system_decision(tk: Take) -> str:
    ev = [e["to"] for e in tk.summary.get("planner_events", [])]
    return "avoid" if "avoid" in ev else ("stop" if "blocked" in ev else "pass")


def box_distance(lo_a, hi_a, lo_b, hi_b) -> float:
    lo_b, hi_b = np.where(np.isfinite(lo_b), lo_b, -1e3), np.where(np.isfinite(hi_b), hi_b, 1e3)
    gap = np.maximum(np.maximum(lo_a - hi_b, lo_b - hi_a), 0.0)
    return float(np.linalg.norm(gap)) if np.any(gap > 0) else -float(np.min(np.minimum(hi_a - lo_b, hi_b - lo_a)))


def clearance(tk: Take) -> float:
    from methods.planner import Planner
    pl = Planner(tk.cfg)
    fixed = [t for t in tk.truth if t.kind in ("obstacle", "lamp", "ball")]
    best = np.inf
    for row in tk.rail[:: max(1, len(tk.rail) // 2000)]:
        p = row[1:4]
        for lo, hi in (pl.pf.head_box(p), pl.pf.carriage_sweep(p[0], p[0])):
            for t in fixed:
                if t.active(row[0]):
                    best = min(best, box_distance(lo, hi, t.lo, t.hi))
    return best


def decisions(takes: list[Take]) -> dict:
    runs = []
    for tk in takes:
        if tk.test or "replay_of" in tk.summary:
            continue
        o, s = oracle(tk), system_decision(tk)
        first = next((d for d in tk.decisions if d["state"] in ("blocked", "avoid")), None)
        obs = [t for t in tk.truth if t.kind == "obstacle"]
        dist = min((t.lo[0] - first["pose"][0] for t in obs), default=None) if first else None
        runs.append({"take": tk.name, "system": s, "oracle": o["decision"], "correct": s == o["decision"],
                     "clearance_m": clearance(tk), "decision_distance_m": dist,
                     "travel_m": float(np.sum(np.linalg.norm(np.diff(tk.rail[:, 1:4], axis=0), axis=1))) if len(tk.rail) > 1 else 0.0,
                     "end_x_m": float(tk.rail[-1, 1]) if len(tk.rail) else 0.0})
    n = len(runs)
    k = sum(r["correct"] for r in runs)
    hits = sum(r["clearance_m"] < 0 for r in runs)
    return {"runs": runs, "n": n, "correct": k, "correct_ci": wilson(k, n), "collisions": hits,
            "collision_upper": (wilson(hits, n)[1] if n else np.nan)}


def waits(takes: list[Take]) -> dict:
    onsets, justified, entries, served, latency, still = 0, 0, 0, 0, [], []
    for tk in takes:
        prev = None
        for d in tk.decisions:
            if d["state"] == "wait" and prev != "wait":
                onsets += 1
                justified += any(a - 1.0 <= d["t"] <= b + 1.0 for a, b in tk.events.get("person", []))
            prev = d["state"]
        for a, b in tk.events.get("person", []):
            entries += 1
            w = next((d for d in tk.decisions if a <= d["t"] <= b and d["state"] == "wait"), None)
            if w is None:
                continue
            served += 1
            latency.append(w["t"] - a)
            after = tk.rail[tk.rail[:, 0] >= w["t"]] if len(tk.rail) else np.zeros((0, 5))
            if len(after) > 3:
                v = np.linalg.norm(np.diff(after[:, 1:4], axis=0), axis=1)
                k = np.flatnonzero(v < 1e-5)
                if k.size:
                    still.append(after[k[0], 0] - w["t"])
    return {"onsets": onsets, "justified": justified, "precision_ci": wilson(justified, onsets),
            "entries": entries, "served": served, "recall_ci": wilson(served, entries),
            "latency_s": np.array(latency), "still_s": np.array(still)}


def light(takes: list[Take]) -> dict:
    tp = n_on = n_true = 0
    latency, reads = [], []
    for tk in takes:
        truth_on = [a for name in ("light", "flashing") for a, _ in tk.events.get(name, [])]
        lit = [iv for name in ("light", "flashing") for iv in tk.events.get(name, [])]
        on = [float(r["t"]) for r in tk.camera if r.get("onset") == "1"]
        n_on += len(on)
        n_true += len(truth_on)
        tp += sum(any(a - 0.2 <= t <= b + 1.0 for a, b in lit) for t in on)
        latency += [min(t - a for t in on if -0.2 <= t - a <= 1.0) for a in truth_on if any(-0.2 <= t - a <= 1.0 for t in on)]
        for r in tk.readings:
            if r.get("correct") in (None, ""):
                continue
            t = float(r["t"])
            reads.append({"take": tk.name, "t_s": t - tk.t0, "share": int(r["correct"]) / max(int(r["of"]), 1),
                          "margin": float(r["min_margin"]) if r.get("min_margin") not in (None, "") else np.nan,
                          "correct": int(r["correct"]), "accepted": int(r["accepted"]), "flagged": int(r["light"]) or int(r["flashing"]),
                          "light_true": int(tk.during("light", t) or tk.during("flashing", t)), "text": r["text"]})
    share = np.array([r["share"] for r in reads])
    acc = np.array([r["accepted"] for r in reads], bool)
    lit = np.array([r["light_true"] for r in reads], bool)

    def mean_ci(x):
        x = np.asarray(x, float)
        if len(x) < 2:
            return (float(x.mean()) if len(x) else np.nan, np.nan, np.nan, len(x))
        b = np.random.default_rng(0).choice(x, size=(2000, len(x))).mean(axis=1)
        return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)), len(x)
    return {"onsets": n_on, "truth": n_true, "precision_ci": wilson(tp, n_on), "precision": tp / n_on if n_on else np.nan,
            "recall": len(latency) / n_true if n_true else np.nan, "recall_ci": wilson(len(latency), n_true),
            "latency_s": np.array(latency), "reads": reads,
            "dark": mean_ci(share[~lit]) if len(reads) else mean_ci([]), "lit_accepted": mean_ci(share[lit & acc]),
            "lit_rejected": mean_ci(share[lit & ~acc]), "accepted": mean_ci(share[acc]), "all": mean_ci(share),
            "three_accepted": wilson(int(np.sum((share[acc] * 8) >= 3)), int(acc.sum())) if len(reads) else (np.nan, np.nan),
            "selective": selective(reads)}


def selective(reads: list[dict]) -> dict:
    m = np.array([r["margin"] for r in reads], float)
    y = np.array([r["share"] for r in reads], float)
    ok = np.isfinite(m)
    m, y = m[ok], y[ok]
    if not len(m):
        return {"coverage": np.zeros(0), "accuracy": np.zeros(0), "auc": np.nan}
    order = np.argsort(-m, kind="stable")
    acc = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    cov = np.arange(1, len(y) + 1) / len(y)
    good = y >= 3 / 8
    auc = mt.roc(m, good)[2] if good.any() and (~good).any() else np.nan
    return {"coverage": cov, "accuracy": acc, "auc": auc, "best_decile": float(acc[max(0, len(acc) // 10 - 1)])}


def classification(takes: list[Take]) -> dict:
    rows = []
    for tk in takes:
        for lk in tk.looks:
            truth, objs, _, pairs = match_look(tk, lk)
            for i, j, _ in pairs:
                t, o = truth[i], objs[j]
                rows.append({"take": tk.name, "truth": t.id, "kind": t.kind, "material_true": t.material,
                             "material_pred": o.get("material"), "motion_true": t.motion, "motion_pred": o.get("motion"),
                             "person_true": t.kind == "person", "person_pred": bool(o.get("person")), "static_pred": o["kind"] == "static"})
    out = {"rows": rows}
    for head in ("material", "motion"):
        lab = [r for r in rows if r[f"{head}_true"] and r[f"{head}_pred"] is not None]
        k = sum(r[f"{head}_true"] == r[f"{head}_pred"] for r in lab)
        out[head] = {"n": len(lab), "acc": k / len(lab) if lab else np.nan, "ci": wilson(k, len(lab)),
                     "classified": len(lab) / max(sum(1 for r in rows if r[f"{head}_true"]), 1)}
    return out


def cognition(takes: list[Take]) -> dict:
    chirps, gaps, plans, info, dur, skipped = {}, [], {}, 0.0, 0.0, 0
    frame_s = []
    for tk in takes:
        t = np.array([lk["t"] for lk in tk.looks if lk["chirp"] == tk.looks[0]["chirp"]])
        if len(t) > 1:
            gaps += list(np.diff(t))
            dur += t[-1] - t[0]
            frame_s.append(np.median(np.diff(t)))
        for lk in tk.looks:
            chirps[lk["chirp"]] = chirps.get(lk["chirp"], 0) + 1
            plan = "+".join(lk.get("looks") or [lk["chirp"]])
            plans[plan] = plans.get(plan, 0) + 1
            info += float(lk.get("info_bits", 0.0) or 0.0)
        skipped += int(tk.looks[-1].get("skipped", 0) or 0)
    gaps = np.asarray(gaps)
    f = float(np.median(frame_s)) if frame_s else np.nan
    return {"chirps": chirps, "plans": plans, "gaps_s": gaps, "frame_s": f, "gap_max_s": float(gaps.max()) if gaps.size else np.nan,
            "blind_share": float(np.mean(gaps > 1.5 * f)) if gaps.size else np.nan, "info_bits_per_s": info / dur if dur else np.nan,
            "skipped": skipped, "looks": sum(len(tk.looks) for tk in takes)}


def latency(takes: list[Take]) -> dict:
    proc, result, camera, decide, stops = [], [], [], [], []
    for tk in takes:
        for lk in tk.looks:
            proc.append(lk["ms"]["total"] if isinstance(lk["ms"], dict) else lk["ms"])
            if "t_ready" in lk:
                result.append(1e3 * (lk["t_ready"] - lk["t"]))
        camera += [float(r["radar_age_ms"]) for r in tk.camera if r.get("radar_age_ms") not in (None, "")]
        by_frame = {lk["frame"]: lk["t"] for lk in tk.looks}
        decide += [1e3 * (d["t"] - by_frame[d["frame"]]) for d in tk.decisions if d.get("frame") in by_frame]
        stops += [s["time_s"] for s in tk.rail_info.get("stops", [])]
    stats = lambda x: (float(np.median(x)), float(np.percentile(x, 95))) if len(x) else (np.nan, np.nan)
    stages = {}
    for tk in takes:
        for lk in tk.looks:
            if isinstance(lk["ms"], dict):
                for k, v in lk["ms"].items():
                    stages.setdefault({"erex_fdr": "infinirex_fdr"}.get(k, k), []).append(v)
    return {"processing_ms": stats(proc), "result_ms": stats(result), "camera_ms": stats(camera),
            "decision_ms": stats(decide), "rail_stop_s": stats(stops), "stages": {k: stats(v) for k, v in stages.items()},
            "processing": np.asarray(proc), "result": np.asarray(result), "stages_raw": {k: np.asarray(v) for k, v in stages.items()}}


def obstacle_boxes(takes: list[Take]) -> dict:
    rows = []
    for tk in takes:
        for lk in tk.looks:
            truth, objs, _, pairs = match_look(tk, lk)
            for i, j, _ in pairs:
                t, o = truth[i], objs[j]
                if t.kind != "obstacle":
                    continue
                lo, hi = np.asarray(o["lo"], float), np.asarray(o["hi"], float)
                inter = np.prod(np.clip(np.minimum(hi[:2], t.hi[:2]) - np.maximum(lo[:2], t.lo[:2]), 0, None))
                union = np.prod(hi[:2] - lo[:2]) + np.prod(t.hi[:2] - t.lo[:2]) - inter
                rows.append({"take": tk.name, "truth": t.id, "distance_m": t.lo[0] - lk["pose"][0], "near_err_m": lo[0] - t.lo[0],
                             "iou_xy": inter / union if union > 0 else 0.0,
                             "contains": bool(np.all(lo[:2] <= t.lo[:2]) and np.all(hi[:2] >= t.hi[:2]))})
    c = sum(r["contains"] for r in rows)
    return {"rows": rows, "iou_median": float(np.median([r["iou_xy"] for r in rows])) if rows else np.nan,
            "contains_ci": wilson(c, len(rows)), "contains": c / len(rows) if rows else np.nan}


def regions(takes: list[Take]) -> dict:
    hs = ("step", "0.5", "1.0")
    out, series, widths = {}, {h: [] for h in hs}, {}
    for h in hs:
        x = [1 - m for tk in takes for lk in tk.looks for m in (lk.get("misses") or {}).get(h, [])]
        out[h] = (float(np.mean(x)) if x else np.nan, *block_bootstrap(x), len(x))
    runs_p = {}
    for h in hs:
        p = [mt.runs_test([m for lk in tk.looks for m in (lk.get("misses") or {}).get(h, [])]) for tk in takes]
        runs_p[h] = float(np.nanmedian(p)) if np.any(np.isfinite(p)) else np.nan
    for tk in takes:
        for lk in tk.looks:
            t = lk["t"] - tk.t0
            for h in hs:
                if h in (lk.get("coverage") or {}):
                    series[h].append((tk.name, t, lk["coverage"][h], (lk.get("alpha_t") or {}).get(h, np.nan)))
            for tr in lk["tracks"]:
                g = tr.get("region") or {}
                if tr["status"] != "confirmed" or not g.get("calibrated") or g.get("lo") is None:
                    continue
                lo, hi = np.asarray(g["lo"][:2], float), np.asarray(g["hi"][:2], float)
                if np.all(np.isfinite(np.r_[lo, hi])):
                    lab = ((tr.get("classes") or {}).get("material") or {}).get("label") or "unclassified"
                    widths.setdefault(lab, []).append((hi[0] - lo[0], tr["range_m"] * (hi[1] - lo[1])))
    alpha = takes[0].cfg["tracking"]["alpha"] if takes else np.nan
    final = {h: float(np.nanmean([q[3] for q in series[h][-50:]])) if series[h] else np.nan for h in hs}
    seqs = score_sequences(takes)
    return {"coverage": out, "nominal": 1 - alpha, "alpha": alpha, "series": series, "runs_p": runs_p,
            "alpha_t": final, "widths": {k: np.array(v) for k, v in widths.items()},
            "exchange": mt.exchangeability([np.array([q for q, _ in v]) for v in seqs.values()])}


def score_sequences(takes: list[Take]) -> dict:
    out = {}
    for tk in takes:
        for lk in tk.looks:
            regs = {tr["id"]: tr["region"] for tr in lk["tracks"] if (tr.get("region") or {}).get("calibrated")}
            for d in lk["dets"]:
                g = regs.get(d.get("track"))
                if d.get("label") != "core" or g is None or g.get("lo") is None:
                    continue
                lo, hi = np.asarray(g["lo"][:2], float), np.asarray(g["hi"][:2], float)
                if np.all(np.isfinite(np.r_[lo, hi])):
                    z = np.array([d["range_m"], np.radians(d["theta_deg"])])
                    score = float(np.max(np.abs(2 * (z - lo) / np.maximum(hi - lo, 1e-9) - 1)))
                    out.setdefault(f"{tk.name}:{d['track']}", []).append((score, int(score > 1)))
    return out


def distance(o: dict, t: Truth) -> float:
    if t.kind == "person":
        return gap_xy(o["lo"], o["hi"], t.lo, t.hi)
    c = (np.asarray(o["lo"][:2], float) + np.asarray(o["hi"][:2], float)) / 2
    return float(np.hypot(*(c - t.centre)))


def _frame(tk: Take, t: float, truth, objs, S, pairs) -> dict:
    gt = [f"{tk.name}:{q.id}" for q in truth]
    D = np.array([[distance(o, q) for o in objs] for q in truth]).reshape(len(truth), len(objs))
    gs = mt.gospa(D, GOSPA_C_M)
    gs["assign"] = {gt[i]: objs[j]["id"] for i, j in gs.pop("pairs")}
    return {"t": t, "gt": gt, "tr": [f"{tk.name}:{o['id']}" for o in objs], "sim": S,
            "hit": {truth[i].id: objs[j]["id"] for i, j, _ in pairs}, "kind": [q.kind for q in truth], "gospa": gs}


def _scores(frames: list[dict], kinds) -> dict:
    out = {"hota": mt.hota(frames), "clear": mt.clear(frames), "idf1": mt.idf1(frames), "objects": {},
           "segments": [{"HOTA": mt.hota(q)["HOTA"], "MOTA": mt.clear(q)["MOTA"], "IDF1": mt.idf1(q)["IDF1"],
                         "GOSPA": float(np.mean([f["gospa"]["gospa"] for f in q])),
                         "OSPA": float(np.mean([f["gospa"]["ospa"] for f in q]))} for q in mt.segments(frames)]}
    g = [f["gospa"] for f in frames]
    pairs = sum(len(x["assign"]) for x in g)
    sw = mt.switches(frames)
    out["gospa"] = {"gospa": float(np.mean([x["gospa"] for x in g])), "ospa": float(np.mean([x["ospa"] for x in g])),
                    "loc_rms": float(np.sqrt(sum(x["loc"] for x in g) / max(pairs, 1))),
                    "missed": float(np.mean([x["n_missed"] for x in g])), "false": float(np.mean([x["n_false"] for x in g])),
                    "switches": sw, "t_gospa": float(np.sqrt((sum(x["loc"] + x["missed"] + x["false"] for x in g)
                                                             + GOSPA_C_M ** 2 * sw) / max(len(g), 1)))}
    for k in kinds:
        ids, n = [], 0
        for f in frames:
            for g, kind in zip(f["gt"], f["kind"]):
                if kind == k:
                    n += 1
                    ids.append(f["hit"].get(g.split(":", 1)[1] if ":" in g else g))
        got = [i for i in ids if i is not None]
        counts = {i: got.count(i) for i in set(got)}
        out["objects"][k] = {"frames": n, "tracked": len(got) / max(n, 1), "ci": wilson(len(got), n), "ids": len(counts),
                             "longest": max(counts.values()) / max(n, 1) if counts else 0.0}
    return out


def tracking(takes: list[Take]) -> dict:
    frames = {"look": [], "live": []}
    offset, dist = 0.0, []
    for tk in takes:
        by_frame = {lk["frame"]: lk for lk in tk.looks}
        for lk in tk.looks:
            truth, objs, S, pairs = match_look(tk, lk)
            frames["look"].append(_frame(tk, offset + lk["t"] - tk.t0, truth, objs, S, pairs))
            for i, j, _ in pairs:
                c = (np.asarray(objs[j]["lo"][:2], float) + np.asarray(objs[j]["hi"][:2], float)) / 2
                dist.append((offset + lk["t"] - tk.t0, truth[i].kind, float(np.hypot(*(c - truth[i].centre)))))
        rows = [(float(r["t"]), r.get("radar_frame")) for r in tk.camera if r.get("radar_frame") not in (None, "")]
        if rows:
            ts = np.array([q[0] for q in rows])
            for tg in np.arange(ts[0], ts[-1], 0.1):
                k = int(np.searchsorted(ts, tg, side="right") - 1)
                lk = by_frame.get(int(float(rows[k][1])))
                if lk is None:
                    continue
                truth, objs, S, pairs = match_look(tk, lk, t=tg)
                frames["live"].append(_frame(tk, offset + tg - tk.t0, truth, objs, S, pairs))
        offset += tk.looks[-1]["t"] - tk.t0 + 10.0
    kinds = sorted({q.kind for tk in takes for q in tk.truth})
    out = {v: _scores(f, kinds) for v, f in frames.items() if f}
    out["identity"] = [{"t": f["t"], "gt": [g for g in f["gt"]], "hit": f["hit"],
                        "gospa": {k: f["gospa"][k] for k in ("loc", "missed", "false", "gospa")}} for f in frames["look"]]
    out["runs"] = [(tk.name, tk.looks[-1]["t"] - tk.t0) for tk in takes]
    out["kinds"] = kinds
    motp = {}
    for k in kinds:
        q = np.array([(t, d) for t, kind, d in dist if kind == k and kind != "person"])
        if len(q):
            seg = [q[(q[:, 0] >= a) & (q[:, 0] < a + 10.0), 1] for a in np.arange(0, q[-1, 0], 10.0)]
            motp[k] = {"mean_m": float(q[:, 1].mean()), "segments_m": [float(v.mean()) for v in seg if len(v) >= 20]}
    out["motp"] = motp
    return out


def motion(takes: list[Take]) -> dict:
    import json
    import paths
    runs = []
    for tk in takes:
        R = tk.rail
        v = np.r_[0.0, np.linalg.norm(np.diff(R[:, 1:4], axis=0), axis=1) / np.maximum(np.diff(R[:, 0]), 1e-6)] if len(R) > 1 else np.zeros(0)
        moving = v > 1e-4
        states = {}
        for a, b in zip(tk.decisions, tk.decisions[1:]):
            states[a["state"]] = states.get(a["state"], 0.0) + b["t"] - a["t"]
        drift = {}
        for lk in tk.looks:
            if np.hypot(*lk["v_rail"][:2]) < 1e-3:
                continue
            for o in lk["objects"]:
                if o["kind"] == "static" and o.get("live") and not o.get("known"):
                    drift.setdefault(o["id"], []).append((np.asarray(o["lo"][:2]) + np.asarray(o["hi"][:2])) / 2)
        sd = [float(np.linalg.norm(np.std(np.array(q), axis=0))) for q in drift.values() if len(q) >= 20]
        obs = [q for q in tk.truth if q.kind == "obstacle"]
        approach = [(lk["t"] - tk.t0, min(q.lo[0] for q in obs) - lk["pose"][0]) for lk in tk.looks] if obs else []
        stops = tk.rail_info.get("stops", [])
        runs.append({"take": tk.name, "test": tk.test, "travel_m": float(np.sum(v[1:] * np.diff(R[:, 0]))) if len(R) > 1 else 0.0,
                     "moving_s": float(np.sum(np.diff(R[:, 0])[moving[1:]])) if len(R) > 1 else 0.0,
                     "speed_mm_s": float(1e3 * np.median(v[moving])) if moving.any() else np.nan,
                     "stops": len(stops), "stop_mm": float(1e3 * max((q["distance_m"] for q in stops), default=np.nan)),
                     "stop_s": float(np.median([q["time_s"] for q in stops])) if stops else np.nan, "states_s": states,
                     "drift_mm": float(1e3 * np.median(sd)) if sd else np.nan, "drift_objects": len(sd),
                     "rail": (R[:, 0] - tk.t0, R[:, 1], v) if len(R) else None, "approach": approach,
                     "trace": [(lk["t"] - tk.t0, lk["pose"][0], lk["pose"][1]) for lk in tk.looks]})
    scale = {}
    for tk in takes:
        f = paths.reference_dir(tk.name.split("/")[0]) / "rail_scale.json"
        if f.exists():
            scale = json.loads(f.read_text())
    return {"runs": runs, "rail_scale": scale}


def camera(takes: list[Take]) -> dict:
    from backend.reader_process import correct_chars
    runs = []
    for tk in takes:
        text = tk.scene.get("text", "")
        t = np.array([float(r["t"]) - tk.t0 for r in tk.camera])
        light = np.array([int(r.get("light") or 0) for r in tk.camera])
        stable, last, rows = [], [], []
        for r in tk.readings:
            tr = float(r["t"]) - tk.t0
            if r.get("accepted") == "1" and r["text"]:
                last = (last + [r["text"]])[-15:]
            best = max(set(last), key=last.count) if last else ""
            stable.append((tr, correct_chars(best, text) if text else 0))
            rows.append({"t_s": tr, "text": r["text"], "role": r.get("role", ""), "accepted": int(r["accepted"])})
        roles, crops, last_crop = {}, [], {}
        first = int(tk.camera[0]["camera_frame"]) if tk.camera else 0
        for r in tk.camera:
            role = r.get("role") or "none"
            roles[role] = roles.get(role, 0) + 1
            tr = float(r["t"]) - tk.t0
            if role in ("ball", "person", "collider") and r.get("crop_px") and tr - last_crop.get(role, -1e9) >= 2.0:
                last_crop[role] = tr
                crops.append({"t_s": tr, "pos": int(r["camera_frame"]) - first, "role": role,
                              "rect": [float(v) for v in r["crop_px"].split(";")]})
        video = Path(tk.summary.get("replay_of") or tk.folder) / "camera.mp4"
        final = stable[-1][1] if stable else 0
        runs.append({"take": tk.name, "t": t, "light": light, "events": {k: [[a - tk.t0, b - tk.t0] for a, b in v]
                                                                             for k, v in tk.events.items()},
                     "stable": stable, "final_correct": final, "letters": len(text), "reads": rows, "roles": roles,
                     "crops": crops, "video": str(video)})
    return {"runs": runs}
