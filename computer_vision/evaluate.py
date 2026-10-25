"""
evaluate.py - both readers on every dataset frame (raw, and after Radar-Anchored Glare Removal while the lamp is on),
the reading metrics per condition, character confusions, confidence, run time and alignment, as predictions.csv and
tables.tex.

    python evaluate.py           (python evaluate.py --again reads every take again)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import json  # noqa: E402

import numpy as np  # noqa: E402

import complexity  # noqa: E402
import dataset  # noqa: E402
from backend import glare as gl  # noqa: E402
from backend import pipeline  # noqa: E402
from backend.settings import CACHE, CALIB, MODELS, OUTPUT, load_config, take_dir  # noqa: E402
from metrics import align, auroc, counts, stable  # noqa: E402
from readers import cnn_reader, metricsight  # noqa: E402

READERS = {"MetricSight": metricsight, "CNN": cnn_reader}
CONDITIONS = (("text", "Text Only"), ("lamp_off", "Lamp Off"), ("lamp_on", "Lamp On"),
              ("lamp_removed", "Lamp On, Glare Removed"))
PREDICTIONS = OUTPUT / "predictions.csv"
TABLES = OUTPUT / "tables.tex"
COLUMNS = ["take", "scene", "k", "lamp", "reader", "condition", "text", "cer", "S", "D", "I", "exact", "confidence",
           "ms", "boxes", "line_box"]
STABLE_N = 15


def read_take(name: str, frames: list[dict], cfg, proj) -> list[dict]:
    want = {int(r["k"]): r for r in frames if r["text_found"] == 1}
    out = []
    for k, rgb in dataset.video_frames(take_dir(dataset.SESSION, name), set(want)):
        r = want[k]
        images = {"raw": rgb}
        if r["lamp"] == 1 and r["glare_u"] != "":
            g = gl.Glare(-1, (r["glare_u"], r["glare_v"]), r["glare_core_px"], json.loads(r["glare_fit"]), 0.0)
            images["removed"] = gl.clean(rgb, g, pipeline.fast_luma(rgb) >= cfg["camera"]["clip_level"])
        for condition, img in images.items():
            for reader, mod in READERS.items():
                rd = mod.read(img, (r["centre_u"], r["centre_v"]), r["text_range_m"], proj.f, cfg)
                c = counts(dataset.TRUTH, rd.text)
                out.append({"take": name, "scene": r["scene"], "k": k, "lamp": int(r["lamp"]), "reader": reader,
                            "condition": condition, "text": rd.text, "cer": round((c["S"] + c["D"] + c["I"]) / c["N"], 4),
                            "S": c["S"], "D": c["D"], "I": c["I"], "exact": int(rd.text == dataset.TRUTH),
                            "confidence": json.dumps(np.round(rd.margins, 4).tolist()), "ms": round(rd.ms, 2),
                            "boxes": json.dumps(np.round(rd.boxes, 1).tolist()),
                            "line_box": json.dumps(None if rd.line_box is None else np.round(rd.line_box, 1).tolist())})
    return out


def predictions(rows: list[dict], again: bool) -> list[dict]:
    from backend.alignment import Projector
    cfg, proj = load_config(), Projector()
    deps = [Path(__file__), MODELS / "metricsight_ridge.npz", MODELS / "cnn_chars.npz",
            HERE / "readers" / "metricsight.py", HERE / "readers" / "cnn_reader.py", HERE / "readers" / "segment.py",
            HERE / "backend" / "glare.py"]
    out = []
    for name in dataset.TAKES:
        cache = CACHE / f"predictions_{name}.csv"
        if again or not dataset.fresh(cache, deps + [CACHE / f"dataset_{name}.csv"]):
            print(f"  predictions: {name} ...", flush=True)
            dataset.write(cache, read_take(name, [r for r in rows if r["take"] == name], cfg, proj), COLUMNS)
        out += dataset.read(cache)
    for p in out:
        p["text"] = "" if p["text"] == "" else str(p["text"])
    dataset.write(PREDICTIONS, out, COLUMNS)
    return out


def condition_of(p: dict) -> str:
    if p["scene"] == "text":
        return "text"
    if p["condition"] == "removed":
        return "lamp_removed"
    return "lamp_on" if p["lamp"] == 1 else "lamp_off"


def add_stable(preds: list[dict]):
    groups = {}
    for p in preds:
        groups.setdefault((p["take"], p["reader"], p["condition"]), []).append(p)
    for g in groups.values():
        g.sort(key=lambda p: p["k"])
        for p, s in zip(g, stable([p["text"] for p in g], STABLE_N)):
            c = counts(dataset.TRUTH, s)
            p["stable"], p["stable_cer"], p["stable_exact"] = s, (c["S"] + c["D"] + c["I"]) / c["N"], int(s == dataset.TRUTH)


def paired(preds: list[dict]) -> list[dict]:
    removed = {(p["take"], p["k"], p["reader"]) for p in preds if p["condition"] == "removed"}
    return [p for p in preds if condition_of(p) != "lamp_on" or (p["take"], p["k"], p["reader"]) in removed]


def summary(preds: list[dict]) -> dict:
    out = {}
    for key, _ in CONDITIONS:
        for reader in READERS:
            sel = [p for p in preds if condition_of(p) == key and p["reader"] == reader]
            if not sel:
                continue
            n_chars = len(sel) * len(dataset.TRUTH)
            out[key, reader] = {"frames": len(sel), "cer": 100 * np.mean([p["cer"] for p in sel]),
                                "word": 100 * np.mean([p["exact"] for p in sel]),
                                "stable_cer": 100 * np.mean([p["stable_cer"] for p in sel]),
                                "stable_word": 100 * np.mean([p["stable_exact"] for p in sel]),
                                "S": 100 * sum(p["S"] for p in sel) / n_chars, "D": 100 * sum(p["D"] for p in sel) / n_chars,
                                "I": 100 * sum(p["I"] for p in sel) / n_chars,
                                "ms": float(np.median([p["ms"] for p in sel]))}
    return out


def per_take(preds: list[dict]) -> dict:
    out = {}
    for name, scene in dataset.TAKES.items():
        if scene != "light":
            continue
        for reader in READERS:
            for key in ("lamp_on", "lamp_removed"):
                sel = [p for p in preds if p["take"] == name and p["reader"] == reader and condition_of(p) == key]
                out[name, reader, key] = {"frames": len(sel), "cer": 100 * np.mean([p["cer"] for p in sel]),
                                          "stable_cer": 100 * np.mean([p["stable_cer"] for p in sel])}
    return out


def characters(preds: list[dict]) -> dict:
    out = {}
    for reader in READERS:
        pairs, conf, ok = [], [], []
        for p in preds:
            if p["reader"] != reader or p["condition"] != "raw":
                continue
            margins = json.loads(p["confidence"])
            for t, r, j in align(dataset.TRUTH, p["text"]):
                pairs.append((t, r))
                if j >= 0 and j < len(margins):
                    conf.append(margins[j])
                    ok.append(t == r)
        out[reader] = {"pairs": pairs, "confidence": np.array(conf), "correct": np.array(ok, bool),
                       "auroc": auroc(np.array(conf), np.array(ok, bool))}
    return out


def scene_image(points: list[dict], radius: int = 70) -> np.ndarray:
    from backend.imgio import read_png
    from backend.settings import ROOT
    shots = [read_png(ROOT / q["image"]) for q in points]
    out = np.median(np.stack(shots), axis=0)
    h, w = out.shape[:2]
    yy, xx = np.ogrid[0:h, 0:w]
    for img, q in zip(shots, points):
        u, v = q["uv_px"]
        r = 2 * radius if q["label"] == "big CR" else radius
        alpha = np.clip((r - np.hypot(xx - u, yy - v)) / (0.3 * r), 0, 1)[..., None]
        out = out * (1 - alpha) + img * alpha
    return out.astype(np.uint8)


def alignment_result(rows: list[dict], preds: list[dict]) -> dict:
    a = json.loads((CALIB / "alignment.json").read_text())
    f = a["params"]["f"]
    by_k = {(r["take"], r["k"]): r for r in rows}
    word = []
    for p in preds:
        if p["reader"] == "MetricSight" and p["condition"] == "raw" and p["lamp"] == 0 and p["line_box"] != "null":
            r = by_k[p["take"], p["k"]]
            x0, y0, x1, y1 = json.loads(p["line_box"])
            if p["cer"] <= 0.25:
                word.append(((x0 + x1) / 2 - r["centre_u"], (y0 + y1) / 2 - r["centre_v"], r["text_range_m"]))
    lamp = []
    for name, scene in dataset.TAKES.items():
        if scene == "light":
            s = json.loads((take_dir(dataset.SESSION, name) / "summary.json").read_text())
            for e in s.get("events", []):
                if e.get("source") and e.get("core_uv"):
                    src = e["source"]
                    lamp.append((e["core_uv"][0] - src["anchor_uv"][0], e["core_uv"][1] - src["anchor_uv"][1], src["range_m"]))
    word, lamp = np.array(word), np.array(lamp).reshape(-1, 3)
    return {"model": a, "f": f, "scene": scene_image(a["points"]), "word_px": word[:, :2], "word_cm": 100 * word[:, :2] * word[:, 2:] / f,
            "lamp_px": lamp[:, :2], "lamp_cm": 100 * lamp[:, :2] * lamp[:, 2:] / f}


def veil_example(rows: list[dict]) -> dict:
    from backend.alignment import Projector
    cfg, proj = load_config(), Projector()
    lit = [r for r in rows if r["take"] == "light_3" and r["lamp"] == 1 and r["glare_u"] != ""]
    r = lit[len(lit) // 2]
    k, rgb = next(dataset.video_frames(take_dir(dataset.SESSION, "light_3"), {int(r["k"])}))
    clipped = pipeline.fast_luma(rgb) >= cfg["camera"]["clip_level"]
    g = dataset.glare_fit(rgb, clipped, (r["glare_u"], r["glare_v"]), json.loads(r["objects"]), proj, cfg)
    return {"take": "light_3", "k": k, "radii": np.asarray(g.profile["radii"]), "levels": [np.asarray(x) for x in g.profile["levels"]],
            "fits": g.fits}


def showcase(rows: list[dict], preds: list[dict]) -> dict:
    from backend.alignment import Projector
    cfg, proj = load_config(), Projector()
    by = {(p["k"], p["reader"], p["condition"]): p for p in preds if p["take"] == "light_3"}
    cand = [r for r in rows if r["take"] == "light_3" and r["lamp"] == 1 and r["lamp_radar"] == 1 and r["glare_u"] != ""
            and (r["k"], "MetricSight", "raw") in by
            and by[r["k"], "MetricSight", "raw"]["text"] == by[r["k"], "MetricSight", "raw"].get("stable")]
    r = cand[len(cand) // 2]
    k, rgb = next(dataset.video_frames(take_dir(dataset.SESSION, "light_3"), {int(r["k"])}))
    g = gl.Glare(-1, (r["glare_u"], r["glare_v"]), r["glare_core_px"], json.loads(r["glare_fit"]), 0.0)
    clean = gl.clean(rgb, g, pipeline.fast_luma(rgb) >= cfg["camera"]["clip_level"])
    centre = (r["centre_u"], r["centre_v"])
    out = {"k": k, "row": r, "raw": rgb, "clean": clean, "f": proj.f, "objects": json.loads(r["objects"]),
           "bars": [proj.bar(o["range_m"], o["azimuth_deg"], cfg["alignment"]["bar_z_m"]) for o in json.loads(r["objects"])]}
    for cond, img in (("raw", rgb), ("removed", clean)):
        for reader, mod in READERS.items():
            out[reader, cond] = mod.read(img, centre, r["text_range_m"], proj.f, cfg, keep_debug=(reader == "MetricSight"))
    return out


def live_timing() -> dict:
    fps, radar = [], []
    for name in dataset.TAKES:
        s = json.loads((take_dir(dataset.SESSION, name) / "summary.json").read_text())
        fps.append(s["stream"]["fps_measured"])
    rows = dataset.read(dataset.TABLE)
    radar = [r["radar_ms"] for r in rows if r["radar_ms"] != ""]
    return {"camera_fps": float(np.median(fps)), "radar_ms": float(np.median(radar)), "radar_ms_p95": float(np.percentile(radar, 95))}


def run(again: bool = False) -> dict:
    rows = dataset.build(again)
    preds = predictions(rows, again)
    add_stable(preds)
    fair = paired(preds)
    res = {"rows": rows, "preds": fair, "summary": summary(fair), "per_take": per_take(fair), "chars": characters(fair),
           "alignment": alignment_result(rows, fair), "live": live_timing(), "complexity": complexity.measure(),
           "veil": veil_example(rows), "showcase": showcase(rows, fair)}
    write_tables(res)
    return res


def write_tables(res: dict):
    S, rows = res["summary"], res["rows"]
    t = ["% \\input{tables.tex} once, then \\cvDataset, \\cvReading, \\cvErrors, \\cvLight, \\cvConfidence, \\cvTiming, "
         "\\cvAlignment where the table belongs", ""]

    def table(name, spec, head, body):
        t.extend([f"\\newcommand{{\\{name}}}{{%", f"\\begin{{tabular}}{{{spec}}}", "\\toprule", head + " \\\\",
                  "\\midrule", *[b + " \\\\" for b in body], "\\bottomrule", "\\end{tabular}", "}", ""])

    body = []
    for name, scene in dataset.TAKES.items():
        r = [x for x in rows if x["take"] == name]
        lamp = [x for x in r if x["lamp"] == 1]
        body.append(f"{name.replace('_', '\\_')} & {'text only' if scene == 'text' else 'lamp switched'} & {len(r)} & "
                    f"{100 * np.mean([x['text_found'] == 1 for x in r]):.1f} & {len(lamp)} & "
                    f"{sum(x['glare_u'] != '' for x in lamp)} & "
                    + (f"{100 * np.mean([x['lamp_radar'] == 1 for x in lamp]):.1f}" if lamp else "--"))
    table("cvDataset", "llrrrrr", "Take & Scene & Frames & Board found [\\%] & Lamp on & Veil fitted & Lamp found [\\%]", body)

    body = []
    for key, label in CONDITIONS:
        cells = []
        for reader in READERS:
            v = S.get((key, reader))
            cells += [f"{v['cer']:.1f}", f"{v['word']:.1f}", f"{v['stable_cer']:.1f}", f"{v['stable_word']:.1f}"] if v else ["--"] * 4
        n = S[key, "MetricSight"]["frames"]
        body.append(f"{label} & {n} & " + " & ".join(cells))
    table("cvReading", "lrrrrrrrrr",
          " & & \\multicolumn{4}{c}{MetricSight} & \\multicolumn{4}{c}{CNN} \\\\\nCondition & Frames & CER & Word & "
          "CER$_s$ & Word$_s$ & CER & Word & CER$_s$ & Word$_s$", body)

    body = []
    for key, label in CONDITIONS:
        for reader in READERS:
            v = S.get((key, reader))
            if v:
                body.append(f"{label} & {reader} & {v['S']:.1f} & {v['D']:.1f} & {v['I']:.1f} & {v['cer']:.1f}")
    table("cvErrors", "llrrrr", "Condition & Reader & Substitutions & Deletions & Insertions & CER", body)

    body = []
    for name, scene in dataset.TAKES.items():
        if scene != "light":
            continue
        cells = []
        for reader in READERS:
            a, b = res["per_take"][name, reader, "lamp_on"], res["per_take"][name, reader, "lamp_removed"]
            cells += [f"{a['cer']:.1f}", f"{b['cer']:.1f}", f"{a['stable_cer']:.1f}", f"{b['stable_cer']:.1f}"]
        body.append(f"{name.replace('_', '\\_')} & {res['per_take'][name, 'MetricSight', 'lamp_on']['frames']} & " + " & ".join(cells))
    a = {r: (S["lamp_on", r], S["lamp_removed", r]) for r in READERS}
    body.append(f"all & {a['MetricSight'][0]['frames']} & " + " & ".join(
        f"{x:.1f}" for r in READERS for x in (a[r][0]['cer'], a[r][1]['cer'], a[r][0]['stable_cer'], a[r][1]['stable_cer'])))
    table("cvLight", "lrrrrrrrrr",
          " & & \\multicolumn{4}{c}{MetricSight} & \\multicolumn{4}{c}{CNN} \\\\\nTake & Frames & Raw & Removed & "
          "Raw$_s$ & Removed$_s$ & Raw & Removed & Raw$_s$ & Removed$_s$", body)

    body = []
    for reader in READERS:
        c = res["chars"][reader]
        keep = np.argsort(-c["confidence"])[: int(0.8 * len(c["confidence"]))]
        body.append(f"{reader} & {len(c['confidence'])} & {100 * c['correct'].mean():.1f} & "
                    f"{100 * c['correct'][keep].mean():.1f} & {c['auroc']:.3f}")
    table("cvConfidence", "lrrrr", "Reader & Characters & Accuracy & Accuracy at 80\\,\\% coverage & AUROC", body)

    body = []
    for reader in READERS:
        ms = [p["ms"] for p in res["preds"] if p["reader"] == reader and p["condition"] == "raw"]
        size = sum(v.size for k, v in np.load(MODELS / ("metricsight_ridge.npz" if reader == "MetricSight" else "cnn_chars.npz")).items()
                   if k not in ("lam", "test_accuracy"))
        c = [x for x in res["complexity"] if x["reader"] == reader]
        body.append(f"{reader} & {np.median(ms):.1f} & {np.percentile(ms, 95):.1f} & {c[0]['ms']:.0f} / {c[-1]['ms']:.0f} & "
                    f"{size / 1000:.1f}\\,k")
    L = res["live"]
    body.append(f"Radar (Infini-Rex) & {L['radar_ms']:.1f} & {L['radar_ms_p95']:.1f} & -- & --")
    table("cvTiming", "lrrrr", "Component & Median [ms] & 95th pct. [ms] & 0.5\\,m / 2\\,m [ms] & Parameters", body)

    A = res["alignment"]
    m = A["model"]
    body = [f"Calibration points (fit) & {len(m['error_px'])} & {m['rms_px']:.1f} & -- & --",
            f"Calibration points (leave one out) & {len(m['loo_error_px'])} & {m['loo_rms_px']:.1f} & -- & --",
            f"Word on the ball (independent) & {len(A['word_px'])} & {np.sqrt((A['word_px'] ** 2).sum(1).mean()):.1f} & "
            f"{np.median(A['word_cm'][:, 0]):+.1f} & {np.median(A['word_cm'][:, 1]):+.1f}",
            f"Phone flash (independent) & {len(A['lamp_px'])} & {np.sqrt((A['lamp_px'] ** 2).sum(1).mean()):.1f} & "
            f"{np.median(A['lamp_cm'][:, 0]):+.1f} & {np.median(A['lamp_cm'][:, 1]):+.1f}"]
    table("cvAlignment", "lrrrr", "Test & Points & RMS [px] & $\\Delta u$ [cm] & $\\Delta v$ [cm]", body)
    TABLES.write_text("\n".join(t), encoding="utf-8")


if __name__ == "__main__":
    r = run(again="--again" in sys.argv)
    for (key, reader), v in r["summary"].items():
        print(f"  {key:<13} {reader:<12} frames {v['frames']:5d}  CER {v['cer']:5.1f}  word {v['word']:5.1f}  "
              f"stable CER {v['stable_cer']:5.1f}  stable word {v['stable_word']:5.1f}")
