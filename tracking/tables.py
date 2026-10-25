"""
tables.py - the results as booktabs tables, one LaTeX macro each (\\trDataset, \\trTracking, \\trObjects,
\\trEstimation, \\trRegions, \\trDetection, \\trClassification, \\trReading, \\trTiming), for output/tables.tex.

    from tables import render
"""
from __future__ import annotations

import numpy as np

VIEWS = (("look", "Per Look"), ("live", "Live"))
STAGES = (("range_doppler", "Range--Doppler"), ("infinirex_fdr", "Infini-Rex + e-BH"), ("predict", "Prediction"),
          ("track", "Association + KalmanCQR"), ("material", "M$^2$-SRC"), ("cognitive", "Chirp choice"),
          ("total", "Total"))


def f(x, d=1, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}"


def macro(name, spec, head, body):
    rows = "\n".join(" & ".join(r) + r" \\" if r != "mid" else r"\midrule" for r in body)
    return (f"\\newcommand{{\\{name}}}{{%\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
            f"{rows}\n\\bottomrule\n\\end{{tabular}}\n}}\n")


def render(res: dict) -> str:
    T = res["takes"]
    out = ["% \\input{tables.tex} once, then \\trDataset, \\trTracking, \\trObjects, \\trEstimation, \\trRegions, "
           "\\trDetection, \\trClassification, \\trReading, \\trTiming, \\trPolicies where the table belongs\n"]
    body = []
    for k, r in T.items():
        tr = r["truth"]
        body.append([k.replace("_", r"\_"), r["label"], f(r["duration_s"], 1), str(len(r["looks"])), str(r["camera_frames"]),
                     f(np.mean([q["ball"]["found"] for q in tr]), 0, True),
                     f(np.mean([q["person"]["found"] and not q["person"]["partial"] for q in tr]), 0, True),
                     f(np.mean([q["person"]["partial"] for q in tr]), 0, True)])
    out.append(macro("trDataset", "llrrrrrr", r"Take & Chirps & Seconds & Radar looks & Camera frames & Ball (\%) & "
                     r"Person (\%) & Person at edge (\%)", body))
    cols = [(k, v) for k in T for v, _ in VIEWS]
    head = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{T[k]['label']}}}" for k in T) + r" \\" + "\n" + \
        "Metric & " + " & ".join(lbl for _ in T for _, lbl in VIEWS)
    keys = [("HOTA", "hota", "HOTA"), ("DetA", "hota", "DetA"), ("AssA", "hota", "AssA"), ("LocA", "hota", "LocA"),
            ("MOTA", "clear", "MOTA"), ("MOTP", "clear", "MOTP"), ("IDF1", "idf1", "IDF1")]
    body = [[n] + [f(T[k]["mot"][v][g][m], 1, True) for k, v in cols] for n, g, m in keys]
    body += [[n] + [str(T[k]["mot"][v]["clear"][m]) for k, v in cols] for n, m in
             (("ID switches", "IDSW"), ("False positives", "FP"), ("Misses", "FN"), ("Ground truth", "GT"))]
    body += ["mid"]
    body += [[n] + [f(T[k]["mot"][v]["gospa"][m], d) for k, v in cols] for n, m, d in
             ((r"GOSPA ($^\circ$)", "gospa", 2), (r"\quad localisation RMS ($^\circ$)", "loc_rms", 2),
              (r"\quad missed per look", "missed", 2), (r"\quad false per look", "false", 2),
              (r"OSPA ($^\circ$)", "ospa", 2), ("Track switches (T-GOSPA)", "switches", 1),
              (r"T-GOSPA per look ($^\circ$)", "t_gospa", 2))]
    out.append(macro("trTracking", "l" + "r" * len(cols), head, body))
    body = []
    for k in T:
        for v, lbl in VIEWS:
            for o, s in T[k]["mot"][v]["objects"].items():
                body.append([T[k]["label"], lbl, o, str(s["frames"]), f(s["tracked"], 1, True),
                             f"{f(s['ci'][0], 0, True)}--{f(s['ci'][1], 0, True)}", str(s["ids"]), f(s["longest"], 1, True)])
    out.append(macro("trObjects", "lllrrcrr", r"Chirps & View & Object & Frames & Tracked (\%) & 95\,\% CI & IDs & "
                     r"Longest ID (\%)", body))
    body = []
    for k in T:
        for o, e in T[k]["estimation"].items():
            body.append([T[k]["label"], o, f(e["abs_median"], 2), f(e["abs_q90"], 2), f(e["live_abs_median"], 2),
                         f(e["bias"], 2), f(e["z_rms"], 2),
                         f"{f(100 * e['range_bias'], 1)} / {f(100 * e['range_std'], 1)}" if o == "ball" else "--"])
    out.append(macro("trEstimation", "llrrrrrc", r"Chirps & Object & $|\Delta\theta|$ median ($^\circ$) & "
                     r"90\,\% ($^\circ$) & Live median ($^\circ$) & Bias ($^\circ$) & RMS $z$ & Range bias / sd (cm)", body))
    body = []
    for k in T:
        g, e = T[k]["regions"], T[k]["estimation"]
        c = g["coverage_radar"]
        body.append([T[k]["label"], f(c.get("step"), 2), f(c.get("0.5"), 2), f(c.get("1.0"), 2),
                     f(e["ball"]["coverage_camera"], 2), f(e["person"]["coverage_camera"], 2),
                     f(np.median(e["ball"]["iou"]) if len(e["ball"]["iou"]) else np.nan, 2),
                     f(np.median(e["person"]["iou"]) if len(e["person"]["iou"]) else np.nan, 2),
                     f(g["exchange"]["ks_p"], 2), f(g["exchange"]["lag1"], 2), f(g["runs_p_median"], 2)])
    out.append(macro("trRegions", "lrrrrrrrrrr", r" & \multicolumn{3}{c}{Radar coverage} & \multicolumn{2}{c}{Camera coverage} & "
                     r"\multicolumn{2}{c}{IoU median} & \multicolumn{3}{c}{Exchangeability} \\" + "\n" +
                     r"Chirps & Next & 0.5\,s & 1\,s & Ball & Person & Ball & Person & KS $p$ & Lag-1 & Runs $p$", body))
    body = []
    for k in T:
        d = T[k]["detection"]
        body.append([T[k]["label"], f(d["fdr"], 3), f"{f(d['fdr_ci'][0], 3)}--{f(d['fdr_ci'][1], 3)}", f(d["precision"], 3),
                     f(d["recall_ball"], 2), f(d["recall_person"], 2), f(d["dets_per_look"], 1)])
    out.append(macro("trDetection", "lrcrrrr", r"Chirps & FDR ($\alpha = 0.1$) & 95\,\% CI & Precision & Recall ball & "
                     r"Recall person & Detections per look", body))
    body = []
    C = res["classification"]
    for h, c in C.items():
        body.append([rf"\multicolumn{{6}}{{l}}{{\textit{{{h} head}}}}"])
        for r in c["per_class"]:
            body.append([r["class"], str(r["n"]), f(r["precision"], 1, True), f(r["recall"], 1, True), f(r["f1"], 1, True),
                         f(r["auc"], 1, True)])
        body.append([r"Accuracy / vote / macro-F1", "", f(c["accuracy"], 1, True), f(c["vote_accuracy"], 1, True),
                     f(c["macro_f1"], 1, True), ""])
        body.append([rf"Cross-entropy (bit, chance {c['chance_bits']:.2f})", "", "", "", f(c["cross_entropy_bits"], 2), ""])
    out.append(macro("trClassification", "lrrrrr", r"Class & $n$ & P & R & F1 & AUC", body))
    body = []
    for k in T:
        g = T[k]["reading"]
        body.append([T[k]["label"], str(len(g["reads"])), f(g["mean_share"], 1, True), f(g["three"], 1, True),
                     f"{g['final']} of {g['letters']}", f(g["auc"], 2)])
    out.append(macro("trReading", "lrrrrr", r"Chirps & Reads & Letters right (\%) & Reads with $\geq$ 3 letters (\%) & "
                     r"Stable reading at the end & AUROC", body))
    body = []
    for s, lbl in STAGES:
        body.append([lbl] + [x for k in T for x in (f(np.nanmedian(T[k]["timing"]["ms"][s]), 1),
                                                    f(np.nanquantile(T[k]["timing"]["ms"][s], 0.9), 1))])
    body.append(["Radar latency (fitted)"] + [x for k in T for x in (f(T[k]["timing"]["radar_latency_ms"], 0), "")])
    head = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{T[k]['label']}}}" for k in T) + r" \\" + "\n" + \
        "Stage (ms) & " + " & ".join("Median & 90\\,\\%" for _ in T)
    out.append(macro("trTiming", "l" + "r" * 2 * len(T), head, body))
    from plots import policy_numbers
    P = {k: policy_numbers(r) for k, r in res["policies"].items()}
    names = list(next(iter(P.values())))
    body = [[r["label"]] + [f(P[k][m], 1) for m in names] for k, r in res["policies"].items()]
    out.append(macro("trPolicies", "l" + "r" * len(names), "Look policy & " + " & ".join(names), body))
    return "\n".join(out)
