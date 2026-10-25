"""
tables.py - the results as booktabs tables, one LaTeX macro each, for output/tables.tex: the method (\\obDataset,
\\obTracking, \\obScene, \\obSceneQuality, \\obDetection, \\obMotion, \\obSafety, \\obCamera, \\obClassification,
\\obCognition, \\obTiming), \\obCompare (the method against its baselines), \\obDoa (the DoA study), \\obLight (the light
monitor on every camera video).

    from tables import render_all
"""
from __future__ import annotations

import numpy as np

KINDS = (("obstacle", "Obstacles"), ("ball", "Ball"), ("person", "Person"), ("lamp", "Lamp"))
STAGES = (("range_doppler", "Range--Doppler"), ("infinirex_fdr", "Infini-Rex"), ("predict", "Prediction"),
          ("place", "Placing in the world"), ("track", "Association + KalmanCQR"), ("material", "M$^2$-SRC"),
          ("world", "World model"),
          ("choice", "Chirp choice"), ("total", "Total"))


def f(x, d=1, pct=False):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "--"
    return f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}"


def ci(lo_hi, d=0, pct=True):
    if not np.all(np.isfinite(np.asarray(lo_hi, float))):
        return "--"
    return f"{f(lo_hi[0], d, pct)}--{f(lo_hi[1], d, pct)}"


def name(run: str) -> str:
    return run.split("/")[-1].replace("_", r"\_")


def macro(cmd, spec, head, body):
    rows = "\n".join(" & ".join(r) + r" \\" if r != "mid" else r"\midrule" for r in body)
    return (f"\\newcommand{{\\{cmd}}}{{%\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
            f"{rows}\n\\bottomrule\n\\end{{tabular}}\n}}\n")


def render(res: dict, p: str = "ob") -> str:
    out = []
    body = [[name(r["name"]) + (" (test)" if r["test"] else ""), f(r["duration_s"], 1), str(r["looks"]),
             str(r["camera_frames"]), ", ".join(sorted(set(r["objects"]))), str(r["person_entries"]), str(r["light_events"]),
             f(r["travel_m"], 2), f(r["PQ"], 1, True)] for r in res["runs"]]
    out.append(macro(p + "Dataset", "lrrrlrrrr", r"Run & Seconds & Radar looks & Camera frames & Objects & Person entries & "
                     r"Light events & Travel (m) & Scene PQ (\%)", body))

    tr = res["tracking"]
    views = [v for v in ("look", "live") if v in tr]
    lbl = {"look": "Per look", "live": "Real time"}
    keys = [("HOTA", "hota", "HOTA"), ("DetA", "hota", "DetA"), ("AssA", "hota", "AssA"), ("LocA", "hota", "LocA"),
            ("MOTA", "clear", "MOTA"), ("IDF1", "idf1", "IDF1")]
    body = [[n] + [f(tr[v][g][m], 1, True) for v in views] for n, g, m in keys]
    body += [[f"MOTP, {k}: centre error (m)"] + [f(v["mean_m"], 3)] + ["--"] * (len(views) - 1) for k, v in tr["motp"].items()]
    body += [[n] + [f(tr[v]["gospa"][m], d) for v in views] for n, m, d in
             (("GOSPA (m, $c$ = 0.5 m)", "gospa", 3), ("OSPA (m)", "ospa", 3), ("Localisation RMS (m)", "loc_rms", 3),
              ("Missed per look", "missed", 2), ("False per look", "false", 2), ("Track switches", "switches", 1),
              ("Trajectory GOSPA (m)", "t_gospa", 3))]
    body += [[n] + [str(tr[v]["clear"][m]) for v in views] for n, m in
             (("ID switches", "IDSW"), ("False positives", "FP"), ("Misses", "FN"), ("Ground truth", "GT"))]
    body.append("mid")
    for k in tr["kinds"]:
        cells = []
        for v in views:
            o = tr[v]["objects"][k]
            cells.append(f"{f(o['tracked'], 0, True)}\\,\\% / {o['ids']} / {f(o['longest'], 0, True)}\\,\\%" if o["frames"] else "--")
        body.append([f"{k.capitalize()}: tracked / IDs / longest ID"] + cells)
    out.append(macro(p + "Tracking", "l" + "r" * len(views), "Metric (measured boxes) & " + " & ".join(lbl[v] for v in views),
                     body))

    u, s = res["understanding"], res["separation"]
    body = []
    for k, lbl in KINDS:
        if k in u["kinds"]:
            q = u["kinds"][k]
            body.append([lbl, str(q["tp"] + q["fn"]), f(q["recall"], 1, True), ci(q["recall_ci"]), f(q["sq"], 1, True),
                         f(q["material_acc"], 1, True), f(q["person_acc"], 1, True)])
    body += ["mid", ["All objects", str(u["TP"] + u["FN"]), f(u["TP"] / max(u["TP"] + u["FN"], 1), 1, True), "--",
                     f(u["SQ"], 1, True), "--", "--"]]
    out.append(macro(p + "Scene", "lrrcrrr", r"Object & Truth looks & Found as its kind (\%) & 95\,\% CI & "
                     r"Located SQ (\%) & Material (\%) & Person label (\%)", body) +
               macro(p + "SceneQuality", "lrc", r"Score & Value (\%) & 95\,\% CI",
                     [["Scene PQ = SQ $\\times$ RQ", f(u["PQ"], 1, True), ci(u["PQ_ci"], 1)],
                      ["Recognition RQ", f(u["RQ"], 1, True), "--"], ["Localisation SQ", f(u["SQ"], 1, True), "--"],
                      ["Scene PQ with kind and material", f(u["PQ_full"], 1, True), ci(u["PQ_full_ci"], 1)],
                      ["Close pairs separated", f(s["rate"], 1, True) + f" ({s['separated']}/{s['pairs']})", ci(s["rate_ci"], 1)],
                      ["Close pairs merged", str(s["merged"]), "--"]]))

    d, g, b = res["detection"], res["regions"], res["boxes"]
    cov = g["coverage"]
    body = [["FDR (Infini-Rex)", f(d["fdr"], 3), f"{f(d['fdr_ci'][0], 3)}--{f(d['fdr_ci'][1], 3)}",
             f"$\\le$ {f(d['alpha'], 2)}"],
            ["Detections per look", f(d["per_look"], 1), "--", "--"]]
    body += [[f"KalmanCQR coverage, {lbl}", f(cov[h][0], 3), f"{f(cov[h][1], 3)}--{f(cov[h][2], 3)}", f(g["nominal"], 2)]
             for h, lbl in (("step", "next look"), ("0.5", "0.5 s"), ("1.0", "1 s"))]
    body += [[f"ACI level $\\varepsilon_t$ at the end, {lbl}", f(g["alpha_t"][h], 3), "--", f(g["alpha"], 2)]
             for h, lbl in (("step", "next look"), ("0.5", "0.5 s"), ("1.0", "1 s"))]
    body += [[f"Runs test of the misses ($p$), {lbl}", f(g["runs_p"][h], 2), "--", "$>$ 0.05"]
             for h, lbl in (("step", "next look"), ("0.5", "0.5 s"), ("1.0", "1 s"))]
    body += [[f"Next-look region, {k}: range / across (cm, median)",
              f"{f(100 * np.median(v[:, 0]), 0)} / {f(100 * np.median(v[:, 1]), 0)}", "--", "--"] for k, v in g["widths"].items()]
    body += [["Obstacle box IoU (median)", f(b["iou_median"], 2), "--", "--"],
             ["Obstacle inside its box", (f(b["contains"], 1, True) + r"\,\%" if np.isfinite(b["contains"]) else "--"), ci(b["contains_ci"]), "--"]]
    out.append(macro(p + "Detection", "lrcc", r"Quantity & Value & 95\,\% CI & Nominal", body))

    mo = res["motion"]
    body = []
    for r in mo["runs"]:
        st = r["states_s"]
        tot = max(sum(st.values()), 1e-9)
        body.append([name(r["take"]), f(r["travel_m"], 3), f(r["speed_mm_s"], 1), str(r["stops"]), f(r["stop_mm"], 1),
                     f(r["drift_mm"], 1), f(st.get("wait", 0) / tot, 0, True),
                     f((st.get("cruise", 0) + st.get("approach", 0) + st.get("return", 0)) / tot, 0, True)])
    sc = mo["rail_scale"]
    if sc:
        body += ["mid", [r"Rail scale (radar)", f(sc["scale"], 4), "", "", f(sc["residual_rms_mm"], 2),
                         f(sc["home_repeat_mm"], 2), "", ""]]
    out.append(macro(p + "Motion", "lrrrrrrr", r"Run & Travel (m) & Speed (mm/s) & Stops & Stop dist.\ (mm) & "
                     r"Static drift (mm) & Waiting (\%) & Driving (\%)", body))

    w, li = res["waits"], res["light"]
    med = lambda x: f(np.median(x), 2) if len(x) else "--"
    body = [["Waits started while a person was there", f"{w['justified']}/{w['onsets']}", ci(w["precision_ci"])],
            ["Person entries that made it wait", f"{w['served']}/{w['entries']}", ci(w["recall_ci"])],
            ["Person entry $\\to$ wait (s, median)", med(w["latency_s"]), "--"],
            ["Wait $\\to$ rail still (s, median)", med(w["still_s"]), "--"],
            ]
    out.append(macro(p + "Safety", "lrc", r"Event & Value & 95\,\% CI (\%)", body))

    share = lambda v, n: f"{f(v, 0, True)}\\,\\% of {n}" if n else "none"
    body = [["Light onsets that were real", share(li["precision"], li["onsets"]), ci(li["precision_ci"])],
            ["Real light events detected", share(li["recall"], li["truth"]), ci(li["recall_ci"])],
            ["Light on $\\to$ detected (s, median)", med(li["latency_s"]), "--"], "mid"]
    for key, lbl_ in (("dark", "Letters read right, no light"), ("lit_accepted", "Letters read right, light, kept"),
                      ("lit_rejected", "Letters read right, light, filtered out")):
        m = li[key]
        body.append([lbl_ + f" ({m[3]} reads)", f(m[0], 1, True) + r"\,\%" if m[3] else "--", ci((m[1], m[2]))])
    body.append(["Reads with $\\ge$ 3 letters right (kept)", "--", ci(li["three_accepted"])])
    body.append(["Reader confidence vs.\\ correctness (AUROC)", f(li["selective"]["auc"], 2), "--"])
    body.append("mid")
    for r in res["camera"]["runs"]:
        body.append([f"Stable reading at the end, {name(r['take'])}", f"{r['final_correct']}/{r['letters']} letters", "--"])
    out.append(macro(p + "Camera", "lrc", r"Camera & Value & 95\,\% CI (\%)", body))

    c = res["classification"]
    body = [[h.capitalize(), str(c[h]["n"]), f(c[h]["acc"], 1, True), ci(c[h]["ci"]), f(c[h]["classified"], 1, True)]
            for h in ("material", "motion")]
    out.append(macro(p + "Classification", "lrrcr", r"Head & Looks with a label & Correct (\%) & 95\,\% CI & "
                     r"Labelled (\%)", body))

    cg = res["cognition"]
    n = max(cg["looks"], 1)
    body = [[p.replace("+", " + "), str(k), f(k / n, 1, True)] for p, k in sorted(cg["plans"].items(), key=lambda kv: -kv[1])]
    body += ["mid", ["Longest gap between search looks (ms)", f(1e3 * cg["gap_max_s"], 0), "--"],
             ["Gaps $>$ 1.5 frames (blind)", "--", f(cg["blind_share"], 1, True)],
             ["Information (bit/s)", f(cg["info_bits_per_s"], 1), "--"], ["Frames skipped (late)", str(cg["skipped"]), "--"]]
    out.append(macro(p + "Cognition", "lrr", r"Chirps processed per look & Looks & Share (\%)", body))

    lat = res["latency"]
    body = [[lbl, f(lat["stages"][k][0], 1), f(lat["stages"][k][1], 1)] for k, lbl in STAGES if k in lat["stages"]]
    body += ["mid", ["Frame $\\to$ result", f(lat["result_ms"][0], 0), f(lat["result_ms"][1], 0)],
             ["Result age at the camera", f(lat["camera_ms"][0], 0), f(lat["camera_ms"][1], 0)],
             ["Result $\\to$ planner decision", f(lat["decision_ms"][0], 1), f(lat["decision_ms"][1], 1)],
             ["Rail stop command $\\to$ still", f(1e3 * lat["rail_stop_s"][0], 0), f(1e3 * lat["rail_stop_s"][1], 0)]]
    out.append(macro(p + "Timing", "lrr", r"Stage (ms) & Median & 95\,\%", body))
    return "\n".join(out)


GROUPS = (("final", "Method (Infini-Rex)"), ("ctrex", "CT-Rex detector"), ("cfar", "OS-CFAR detector"),
          ("baseline", "Classical chain"))
COMPARE = (("Scene PQ (\\%)", lambda r: r["understanding"]["PQ"], True), ("HOTA (\\%)", lambda r: r["tracking"]["look"]["hota"]["HOTA"], True),
           ("DetA (\\%)", lambda r: r["tracking"]["look"]["hota"]["DetA"], True),
           ("AssA (\\%)", lambda r: r["tracking"]["look"]["hota"]["AssA"], True),
           ("MOTA (\\%)", lambda r: r["tracking"]["look"]["clear"]["MOTA"], True),
           ("IDF1 (\\%)", lambda r: r["tracking"]["look"]["idf1"]["IDF1"], True),
           ("GOSPA (m)", lambda r: r["tracking"]["look"]["gospa"]["gospa"], False),
           ("ID switches", lambda r: r["tracking"]["look"]["clear"]["IDSW"], None),
           ("Detections per look on a measured object", lambda r: f(r["detection"]["true_per_look"], 1), None),
           ("Detections per look on nothing known", lambda r: f(r["detection"]["false_per_look"], 1), None),
           ("FDR", lambda r: r["detection"]["fdr"], False),
           ("Obstacles found as static (\\%)", lambda r: r["understanding"]["kinds"]["obstacle"]["recall"], True),
           ("Ball found as a mover (\\%)", lambda r: r["understanding"]["kinds"]["ball"]["recall"], True),
           ("Person found as a mover or person (\\%)", lambda r: r["understanding"]["kinds"]["person"]["recall"], True),
           ("Material right (\\%)", lambda r: r["classification"]["material"]["acc"], True),
           ("Motion right (\\%)", lambda r: r["classification"]["motion"]["acc"], True),
           ("Person entries that made it wait", lambda r: f"{r['waits']['served']}/{r['waits']['entries']}", None))


def compare(res: dict) -> str:
    cols = [(k, lbl) for k, lbl in GROUPS if k in res]
    body = []
    for lbl, get, pct in COMPARE:
        row = [lbl]
        for k, _ in cols:
            try:
                v = get(res[k])
            except (KeyError, TypeError):
                v = None
            row.append("--" if v is None else v if isinstance(v, str) else str(v) if pct is None else f(v, 1 if pct else 3, bool(pct)))
        body.append(row)
    return macro("obCompare", "l" + "r" * len(cols), "Metric & " + " & ".join(lbl for _, lbl in cols), body)


def render_all(res: dict) -> str:
    out = ["% \\input{tables.tex} once, then \\obCompare and \\obDoa; the method: \\obDataset, \\obTracking, \\obScene, "
           "\\obSceneQuality, \\obDetection, \\obMotion, \\obSafety, \\obCamera, \\obClassification, \\obCognition, "
           "\\obTiming\n", compare(res)]
    if "final" in res:
        out.append(render(res["final"], "ob"))
    if res.get("doa"):
        out.append(doa(res["doa"]))
    if res.get("light_videos"):
        li = res["light_videos"]
        med = f(float(np.median(li["latency_s"])), 2) if len(li["latency_s"]) else "--"
        out.append(macro("obLight", "lr", r"Light monitor on every camera video & Value",
                         [["Runs", str(len(li["runs"]))], ["Onsets while a light was on", f"{li['real']}/{li['onsets']}"],
                          ["Annotated lights detected within 1 s", f"{li['found']}/{li['truth']}"],
                          ["Flashing reported while it flashed", f"{li['flashing_found']}/{li['flashing_truth']}"],
                          ["Light on $\\to$ onset (s, median)", med]]))
    return "\n".join(out)


def doa(d: dict) -> str:
    kinds = [k for k, _ in KINDS if any(k in v["recall"] for v in d.values())]
    body = [[v["label"] + (" (classical)" if v["classical"] else ""), f(v["detections"], 1)] +
            [f(v["recall"].get(k), 1, True) for k in kinds] + [f(v["fdr"], 3), f(v["ms"], 1)] for v in d.values()]
    head = "Method & Detections per look & " + " & ".join(f"{lbl} found (\\%)" for k, lbl in KINDS if k in kinds) + \
        " & False share & ms per look"
    return macro("obDoa", "l" + "r" * (len(kinds) + 3), head, body)
