"""
evaluate.py - leave-one-take-out on the training sessions and the test on the unseen session B, plus per-class scores,
distances to the decision boundary, learning curve, solver convergence and timing; writes predictions.csv and
tables.tex and saves the final model.

    python evaluate.py
"""
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from material_classification.classifier import ClarabelLasso, MaterialMotionModel, duality_gap, fista_steps, objective  # noqa: E402
from material_classification.dataset import HOP, RAW, TABLE, WINDOW, Radar, read_table, sessions  # noqa: E402
from material_classification.features import DETECT_DB  # noqa: E402

OUT = HERE / "output"
MODEL = HERE / "models" / "material_motion_model.json"
TEST_SESSION = "sessionB"
HEADS = ("motion", "material")
ORDER = {"motion": ["static", "rigid", "articulated", "distributed"], "material": ["metal", "dielectric", "waterlike"]}
ITERS = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000]
TIMING_TAKE = ("sessionA", "metal-static/gold8cm_aimed_take1")
BATCH = 27


def subset(t, mask):
    return {k: v[mask] for k, v in t.items()}


def objects(t):
    return t["detected"] & (t["motion"] != "none")


def columns(table, names):
    return np.stack([np.asarray(table[n], float) for n in names], axis=1)


def head_outputs(model, table):
    out = {}
    for head in HEADS:
        clf = getattr(model, head)
        labels, conf, res = clf.predict(columns(table, model.features[head]))
        full = np.full((len(labels), len(ORDER[head])), np.nan)
        for j, c in enumerate(clf.classes):
            full[:, ORDER[head].index(c)] = res[:, j]
        out[head] = labels, conf, full
    return out


def evaluate(train, test=None, solver="fista"):
    data = train if test is None else test
    n = len(data["take"])
    out = {}
    for head in HEADS:
        out[head + "_pred"] = np.full(n, "none", object)
        out[head + "_conf"] = np.zeros(n)
        out[head + "_res"] = np.full((n, len(ORDER[head])), np.nan)
    learn = objects(train)
    folds = [None] if test is not None else sorted(set(data["take"]))
    for take in folds:
        keep = learn if take is None else learn & (train["take"] != take)
        target = data["detected"] if take is None else data["detected"] & (data["take"] == take)
        if not target.any():
            continue
        model = MaterialMotionModel(solver=solver).fit(subset(train, keep))
        for head, (labels, conf, res) in head_outputs(model, subset(data, target)).items():
            out[head + "_pred"][target], out[head + "_conf"][target], out[head + "_res"][target] = labels, conf, res
    out["joint_pred"] = np.where(out["motion_pred"] == "none", "none", out["material_pred"] + "-" + out["motion_pred"])
    r = {**data, **out, "detected_object": objects(data)}
    for head in HEADS:
        r[head + "_distance"] = boundary_distance(r, head)
    return r


def boundary_distance(r, head):
    R = r[head + "_res"]
    d = np.full(len(R), np.nan)
    for i, c in enumerate(r[head]):
        if r["detected_object"][i] and c in ORDER[head]:
            t = ORDER[head].index(c)
            others = np.delete(R[i], t)
            if np.isfinite(R[i, t]) and np.isfinite(others).any():
                d[i] = (np.nanmin(others) - R[i, t]) / np.sqrt(2)
    return d


def summary(r):
    obj, empty, det = r["motion"] != "none", r["motion"] == "none", r["detected_object"]
    mo = r["motion"][det] == r["motion_pred"][det]
    ma = r["material"][det] == r["material_pred"][det]
    votes = [Counter(r["joint_pred"][r["take"] == t]).most_common(1)[0][0] == r["joint"][r["take"] == t][0]
             for t in dict.fromkeys(r["take"])]
    nums = {"detect": r["detected"][obj].mean(), "false_alarm": r["detected"][empty].mean(), "motion": mo.mean(),
            "material": ma.mean(), "both": (mo & ma).mean(), "joint_all": np.mean(r["joint"] == r["joint_pred"]),
            "takes_vote": np.mean(votes), "takes": len(votes), "windows": len(obj),
            **{f"{h}_f1": np.mean([c["f1"] for c in per_class(r, h)]) for h in HEADS}}
    lines = [f"windows {len(obj)} ({empty.sum()} empty room), detected {nums['detect'] * 100:.1f} %, false alarm "
             f"{nums['false_alarm'] * 100:.1f} % (threshold {DETECT_DB:.0f} dB)",
             f"motion {nums['motion'] * 100:.1f} %, material {nums['material'] * 100:.1f} %, both "
             f"{nums['both'] * 100:.1f} %, joint label {nums['joint_all'] * 100:.1f} %, takes by vote "
             f"{nums['takes_vote'] * 100:.0f} % of {len(votes)}"]
    return lines, nums


def take_table(r):
    rows = []
    for take in dict.fromkeys(r["take"]):
        m = r["take"] == take
        vote = Counter(r["joint_pred"][m]).most_common(1)[0][0]
        rows.append({"take": take, "true": r["joint"][m][0], "windows": int(m.sum()),
                     "correct": float(np.mean(r["joint"][m] == r["joint_pred"][m])), "vote": vote,
                     "vote_ok": int(vote == r["joint"][m][0])})
    return rows


def roc(score, positive):
    order = np.argsort(-score, kind="stable")
    s, y = score[order], positive[order]
    cut = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tpr = np.r_[0, np.cumsum(y)[cut] / max(y.sum(), 1)]
    fpr = np.r_[0, np.cumsum(~y)[cut] / max((~y).sum(), 1)]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def per_class(r, head):
    det = r["detected_object"]
    true, pred = r[head][det], r[head + "_pred"][det]
    score = np.nan_to_num(-r[head + "_res"][det], nan=-1e9)
    rows = []
    for j, c in enumerate(ORDER[head]):
        if not np.any(true == c):
            continue
        tp, fp, fn = np.sum((pred == c) & (true == c)), np.sum((pred == c) & (true != c)), np.sum((pred != c) & (true == c))
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        fpr, tpr, auc = roc(score[:, j], true == c)
        rows.append({"class": c, "n": int(np.sum(true == c)), "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / max(precision + recall, 1e-12), "auc": auc, "fpr": fpr, "tpr": tpr})
    return rows


def f1_bootstrap(r, head, n=200, seed=0):
    rng = np.random.default_rng(seed)
    det = r["detected_object"]
    takes = np.array(sorted(set(r["take"][det])))
    rows = {t: np.flatnonzero(det & (r["take"] == t)) for t in takes}
    out = {c: [] for c in ORDER[head]}
    for _ in range(n):
        idx = np.concatenate([rows[t] for t in rng.choice(takes, size=len(takes))])
        true, pred = r[head][idx], r[head + "_pred"][idx]
        for c in ORDER[head]:
            if np.any(true == c):
                tp = np.sum((pred == c) & (true == c))
                out[c].append(2 * tp / max(np.sum(pred == c) + np.sum(true == c), 1))
    return {c: np.array(v) for c, v in out.items() if v}


def learning_curve(train, repeats=3, seed=0):
    rng = np.random.default_rng(seed)
    obj = objects(train)
    takes = {c: sorted(set(train["take"][obj & (train["joint"] == c)])) for c in set(train["joint"][obj])}
    k_max = max(1, max(len(t) for t in takes.values()) - 1)
    lc = {"k": list(range(1, k_max + 1)), "chance": {h: 1 / len(set(train[h][obj])) for h in HEADS}}
    for h in HEADS:
        lc[h] = {"train": [], "test": []}
    for k in lc["k"]:
        runs = {h: {"train": [], "test": []} for h in HEADS}
        for _ in range(repeats):
            hits = {h: {"train": [], "test": []} for h in HEADS}
            for held in sorted(set(train["take"][obj])):
                chosen = [c for t in takes.values() for c in rng.permutation([x for x in t if x != held])[:k]]
                tr = obj & np.isin(train["take"], chosen)
                model = MaterialMotionModel().fit(subset(train, tr))
                sample = np.zeros(len(obj), bool)
                sample[rng.permutation(np.flatnonzero(tr))[:30]] = True
                for part, m in (("train", sample), ("test", obj & (train["take"] == held))):
                    p = model.predict(subset(train, m))
                    for h in HEADS:
                        hits[h][part].extend(p[h] == train[h][m])
            for h in HEADS:
                for part in ("train", "test"):
                    runs[h][part].append(np.mean(hits[h][part]))
        for h in HEADS:
            for part in ("train", "test"):
                lc[h][part].append((np.mean(runs[h][part]), np.std(runs[h][part])))
    return lc


def decide(clf, Y, A):
    res = np.stack([np.linalg.norm(Y - clf.D[:, clf.labels == c] @ A[clf.labels == c], axis=0) for c in clf.classes], 1)
    return np.array(clf.classes, object)[np.argmin(res, axis=1)]


def accuracy_by_iteration(train):
    obj = objects(train)
    hits = {h: {k: [] for k in ITERS} for h in HEADS}
    for take in sorted(set(train["take"][obj])):
        model = MaterialMotionModel().fit(subset(train, obj & (train["take"] != take)))
        held = subset(train, obj & (train["take"] == take))
        for head in HEADS:
            clf = getattr(model, head)
            Y = clf._scale(columns(held, model.features[head])).T
            for k, A in enumerate(fista_steps(clf.D, Y, clf.lam), 1):
                if k in hits[head]:
                    hits[head][k].extend(decide(clf, Y, A) == held[head])
                if k == ITERS[-1]:
                    break
    return {h: [np.mean(hits[h][k]) for k in ITERS] for h in HEADS}


def convergence(model, train, n_sample=200, seed=0):
    rng = np.random.default_rng(seed)
    obj = objects(train)
    pick = np.zeros(len(obj), bool)
    pick[rng.choice(np.flatnonzero(obj), size=min(n_sample, obj.sum()), replace=False)] = True
    sample = subset(train, pick)
    out, Ys = {}, {}
    for head in HEADS:
        clf = getattr(model, head)
        D, lam = clf.D, clf.lam
        Y = Ys[head] = clf._scale(columns(sample, model.features[head])).T
        A_star = ClarabelLasso(D, lam)(Y)
        f_star = objective(D, Y, A_star, lam)
        rel_obj, rel_gap = [], []
        for k, A in enumerate(fista_steps(D, Y, lam), 1):
            if k in ITERS:
                rel_obj.append(np.median((objective(D, Y, A, lam) - f_star) / f_star))
                rel_gap.append(np.median(duality_gap(D, Y, A, lam) / f_star))
            if k == ITERS[-1]:
                break
        out[head] = {"objective": rel_obj, "gap": rel_gap,
                     "clarabel_gap": float(np.median(duality_gap(D, Y, A_star, lam) / f_star))}
    out["level_set"] = level_set_case(model.material, Ys["material"])
    return out


def level_set_case(clf, Y, max_cos=0.9):
    A_star = ClarabelLasso(clf.D, clf.lam)(Y)
    w = int(np.argmax(A_star.max(axis=0)))
    a = A_star[:, w]
    i = int(np.argmax(a))
    pull = clf.D.T @ (Y[:, w] - clf.D[:, i] * a[i])
    pull[np.abs(clf.D.T @ clf.D[:, i]) > max_cos] = -np.inf
    j = int(np.argmax(pull))
    pair = clf.D[:, [i, j]]
    y_pair = Y[:, w] - clf.D @ a + pair @ a[[i, j]]
    free = np.linalg.solve(pair.T @ pair, pair.T @ y_pair - clf.lam)
    path = [(0.0, 0.0)]
    for k, A in enumerate(fista_steps(pair, y_pair[:, None], clf.lam), 1):
        path.append((A[0, 0], A[1, 0]))
        if np.hypot(A[0, 0] - a[i], A[1, 0] - a[j]) < 1e-5 or k == 300:
            break
    return {"D": clf.D, "y": Y[:, w], "lam": clf.lam, "a_star": a, "i": i, "j": j, "free": free,
            "path": np.array(path), "labels": (clf.labels[i], clf.labels[j])}


def timing(model, train, repeats=50):
    radar = Radar()
    session, take = TIMING_TAKE
    radar.background(RAW / session)
    stack_frames = radar.frames(RAW / session / f"{take}.npy", 0, WINDOW + 4 * HOP)
    t0 = time.perf_counter()
    radar.frames(RAW / session / f"{take}.npy", 0, 100)
    frame_ms = (time.perf_counter() - t0) / 100 * 1e3
    t0 = time.perf_counter()
    windows = [radar.fx.window(stack_frames[s:s + WINDOW], radar.background(RAW / session)) for s in range(0, 4 * HOP + 1, HOP)]
    window_ms = (time.perf_counter() - t0) / len(windows) * 1e3
    obj = objects(train)
    rows = [subset(train, np.flatnonzero(obj)[[i]]) for i in range(repeats)]
    clarabel = MaterialMotionModel(solver="clarabel").fit(subset(train, obj))
    clarabel.predict(rows[0])
    out = {"frame_ms": frame_ms, "window_ms": window_ms, "hop_frames": HOP,
           "frame_period_ms": radar.params.frame_period_s * 1e3}
    for name, m in (("fista_ms", model), ("clarabel_ms", clarabel)):
        t0 = time.perf_counter()
        for row in rows:
            m.predict(row)
        out[name] = (time.perf_counter() - t0) / repeats * 1e3
    batch = subset(train, np.flatnonzero(obj)[:BATCH])
    t0 = time.perf_counter()
    model.predict(batch)
    out["fista_batch_ms"] = (time.perf_counter() - t0) / BATCH * 1e3
    return out


def solver_study(train, cv):
    t0 = time.perf_counter()
    clarabel = evaluate(train, solver="clarabel")
    det = cv["detected_object"]
    return {"clarabel": clarabel, "clarabel_s": time.perf_counter() - t0,
            "agree": {h: np.mean(cv[h + "_pred"][det] == clarabel[h + "_pred"][det]) for h in HEADS},
            "iterations": accuracy_by_iteration(train)}


def write_predictions(results):
    keys = ["session", "take", "t_start_s", "detected", "joint", "joint_pred"]
    for h in HEADS:
        keys += [h, h + "_pred", h + "_conf", h + "_distance"]
    with open(OUT / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation", *keys])
        for name, r in results.items():
            for i in range(len(r["take"])):
                writer.writerow([name] + [round(float(r[k][i]), 4) if isinstance(r[k][i], (float, np.floating))
                                          else r[k][i] for k in keys])


def pct(x):
    return f"{100 * x:.1f}"


def tables(train, test, cv, tb, solvers, conv, times):
    def macro(name, body):
        return f"\\newcommand{{\\{name}}}{{%\n{body}}}\n\n"

    sessions_rows = ""
    for data, use in ((train, "training"), (test, "test")):
        if data is None:
            continue
        for s in dict.fromkeys(data["session"]):
            m = data["session"] == s
            sessions_rows += (f"{s} & {use} & {len(set(data['take'][m]))} & {m.sum()} & "
                              f"{np.sum(m & objects(data))} & {np.sum(m & (data['motion'] == 'none'))} \\\\\n")
    out = macro("mcDataset", "\\begin{tabular}{llrrrr}\n\\toprule\nSession & Use & Takes & Windows & Objects & Empty \\\\\n"
                "\\midrule\n" + sessions_rows + "\\bottomrule\n\\end{tabular}\n")

    cols = [("Leave one take out", cv)] + ([("Test on B", tb)] if tb is not None else [])
    stats = [summary(r)[1] for _, r in cols]
    rows = [("Object detected", "detect"), ("False alarm", "false_alarm"), ("Motion", "motion"),
            ("Material", "material"), ("Motion and material", "both"), ("Joint label", "joint_all"),
            ("Macro-F1 motion", "motion_f1"), ("Macro-F1 material", "material_f1"), ("Takes by vote", "takes_vote")]
    body = "".join(f"{name} & " + " & ".join(pct(s[k]) for s in stats) + " \\\\\n" for name, k in rows)
    out += macro("mcResults", f"\\begin{{tabular}}{{l{'r' * len(cols)}}}\n\\toprule\n & "
                 + " & ".join(c for c, _ in cols) + " \\\\\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n")

    body = ""
    for h in HEADS:
        body += f"\\multicolumn{{{2 + 4 * len(cols)}}}{{l}}{{\\textit{{{h} head}}}} \\\\\n"
        tabs = [{c["class"]: c for c in per_class(r, h)} for _, r in cols]
        for c in ORDER[h]:
            if c in tabs[0]:
                body += f"{c} & {tabs[0][c]['n']} & " + " & ".join(
                    " & ".join(pct(t[c][k]) if c in t else "--" for k in ("precision", "recall", "f1", "auc"))
                    for t in tabs) + " \\\\\n"
    group = " & ".join(f"\\multicolumn{{4}}{{c}}{{{c}}}" for c, _ in cols)
    sub = " & ".join(["P & R & F1 & AUC"] * len(cols))
    out += macro("mcPerClass", f"\\begin{{tabular}}{{lr{'rrrr' * len(cols)}}}\n\\toprule\n & & {group} \\\\\n"
                 f"Class & $n$ & {sub} \\\\\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n")

    cl = solvers["clarabel"]
    det = cv["detected_object"]
    acc = {name: {h: np.mean(r[h][det] == r[h + "_pred"][det]) for h in HEADS} for name, r in (("f", cv), ("c", cl))}
    k300 = ITERS.index(300)
    out += macro("mcSolvers", "\\begin{tabular}{lrr}\n\\toprule\n & FISTA (300 it.) & Clarabel \\\\\n\\midrule\n"
                 f"Motion accuracy (\\%) & {pct(acc['f']['motion'])} & {pct(acc['c']['motion'])} \\\\\n"
                 f"Material accuracy (\\%) & {pct(acc['f']['material'])} & {pct(acc['c']['material'])} \\\\\n"
                 f"Same prediction motion / material (\\%) & \\multicolumn{{2}}{{c}}{{{pct(solvers['agree']['motion'])} / "
                 f"{pct(solvers['agree']['material'])}}} \\\\\n"
                 f"Relative objective gap & {conv['motion']['objective'][k300]:.1e} & 0 \\\\\n"
                 f"Relative duality gap & {conv['motion']['gap'][k300]:.1e} & {conv['motion']['clarabel_gap']:.1e} \\\\\n"
                 f"Time per window, one window (ms) & {times['fista_ms']:.1f} & {times['clarabel_ms']:.1f} \\\\\n"
                 f"Time per window, batch of {BATCH} (ms) & {times['fista_batch_ms']:.1f} & -- \\\\\n"
                 "\\bottomrule\n\\end{tabular}\n")

    per_window = {s: times["hop_frames"] * times["frame_ms"] + times["window_ms"] + times[s + "_ms"]
                  for s in ("fista", "clarabel")}
    budget = times["hop_frames"] * times["frame_period_ms"]
    out += macro("mcTiming", "\\begin{tabular}{lrr}\n\\toprule\nStep & FISTA (ms) & Clarabel (ms) \\\\\n\\midrule\n"
                 f"Frame processing ({times['hop_frames']} new frames) & \\multicolumn{{2}}{{c}}"
                 f"{{{times['hop_frames'] * times['frame_ms']:.1f}}} \\\\\n"
                 f"Features of the window & \\multicolumn{{2}}{{c}}{{{times['window_ms']:.1f}}} \\\\\n"
                 f"Classification (both heads) & {times['fista_ms']:.1f} & {times['clarabel_ms']:.1f} \\\\\n"
                 f"\\midrule\nTotal per window & {per_window['fista']:.1f} & {per_window['clarabel']:.1f} \\\\\n"
                 f"Available (one window every {budget / 1e3:.1f} s) & \\multicolumn{{2}}{{c}}{{{budget:.0f}}} \\\\\n"
                 "\\bottomrule\n\\end{tabular}\n")
    (OUT / "tables.tex").write_text("% \\input{tables.tex} once, then \\mcDataset, \\mcResults, \\mcPerClass, "
                                    "\\mcSolvers, \\mcTiming where the table belongs\n\n" + out, encoding="utf-8")


def run():
    table = read_table(TABLE)
    names = [s.name for s in sessions()]
    train = subset(table, table["session"] != TEST_SESSION)
    test = subset(table, table["session"] == TEST_SESSION) if TEST_SESSION in names else None

    cv = evaluate(train)
    tb = evaluate(train, test) if test is not None else None
    model = MaterialMotionModel().fit(subset(train, objects(train)))
    model.settings = {"name": "M2-SRC", "trained_on": sorted(set(train["session"])), "window": WINDOW, "hop": HOP}
    model.save(MODEL)

    results = {"crossval": cv, **({"test": tb} if tb is not None else {})}
    for name, r in results.items():
        print(f"[{name}] " + " | ".join(summary(r)[0]))
    print("learning curve ...")
    curve = learning_curve(train)
    print("solvers ...")
    solvers = solver_study(train, cv)
    conv = convergence(model, train)
    times = timing(model, train)
    write_predictions(results)
    tables(train, test, cv, tb, solvers, conv, times)
    return {"train": train, "test": test, "cv": cv, "tb": tb, "model": model, "curve": curve, "solvers": solvers,
            "convergence": conv, "timing": times}


if __name__ == "__main__":
    run()
