"""
plots.py - every thesis figure of the experiment as a PDF in output/figures/ (ARTIFACT-TeX style), and the live
dashboard used by run_realtime_sensing.py.

    python run_all.py        (draws all figures; this file is imported, not run)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from visualization import style
from material_classification.evaluate import HEADS, ITERS, ORDER, f1_bootstrap, objects, per_class
from material_classification.features import MATERIAL_FEATURES, MOTION_FEATURES
from visualization.style import (ACCENT, DOUBLE, GOOD, INK, LOSS, MATERIAL_COLORS, MARKERS, MOTION_COLORS, MUTED,
                                           PRIMARY, SEQUENTIAL, bold, tex, title)

COLORS = {"motion": MOTION_COLORS, "material": MATERIAL_COLORS}
EVALS = (("cv", "leave one take out", PRIMARY), ("tb", "test on session B", ACCENT))


def _labels(names):
    return [tex(n) for n in names]


def _matrix(ax, true, pred, rows, cols, name):
    count = np.array([[np.sum((true == r) & (pred == c)) for c in cols] for r in rows], float)
    n = count.sum(axis=1, keepdims=True)
    share = count / np.maximum(n, 1)
    im = ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.tick_params(length=0, top=False, right=False)
    ax.set_xticks(range(len(cols)), _labels(cols), rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), [tex(f"{r} ({int(k)})") for r, k in zip(rows, n[:, 0])])
    for i in range(len(rows)):
        for j in range(len(cols)):
            if count[i, j]:
                text = f"{share[i, j] * 100:.0f}" if share[i, j] >= 0.005 else "$<$1"
                ax.text(j, i, text, ha="center", va="center", fontsize=6.5,
                        color="white" if share[i, j] > 0.55 else INK)
    title(ax, f"{name}: {np.mean(true == pred) * 100:.1f}\\,%")
    return im


def confusion(res, pdf):
    evals = [(key, name) for key, name, _ in EVALS if res[key] is not None]
    fig, axs = plt.subplots(len(evals), 2, figsize=(DOUBLE, 2.9 * len(evals)), layout="constrained", squeeze=False,
                            gridspec_kw={"width_ratios": [4, 3]})
    for row, (key, name) in zip(axs, evals):
        r = res[key]
        det = r["detected_object"]
        for ax, head in zip(row, HEADS):
            present = [c for c in ORDER[head] if np.any(r[head][det] == c)]
            im = _matrix(ax, r[head][det], r[head + "_pred"][det], present, present, f"{head.capitalize()}, {name}")
        row[0].set_ylabel("True class (windows)")
    for ax in axs[-1]:
        ax.set_xlabel("Predicted class")
    fig.colorbar(im, ax=axs, shrink=0.6, label="Share of the true class")
    style.save(pdf, fig)

    fig, axs = plt.subplots(1, len(evals), figsize=(DOUBLE, 3.6), layout="constrained", squeeze=False)
    for ax, (key, name) in zip(axs[0], evals):
        r = res[key]
        rows = sorted(set(r["joint"]), key=lambda s: (s == "none", s))
        pred = np.where(np.isin(r["joint_pred"], rows), r["joint_pred"], "other")
        cols = rows + (["other"] if np.any(pred == "other") else [])
        im = _matrix(ax, r["joint"], pred, rows, cols, f"Joint label, {name}")
        ax.set_xlabel("Predicted label")
    axs[0, 0].set_ylabel("True label (windows)")
    fig.colorbar(im, ax=axs, shrink=0.7, label="Share of the true label")
    style.save(pdf, fig)


def scores(res, pdf):
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), layout="constrained", gridspec_kw={"width_ratios": [4, 3]})
    evals = [e for e in EVALS if res[e[0]] is not None]
    for ax, head in zip(axs, HEADS):
        x = np.arange(len(ORDER[head]))
        for k, (key, name, color) in enumerate(evals):
            f1 = f1_bootstrap(res[key], head)
            classes = [c for c in ORDER[head] if c in f1]
            pos = [ORDER[head].index(c) + (k - (len(evals) - 1) / 2) * 0.26 for c in classes]
            bp = ax.boxplot([f1[c] * 100 for c in classes], positions=pos, widths=0.16, patch_artist=True,
                            showfliers=False, medianprops={"color": "white", "lw": 1.1},
                            whiskerprops={"color": INK, "lw": 0.6}, capprops={"color": INK, "lw": 0.6},
                            boxprops={"edgecolor": INK, "lw": 0.5})
            for box in bp["boxes"]:
                box.set_facecolor(color)
            observed = {c["class"]: c["f1"] for c in per_class(res[key], head)}
            ax.plot(pos, [observed[c] * 100 for c in classes], "D", ms=2.5, color=INK, zorder=4)
        ax.set_xticks(x, _labels(ORDER[head]))
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.set_ylim(0, 108)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel(r"$F_1$-Score (\%)")
        ax.grid(axis="x", visible=False)
        title(ax, f"{head.capitalize()} head")
    handles = [Patch(facecolor=color, edgecolor=INK, lw=0.5, label=tex(name)) for _, name, color in evals]
    handles.append(Line2D([], [], ls="", marker="D", ms=2.5, color=INK, label="observed"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, title="200 resamplings of the takes",
               title_fontsize=7)
    style.save(pdf, fig)


def roc(res, pdf):
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 3.1), layout="constrained")
    for ax, head in zip(axs, HEADS):
        ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=0.8)
        curves = {key: {c["class"]: c for c in per_class(res[key], head)} for key, _, _ in EVALS if res[key] is not None}
        for c in ORDER[head]:
            if c not in curves["cv"]:
                continue
            aucs = []
            for key, ls in (("cv", "-"), ("tb", "--")):
                if key in curves and c in curves[key]:
                    ax.plot(curves[key][c]["fpr"], curves[key][c]["tpr"], color=COLORS[head][c], ls=ls, lw=1.1)
                    aucs.append(f"{curves[key][c]['auc']:.2f}")
            ax.plot([], [], color=COLORS[head][c], label=tex(f"{c} ({' / '.join(aucs)})"))
        ax.plot([], [], color=INK, ls="-", label="leave one take out")
        if "tb" in curves:
            ax.plot([], [], color=INK, ls="--", label="test on session B")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", title=tex("class (AUC)"), title_fontsize=7)
        title(ax, f"{head.capitalize()} head, one vs. rest")
    style.save(pdf, fig)


def margin(res, pdf):
    r = res["cv"]
    det = r["detected_object"]
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 5.6), layout="constrained")
    for col, head in enumerate(HEADS):
        R = r[head + "_res"][det]
        true = r[head][det]
        t_idx = np.array([ORDER[head].index(c) for c in true])
        r_true = R[np.arange(len(R)), t_idx]
        r_other = np.array([np.nanmin(np.delete(row, t)) for row, t in zip(R, t_idx)])
        ax = axs[0, col]
        top = np.nanmax([r_true.max(), r_other.max()]) * 1.03
        ax.fill_between([0, top], [0, top], top, color=LOSS, alpha=0.07, lw=0)
        ax.fill_between([0, top], 0, [0, top], color=GOOD, alpha=0.10, lw=0)
        ax.plot([0, top], [0, top], color=INK, lw=0.8)
        for c in ORDER[head]:
            m = true == c
            if m.any():
                ax.scatter(r_other[m], r_true[m], s=5, color=COLORS[head][c], marker=MARKERS[c], alpha=0.7,
                           lw=0.3, label=tex(c))
        ax.text(0.04 * top, 0.96 * top, "wrong", va="top")
        ax.text(0.5 * top, 0.04 * top, "correct")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_xlabel(r"Residual of the best other class $\min_{c \neq t} \| y - D a_c \|$")
        ax.set_ylabel(r"Residual of the true class $\| y - D a_t \|$")
        ax.legend(loc="lower left", markerscale=2)
        title(ax, f"{head.capitalize()} head: decision boundary")

        d = r[head + "_distance"][det]
        d = d[np.isfinite(d)]
        ax = axs[1, col]
        bins = np.linspace(d.min(), d.max(), 40)
        ax.hist(d[d >= 0], bins=bins, color=GOOD, label=tex(f"correct ({np.mean(d >= 0) * 100:.1f} %)"))
        ax.hist(d[d < 0], bins=bins, color=LOSS, label=tex(f"wrong ({np.mean(d < 0) * 100:.1f} %)"))
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlabel(r"Signed distance to the boundary $(r_{\mathrm{other}} - r_{\mathrm{true}})/\sqrt{2}$")
        ax.set_ylabel("Windows")
        ax.legend(loc="upper left")
        title(ax, f"{head.capitalize()} head: margin")
    style.save(pdf, fig)


def confidence(res, pdf):
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 5.4), layout="constrained", gridspec_kw={"width_ratios": [4, 3]})
    for col, head in enumerate(HEADS):
        ax = axs[0, col]
        for key, name, color in EVALS:
            r = res[key]
            if r is None:
                continue
            det = r["detected_object"]
            ok = r[head][det] == r[head + "_pred"][det]
            conf = r[head + "_conf"][det]
            thresholds = np.linspace(0, 0.98, 50)
            kept = [np.mean(conf >= t) * 100 for t in thresholds]
            acc = [np.mean(ok[conf >= t]) * 100 if np.any(conf >= t) else np.nan for t in thresholds]
            ax.plot(kept, acc, color=color, label=tex(name))
        ax.set_xlim(100, 0)
        ax.set_xlabel(r"Windows kept by the confidence threshold (\%)")
        ax.set_ylabel(r"Accuracy of the kept windows (\%)")
        ax.legend(loc="lower left")
        title(ax, f"{head.capitalize()} head: selective decisions")

        r = res["cv"]
        det = r["detected_object"]
        present = [c for c in ORDER[head] if np.any(r[head][det] == c)]
        true, pred, conf = r[head][det], r[head + "_pred"][det], r[head + "_conf"][det]
        mean = np.full((len(present), len(present)), np.nan)
        for i, a in enumerate(present):
            for j, b in enumerate(present):
                m = (true == a) & (pred == b)
                if m.any():
                    mean[i, j] = conf[m].mean()
        ax = axs[1, col]
        im = ax.imshow(mean, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
        for i in range(len(present)):
            for j in range(len(present)):
                if np.isfinite(mean[i, j]):
                    ax.text(j, i, f"{mean[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if mean[i, j] > 0.55 else INK)
        ax.grid(False)
        ax.tick_params(length=0, top=False, right=False)
        ax.set_xticks(range(len(present)), _labels(present), rotation=35, ha="right")
        ax.set_yticks(range(len(present)), _labels(present))
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        title(ax, f"{head.capitalize()} head: mean confidence")
    fig.colorbar(im, ax=axs[1, :], shrink=0.8, label="Mean confidence")
    style.save(pdf, fig)


def _boxes(ax, values, groups, classes, colors):
    data = [values[(groups == c) & np.isfinite(values)] for c in classes]
    keep = [i for i, v in enumerate(data) if len(v)]
    bp = ax.boxplot([data[i] for i in keep], positions=keep, widths=0.28, patch_artist=True, showfliers=False,
                    medianprops={"color": "white", "lw": 1.1}, whiskerprops={"color": INK, "lw": 0.6},
                    capprops={"color": INK, "lw": 0.6}, boxprops={"edgecolor": INK, "lw": 0.5})
    for box, i in zip(bp["boxes"], keep):
        box.set_facecolor(colors[classes[i]])
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)


def features(res, pdf):
    data = {k: np.concatenate([res[d][k] for d in ("train", "test") if res[d] is not None]) for k in res["train"]}
    m = objects(data)
    fig, axs = plt.subplots(4, 4, figsize=(DOUBLE, 7.0), layout="constrained")
    for block, (head, names) in enumerate((("motion", MOTION_FEATURES), ("material", MATERIAL_FEATURES))):
        panels = axs[2 * block:2 * block + 2].flat
        for ax, name in zip(panels, names):
            _boxes(ax, data[name][m], data[head][m], ORDER[head], COLORS[head])
            title(ax, r"$\mathtt{" + tex(name) + "}$")
        legend_ax = axs[2 * block + 1, 3]
        legend_ax.axis("off")
        legend_ax.legend(handles=[Patch(facecolor=COLORS[head][c], edgecolor=INK, lw=0.5, label=tex(c))
                                  for c in ORDER[head]], loc="center", title=bold(f"{head} features"))
    title(fig, "Features per Class")
    style.save(pdf, fig)


def _pca(train, test, names):
    X = np.stack([train[n] for n in names], 1)
    med = np.nanmedian(X, 0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    _, S, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    project = lambda T: (np.where(np.isnan(T), med, T) - mu) / sd @ Vt[:2].T  # noqa: E731
    Xt = np.stack([test[n] for n in names], 1) if test is not None else None
    return project(X), project(Xt) if Xt is not None else None, S ** 2 / np.sum(S ** 2) * 100


def embedding(res, pdf):
    train = {k: v[objects(res["train"])] for k, v in res["train"].items()}
    test = {k: v[objects(res["test"])] for k, v in res["test"].items()} if res["test"] is not None else None
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.7), layout="constrained")
    panels = (("Motion features", MOTION_FEATURES, "motion"), ("Material features", MATERIAL_FEATURES, "material"),
              ("All features", MOTION_FEATURES + MATERIAL_FEATURES, "material"))
    for ax, (name, names, head) in zip(axs, panels):
        P, Q, ev = _pca(train, test, names)
        for c in ORDER[head]:
            for data, Z, filled in ((train, P, True), (test, Q, False)):
                if data is None:
                    continue
                m = data[head] == c
                marker = [MARKERS[x] for x in data["motion"][m]] if name == "All features" else [MARKERS[c]] * m.sum()
                for mk in set(marker):
                    sel = np.array(marker) == mk
                    ax.scatter(Z[m][sel, 0], Z[m][sel, 1], s=6, marker=mk, lw=0.5, alpha=0.75,
                               color=COLORS[head][c] if filled else "none", edgecolors=COLORS[head][c])
        ax.set_xlabel(tex(f"PC 1 ({ev[0]:.0f} %)"))
        ax.set_ylabel(tex(f"PC 2 ({ev[1]:.0f} %)"))
        ax.set_xticks([])
        ax.set_yticks([])
        title(ax, name)
    handles = [Line2D([], [], ls="", marker=MARKERS[c], color=MOTION_COLORS[c], label=tex(c)) for c in ORDER["motion"]]
    handles += [Line2D([], [], ls="", marker="s", color=MATERIAL_COLORS[c], label=tex(c)) for c in ORDER["material"]]
    handles += [Line2D([], [], ls="", marker="o", color=INK, label="training (A, A2)"),
                Line2D([], [], ls="", marker="o", mfc="none", color=INK, label="test (B)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=9, columnspacing=1.0, handletextpad=0.2)
    style.save(pdf, fig)


def learning(res, pdf):
    lc = res["curve"]
    k = np.array(lc["k"])
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), layout="constrained")
    for ax, head in zip(axs, HEADS):
        for part, color, ls, name in (("test", PRIMARY, "-", "held-out takes"), ("train", MUTED, "--", "training takes")):
            m, sd = (np.array(v) * 100 for v in zip(*lc[head][part]))
            ax.fill_between(k, m - sd, np.minimum(m + sd, 100), color=color, alpha=0.15, lw=0)
            ax.plot(k, m, color=color, ls=ls, marker="o", label=name)
        ax.axhline(lc["chance"][head] * 100, color=MUTED, ls=":", lw=0.8, label="chance")
        ax.set_ylim(0, 102)
        ax.set_xticks(k)
        ax.set_xlabel("Training takes per class (at most)")
        ax.set_ylabel(r"Accuracy (\%)")
        ax.legend(loc="lower right")
        title(ax, f"{head.capitalize()} head")
    style.save(pdf, fig)


def _level_set(ax, case):
    D, y, lam, a = case["D"], case["y"], case["lam"], case["a_star"]
    i, j, path, free = case["i"], case["j"], case["path"], case["free"]
    span = max(a[i], abs(free[0]), abs(free[1]), np.abs(path).max()) * 1.3
    u = np.linspace(-0.15 * span, span, 220)
    v = np.linspace(min(free[1], 0) - 0.15 * span, 0.75 * span, 220)
    U, V = np.meshgrid(u, v)
    rest = D @ a - D[:, i] * a[i] - D[:, j] * a[j]
    R = y[:, None, None] - rest[:, None, None] - D[:, i, None, None] * U - D[:, j, None, None] * V
    F = 0.5 * np.sum(R ** 2, axis=0) + lam * (U + V)
    levels = F.min() + (F.max() - F.min()) * np.linspace(0, 1, 16)[1:] ** 2
    ax.contour(U, V, F, levels=levels, cmap=SEQUENTIAL, linewidths=0.7)
    ax.fill_between(u, v[0], 0, color=MUTED, alpha=0.18, lw=0, label=r"infeasible ($a_j < 0$)")
    ax.fill_betweenx(v, u[0], 0, color=MUTED, alpha=0.18, lw=0)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axvline(0, color=INK, lw=0.7)
    ax.plot(*free, marker="o", ms=5, mfc="white", mec=INK, ls="", label="unconstrained minimum")
    ax.plot(path[:, 0], path[:, 1], color=PRIMARY, lw=1.0, marker="o", ms=1.8, label=f"FISTA ({len(path) - 1} steps)")
    ax.plot(a[i], a[j], marker="*", ms=9, color=ACCENT, ls="", label="optimum (Clarabel)")
    ax.set_xlim(u[0], u[-1])
    ax.set_ylim(v[0], v[-1])
    ax.set_xlabel(f"$a_i$, atom {i} ({case['labels'][0]})")
    ax.set_ylabel(f"$a_j$, atom {j} ({case['labels'][1]})")
    ax.legend(loc="upper right", fontsize=6.5)
    title(ax, r"Level sets of $\tfrac12\|y - Da\|^2 + \lambda \mathbf{1}^\top a$")


def optimization(res, pdf):
    conv, it = res["convergence"], res["solvers"]["iterations"]
    clarabel = res["solvers"]["clarabel"]
    det = res["cv"]["detected_object"]
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 5.6), layout="constrained")
    _level_set(axs[0, 0], conv["level_set"])
    styles = {"motion": "-", "material": "--"}
    panels = ((axs[0, 1], "objective", r"$(f_k - f^\star)/f^\star$", "Objective"),
              (axs[1, 0], "gap", r"Duality gap $/ f^\star$", "Duality gap"))
    for ax, key, ylabel, name in panels:
        for head in HEADS:
            ax.loglog(ITERS, np.maximum(conv[head][key], 1e-12), color=PRIMARY, ls=styles[head], marker="o",
                      ms=2.5, label=f"FISTA, {head} head")
        if key == "gap":
            for head in HEADS:
                ax.axhline(conv[head]["clarabel_gap"], color=ACCENT, ls=styles[head], lw=0.9,
                           label=f"Clarabel, {head} head")
        ax.axvline(300, color=MUTED, ls=":", lw=0.8)
        ax.set_xlabel("Iterations $k$")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right" if key == "objective" else "center left")
        title(ax, f"{name} vs. Iterations")
    ax = axs[1, 1]
    for head in HEADS:
        ax.semilogx(ITERS, np.array(it[head]) * 100, color=PRIMARY, ls=styles[head], marker="o", ms=2.5,
                    label=f"FISTA, {head} head")
        exact = np.mean(clarabel[head][det] == clarabel[head + "_pred"][det]) * 100
        ax.axhline(exact, color=ACCENT, ls=styles[head], lw=0.9, label=f"Clarabel, {head} head")
    ax.axvline(300, color=MUTED, ls=":", lw=0.8)
    ax.set_xlabel("Iterations $k$")
    ax.set_ylabel(r"Accuracy, leave one take out (\%)")
    ax.legend(loc="center right")
    title(ax, "Accuracy vs. iterations")
    style.save(pdf, fig)


def timing(res, pdf):
    t = res["timing"]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), layout="constrained", gridspec_kw={"width_ratios": [2, 3]})
    ax = axs[0]
    bars = (("FISTA\n27 windows", t["fista_batch_ms"], PRIMARY), ("FISTA\n1 window", t["fista_ms"], PRIMARY),
            ("Clarabel\n1 window", t["clarabel_ms"], ACCENT))
    ax.bar(range(3), [b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
    top = max(b[1] for b in bars)
    for x, (_, v, _) in enumerate(bars):
        ax.text(x, v + 0.03 * top, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_ylim(0, top * 1.15)
    ax.set_xticks(range(3), [b[0] for b in bars])
    ax.set_ylabel("Time per window (ms)")
    ax.grid(axis="x", visible=False)
    title(ax, "Solving both heads")

    ax = axs[1]
    frames = t["hop_frames"] * t["frame_ms"]
    budget = t["hop_frames"] * t["frame_period_ms"]
    for y, (name, solve, color) in enumerate((("FISTA", t["fista_ms"], PRIMARY), ("Clarabel", t["clarabel_ms"], ACCENT))):
        left = 0
        for width, shade, part in ((frames, SEQUENTIAL(0.25), f"{t['hop_frames']} new frames"),
                                   (t["window_ms"], MUTED, "features"), (solve, color, "classification")):
            label = f"{part}, {name}" if part == "classification" else (part if y == 0 else None)
            ax.barh(y, width, left=left, color=shade, height=0.5, label=label)
            left += width
        ax.text(budget * 1.02, y, f"{left:.0f} ms", va="center", fontsize=6.5)
    ax.axvline(budget, color=INK, ls="--", lw=0.8, label=tex(f"time per new window ({budget:.0f} ms)"))
    ax.set_yticks([0, 1], ["FISTA", "Clarabel"])
    ax.set_xlim(0, budget * 1.15)
    ax.set_xlabel("Time per new window (ms)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=3, columnspacing=1.0)
    title(ax, "Real-time budget")
    style.save(pdf, fig)


def takes(res, pdf):
    from matplotlib.colors import to_rgb
    for key, name, _ in EVALS:
        r = res[key]
        if r is None:
            continue
        names = sorted(dict.fromkeys(r["take"]), key=lambda t: (r["joint"][r["take"] == t][0] == "none",
                                                               r["joint"][r["take"] == t][0], t))
        width = max(int(np.sum(r["take"] == t)) for t in names)
        img = np.ones((3 * len(names), width, 3))
        rows, correct = [], []
        for i, take in enumerate(names):
            m = np.flatnonzero(r["take"] == take)
            m = m[np.argsort(r["t_start_s"][m])]
            for j, k in enumerate(m):
                img[3 * i, j] = to_rgb(MOTION_COLORS.get(r["motion_pred"][k], MUTED))
                img[3 * i + 1, j] = to_rgb(MATERIAL_COLORS.get(r["material_pred"][k], MUTED))
            rows.append(tex(f"{r['joint'][m[0]]}  {take.split('/')[-1]}"))
            correct.append(np.mean(r["joint"][m] == r["joint_pred"][m]) * 100)
        fig, ax = plt.subplots(figsize=(DOUBLE, 0.14 * len(names) + 1.3), layout="constrained")
        ax.imshow(img, aspect="auto", interpolation="nearest", extent=(0, width, 3 * len(names), 0))
        ax.grid(False)
        ax.tick_params(length=0, top=False, right=False)
        ax.set_yticks(3 * np.arange(len(names)) + 1, rows, fontsize=5.5)
        for i, c in enumerate(correct):
            ax.plot(width + 0.5, 3 * i + 1, "s", ms=3, color=GOOD if c >= 50 else LOSS, clip_on=False)
            ax.text(width + 1.0, 3 * i + 1, f"{c:.0f}", va="center", fontsize=5.5)
        ax.set_xlim(0, width)
        ax.set_xlabel("Window (1.6 s, one every 0.4 s)")
        title(ax, f"Every take, {name}")
        handles = [Patch(color=MOTION_COLORS[c], label=tex(c)) for c in ORDER["motion"]]
        handles += [Patch(color=MATERIAL_COLORS[c], label=tex(c)) for c in ORDER["material"]]
        handles += [Patch(color=MUTED, label="nothing detected"), Patch(color=GOOD, label="take mostly correct"),
                    Patch(color=LOSS, label="take mostly wrong")]
        fig.legend(handles=handles, loc="outside lower center", ncol=5, columnspacing=1.0, title_fontsize=7,
                   title="upper strip: predicted motion, lower strip: predicted material, right: windows correct (\\%)")
        style.save(pdf, fig)


def _spectrogram(ax, w):
    vel, t, db = w.spectrogram
    im = ax.imshow(db - db.max(), origin="lower", aspect="auto", cmap=SEQUENTIAL, vmin=-40, vmax=0,
                   extent=(t[0], t[-1] if t.size > 1 else 1, vel[0], vel[-1]))
    ax.set_ylim(-1.5, 1.5)
    ax.grid(False)
    return im


def microdoppler(windows, pdf):
    cats = sorted(windows, key=lambda s: (s == "none", s))
    fig, axs = plt.subplots(2, 4, figsize=(DOUBLE, 3.6), layout="constrained", sharex=True, sharey=True)
    for ax, c in zip(axs.flat, cats):
        im = _spectrogram(ax, windows[c])
        title(ax, c)
    for ax in axs.flat[len(cats):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("Radial velocity (m/s)")
    for ax in axs[-1]:
        ax.set_xlabel("Time (s)")
    fig.colorbar(im, ax=axs, shrink=0.7, label="Power (dB, peak = 0)")
    style.save(pdf, fig)


class Dashboard:

    def __init__(self, extractor, model, live=True):
        style.use()
        if live:
            plt.ion()
        self.fx, self.model = extractor, model
        self.fig, axs = plt.subplots(2, 3, figsize=(DOUBLE, 4.6), layout="constrained")
        (self.a_map, self.a_md, self.a_rp), (self.a_mo, self.a_ma, self.a_p) = axs
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "closed", True))
        self.fig.canvas.mpl_connect("key_press_event", self._key)

    def _key(self, event):
        if event.key == "s":
            import time
            from pathlib import Path
            path = Path("output") / time.strftime("dashboard_%Y%m%d_%H%M%S.pdf")
            path.parent.mkdir(exist_ok=True)
            style.save(path, self.fig, close=False)
            print(f"  snapshot -> {path}")

    def update(self, w, label, smooth, heading=None, truth=None):
        geo, rng = self.fx.geo, self.fx.range_axis
        ok = geo.uy_valid
        a0 = int(np.argmin(np.abs(geo.uy_grid - np.sin(np.radians(w.azimuth_deg)))))
        for ax in (self.a_map, self.a_md, self.a_rp, self.a_mo, self.a_ma, self.a_p):
            ax.clear()

        ax = self.a_map
        ax.imshow(w.change_map_db[:, ok], origin="lower", aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=40,
                  extent=(geo.uy_grid[ok][0], geo.uy_grid[ok][-1], rng[0], rng[-1]))
        if w.detected:
            ax.plot(np.sin(np.radians(w.azimuth_deg)), w.range_m, "o", ms=8, mfc="none", mec=ACCENT, mew=1.2)
        ax.grid(False)
        ax.set_xlabel(r"Azimuth $\sin\theta$")
        ax.set_ylabel("Range (m)")
        title(ax, "Change against the empty room")

        _spectrogram(self.a_md, w)
        self.a_md.set_xlabel("Time (s)")
        self.a_md.set_ylabel("Radial velocity (m/s)")
        title(self.a_md, "Micro-Doppler")

        ax = self.a_rp
        ax.plot(rng, w.change_map_db[:, a0], color=PRIMARY, label="change")
        ax.axhline(8, color=MUTED, ls=":", lw=0.8, label="threshold")
        if w.detected:
            ax.axvline(w.range_m, color=ACCENT, ls="--", lw=0.8, label="object")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Change (dB)")
        ax.legend(loc="upper right")
        title(ax, f"Object at {w.range_m:.2f} m, {w.azimuth_deg:+.0f} deg" if w.detected
              else "No Object (Change Below 8 dB)")

        shares = {}
        for ax, head in ((self.a_mo, "motion"), (self.a_ma, "material")):
            clf = getattr(self.model, head)
            title(ax, f"{head.capitalize()} head: sparse code")
            ax.set_xlabel("Training windows, by class")
            ax.set_ylabel("Weight")
            if not w.detected:
                ax.text(0.5, 0.5, "no object", ha="center", va="center", transform=ax.transAxes)
                continue
            code, share = clf.explain(np.array([[w.features[n] for n in self.model.features[head]]]))
            order = np.argsort([clf.classes.index(c) for c in clf.labels], kind="stable")
            colors = [COLORS[head][c] for c in clf.labels[order]]
            nz = np.flatnonzero(code[order] > 1e-6)
            ax.vlines(nz, 0, code[order][nz], colors=[colors[i] for i in nz], lw=0.8)
            ax.scatter(nz, code[order][nz], s=9, c=[colors[i] for i in nz], zorder=3)
            ax.set_xlim(-0.5, len(order) - 0.5)
            ax.set_ylim(0, max(code.max() * 1.6, 1e-3))
            ax.legend(handles=[Patch(color=COLORS[head][c], label=tex(c)) for c in clf.classes], loc="upper center",
                      ncol=len(clf.classes), fontsize=6.5, handlelength=0.8, columnspacing=0.6)
            shares[head] = clf.classes, share

        ax = self.a_p
        y, ticks = 0.0, []
        for head in HEADS:
            if head in shares:
                for c, v in zip(*shares[head]):
                    ax.barh(y, v * 100, color=COLORS[head][c], height=0.7)
                    ticks.append((y, tex(c)))
                    y -= 1
                y -= 0.6
        ax.set_yticks([t for t, _ in ticks], [c for _, c in ticks])
        ax.set_xlim(0, 100)
        ax.set_xlabel(r"Share of the sparse code (\%)")
        ax.grid(axis="y", visible=False)
        if not shares:
            ax.text(50, 0.5, "nothing detected", ha="center", va="center", transform=ax.get_xaxis_transform())
        wrong = truth is not None and label != truth
        ax.set_title(bold(f"Prediction: {label}" + ("" if smooth == label else f" (last 5: {smooth})")
                          + (" (Wrong)" if wrong else "")), fontsize=7.5, color=LOSS if wrong else INK)
        if heading:
            title(self.fig, heading)
        style.finish(self.fig)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if plt.get_backend().lower() != "agg":
            plt.pause(0.001)


def dashboards(windows, model, extractor, folder):
    dash = Dashboard(extractor, model, live=False)
    for cat in sorted(windows, key=lambda s: (s == "none", s)):
        w = windows[cat]
        label = "none"
        if w.detected:
            p = model.predict({k: [v] for k, v in w.features.items()})
            label = f"{p['material'][0]}-{p['motion'][0]}"
        dash.update(w, label, label, heading=f"True class {cat}", truth=cat)
        style.save(folder / f"{cat}.pdf", dash.fig, close=False)
    plt.close(dash.fig)


def readme_figure(res, path):
    r = res["tb"] if res["tb"] is not None else res["cv"]
    name = "unseen session B" if res["tb"] is not None else "leave one take out"
    det = r["detected_object"]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), layout="constrained", gridspec_kw={"width_ratios": [4, 3]})
    for ax, head in zip(axs, HEADS):
        present = [c for c in ORDER[head] if np.any(r[head][det] == c)]
        im = _matrix(ax, r[head][det], r[head + "_pred"][det], present, present, f"{head.capitalize()}, {name}")
        ax.set_xlabel("Predicted class")
    axs[0].set_ylabel("True class (windows)")
    fig.colorbar(im, ax=axs, shrink=0.8, label="Share of the true class")
    style.save(path, fig)
