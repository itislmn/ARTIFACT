"""
plots.py - the thesis figures of the experiment as PDFs in output/figures/ (ARTIFACT-TeX): reading, confusions,
confidence, light, alignment, run time, and the README picture.

    python run_all.py        (draws all figures; this file is imported, not run)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from visualization import style
from visualization.style import ACCENT, DOUBLE, GOAL, GOOD, HATCHES, INK, MUTED, PRIMARY, SEQUENTIAL, SERIES, SINGLE, tex, title

import dataset
from evaluate import CONDITIONS, READERS
from metrics import risk_coverage

COLOR = {"MetricSight": PRIMARY, "CNN": ACCENT}
LETTERS = list(dict.fromkeys(dataset.TRUTH))


def _bars(ax, res, key, stable_key, ylabel):
    labels = [label for k, label in CONDITIONS]
    x = np.arange(len(labels))
    for i, reader in enumerate(READERS):
        v = [res["summary"].get((k, reader), {}) for k, _ in CONDITIONS]
        ax.bar(x + (i - 0.5) * 0.36, [s.get(key, np.nan) for s in v], 0.34, color=COLOR[reader], label=f"{reader}, per frame")
        ax.plot(x + (i - 0.5) * 0.36, [s.get(stable_key, np.nan) for s in v], "D", color=INK, ms=3.5,
                mfc="white", label="Stable reading (last 15)" if i == 0 else None)
    ax.set_xticks(x, [label.replace(", ", ",\n") for label in labels])
    ax.set_ylabel(ylabel)


def _letters(pairs):
    hit = {c: [0, 0] for c in LETTERS}
    for t, r in pairs:
        if t:
            hit[t][0] += t == r
            hit[t][1] += 1
    return [100 * hit[c][0] / max(hit[c][1], 1) for c in LETTERS]


def reading(res, pdf):
    fig, axs = plt.subplots(1, 2, figsize=style.size(DOUBLE, 0.38), layout="constrained")
    _bars(axs[0], res, "cer", "stable_cer", "Character error rate [\\%]")
    title(axs[0], "Character Error Rate")
    x = np.arange(len(LETTERS))
    for i, reader in enumerate(READERS):
        v = _letters(res["chars"][reader]["pairs"])
        axs[1].bar(x + (i - 0.5) * 0.36, v, 0.34, color=COLOR[reader])
        for xi, y in zip(x + (i - 0.5) * 0.36, v):
            axs[1].text(xi, y + 1.5, f"{y:.0f}", ha="center", fontsize=6)
    axs[1].set_xticks(x, LETTERS)
    axs[1].set_ylim(0, 110)
    axs[1].set_xlabel("True letter")
    axs[1].set_ylabel("Read correctly [\\%]")
    title(axs[1], "Accuracy per Letter (Raw Frames)")
    style.legend_below(fig, axs)
    style.save(pdf, fig)


def _confusion(ax, pairs, reader):
    cols = LETTERS + ["Other", "Missed"]
    count = np.zeros((len(LETTERS), len(cols)))
    for t, r in pairs:
        if not t:
            continue
        j = cols.index(r) if r in LETTERS else (len(cols) - 1 if r == "" else len(cols) - 2)
        count[LETTERS.index(t), j] += 1
    share = count / np.maximum(count.sum(1, keepdims=True), 1)
    im = ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.tick_params(length=0, top=False, right=False)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(LETTERS)), [f"{c} ({int(n)})" for c, n in zip(LETTERS, count.sum(1))])
    for i in range(len(LETTERS)):
        for j in range(len(cols)):
            if count[i, j]:
                ax.text(j, i, f"{100 * share[i, j]:.0f}" if share[i, j] >= 0.005 else "$<$1", ha="center", va="center",
                        fontsize=6.5, color="white" if share[i, j] > 0.55 else INK)
    title(ax, f"{reader}: {100 * np.trace(share[:, :len(LETTERS)]) / len(LETTERS):.1f}\\,\\% Mean Recall")
    return im


def confusions(res, pdf):
    fig, axs = plt.subplots(1, 2, figsize=style.size(DOUBLE, 0.36), layout="constrained")
    for ax, reader in zip(axs, READERS):
        im = _confusion(ax, res["chars"][reader]["pairs"], reader)
        ax.set_xlabel("Read as")
    axs[0].set_ylabel("True letter (characters)")
    fig.colorbar(im, ax=axs, shrink=0.8, label="Share of the true letter")
    style.save(pdf, fig)
    fig, axs = plt.subplots(1, 2, figsize=style.size(DOUBLE, 0.36), layout="constrained")
    labels = [label for _, label in CONDITIONS]
    x = np.arange(len(labels))
    for i, reader in enumerate(READERS):
        bottom = np.zeros(len(labels))
        for kind, name, h in (("S", "Substitutions", HATCHES[0]), ("D", "Deletions", HATCHES[1]), ("I", "Insertions", HATCHES[3])):
            v = np.array([res["summary"].get((k, reader), {}).get(kind, np.nan) for k, _ in CONDITIONS])
            axs[0].bar(x + (i - 0.5) * 0.36, v, 0.34, bottom=bottom, color=COLOR[reader], hatch=h, edgecolor="white",
                       label=f"{reader}" if kind == "S" else None)
            bottom += np.nan_to_num(v)
    axs[0].set_xticks(x, [label.replace(", ", ",\n") for label in labels])
    axs[0].set_ylabel("Errors per true character [\\%]")
    handles = [Patch(color=COLOR[r], label=r) for r in READERS] + \
              [Patch(facecolor="white", edgecolor=INK, hatch=h, label=n) for n, h in
               (("Substitutions", HATCHES[0]), ("Deletions", HATCHES[1]), ("Insertions", HATCHES[3]))]
    title(axs[0], "Error Types")
    top = {}
    for reader in READERS:
        for t, r in res["chars"][reader]["pairs"]:
            if t != r:
                top.setdefault((t or "-", r or "-"), {}).setdefault(reader, 0)
                top[t or "-", r or "-"][reader] += 1
    best = sorted(top, key=lambda k: -sum(top[k].values()))[:10][::-1]
    y = np.arange(len(best))
    for i, reader in enumerate(READERS):
        axs[1].barh(y + (i - 0.5) * 0.38, [top[k].get(reader, 0) for k in best], 0.36, color=COLOR[reader], label=reader)
    axs[1].set_yticks(y, [f"{t} $\\rightarrow$ {r}" for t, r in best])
    axs[1].set_xlabel("Characters (raw readings, all frames)")
    title(axs[1], "Most Frequent Errors ($-$: None)")
    fig.legend(handles=handles, loc="outside lower center", ncols=5)
    style.save(pdf, fig)


def confidence(res, pdf):
    fig, axs = plt.subplots(1, 3, figsize=style.size(DOUBLE, 0.32), layout="constrained")
    for reader in READERS:
        c = res["chars"][reader]
        cov, acc = risk_coverage(c["confidence"], c["correct"])
        axs[0].plot(100 * cov, 100 * acc, color=COLOR[reader], label=f"{reader} (AUROC {c['auroc']:.2f})")
    axs[0].set_xlabel("Characters accepted [\\%]")
    axs[0].set_ylabel("Accuracy of accepted [\\%]")
    axs[0].legend(loc="lower left")
    title(axs[0], "Accuracy vs. Coverage")
    for ax, reader in zip(axs[1:], READERS):
        c = res["chars"][reader]
        bins = np.linspace(c["confidence"].min(), np.percentile(c["confidence"], 99.5), 30)
        ax.hist(c["confidence"][c["correct"]], bins, color=GOOD, alpha=0.8, label="Correct")
        ax.hist(c["confidence"][~c["correct"]], bins, color=ACCENT, alpha=0.8, label="Wrong")
        ax.set_xlabel("Confidence (best minus second class)")
        ax.set_yscale("log")
        ax.set_ylabel("Characters")
        ax.legend(loc="upper left")
        title(ax, f"{reader} Confidence")
    style.save(pdf, fig)


def _rolling(v, n=15):
    v = np.asarray(v, float)
    return np.convolve(v, np.ones(n) / n, mode="same")


def light(res, pdf):
    fig, axs = plt.subplots(3, 1, figsize=style.size(DOUBLE, 0.62), layout="constrained", sharex=True)
    for ax, name in zip(axs, [n for n, s in dataset.TAKES.items() if s == "light"]):
        rows = [r for r in res["rows"] if r["take"] == name]
        t = np.array([r["t_s"] for r in rows])
        lamp = np.array([r["lamp"] for r in rows]) == 1
        for a, b in _spans(t, lamp):
            ax.axvspan(a, b, color=GOAL, alpha=0.12, lw=0, label="Lamp on")
        for reader in READERS:
            for cond, ls in (("raw", "-"), ("removed", "--")):
                p = sorted((q for q in res["preds"] if q["take"] == name and q["reader"] == reader and q["condition"] == cond),
                           key=lambda q: q["k"])
                if cond == "removed":
                    raw = {q["k"]: 100 * (1 - q["cer"]) for q in res["preds"] if q["take"] == name and q["reader"] == reader
                           and q["condition"] == "raw" and q["lamp"] == 0}
                    p = sorted(p + [{"k": k, "cer": 1 - v / 100} for k, v in raw.items()], key=lambda q: q["k"])
                ax.plot([q["k"] / dataset.FPS for q in p], _rolling([100 * max(0, 1 - q["cer"]) for q in p]), ls,
                        color=COLOR[reader], lw=1.0, label=f"{reader}, " + ("raw frame" if cond == "raw" else "glare removed"))
        ax.set_ylim(-5, 105)
        ax.set_ylabel("Characters correct [\\%]")
        title(ax, f"{tex(name)}: Reading Over Time (1 s Mean)")
    axs[-1].set_xlabel("Time [s]")
    style.legend_below(fig, axs, ncols=5)
    style.save(pdf, fig)
    fig, axs = plt.subplots(1, 2, figsize=style.size(DOUBLE, 0.36), layout="constrained")
    takes = [n for n, s in dataset.TAKES.items() if s == "light"] + ["all"]
    x = np.arange(len(takes))
    for ax, key, name in ((axs[0], "cer", "Per Frame"), (axs[1], "stable_cer", "Stable Reading")):
        for i, reader in enumerate(READERS):
            for j, cond in enumerate(("lamp_on", "lamp_removed")):
                v = [res["per_take"][t, reader, cond][key] if t != "all" else res["summary"][cond, reader][key] for t in takes]
                ax.bar(x + (2 * i + j - 1.5) * 0.19, v, 0.18, color=COLOR[reader], hatch=HATCHES[2 * j], edgecolor="white",
                       label=f"{reader}, " + ("raw frame" if j == 0 else "glare removed"))
        ax.set_xticks(x, [tex(t) for t in takes])
        ax.set_ylabel("Character error rate [\\%]")
        title(ax, f"Lamp On, {name}")
    style.legend_below(fig, axs, ncols=4)
    style.save(pdf, fig)
    v = res["veil"]
    fig, ax = plt.subplots(figsize=style.size(SINGLE, 0.75), layout="constrained")
    for c, (name, col) in enumerate((("Red", ACCENT), ("Green", GOOD), ("Blue", PRIMARY))):
        b, a, p = v["fits"][c]
        ax.plot(v["radii"], v["levels"][c], "o", ms=2.5, color=col, label=f"{name} ring median")
        rr = np.linspace(v["radii"].min(), v["radii"].max(), 200)
        ax.plot(rr, b + a * rr ** -p, color=col, lw=1, label=f"$b + a\\rho^{{-{p:.2f}}}$")
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000], ["100", "200", "500", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("Distance $\\rho$ from the lamp core [px]")
    ax.set_ylabel("Brightness")
    style.legend_below(fig, [ax], ncols=2)
    title(ax, f"Veil Fit ({tex(v['take'])}, frame {v['k']})")
    style.save(pdf, fig)


def _spans(t, on):
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    return [(t[a], t[min(b, len(t)) - 1]) for a, b in zip(edges[::2], edges[1::2])]


def alignment(res, pdf):
    A = res["alignment"]
    m = A["model"]
    from backend.alignment import NAMES, project, right_handed
    p = np.array([m["params"][n] for n in NAMES])
    dz = m.get("height_correction_mm", {})
    xyz = np.array([q["xyz_m"] for q in m["points"]], float)
    xyz[:, 2] += [dz.get(q["label"], 0) / 1000 for q in m["points"]]
    uv = np.array([q["uv_px"] for q in m["points"]], float)
    pr = project(p, right_handed(xyz), (m["image_size_wh"][0] / 2, m["image_size_wh"][1] / 2))[0]
    fig, axs = plt.subplots(1, 2, figsize=style.size(DOUBLE, 0.36), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    ax.imshow(A["scene"], extent=(0, m["image_size_wh"][0], m["image_size_wh"][1], 0), alpha=0.75, zorder=0)
    ax.grid(False)
    for (a, b), (c, d) in zip(uv, pr):
        ax.plot([a, c], [b, d], color="white", lw=1.2)
    ax.plot(uv[:, 0], uv[:, 1], "*", color=GOAL, ms=7, label="Clicked reflector apex")
    ax.plot(pr[:, 0], pr[:, 1], "o", color=PRIMARY, mfc="white", mew=1.3, ms=5, label="Radar point projected")
    ax.set_xlim(0, m["image_size_wh"][0])
    ax.set_ylim(m["image_size_wh"][1], 0)
    ax.set_aspect("equal")
    ax.set_xlabel("$u$ [px]")
    ax.set_ylabel("$v$ [px]")
    style.legend_below(fig, [ax], ncols=2)
    title(ax, "Reprojection on the Scene")
    ax = axs[1]
    data = [np.array(m["error_px"]), np.array(m["loo_error_px"]),
            np.hypot(*A["word_px"].T), np.hypot(*A["lamp_px"].T)]
    labels = ["Fit", "Leave one\nout", "Word on\nball", "Phone\nflash"]
    box = ax.boxplot(data, widths=0.28, patch_artist=True, medianprops={"color": "white"}, showfliers=False)
    for patch, col in zip(box["boxes"], [MUTED, PRIMARY, PRIMARY, PRIMARY]):
        patch.set_facecolor(col)
    for i, d in enumerate(data):
        rng = np.random.default_rng(i)
        d = d if len(d) <= 200 else rng.choice(d, 200, replace=False)
        ax.plot(np.full(len(d), i + 1) + rng.uniform(-0.08, 0.08, len(d)), d, ".", color=INK, ms=2)
    ax.set_xticks(range(1, 5), labels)
    ax.set_ylabel("Image error [px]")
    title(ax, "Calibration Points and Unseen Targets")
    style.save(pdf, fig)


def timing(res, pdf):
    fig, axs = plt.subplots(1, 2, figsize=style.size(DOUBLE, 0.34), layout="constrained")
    data = [[p["ms"] for p in res["preds"] if p["reader"] == r and p["condition"] == "raw"] for r in READERS]
    box = axs[0].boxplot(data, widths=0.28, patch_artist=True, medianprops={"color": "white"}, showfliers=False)
    for patch, r in zip(box["boxes"], READERS):
        patch.set_facecolor(COLOR[r])
    axs[0].axhline(1000 / dataset.FPS, color=MUTED, ls="--", lw=0.8, label="Camera frame period")
    axs[0].axhline(res["live"]["radar_ms"], color=INK, ls=":", lw=0.8, label="Radar Infini-Rex per frame")
    axs[0].set_xticks([1, 2], list(READERS))
    axs[0].set_ylabel("Time per reading [ms]")
    axs[0].legend(loc="upper left", fontsize=6.5)
    title(axs[0], "Recorded Frames (One Core)")
    for r in READERS:
        c = [x for x in res["complexity"] if x["reader"] == r]
        R = [x["range_m"] for x in c]
        axs[1].plot(R, [x["ms"] for x in c], "o-", color=COLOR[r], label=r)
        axs[1].fill_between(R, [x["ms_q25"] for x in c], [x["ms_q75"] for x in c], color=COLOR[r], alpha=0.15, lw=0)
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Board range $R$ [m]")
    axs[1].set_ylabel("Time per reading [ms]")
    axs[1].legend(loc="upper right")
    title(axs[1], "Cost vs. Range (Synthetic, Unseen Fonts)")
    style.save(pdf, fig)


def _frame(ax, img, sc, boxes, panel):
    h, w = img.shape[:2]
    ax.imshow(img[::2, ::2], extent=(0, w, h, 0))
    for bar in sc["bars"]:
        if bar is not None:
            ax.plot(bar[:, 0], bar[:, 1], color=SERIES[2], lw=1.4)
    for a, b, c, d in boxes:
        ax.add_patch(plt.Rectangle((a, b), c - a, d - b, fill=False, color=PRIMARY, lw=0.9))
    r = sc["row"]
    ax.plot(r["glare_u"], r["glare_v"], "*", color=GOAL, ms=9, mec="white", mew=0.5)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    title(ax, panel)


def _reads(ax, sc, cond):
    for i, reader in enumerate(READERS):
        text = sc[reader, cond].text or "-"
        ax.text(0.02, 0.05 + 0.13 * (1 - i), f"{reader}: {tex(text)}", transform=ax.transAxes, fontsize=7,
                color="white", bbox={"facecolor": COLOR[reader], "edgecolor": "none", "pad": 1.5})


def readme_figure(res, path):
    sc, A = res["showcase"], res["alignment"]
    m = A["model"]
    fig = plt.figure(figsize=(DOUBLE, 3.1), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.3, 1])
    axs = top.subplots(1, 3)
    ax = axs[0]
    w, h = m["image_size_wh"]
    ax.imshow(A["scene"][::2, ::2], extent=(0, w, h, 0))
    from backend.alignment import NAMES, project, right_handed
    pr = project(np.array([m["params"][n] for n in NAMES]), right_handed(
        np.array([q["xyz_m"] for q in m["points"]]) + np.array([[0, 0, m.get("height_correction_mm", {}).get(q["label"], 0) / 1000]
                                                                for q in m["points"]])), (w / 2, h / 2))[0]
    uv = np.array([q["uv_px"] for q in m["points"]], float)
    ax.plot(uv[:, 0], uv[:, 1], "*", color=GOAL, ms=7, mec="white", mew=0.4)
    ax.plot(pr[:, 0], pr[:, 1], "o", color=SERIES[2], mfc="none", mew=1.2, ms=6)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    title(ax, "1 Radar-Camera Alignment")
    _frame(axs[1], sc["raw"], sc, sc["MetricSight", "raw"].boxes, "2 Lamp On, Raw Frame")
    _reads(axs[1], sc, "raw")
    _frame(axs[2], sc["clean"], sc, sc["MetricSight", "removed"].boxes, "3 Glare Removed")
    _reads(axs[2], sc, "removed")
    d = sc["MetricSight", "raw"].debug
    a, b = d["line"]
    axs = bottom.subplots(1, 3, width_ratios=[1, 1, 1.3])
    axs[0].imshow(np.clip(d["luma"], 0, 255), cmap="gray", aspect="auto")
    axs[0].axhspan(a, b, color=PRIMARY, alpha=0.15, lw=0)
    title(axs[0], f"4 Metric Image, {load_mm():.1f} mm per Pixel")
    axs[1].imshow(d["ink"], cmap=SEQUENTIAL, aspect="auto", interpolation="nearest")
    for c0, c1 in d["cols"]:
        axs[1].add_patch(plt.Rectangle((c0, a), c1 - c0, b - a, fill=False, color=GOAL, lw=0.9))
    title(axs[1], "5 Censored Ink and Letter Cuts")
    crops = d["crops"]
    strip = np.concatenate([np.pad(c, 3) for c in crops], axis=1) if len(crops) else np.zeros((38, 38))
    axs[2].imshow(strip, cmap=SEQUENTIAL, aspect="equal", interpolation="nearest")
    text = sc["MetricSight", "raw"].text
    for i, ch in enumerate(text):
        axs[2].text(19 + 38 * i, 42, tex(ch), ha="center", va="top", fontsize=8, color=PRIMARY)
    axs[2].set_ylim(56, -2)
    title(axs[2], "6 Letters $32\\times32$, Ridge Classes")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    style.save(path, fig)


def load_mm():
    from backend.settings import load_config
    return load_config()["text"]["mm_per_px"]


def _mosaic(X, cols):
    rows = len(X) // cols
    out = np.ones((rows * 34, cols * 34))
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        out[r * 34 + 1:r * 34 + 33, c * 34 + 1:c * 34 + 33] = 1 - X[i]
    return out


def training_data(res, pdf):
    from readers import cnn_reader, metricsight
    from readers.glyphs import TEST_FONTS, TRAIN_FONTS, characters
    names = {"segoesc.ttf": "Segoe Script", "georgia.ttf": "Georgia", "BRADHITC.TTF": "Bradley Hand"}
    fig, ax = plt.subplot_mosaic([["train", "acc"], ["test", "acc"]], figsize=style.size(DOUBLE, 0.4),
                                 layout="constrained", width_ratios=[1.5, 1], height_ratios=[4, 3])
    axs = [ax["train"], ax["test"], ax["acc"]]
    train = characters(64, np.random.default_rng(1))[0]
    test = [characters(1200, np.random.default_rng(2), [f]) for f in TEST_FONTS]
    for ax, img, text in ((axs[0], _mosaic(train, 16), f"Training: {len(TRAIN_FONTS)} Fonts, Distorted"),
                          (axs[1], _mosaic(np.concatenate([X[:16] for X, _ in test]), 16), "Test: Unseen Fonts")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        title(ax, text)
    axs[1].set_yticks(34 * np.arange(len(TEST_FONTS)) + 17, [names.get(f, f) for f in TEST_FONTS])
    net = cnn_reader.LeNet(seed=1)
    weights = np.load(cnn_reader.MODEL)
    for k in net.p:
        net.p[k] = weights[k]
    W = np.load(metricsight.MODEL)["W"]
    acc = {"MetricSight": [100 * np.mean(metricsight.classify(W, X)[0] == y) for X, y in test],
           "CNN": [100 * np.mean(np.concatenate([net.forward(X[j:j + 500, None])[0].argmax(1)
                                                  for j in range(0, len(y), 500)]) == y) for X, y in test]}
    yy = np.arange(len(TEST_FONTS))
    for k, r in enumerate(READERS):
        axs[2].barh(yy + (0.2 if k else -0.2), acc[r], height=0.38, color=COLOR[r], label=r)
        for y0, a in zip(yy, acc[r]):
            axs[2].text(a + 1, y0 + (0.2 if k else -0.2), f"{a:.0f}", va="center", fontsize=6.5)
    axs[2].set_yticks(yy, [names.get(f, f) for f in TEST_FONTS])
    axs[2].invert_yaxis()
    axs[2].set_xlim(0, 112)
    axs[2].set_xlabel("Characters Correct [\\%]")
    title(axs[2], "Accuracy, Unseen Fonts")
    style.legend_below(fig, [axs[2]], ncols=2)
    style.save(pdf, fig)


FIGURES = (("reading", reading), ("confusion", confusions), ("confidence", confidence), ("light", light),
           ("alignment", alignment), ("timing", timing), ("training_data", training_data))
