"""
plots.py - the thesis figures of the experiment as PDFs in output/figures/ (ARTIFACT-TeX): scene, tracking (space-time
maps, HOTA, identities, accuracy), information, prediction regions, detection, classification, run time, and the
README picture.

    python plots.py          (redraws from output/cache/results.pkl; python run_all.py computes everything)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import dataset  # noqa: E402
import paths  # noqa: E402
from visualization import style  # noqa: E402
from visualization.style import (ACCENT, DOUBLE, GOAL, INK, LOSS, MATERIAL_COLORS, MOTION_COLORS, MUTED,  # noqa: E402
                                 NORM, PRIMARY, SEQUENTIAL, SERIES, tex, title)

FIGURES = paths.OUTPUT / "figures"
TAKE_COLOR = dict(zip(dataset.TAKES, (PRIMARY, ACCENT)))
OBJ_MARKER = {"ball": "o", "person": "^"}
ALPHA = 0.1
EPSILON = 0.1


def _box(ax, data, pos, color, width=0.28):
    data = [np.asarray(d, float)[np.isfinite(np.asarray(d, float))] for d in data]
    keep = [i for i, d in enumerate(data) if len(d)]
    if not keep:
        return
    b = ax.boxplot([data[i] for i in keep], positions=[pos[i] for i in keep], widths=width, patch_artist=True,
                   showfliers=False, medianprops={"color": "white", "lw": 1.2}, whiskerprops={"color": color},
                   capprops={"color": color}, boxprops={"facecolor": color, "edgecolor": color})
    return b


def _gaps(ax, r, horizontal=True, **kw):
    t = np.asarray(r["t_arr"])
    for a, b in zip(t[:-1], t[1:]):
        if b - a > 0.5:
            (ax.axvspan if horizontal else ax.axhspan)(a, b, color=MUTED, alpha=0.18, lw=0, **kw)


def _label_color(tr, head="material"):
    lab = ((tr.get("classes") or {}).get(head) or {}).get("label")
    return (MATERIAL_COLORS if head == "material" else MOTION_COLORS).get(lab, MUTED)


def _takes(res):
    return list(res["takes"].items())


def wedge(tr, n=12):
    lo, hi = tr["region"]["lo"], tr["region"]["hi"]
    if not tr["region"]["calibrated"] or not np.all(np.isfinite(np.asarray(list(lo[:2]) + list(hi[:2]), float))):
        return None
    th = np.linspace(*np.clip([lo[1], hi[1]], -np.radians(60), np.radians(60)), n)
    rr = np.r_[np.full(n, max(lo[0], 0.3)), np.full(n, min(hi[0], 3.2))]
    a = np.r_[th, th[::-1]]
    return np.c_[rr * np.sin(a), rr * np.cos(a)]


def spacetime(ax, r, every=5, simple=False, Y=(-2.4, 1.2), X=(0.0, 3.0)):
    recs, t = r["recs"], np.asarray(r["t_arr"])
    paths_ = {}
    for k, rec in enumerate(recs):
        for tr in rec["tracks"]:
            if tr["status"] != "confirmed":
                continue
            paths_.setdefault(tr["id"], []).append((t[k], tr["y"], tr["x"], _label_color(tr)))
            w = wedge(tr) if k % every == 0 else None
            if w is not None:
                w = np.c_[np.full(len(w), t[k]), np.clip(w[:, 0], *Y), np.clip(w[:, 1], *X)]
                ax.add_collection3d(Poly3DCollection([w], facecolor=_label_color(tr), alpha=0.13,
                                                     edgecolor=_label_color(tr, "motion"), lw=0.3))
    if not simple:
        d = np.array([(t[k], x["range_m"] * np.sin(np.radians(x["theta_deg"])), x["range_m"] * np.cos(np.radians(x["theta_deg"])))
                      for k, rec in enumerate(recs) if k % 2 == 0 for x in rec["dets"]], float).reshape(-1, 3)
        d = d[(d[:, 1] >= Y[0]) & (d[:, 1] <= Y[1]) & (d[:, 2] >= X[0]) & (d[:, 2] <= X[1])]
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=0.4, color=INK, alpha=0.25, depthshade=False, lw=0)
    b = np.array(r["truth_az"]["ball"], float)
    if len(b):
        a = np.radians(b[:, 1])
        ax.plot(b[:, 0], dataset.BALL_RANGE_M * np.sin(a), dataset.BALL_RANGE_M * np.cos(a), color=GOAL, lw=0.7, ls="-.")
    for seg in paths_.values():
        p = np.array([s[:3] for s in seg], float)
        for i in range(len(p) - 1):
            if p[i + 1, 0] - p[i, 0] < 0.6:
                ax.plot(p[i:i + 2, 0], np.clip(p[i:i + 2, 1], *Y), np.clip(p[i:i + 2, 2], *X), color=seg[i + 1][3], lw=1.1)
    ax.plot([0, t[-1]], [0, 0], [0, 0], color=NORM, lw=2.0)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(*Y)
    ax.set_zlim(*X)
    ax.set_xlabel("Time (s)", labelpad=2)
    ax.set_ylabel("Sideways $y$ (m)", labelpad=-1)
    ax.set_zlabel("Forward $x$ (m)", labelpad=-3)
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((2.6, 1.3, 1.0), zoom=1.32)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)
        a._axinfo["grid"]["linewidth"] = 0.3
    ax.tick_params(pad=-2)


def _spacetime_legend(fig, simple=False):
    h = [Line2D([], [], color=MATERIAL_COLORS[m], lw=1.4, label=f"track: {m}") for m in ("dielectric", "waterlike")]
    h += [Line2D([], [], color=MUTED, lw=1.4, label="track: unclassified"),
          Patch(facecolor=MUTED, alpha=0.3, edgecolor=INK, lw=0.3, label=r"next-look region (KalmanCQR, 90\,\%)"),
          Line2D([], [], color=GOAL, ls="-.", lw=0.8, label="camera: ball (taped range)"),
          Line2D([], [], color=NORM, lw=2, label="radar (still)")]
    if not simple:
        h += [Line2D([], [], marker=".", ls="", color=INK, alpha=0.5, label="detection (FDR 0.1)")]
    fig.legend(handles=h, loc="outside lower center", ncols=4)


def scene(res, pdf):
    import make_video
    picks = [("final_fixed_1", 8.0), ("final_fixed_1", 26.5), ("run_1", 50.0), ("run_1", 66.0)]
    for page in (picks[:2], picks[2:]):
        fig, axs = plt.subplots(2, 1, figsize=(DOUBLE, 2 * DOUBLE * 720 / 1280 + 0.5), layout="constrained")
        for ax, (take, ts) in zip(axs, page):
            ax.imshow(make_video.still(take, ts))
            ax.axis("off")
            title(ax, f"{dataset.TAKES[take]}, $t$ = {ts:.1f} s")
        style.save(pdf, fig)

    from backend.alignment import Projector
    proj = {k: Projector(dataset.alignment(k)) for k in dataset.TAKES}
    rc = paths.load_config()["radar"]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 3.1), layout="constrained", sharey=True)
    for ax, (take, r) in zip(axs, _takes(res)):
        a = np.radians(np.linspace(-60, 60, 200))
        ax.plot(rc["r_max_m"] * np.sin(a), rc["r_max_m"] * np.cos(a), color=NORM, lw=0.5)
        for s in (-1, 1):
            ax.plot([0, s * rc["r_max_m"] * np.sin(np.radians(60))], [0, rc["r_max_m"] * np.cos(np.radians(60))], color=NORM, lw=0.5)
        ax.axvline(rc["y_min_m"], color=MUTED, ls=":", lw=0.8)
        ax.axvline(rc["y_max_m"], color=MUTED, ls=":", lw=0.8)
        half = np.degrees(np.arctan(proj[take].size_wh[0] / 2 / proj[take].f))
        for s in (-1, 1):
            ax.plot([0, 3 * np.sin(np.radians(s * half))], [0, 3 * np.cos(np.radians(s * half))], color=MUTED, ls="--", lw=0.8)
        tracks = {}
        for rec in r["recs"]:
            for tr in rec["tracks"]:
                if tr["status"] == "confirmed":
                    tracks.setdefault(tr["id"], []).append((tr["y"], tr["x"], _label_color(tr)))
        for pts in tracks.values():
            p = np.array([q[:2] for q in pts])
            for i in range(len(p) - 1):
                ax.plot(p[i:i + 2, 0], p[i:i + 2, 1], color=pts[i + 1][2], lw=0.6, alpha=0.8)
        ax.plot(0, dataset.BALL_RANGE_M, marker="*", color=GOAL, ms=9, ls="", mec=INK, mew=0.3)
        ax.plot(0, 0, marker="^", color=INK, ms=7)
        ax.set_aspect("equal")
        edge = rc["r_max_m"] * np.sin(np.radians(60))
        allp = np.array([q[:2] for pts in tracks.values() for q in pts] + [(-edge, rc["r_max_m"]), (edge, 0.0)])
        ax.set_xlim(allp[:, 0].min() - 0.15, allp[:, 0].max() + 0.15)
        ax.set_ylim(-0.1, allp[:, 1].max() + 0.15)
        ax.set_xlabel("Lateral $y$ (m)")
        title(ax, f"{r['label']}: Every Confirmed Track")
    axs[0].set_ylabel("Forward $x$ (m)")
    reading_page(res, pdf)
    h = [Line2D([], [], color=MATERIAL_COLORS[m], lw=1.4, label=m) for m in ("dielectric", "waterlike")]
    h += [Line2D([], [], color=MUTED, lw=1.4, label="not yet classified"),
          Line2D([], [], marker="*", color=GOAL, ls="", ms=7, label="ball, taped"),
          Line2D([], [], marker="^", color=INK, ls="", ms=5, label="radar and camera"),
          Line2D([], [], color=MUTED, ls="--", label="camera view"), Line2D([], [], color=NORM, lw=0.5, label="radar region")]
    fig.legend(handles=h, loc="outside lower center", ncols=4)
    style.save(pdf, fig)


def timeline(axs, r, every=5):
    recs, t = r["recs"], np.asarray(r["t_arr"])
    paths_ = {}
    for k, rec in enumerate(recs):
        d = [(x["theta_deg"], x["range_m"]) for x in rec["dets"]]
        for ax, j in zip(axs, (0, 1)):
            ax.plot([t[k]] * len(d), [q[j] for q in d], ".", ms=1.2, color=INK, alpha=0.3, lw=0)
        for tr in rec["tracks"]:
            if tr["status"] != "confirmed":
                continue
            c = _label_color(tr)
            paths_.setdefault(tr["id"], []).append((t[k], tr["theta_deg"], tr["range_m"], c))
            lo, hi = tr["region"]["lo"], tr["region"]["hi"]
            if k % every == 0 and tr["region"]["calibrated"] and np.all(np.isfinite(lo[:2])):
                axs[0].plot([t[k]] * 2, np.degrees([lo[1], hi[1]]), color=c, alpha=0.3, lw=1.6, solid_capstyle="butt")
                axs[1].plot([t[k]] * 2, [lo[0], hi[0]], color=c, alpha=0.3, lw=1.6, solid_capstyle="butt")
    for pts in paths_.values():
        p = np.array([q[:3] for q in pts], float)
        for i in range(len(p) - 1):
            if p[i + 1, 0] - p[i, 0] < 0.6:
                for ax, j in zip(axs, (1, 2)):
                    ax.plot(p[i:i + 2, 0], p[i:i + 2, j], color=pts[i + 1][3], lw=1.1)
    b = np.array(r["truth_az"]["ball"], float)
    if len(b):
        axs[0].plot(b[:, 0], b[:, 1], color=GOAL, lw=0.7, ls="-.")
        axs[1].axhline(dataset.BALL_RANGE_M, color=GOAL, lw=0.7, ls="-.")
    q = np.array(r["truth_az"]["person"], float)
    if len(q):
        axs[0].plot(q[:, 0], (q[:, 1] + q[:, 2]) / 2, ls="", marker="*", ms=2.5, color=GOAL)
    for ax in axs:
        _gaps(ax, r)
        ax.set_xlim(0, t[-1])
    axs[0].set_ylim(-60, 30)
    axs[1].set_ylim(0.5, 2.6)
    axs[0].set_ylabel(r"Azimuth ($^\circ$)")
    axs[1].set_ylabel("Range (m)")
    axs[1].set_xlabel("Time (s)")


def reading_page(res, pdf):
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), layout="constrained", gridspec_kw={"width_ratios": [1, 1.6, 1.2]})
    ax = axs[0]
    for i, (take, r) in enumerate(_takes(res)):
        g = r["reading"]
        _box(ax, [g["share"]], [i], TAKE_COLOR[take])
        ax.text(i, 1.03, f"{len(g['reads'])} reads", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(res["takes"])), [r["label"] for _, r in _takes(res)])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Letters read right (fraction)")
    title(ax, "Every Read")
    ax = axs[1]
    for take, r in _takes(res):
        q = np.array(r["reading"]["stable"], float)
        if len(q):
            ax.step(q[:, 0], q[:, 1], where="post", color=TAKE_COLOR[take], lw=1.0)
    ax.axhline(len(dataset.TEXT), color=GOAL, ls="-.", lw=0.8)
    ax.set_ylim(0, len(dataset.TEXT) + 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Correct letters (count)")
    title(ax, "Stable Reading (Last 15 Reads)")
    ax = axs[2]
    for take, r in _takes(res):
        g = r["reading"]
        if len(g["coverage"]):
            ax.plot(g["coverage"], g["accuracy"], color=TAKE_COLOR[take], lw=1.0, label=f"AUROC {g['auc']:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Reads kept, most confident first")
    ax.set_ylabel("Letters read right (fraction)")
    title(ax, "Selective Reading")
    ax.legend(loc="upper right")
    h = [Patch(color=TAKE_COLOR[k], label=v) for k, v in dataset.TAKES.items()]
    h += [Line2D([], [], color=GOAL, ls="-.", label=f"all {len(dataset.TEXT)} letters of {dataset.TEXT}")]
    fig.legend(handles=h, loc="outside lower center", ncols=3)
    style.save(pdf, fig)


def running_coverage(res, pdf):
    hs = (("step", "next look", "-"), ("0.5", "0.5 s", "--"), ("1.0", "1 s", ":"))
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), layout="constrained")
    for ax, key in ((axs[0], "coverage"), (axs[1], "alpha_t")):
        for take, r in _takes(res):
            for h, _, ls in hs:
                q = np.array([(lk["t"], lk[key].get(h, np.nan)) for lk in r["looks"]], float)
                ax.plot(q[:, 0], q[:, 1], color=TAKE_COLOR[take], ls=ls, lw=0.9)
        ax.axhline(1 - EPSILON if key == "coverage" else EPSILON, color=GOAL, ls="-.", lw=0.8)
        ax.set_xlabel("Time (s)")
    axs[0].set_ylim(0.6, 1.02)
    axs[0].set_ylabel("Running coverage (fraction)")
    axs[1].set_ylabel(r"Adaptive level $\varepsilon_t$")
    title(axs[0], "Coverage of the Last 200 Predictions")
    title(axs[1], "Adaptive Conformal Inference")
    h = [Line2D([], [], color=TAKE_COLOR[k], label=v) for k, v in dataset.TAKES.items()]
    h += [Line2D([], [], color=INK, ls=ls, label=lbl) for _, lbl, ls in hs]
    h += [Line2D([], [], color=GOAL, ls="-.", label=r"nominal ($1-\varepsilon$ = 0.9, $\varepsilon$ = 0.1)")]
    fig.legend(handles=h, loc="outside lower center", ncols=6)
    style.save(pdf, fig)


def tracking(res, pdf):
    for take, r in _takes(res):
        fig, axs = plt.subplots(2, 1, figsize=(DOUBLE, 3.9), sharex=True, layout="constrained")
        timeline(axs, r)
        title(axs[0], f"{r['label']}: Tracks over Time")
        h = [Line2D([], [], color=MATERIAL_COLORS[m], lw=1.6, label=m) for m in ("dielectric", "waterlike")]
        h += [Line2D([], [], color=MUTED, lw=1.6, label="not yet classified"),
              Line2D([], [], color=MUTED, lw=4, alpha=0.3, label=r"prediction region (90\,\%)"),
              Line2D([], [], color=INK, marker=".", ls="", ms=3, alpha=0.5, label="detection (FDR 0.1)"),
              Line2D([], [], color=GOAL, ls="-.", label="camera: ball (taped range)"),
              Line2D([], [], color=GOAL, marker="*", ls="", ms=4, label="camera: person (azimuth)"),
              Patch(color=MUTED, alpha=0.3, label="radar blind")]
        fig.legend(handles=h, loc="outside lower center", ncols=4)
        style.save(pdf, fig)

    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.5, 0.55]})
    ax = axs[2]
    for i, (take, r) in enumerate(_takes(res)):
        for v, view in enumerate(("look", "live")):
            x = 0.32 * (2 * i + v) - 0.48
            seg = [s["IDSW"] for s in r["mot"][view]["segments"]]
            b = _box(ax, [seg], [x], TAKE_COLOR[take], 0.26)
            if b and view == "live":
                for patch in b["boxes"]:
                    patch.set_facecolor("white")
            ax.plot(x, r["mot"][view]["clear"]["IDSW"] * 10 / r["duration_s"], ls="", marker="D", ms=3.2, color=INK,
                    mfc=TAKE_COLOR[take] if view == "look" else "white", zorder=4)
    ax.set_xticks([])
    ax.set_ylabel("ID switches per 10 s")
    ax.set_ylim(bottom=0)
    title(ax, "ID Switches")
    ax = axs[0]
    for take, r in _takes(res):
        c = r["mot"]["look"]["hota"]["curve"]
        for key, ls in (("HOTA", "-"), ("DetA", "--"), ("AssA", ":")):
            ax.plot(c["alpha"], c[key], color=TAKE_COLOR[take], ls=ls, label=f"{key}, {r['label']}")
    ax.set_xlabel(r"Localisation threshold $\tau$")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    title(ax, "HOTA and Its Parts (Per Look)")
    ax = axs[1]
    keys = ("HOTA", "MOTA", "MOTP", "IDF1")
    for i, (take, r) in enumerate(_takes(res)):
        for v, view in enumerate(("look", "live")):
            pos = [3 * q + 0.32 * (2 * i + v) - 0.48 for q in range(len(keys))]
            seg = r["mot"][view]["segments"]
            col = TAKE_COLOR[take]
            b = _box(ax, [[s[k] for s in seg] for k in keys], pos, col, 0.26)
            if b and view == "live":
                for patch in b["boxes"]:
                    patch.set_facecolor("white")
            whole = [r["mot"][view]["hota"]["HOTA"], r["mot"][view]["clear"]["MOTA"], r["mot"][view]["clear"]["MOTP"],
                     r["mot"][view]["idf1"]["IDF1"]]
            ax.plot(pos, whole, ls="", marker="D", ms=3.2, color=INK, mfc=col if view == "look" else "white", zorder=4)
    ax.set_xticks([3 * q for q in range(len(keys))], keys)
    ax.axhline(0, color=MUTED, lw=0.6, ls=":")
    ax.set_ylabel("Score per 10 s segment")
    title(ax, "Tracking Scores: Per Look and Live")
    h = [Patch(color=TAKE_COLOR[k], label=f"{v}, per look") for k, v in dataset.TAKES.items()]
    h += [Patch(facecolor="white", edgecolor=TAKE_COLOR[k], label=f"{v}, live") for k, v in dataset.TAKES.items()]
    h += [Line2D([], [], color=INK, ls=ls, label=k) for k, ls in (("HOTA", "-"), ("DetA", "--"), ("AssA", ":"))]
    h += [Line2D([], [], marker="D", color=INK, ls="", ms=3, label="whole take")]
    fig.legend(handles=h, loc="outside lower center", ncols=4)
    style.save(pdf, fig)

    gospa_page(res, pdf)

    fig, axs = plt.subplots(2, 1, figsize=(DOUBLE, 3.0), layout="constrained")
    for ax, (take, r) in zip(axs, _takes(res)):
        for row, o in enumerate(dataset.CLASSES):
            fr = [f for f in r["frames"]["look"] if o in f["gt"]]
            ids = [f["hit"].get(o) for f in fr]
            got = [i for i in ids if i is not None]
            best = max(set(got), key=got.count) if got else None
            for f, i in zip(fr, ids):
                if i is None:
                    ax.plot([f["t"], f["t"]], [row - 0.32, row + 0.32], color=LOSS, lw=0.5)
                else:
                    ax.barh(row, 0.1, left=f["t"] - 0.05, height=0.6, color=PRIMARY if i == best else SERIES[2], lw=0)
            for i in set(got):
                ts = [f["t"] for f, q in zip(fr, ids) if q == i]
                if len(ts) >= 10:
                    ax.text(ts[0], row + 0.42, str(i), fontsize=5.5, ha="left", va="bottom", color=INK)
        _gaps(ax, r)
        ax.set_yticks(range(len(dataset.CLASSES)), list(dataset.CLASSES))
        ax.set_ylim(-0.6, 1.9)
        ax.set_xlim(0, r["t_arr"][-1])
        ax.grid(False)
        title(ax, f"{r['label']}: Track per Object")
    axs[-1].set_xlabel("Time (s)")
    h = [Patch(color=PRIMARY, label="the object's longest track"), Patch(color=SERIES[2], label=r"another track (ID above if $\geq$ 1 s)"),
         Line2D([], [], color=LOSS, label="object in view, no track"), Patch(color=MUTED, alpha=0.3, label="radar blind")]
    fig.legend(handles=h, loc="outside lower center", ncols=4)
    style.save(pdf, fig)

    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    for i, (take, r) in enumerate(_takes(res)):
        for j, o in enumerate(dataset.CLASSES):
            e = r["estimation"][o]
            b = _box(ax, [np.abs(e["err"]), np.abs(e["live"])], [3 * j + 0.7 * i - 0.55, 3 * j + 0.7 * i - 0.2], TAKE_COLOR[take], 0.28)
            if b:
                b["boxes"][-1].set_facecolor("white")
    ax.set_xticks([0, 3], list(dataset.CLASSES))
    ax.set_ylabel(r"$|\theta_\mathrm{radar} - \theta_\mathrm{camera}|$ ($^\circ$)")
    title(ax, "Azimuth Error against the Camera")
    ax = axs[1]
    for take, r in _takes(res):
        rng = r["estimation"]["ball"]["range"]
        ax.hist(rng, bins=np.arange(1.40, 1.90, 0.01), color=TAKE_COLOR[take], alpha=0.55, label=r["label"])
    ax.axvline(dataset.BALL_RANGE_M, color=GOAL, ls="-.", label="taped")
    ax.set_xlabel("Range of the ball's track (m)")
    ax.set_ylabel("Looks")
    title(ax, "Ball Range")
    h = [Patch(color=TAKE_COLOR[k], label=f"{v}, per look") for k, v in dataset.TAKES.items()]
    h += [Patch(facecolor="white", edgecolor=TAKE_COLOR[k], label=f"{v}, live") for k, v in dataset.TAKES.items()]
    h += [Line2D([], [], color=GOAL, ls="-.", label="ball range, taped")]
    axs[1].get_legend() and axs[1].get_legend().remove()
    fig.legend(handles=h, loc="outside lower center", ncols=5)
    style.save(pdf, fig)


def gospa_page(res, pdf):
    from evaluate import GOSPA_C_DEG
    parts = (("loc", "localisation", SERIES[2]), ("missed", "missed objects", SERIES[3]), ("false", "false tracks", MUTED))
    fig = plt.figure(figsize=(DOUBLE, 3.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1])
    for row, (take, r) in enumerate(_takes(res)):
        ax = fig.add_subplot(gs[row, 0])
        fr = r["frames"]["look"]
        t = np.array([f["t"] for f in fr])
        grid = np.arange(0, r["t_arr"][-1], 0.5)
        stack = []
        for key, _, _ in parts:
            v = np.array([f["gospa"][key] for f in fr])
            stack.append([v[(t >= x - 1.0) & (t < x + 1.0)].mean() if np.any((t >= x - 1.0) & (t < x + 1.0)) else np.nan
                          for x in grid])
        ax.stackplot(grid, *[np.nan_to_num(sv) for sv in stack], colors=[c for _, _, c in parts], alpha=0.85, lw=0,
                     labels=[lbl for _, lbl, _ in parts])
        _gaps(ax, r)
        ax.set_xlim(0, r["t_arr"][-1])
        ax.set_ylabel(r"GOSPA$^2$ ($^\circ{}^2$, 2 s)")
        title(ax, f"{r['label']}: GOSPA Components")
        if row == 1:
            ax.set_xlabel("Time (s)")
    ax = fig.add_subplot(gs[:, 1])
    for i, (take, r) in enumerate(_takes(res)):
        for v, view in enumerate(("look", "live")):
            x = 0.32 * (2 * i + v) - 0.48
            seg = [s_["GOSPA"] for s_ in r["mot"][view]["segments"]]
            b = _box(ax, [seg], [x], TAKE_COLOR[take], 0.26)
            if b and view == "live":
                for patch in b["boxes"]:
                    patch.set_facecolor("white")
            ax.plot(x, r["mot"][view]["gospa"]["gospa"], ls="", marker="D", ms=3.2, color=INK,
                    mfc=TAKE_COLOR[take] if view == "look" else "white", zorder=4)
    ax.axhline(GOSPA_C_DEG / np.sqrt(2), color=MUTED, ls=":", lw=0.8)
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(0, 4.3)
    ax.set_xticks([])
    ax.set_ylabel(r"GOSPA per look ($^\circ$), 10 s segments")
    title(ax, rf"GOSPA ($c$ = {GOSPA_C_DEG:g}$^\circ$, $p$ = 2)")
    h = [Patch(color=c, alpha=0.85, label=lbl) for _, lbl, c in parts]
    h += [Patch(color=TAKE_COLOR[k], label=f"{v}, per look") for k, v in dataset.TAKES.items()]
    h += [Patch(facecolor="white", edgecolor=TAKE_COLOR[k], label=f"{v}, live") for k, v in dataset.TAKES.items()]
    h += [Patch(color=MUTED, alpha=0.3, label="radar blind"),
          Line2D([], [], color=MUTED, ls=":", label="one object missed per look")]
    fig.legend(handles=h, loc="outside lower center", ncols=4)
    style.save(pdf, fig)


def information(res, pdf):
    fig, axs = plt.subplots(2, 1, figsize=(DOUBLE, 3.6), layout="constrained", sharey=True)
    for ax, (take, r) in zip(axs, _takes(res)):
        for o, ls in (("ball", "-"), ("person", "--")):
            pts = [(lk["t"], lk["objects"][o]["entropy_bits"]) for lk in r["looks"] if o in lk["objects"]]
            if pts:
                p = np.array(pts)
                br = np.r_[0, np.flatnonzero(np.diff(p[:, 0]) > 0.5) + 1, len(p)]
                for a, b in zip(br[:-1], br[1:]):
                    ax.plot(p[a:b, 0], p[a:b, 1], color=TAKE_COLOR[take], ls=ls, lw=0.9, label=o if a == 0 else None)
        _gaps(ax, r)
        ax.set_xlim(0, r["t_arr"][-1])
        ax.set_ylabel("Position entropy (bit)")
        title(ax, f"{r['label']}: Uncertainty of the Matched Track")
    axs[-1].set_xlabel("Time (s)")
    h = [Line2D([], [], color=INK, ls="-", label="ball"), Line2D([], [], color=INK, ls="--", label="person"),
         Patch(color=MUTED, alpha=0.3, label="radar blind")]
    for ax in axs:
        ax.get_legend() and ax.get_legend().remove()
    fig.legend(handles=h, loc="outside lower center", ncols=3)
    style.save(pdf, fig)

    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axs[0]
    for take, r in _takes(res):
        for o, ls in (("ball", "-"), ("person", "--")):
            p = np.array([(lk["t"], lk["objects"][o]["position_bits"], lk["objects"][o]["class_bits"])
                          for lk in r["looks"] if o in lk["objects"]])
            if not len(p):
                continue
            ax.plot(p[:, 0], p[:, 1] + p[:, 2], color=TAKE_COLOR[take], ls=ls, lw=0.8)
        break
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bits known about the object")
    title(ax, f"Known about Each Object ({_takes(res)[0][1]['label']})")
    ax = axs[1]
    for take, r in _takes(res):
        for o in dataset.CLASSES:
            e = r["estimation"][o]
            ok = np.isfinite(e["err"]) & np.isfinite(e["sigma"])
            if ok.sum() < 20:
                continue
            s, err = e["sigma"][ok], e["err"][ok]
            edges = np.quantile(s, np.linspace(0, 1, 7))
            mid, rms = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                m = (s >= a) & (s <= b)
                if m.sum() >= 5:
                    mid.append(np.median(s[m]))
                    rms.append(np.sqrt(np.mean(err[m] ** 2)))
            ax.plot(mid, rms, marker=OBJ_MARKER[o], color=TAKE_COLOR[take], ms=3.2, lw=0.8)
    lim = ax.get_xlim()[1]
    ax.plot([0, 6], [0, 6], color=NORM, lw=0.6, ls=":")
    ax.set_xlim(0, max(lim, 1.5))
    ax.set_ylim(0, None)
    ax.set_xlabel(r"Claimed $\sigma_\theta$ ($^\circ$)")
    ax.set_ylabel(r"Measured RMS error ($^\circ$)")
    title(ax, "Claimed vs. Real Error")
    ax = axs[2]
    z = np.linspace(-6, 6, 200)
    for take, r in _takes(res):
        allz = np.concatenate([r["estimation"][o]["z"] for o in dataset.CLASSES])
        allz = allz[np.isfinite(allz)]
        ax.hist(np.clip(allz, -6, 6), bins=np.linspace(-6, 6, 37), density=True, histtype="step", color=TAKE_COLOR[take], lw=1.0)
    ax.plot(z, np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi), color=NORM, ls=":", lw=0.8)
    ax.set_xlabel(r"$z = \Delta\theta / \sigma_\theta$")
    ax.set_ylabel("Density")
    title(ax, "Normalised Error")
    h = [Line2D([], [], color=TAKE_COLOR[k], label=v) for k, v in dataset.TAKES.items()]
    h += [Line2D([], [], color=INK, ls="-", marker="o", ms=3, label="ball"), Line2D([], [], color=INK, ls="--", marker="^", ms=3, label="person"),
          Line2D([], [], color=NORM, ls=":", label=r"ideal ($\mathcal{N}(0,1)$, equality)")]
    fig.legend(handles=h, loc="outside lower center", ncols=5)
    style.save(pdf, fig)


def regions(res, pdf):
    for take, r in _takes(res):
        fig = plt.figure(figsize=(DOUBLE, 4.4), layout="constrained")
        ax = fig.add_subplot(projection="3d")
        spacetime(ax, r)
        title(ax, f"{r['label']}: Tracks in Space and Time")
        _spacetime_legend(fig)
        style.save(pdf, fig)

    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axs[0]
    for i, (take, r) in enumerate(_takes(res)):
        c = r["regions"]["coverage_radar"]
        ax.plot(np.arange(3) + 0.12 * (2 * i - 1), [c.get("step"), c.get("0.5"), c.get("1.0")], marker="s", ls="",
                color=TAKE_COLOR[take], ms=4)
        for j, o in enumerate(dataset.CLASSES):
            e = r["estimation"][o]
            x = 3 + j + 0.12 * (2 * i - 1)
            ax.errorbar(x, e["coverage_camera"], yerr=[[e["coverage_camera"] - e["coverage_ci"][0]], [e["coverage_ci"][1] - e["coverage_camera"]]],
                        marker=OBJ_MARKER[o], color=TAKE_COLOR[take], ms=4, capsize=2, lw=0.8)
    ax.axhline(1 - EPSILON, color=GOAL, ls="-.", lw=0.9)
    ax.set_xticks(range(5), ["next look", "0.5 s", "1 s", "ball\n(camera)", "person\n(camera)"])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Coverage")
    title(ax, "Coverage of the Prediction Regions")
    ax = axs[1]
    for take, r in _takes(res):
        p = np.array([(lk["t"], lk["alpha_t"].get("step", np.nan)) for lk in r["looks"]], float)
        ax.plot(p[:, 0], p[:, 1], color=TAKE_COLOR[take], lw=0.8)
    ax.axhline(EPSILON, color=GOAL, ls="-.", lw=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Level $\varepsilon_t$ (next look)")
    title(ax, "Adaptive Conformal Level")
    h = [Line2D([], [], color=TAKE_COLOR[k], marker="s", ls="-", ms=3, label=v) for k, v in dataset.TAKES.items()]
    h += [Line2D([], [], color=INK, marker="o", ls="", ms=3, label="ball"), Line2D([], [], color=INK, marker="^", ls="", ms=3, label="person"),
          Line2D([], [], color=GOAL, ls="-.", label=r"nominal ($1-\varepsilon$ = 0.9, $\varepsilon$ = 0.1)")]
    fig.legend(handles=h, loc="outside lower center", ncols=5)
    style.save(pdf, fig)

    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), layout="constrained")
    ax = axs[0]
    for i, (take, r) in enumerate(_takes(res)):
        _box(ax, [r["estimation"][o]["iou"] for o in dataset.CLASSES], [0.35 * (2 * i - 1), 2 + 0.35 * (2 * i - 1)], TAKE_COLOR[take])
    ax.set_xticks([0, 2], ["ball (box)", "person (width)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU with the camera's box")
    title(ax, "Region vs. Camera Box")
    ax = axs[1]
    u = np.linspace(0, 1, 11)
    for take, r in _takes(res):
        p = r["regions"]["exchange"]["p_values"]
        ax.hist(p, bins=u, density=True, histtype="step", color=TAKE_COLOR[take], lw=1.0,
                label=f"{r['label']} (KS $p$ = {r['regions']['exchange']['ks_p']:.2f})")
    ax.axhline(1, color=NORM, ls=":", lw=0.8)
    ax.set_xlabel("Conformal $p$-value of each new score")
    ax.set_ylabel("Density")
    ax.set_ylim(0, 2)
    title(ax, "Exchangeability")
    ax.legend(loc="upper center", fontsize=6.5)
    ax = axs[2]
    for take, r in _takes(res):
        m = np.concatenate([s for s in r["regions"]["miss_sequences"]]) if r["regions"]["miss_sequences"] else np.array([])
        if len(m) > 50:
            run = 1 - np.convolve(m, np.ones(50) / 50, mode="valid")
            ax.plot(np.arange(len(run)) + 50, run, color=TAKE_COLOR[take], lw=0.8)
    ax.axhline(1 - EPSILON, color=GOAL, ls="-.", lw=0.9)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Scored prediction (by track, in time)")
    ax.set_ylabel("Coverage, last 50")
    title(ax, "Stability over Time")
    style.save(pdf, fig)

    running_coverage(res, pdf)


def detection(res, pdf):
    takes = _takes(res)
    fig, axs = plt.subplots(len(takes), 1, figsize=(DOUBLE, 1.7 * len(takes) + 0.5), layout="constrained", sharey=True)
    for ax, (take, r) in zip(np.atleast_1d(axs), takes):
        d = r["detection"]
        t = np.array([r["looks"][k]["t"] for k in d["fdp_t"]])
        f = d["fdp"]
        run = np.array([f[(t >= x - 2.5) & (t <= x + 2.5)].mean() for x in t])
        cut = np.r_[0, np.flatnonzero(np.diff(t) > 0.5) + 1, len(t)]
        for a, b in zip(cut[:-1], cut[1:]):
            ax.plot(t[a:b], run[a:b], color=TAKE_COLOR[take], lw=0.9)
        ax.plot(t, np.cumsum(f) / np.arange(1, len(f) + 1), color=TAKE_COLOR[take], lw=1.4, ls="--")
        ax.axhline(ALPHA, color=GOAL, ls="-.", lw=0.9)
        _gaps(ax, r)
        ax.set_xlim(0, r["t_arr"][-1])
        ax.set_ylabel("FDP (5 s mean)")
        title(ax, f"{r['label']}: Detection FDR")
    np.atleast_1d(axs)[-1].set_xlabel("Time (s)")
    h = [Line2D([], [], color=INK, lw=0.9, label="false discovery proportion, 5 s mean"),
         Line2D([], [], color=INK, lw=1.4, ls="--", label="running FDR since the start"),
         Line2D([], [], color=GOAL, ls="-.", label=r"level $\alpha$ = 0.1"), Patch(color=MUTED, alpha=0.3, label="radar blind")]
    fig.legend(handles=h, loc="outside lower center", ncols=4)
    style.save(pdf, fig)

    fig, ax = plt.subplots(figsize=(DOUBLE, 2.2), layout="constrained")
    for take, r in takes:
        lam = np.array([(lk["t"], lk["lambda"]) for lk in r["looks"]])
        cut = np.r_[0, np.flatnonzero(np.diff(lam[:, 0]) > 0.5) + 1, len(lam)]
        for a, b in zip(cut[:-1], cut[1:]):
            ax.plot(lam[a:b, 0], lam[a:b, 1], color=TAKE_COLOR[take], lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"New-object share $\lambda$")
    title(ax, "Budget of e-BH for New Objects")
    fig.legend(handles=[Line2D([], [], color=TAKE_COLOR[k], label=v) for k, v in dataset.TAKES.items()],
               loc="outside lower center", ncols=2)
    style.save(pdf, fig)

    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), layout="constrained")
    ax = axs[0]
    for i, (take, r) in enumerate(_takes(res)):
        d = r["detection"]
        vals = [(d["precision"], None), (d["recall_ball"], d["n_ball"]), (d["recall_person"], d["n_person"])]
        for j, (v, n) in enumerate(vals):
            x = j + 0.14 * (2 * i - 1)
            if j == 0:
                lo, hi = 1 - d["fdr_ci"][1], 1 - d["fdr_ci"][0]
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=TAKE_COLOR[take], marker="o", ms=4, capsize=2, lw=0.8)
            elif n:
                from metrics import wilson
                lo, hi = wilson(v * n, n)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=TAKE_COLOR[take], marker="o", ms=4, capsize=2, lw=0.8)
            else:
                ax.plot(x, v, color=TAKE_COLOR[take], marker="o", ms=4)
    ax.set_xticks(range(3), ["precision", "recall ball", "recall person"])
    ax.set_ylim(0.4, 1.02)
    title(ax, "Precision and Recall per Look")
    ax = axs[1]
    for i, (take, r) in enumerate(_takes(res)):
        seen = [x for x in r["dets"] if x["in_view"] == 1 and x["moving_there"] != ""]
        tr = [x["snr_db"] for x in seen if x["moving_there"] == 1]
        fa = [x["snr_db"] for x in seen if x["moving_there"] == 0]
        _box(ax, [tr, fa], [0.35 * (2 * i - 1), 2 + 0.35 * (2 * i - 1)], TAKE_COLOR[take])
    ax.set_xticks([0, 2], ["something moves there", "nothing moves there"])
    ax.set_ylabel("SNR (dB)")
    title(ax, "Strength of True and False Detections")
    h = [Patch(color=TAKE_COLOR[k], label=v) for k, v in dataset.TAKES.items()]
    fig.legend(handles=h, loc="outside lower center", ncols=2)
    style.save(pdf, fig)


def _matrix(ax, true, pred, rows, cols, name):
    count = np.array([[np.sum((true == a) & (pred == b)) for b in cols] for a in rows], float)
    n = count.sum(axis=1, keepdims=True)
    share = count / np.maximum(n, 1)
    im = ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.tick_params(length=0, top=False, right=False)
    ax.set_xticks(range(len(cols)), [tex(c) for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), [tex(f"{a} ({int(k)})") for a, k in zip(rows, n[:, 0])])
    for i in range(len(rows)):
        for j in range(len(cols)):
            if count[i, j]:
                ax.text(j, i, f"{share[i, j] * 100:.0f}" if share[i, j] >= 0.005 else "$<$1", ha="center", va="center",
                        fontsize=6.5, color="white" if share[i, j] > 0.55 else INK)
    title(ax, f"{name}: {np.mean(true == pred) * 100:.1f}\\,%")
    return im


def _cols(c, pred):
    order = list(c["classes"]) + ["undecided"]
    return [x for x in order if np.any(pred == x)] + sorted(set(pred) - set(order))


def physics_page(res, pdf):
    from material_classification.dataset import read_table
    T = read_table(paths.MATERIAL / "output" / "dataset.csv")
    known = T["detected"] & (T["motion"] != "none")
    feats = (("reflectivity_db", "Reflectivity (dB)"), ("back_echo_db", "Back echo (dB)"), ("shadow_db", "Shadow (dB)"),
             ("ripple_db", "Ripple (dB)"))
    fig, axs = plt.subplots(1, len(feats), figsize=(DOUBLE, 2.6), layout="constrained")
    for ax, (f, lab) in zip(axs, feats):
        for i, m in enumerate(("metal", "dielectric", "waterlike")):
            _box(ax, [T[f][known & (T["material"] == m)]], [i], MATERIAL_COLORS[m])
        for j, o in enumerate(dataset.CLASSES):
            for i, (take, r) in enumerate(_takes(res)):
                v = [w["features"][f] for w in r["windows"] if w["truth"] == o]
                _box(ax, [v], [3.6 + 1.3 * j + 0.4 * (2 * i - 1)], TAKE_COLOR[take], 0.3)
        ax.set_xticks([0, 1, 2, 3.6, 4.9], ["metal", "dielectric", "waterlike", "ball", "person"], rotation=35)
        ax.axvline(2.8, color=MUTED, lw=0.6, ls=":")
        ax.set_ylabel(lab)
        title(ax, lab.split(" (")[0])
    h = [Patch(color=MATERIAL_COLORS[m], label=f"training: {m}") for m in ("metal", "dielectric", "waterlike")]
    h += [Patch(color=TAKE_COLOR[k], label=v) for k, v in dataset.TAKES.items()]
    fig.legend(handles=h, loc="outside lower center", ncols=5)
    style.title(fig, "Echo Features against the Training Classes")
    style.save(pdf, fig)


def classification(res, pdf):
    C = res["classification"]
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 3.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
    for col, h in enumerate(("material", "motion")):
        c = C[h]
        rows = [x for x in c["classes"] if np.any(c["true"] == x)]
        for row, (key, name) in enumerate((("pred", "every window"), ("vote", "track label (votes)"))):
            pred = c[key]
            im = _matrix(axs[row, col], c["true"], pred, rows, _cols(c, pred), f"{h.capitalize()}, {name}")
    for ax in axs[:, 0]:
        ax.set_ylabel("True class (windows)")
    for ax in axs[-1]:
        ax.set_xlabel("Predicted class")
    fig.colorbar(im, ax=axs, shrink=0.6, label="Share of the true class")
    style.title(fig, "Both takes")
    style.save(pdf, fig)

    for take, label in dataset.TAKES.items():
        fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 3.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
        for col, h in enumerate(("material", "motion")):
            c = C[h]
            sel = c["take"] == take
            rows = [x for x in c["classes"] if np.any(c["true"][sel] == x)]
            for row, (key, name) in enumerate((("pred", "every window"), ("vote", "track label (votes)"))):
                pred = c[key][sel]
                im = _matrix(axs[row, col], c["true"][sel], pred, rows, _cols(c, pred), f"{h.capitalize()}, {name}")
        style.title(fig, f"{label}, {dataset.SOURCE[take][0]} {take}")
        for ax in axs[:, 0]:
            ax.set_ylabel("True class (windows)")
        for ax in axs[-1]:
            ax.set_xlabel("Predicted class")
        fig.colorbar(im, ax=axs, shrink=0.6, label="Share of the true class")
        style.save(pdf, fig)

    physics_page(res, pdf)

    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), layout="constrained", gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axs[0]
    names, x = [], 0
    for h in ("material", "motion"):
        for r in C[h]["per_class"]:
            for j, (k, m) in enumerate((("precision", "o"), ("recall", "s"), ("f1", "D"), ("auc", "^"))):
                ax.plot(x + 0.15 * (j - 1.5), r[k], marker=m, ls="", ms=4,
                        color=(MATERIAL_COLORS if h == "material" else MOTION_COLORS)[r["class"]], mec=INK, mew=0.3)
            names.append(f"{r['class']}\n({r['n']})")
            x += 1
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    title(ax, "Per Class (Every Window, Both Takes Pooled)")
    ax.legend(handles=[Line2D([], [], marker=m, ls="", color=MUTED, mec=INK, mew=0.3, ms=4, label=l)
                       for m, l in (("o", "precision"), ("s", "recall"), ("D", "F1"), ("^", "AUC"))], loc="lower left", ncols=2)
    ax = axs[1]
    for i, h in enumerate(("material", "motion")):
        ax.bar(i - 0.17, C[h]["cross_entropy_bits"], 0.32, color=PRIMARY, label="M$^2$-SRC" if i == 0 else None)
        ax.bar(i + 0.17, C[h]["chance_bits"], 0.32, color="white", edgecolor=MUTED, hatch="///", label="uniform guess" if i == 0 else None)
    ax.set_xticks([0, 1], ["material", "motion"])
    ax.set_ylabel("Cross-entropy (bit)")
    title(ax, "Cross-Entropy")
    ax.legend(loc="upper left")
    style.save(pdf, fig)

    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 3.7), layout="constrained", gridspec_kw={"width_ratios": [1, 1.3]})
    for col, h in enumerate(("material", "motion")):
        c = C[h]
        rows = [x for x in c["classes"] if np.any(c["true"] == x)]
        cols = _cols(c, c["vote"])
        for k, (sel, name) in enumerate(((c["n"] <= 3, "after 1--3 windows"), (c["n"] > 3, "after 4 or more"))):
            im = _matrix(axs[k, col], c["true"][sel], c["vote"][sel], rows, cols, f"{h.capitalize()} label {name}")
    for ax in axs[:, 0]:
        ax.set_ylabel("True class (windows)")
    for ax in axs[-1]:
        ax.set_xlabel("Track label (votes so far)")
    fig.colorbar(im, ax=axs, shrink=0.7, label="Share of the true class")
    style.save(pdf, fig)


def timing(res, pdf):
    from tables import STAGES
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), layout="constrained", gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axs[0]
    for i, (take, r) in enumerate(_takes(res)):
        ms = r["timing"]["ms"]
        _box(ax, [ms[s] for s, _ in STAGES], [2 * q + 0.35 * (2 * i - 1) for q in range(len(STAGES))], TAKE_COLOR[take])
    ax.axhline(100, color=MUTED, ls="--", lw=0.8)
    ax.text(-0.8, 125, "frame period", fontsize=6.5, color=MUTED, ha="left")
    ax.set_ylim(1e-3, 400)
    short = {"range_doppler": "Range-\nDoppler", "infinirex_fdr": "Infini-Rex\n+ e-BH", "predict": "Predict",
             "track": "Track +\nKalmanCQR", "material": "M$^2$-SRC", "cognitive": "Chirp\nchoice", "total": "Total"}
    ax.set_xticks([2 * q for q in range(len(STAGES))], [short[k] for k, _ in STAGES])
    ax.set_yscale("log")
    ax.set_ylabel("Time per look (ms)")
    title(ax, "Run Time per Stage (One Core)")
    ax = axs[1]
    for i, (take, r) in enumerate(_takes(res)):
        lat = r["timing"]["radar_latency_ms"]
        proc = float(np.nanmedian(r["timing"]["ms"]["total"]))
        ax.bar(i, lat, 0.5, color="white", edgecolor=TAKE_COLOR[take], hatch="///")
        ax.bar(i, proc, 0.5, bottom=lat, color=TAKE_COLOR[take])
        ax.text(i, lat + proc + 3, f"{lat + proc:.0f} ms", ha="center", fontsize=7)
    ax.set_ylim(0, 215)
    ax.set_xticks([0, 1], [r["label"] for _, r in _takes(res)])
    ax.set_ylabel("Echo to result (ms, medians)")
    title(ax, "Latency")
    h = [Patch(color=TAKE_COLOR[k], label=v) for k, v in dataset.TAKES.items()]
    h += [Patch(facecolor="white", edgecolor=INK, hatch="///", label="radar to PC (fitted)"),
          Patch(color=INK, label="processing (median)")]
    fig.legend(handles=h, loc="outside lower center", ncols=4)
    style.save(pdf, fig)


POLICY_COLOR = {"run_1_search": ACCENT, "run_1": PRIMARY, "run_1_all": NORM}


def policy_numbers(r) -> dict:
    lat = [rec.get("t_ready", rec["t"] + rec["ms"]["total"] / 1e3) - rec["t"] for rec in r["recs"]]
    return {"HOTA, Per Look (\\%)": 100 * r["mot"]["look"]["hota"]["HOTA"],
            "HOTA, Live (\\%)": 100 * r["mot"]["live"]["hota"]["HOTA"],
            "IDF1, Live (\\%)": 100 * r["mot"]["live"]["idf1"]["IDF1"],
            "Looks per Second": len(r["looks"]) / r["duration_s"],
            "Latency (ms)": 1e3 * float(np.median(lat)),
            "Fast Looks (\\%)": 100 * float(np.mean([lk["waveform"] != "search" for lk in r["looks"]]))}


def policies(res, pdf):
    P = res["policies"]
    nums = {k: policy_numbers(P[k]) for k in dataset.POLICIES}
    names = list(next(iter(nums.values())))
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE, 3.4), layout="constrained")
    for ax, m in zip(axs.ravel(), names):
        for i, k in enumerate(dataset.POLICIES):
            ax.bar(i, nums[k][m], color=POLICY_COLOR[k], width=0.7)
            ax.text(i, nums[k][m], f"{nums[k][m]:.0f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(dataset.POLICIES)), [])
        ax.margins(y=0.15)
        title(ax, m)
    h = [Patch(color=POLICY_COLOR[k], label=v) for k, v in dataset.POLICIES.items()]
    fig.legend(handles=h, loc="outside lower center", ncols=3, title="Same recording (TR2 run\\_1), three look policies")
    style.save(pdf, fig)


FIGS = (("scene", scene), ("tracking", tracking), ("information", information), ("regions", regions),
        ("detection", detection), ("classification", classification), ("timing", timing), ("policies", policies))


def readme_figure(res, path):
    style.use()
    r = res["takes"]["run_1"]
    fig = plt.figure(figsize=(DOUBLE, 3.6), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    spacetime(ax, r, every=4, simple=True)
    _spacetime_legend(fig, simple=True)
    style.save(path, fig)


def draw(res):
    style.use()
    for name, fn in FIGS:
        with style.pdf(FIGURES / f"{name}.pdf") as pdf:
            fn(res, pdf)
        print(f"  figures/{name}.pdf", flush=True)
    readme_figure(res, FIGURES / "readme.svg")
    print("  figures/readme.svg")


if __name__ == "__main__":
    import pickle
    res = pickle.loads((paths.RESULTS / "results.pkl").read_bytes())
    only = sys.argv[1:]
    style.use()
    for name, fn in FIGS:
        if not only or name in only:
            with style.pdf(FIGURES / f"{name}.pdf") as pdf:
                fn(res, pdf)
            print(f"  figures/{name}.pdf", flush=True)
    if not only or "readme" in only:
        readme_figure(res, FIGURES / "readme.svg")
