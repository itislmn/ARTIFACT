"""
plots.py - the thesis figures of the experiment as PDFs (ARTIFACT-TeX): the scene in space and time, tracking, scene
understanding, motion and decisions, camera (light and reading), classification, the system (chirps, blind time, run
time, latency) of the method, its comparison with the baselines on the same recordings, the DoA study and the README
picture.

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
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import paths  # noqa: E402
from visualization import style  # noqa: E402
from visualization.style import (ACCENT, DOUBLE, GOAL, GOOD, INK, LOSS, MATERIAL_COLORS, MOTION_COLORS, MUTED,  # noqa: E402
                                 NORM, PRIMARY, SEQUENTIAL, SERIES, tex, title)

FIGURES = paths.OUTPUT / "figures"
STATE = {"idle": MUTED, "cruise": PRIMARY, "approach": SERIES[2], "return": SERIES[4], "wait": ACCENT, "blocked": NORM,
         "decide": SERIES[5], "avoid": SERIES[3], "verify": SERIES[3], "done": MUTED}
EVENT = {"person": ACCENT, "light": GOAL, "flashing": GOAL}
TRAIL_S = 3.0


def _short(run):
    return r"$\mathtt{" + run.split("/")[-1].replace("_", r"\_") + "}$"


def _label(tr, head="material"):
    return ((tr.get("classes") or {}).get(head) or {}).get("label")


def wedge(tr, pose, n=12):
    g = tr.get("region") or {}
    lo, hi = g.get("lo"), g.get("hi")
    if not g.get("calibrated") or lo is None or hi is None or not np.all(np.isfinite(np.asarray(lo[:2] + hi[:2], float))):
        return None
    th = np.linspace(*np.clip([lo[1], hi[1]], -np.radians(60), np.radians(60)), n)
    r_lo, r_hi = np.clip([lo[0], hi[0]], 0.25, 3.2)
    r = np.r_[np.full(n, r_lo), np.full(n, r_hi)]
    a = np.r_[th, th[::-1]]
    return np.c_[pose[1] + r * np.sin(a), pose[0] + r * np.cos(a)]


def _pane(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)
        a._axinfo["grid"]["linewidth"] = 0.3
    ax.tick_params(pad=-2)


def _column(ax, lo, hi, T, color, alpha, ls="-"):
    y0, y1, x0, x1 = lo[1], hi[1], lo[0], hi[0]
    for t in (0, T):
        ax.plot([t] * 5, [y0, y1, y1, y0, y0], [x0, x0, x1, x1, x0], color=color, lw=0.6, ls=ls, alpha=min(1, alpha * 4))
    for y, x in ((y0, x0), (y1, x0), (y1, x1), (y0, x1)):
        ax.plot([0, T], [y, y], [x, x], color=color, lw=0.4, ls=ls, alpha=min(1, alpha * 3))


def spacetime(ax, looks, truth, Y=(-1.6, 1.6), X=(-0.1, 3.0)):
    t = np.array([q[0] for q in looks])
    T = t[-1] if len(t) else 1.0
    known = {}
    paths_ = {}
    pts = []
    for k, (ti, pose, objs, tracks, dets) in enumerate(looks):
        for o in objs:
            if o.get("known"):
                known[o["id"]] = o
        for tr in tracks:
            if tr["status"] == "confirmed":
                paths_.setdefault(tr["id"], []).append((ti, tr["wy"], tr["wx"], MATERIAL_COLORS.get(_label(tr), MUTED)))
                w = wedge(tr, pose) if k % 16 == 0 else None
                if w is not None:
                    w = np.c_[np.full(len(w), ti), np.clip(w[:, 0], *Y), np.clip(w[:, 1], *X)]
                    ax.add_collection3d(Poly3DCollection([w], facecolor=MATERIAL_COLORS.get(_label(tr), MUTED), alpha=0.08,
                                                         edgecolor=MOTION_COLORS.get(_label(tr, "motion"), MUTED), lw=0.3))
        if k % 2 == 0:
            pts += [(ti, d[1], d[0]) for d in dets]
    p = np.array(pts, float).reshape(-1, 3)
    p = p[(p[:, 1] >= Y[0]) & (p[:, 1] <= Y[1]) & (p[:, 2] >= X[0]) & (p[:, 2] <= X[1])]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.4, color=INK, alpha=0.25, depthshade=False, lw=0)
    for o in known.values():
        _column(ax, o["lo"], o["hi"], T, MUTED, 0.25)
    for q in truth:
        if q.kind != "person":
            _column(ax, q.lo, q.hi, T, GOAL, 0.25, "-.")
    for seg in paths_.values():
        a = np.array([s[:3] for s in seg], float)
        for i in range(len(a) - 1):
            if a[i + 1, 0] - a[i, 0] < 0.6:
                ax.plot(a[i:i + 2, 0], np.clip(a[i:i + 2, 1], *Y), np.clip(a[i:i + 2, 2], *X), color=seg[i + 1][3], lw=1.1)
    ax.plot(t, [q[1][1] for q in looks], [q[1][0] for q in looks], color=NORM, lw=2.0)
    ax.set_xlim(0, T)
    ax.set_ylim(*Y)
    ax.set_zlim(*X)
    ax.set_xlabel("Time (s)", labelpad=2)
    ax.set_ylabel("Sideways $y$ (m)", labelpad=-1)
    ax.set_zlabel("Forward $x$ (m)", labelpad=-3)
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((2.6, 1.3, 1.0), zoom=1.32)
    _pane(ax)


def _states(ax, states, ev, T):
    for name, ivs in ev.items():
        if name in EVENT:
            for a, b in ivs:
                ax.axvspan(a, b, ymin=0.0, ymax=0.35, color=EVENT[name], alpha=0.5 if name == "flashing" else 0.3, lw=0)
    for (a, s, _), (b, _, _) in zip(states, states[1:] + [(T, "", 0)]):
        ax.axvspan(a, b, ymin=0.4, ymax=1.0, color=STATE.get(s, MUTED), lw=0)
    ax.set_xlim(0, T)
    ax.set_yticks([0.17, 0.7], ["events", "planner"])
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.set_xlabel("Time (s)")


def scene(res, pdf):
    for run, looks in res["tracks"].items():
        truth, ev, states = res["truth"][run], res["events"][run], res["states"][run]
        T = looks[-1][0] if looks else 1.0
        fig = plt.figure(figsize=(DOUBLE, 4.8), layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=[5, 0.6])
        ax = fig.add_subplot(gs[0], projection="3d")
        spacetime(ax, looks, truth)
        title(ax, f"{_short(run)} in Space and Time")
        _states(fig.add_subplot(gs[1]), states, ev, T)
        h = [Line2D([], [], color=MATERIAL_COLORS[m], lw=1.4, label=f"track: {m}") for m in ("metal", "dielectric", "waterlike")]
        h += [Line2D([], [], color=MUTED, lw=1.4, label="track: unclassified"),
              Patch(facecolor=MUTED, alpha=0.3, edgecolor=INK, lw=0.3, label="next-look region (KalmanCQR)"),
              Line2D([], [], color=GOAL, ls="-.", lw=0.8, label="measured object"),
              Line2D([], [], color=MUTED, lw=0.8, label="known room"), Line2D([], [], color=NORM, lw=2, label="rail"),
              Line2D([], [], marker=".", ls="", color=INK, alpha=0.5, label="detection")]
        h += [Patch(color=STATE[s], label=s) for s in sorted({s for _, s, _ in states}) if s in STATE]
        h += [Patch(color=EVENT[e], alpha=0.4, label=f"{e} (annotated)") for e in ("person", "light", "flashing") if ev.get(e)]
        fig.legend(handles=h, loc="outside lower center", ncols=6)
        style.save(pdf, fig)


def tracking(res, pdf):
    tr, g, d = res["tracking"], res["regions"], res["detection"]
    views = [v for v in ("look", "live") if v in tr]
    fig, a4 = plt.subplots(2, 2, figsize=(DOUBLE, 4.4), layout="constrained")
    axs = {"scores": a4[0, 0], "motp": a4[0, 1], "cov": a4[1, 0], "fdr": a4[1, 1]}
    ax = axs["scores"]
    keys = ("HOTA", "MOTA", "IDF1")
    for v, view in enumerate(views):
        pos = [2 * q + 0.36 * (2 * v - 1) for q in range(len(keys))]
        b = _box(ax, [[seg[k] for seg in tr[view]["segments"]] for k in keys], pos, PRIMARY, 0.3)
        if b and view == "live":
            for patch in b["boxes"]:
                patch.set_facecolor("white")
        whole = [tr[view]["hota"]["HOTA"], tr[view]["clear"]["MOTA"], tr[view]["idf1"]["IDF1"]]
        ax.plot(pos, whole, ls="", marker="D", ms=3.2, color=INK, mfc=PRIMARY if view == "look" else "white", zorder=4)
    ax.set_xticks([2 * q for q in range(len(keys))], keys)
    ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.0)
    ax.set_ylabel("Score (10 s Segments)")
    title(ax, "Tracking Scores")
    ax = axs["motp"]
    mk = list(tr["motp"])
    _box(ax, [100 * np.asarray(tr["motp"][k]["segments_m"]) for k in mk], list(range(len(mk))), PRIMARY, 0.4)
    ax.plot(range(len(mk)), [100 * tr["motp"][k]["mean_m"] for k in mk], "D", ms=3.2, color=INK, mfc=PRIMARY, zorder=4)
    ax.set_xticks(range(len(mk)), mk)
    ax.set_xlim(-0.7, len(mk) - 0.3)
    ax.set_ylim(0, None)
    ax.set_ylabel("Centre Error (cm)")
    title(ax, "Localisation (MOTP)")
    ax = axs["cov"]
    cov = g["coverage"]
    hs = [("step", "next look"), ("0.5", "0.5 s"), ("1.0", "1 s")]
    for i, (h, lbl) in enumerate(hs):
        m, lo, hi, n = cov[h]
        if np.isfinite(m):
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]] if np.isfinite(lo) else None, fmt="o", color=PRIMARY, capsize=2, ms=4)
            ax.text(i, 0.715, f"n={n}", ha="center", fontsize=6.5, color=MUTED)
    ax.axhline(g["nominal"], color=GOAL, ls="-.", lw=0.8)
    ax.set_xticks(range(3), [l for _, l in hs])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Coverage (Fraction)")
    title(ax, "Prediction Regions (KalmanCQR)")
    ax = axs["fdr"]
    runs = list(dict.fromkeys(r for r, _, _ in d["series"]))
    for i, run in enumerate(runs):
        q = np.array([(t, f) for r, t, f in d["series"] if r == run])
        w = min(50, len(q))
        mean = np.convolve(q[:, 1], np.ones(w), mode="same") / np.convolve(np.ones(len(q)), np.ones(w), mode="same")
        ax.plot(q[:, 0], mean, color=SERIES[i % len(SERIES)], lw=1.0)
    ax.axhline(d["alpha"], color=GOAL, ls="-.", lw=0.8)
    ax.set_ylim(0, max(0.4, 2 * d["alpha"]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("FDP (5 s Mean)")
    title(ax, f"Detections: FDR {d['fdr']:.2f} (Mean FDP)")
    h = [Patch(color=PRIMARY, label="per look"), Patch(facecolor="white", edgecolor=PRIMARY, label="real time"),
         Line2D([], [], marker="D", color=INK, ls="", ms=3, label="whole run"),
         Line2D([], [], color=GOAL, ls="-.", lw=0.8, label=r"target ($1-\varepsilon$, $\alpha$)")]
    h += [Line2D([], [], color=SERIES[i % len(SERIES)], lw=1.0, label=_short(run)) for i, run in enumerate(runs)]
    fig.legend(handles=h, loc="outside lower center", ncols=7)
    style.save(pdf, fig)
    timeline(res, pdf)

    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 4.4), layout="constrained")
    axs = axs.ravel()
    hs = (("step", "next look", "-"), ("0.5", "0.5 s", "--"), ("1.0", "1 s", ":"))
    for ax, k, ylab in ((axs[0], 2, "Running Coverage (Fraction)"), (axs[1], 3, r"Adaptive Level $\varepsilon_t$")):
        for h, lbl, ls in hs:
            q = [x for x in g["series"][h]]
            for i, run in enumerate(dict.fromkeys(x[0] for x in q)):
                a = np.array([(x[1], x[k]) for x in q if x[0] == run], float)
                ax.plot(a[:, 0], a[:, 1], color=SERIES[i % len(SERIES)], ls=ls, lw=0.9, label=lbl if i == 0 else None)
        ax.axhline(g["nominal"] if k == 2 else g["alpha"], color=GOAL, ls="-.", lw=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylab)
    axs[0].set_ylim(0.6, 1.02)
    title(axs[0], "Stability over Time")
    title(axs[1], "Adaptive Conformal Inference")
    ax = axs[2]
    ex = g["exchange"]
    if len(ex["p_values"]):
        ax.hist(ex["p_values"], bins=np.linspace(0, 1, 11), density=True, histtype="step", color=PRIMARY, lw=1.2)
    ax.axhline(1, color=INK, ls=":", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.0)
    ax.set_xlabel("Conformal $p$-Value of Each New Score")
    ax.set_ylabel("Density")
    title(ax, f"Exchangeability (KS $p$ = {ex['ks_p']:.2f})")
    ax = axs[3]
    labs = [k for k in ("dielectric", "metal", "waterlike", "unclassified") if k in g["widths"]]
    for i, k in enumerate(labs):
        for j in (0, 1):
            b = _box(ax, [100 * g["widths"][k][:, j]], [3 * i + j - 0.5], MATERIAL_COLORS.get(k, MUTED), 0.7)
            if b and j == 1:
                b["boxes"][0].set_facecolor("white")
    ax.set_yscale("log")
    ax.set_xticks([3 * i for i in range(len(labs))], labs)
    ax.set_ylabel("Region Size (cm)")
    title(ax, "Next-Look Region Size")
    h = [Line2D([], [], color=INK, ls=ls, lw=0.9, label=lbl) for _, lbl, ls in hs]
    h += [Line2D([], [], color=GOAL, ls="-.", lw=0.8, label=r"target ($1-\varepsilon$, $\varepsilon$)"),
          Line2D([], [], color=INK, ls=":", lw=0.8, label="uniform"), Patch(color=MUTED, label="region: in range"),
          Patch(facecolor="white", edgecolor=MUTED, label="region: across")]
    fig.legend(handles=h, loc="outside lower center", ncols=7)
    p_ = g["runs_p"]
    style.title(fig, "Prediction Regions")
    style.save(pdf, fig)

    gospa_page(res, pdf)


def timeline(res, pdf):
    for run, looks in res["tracks"].items():
        truth, ev = res["truth"][run], res["events"][run]
        T = looks[-1][0] if looks else 1.0
        fig, axs = plt.subplots(2, 1, figsize=(DOUBLE, 3.8), layout="constrained", sharex=True)
        for ax, k, ylab, lim in ((axs[0], 0, "Sideways $y$ (m)", (-1.0, 1.0)), (axs[1], 1, "Forward $x$ (m)", (0.0, 2.8))):
            for a, b in ev.get("person", []):
                ax.axvspan(a, b, color=EVENT["person"], alpha=0.10, lw=0)
            for q in truth:
                if q.kind != "person":
                    ax.axhspan(*((q.lo[1], q.hi[1]) if k == 0 else (q.lo[0], q.hi[0])), color=GOAL, alpha=0.12, lw=0)
            p = np.array([(t, d[1 - k]) for t, _, _, _, dets in looks for d in dets], float).reshape(-1, 2)
            ax.scatter(p[:, 0], p[:, 1], s=0.8, color=INK, alpha=0.3, lw=0)
            paths = {}
            for i, (t, pose, objs, tracks, _) in enumerate(looks):
                for tr in tracks:
                    if tr["status"] != "confirmed":
                        continue
                    c = MATERIAL_COLORS.get(_label(tr), MUTED)
                    paths.setdefault(tr["id"], []).append((t, tr["wy"] if k == 0 else tr["wx"], c))
                    w = wedge(tr, pose) if i % 2 == 0 else None
                    if w is not None:
                        ax.plot([t, t], [w[:, k].min(), w[:, k].max()], color=c, alpha=0.2, lw=1.4, solid_capstyle="butt")
            for seg in paths.values():
                segs = [[(a[0], a[1]), (b[0], b[1])] for a, b in zip(seg, seg[1:]) if b[0] - a[0] < 0.6]
                ax.add_collection(LineCollection(segs, colors=[b[2] for a, b in zip(seg, seg[1:]) if b[0] - a[0] < 0.6], lw=1.1))
            if k == 1:
                ax.plot([q[0] for q in looks], [q[1][0] for q in looks], color=NORM, lw=1.6)
            ax.set_ylim(*lim)
            ax.set_ylabel(ylab)
        axs[1].set_xlim(0, T)
        axs[1].set_xlabel("Time (s)")
        title(axs[0], f"{_short(run)}: Tracks over Time")
        h = [Line2D([], [], color=MATERIAL_COLORS[m], lw=1.4, label=m) for m in ("metal", "dielectric", "waterlike")]
        h += [Line2D([], [], color=MUTED, lw=1.4, label="not yet classified"),
              Patch(color=MUTED, alpha=0.3, label=r"prediction region (90\,\%)"),
              Line2D([], [], marker=".", ls="", color=INK, alpha=0.5, label="detection (FDR 0.1)"),
              Patch(color=GOAL, alpha=0.25, label="measured object"), Patch(color=EVENT["person"], alpha=0.2, label="person in view"),
              Line2D([], [], color=NORM, lw=1.6, label="radar (rail)")]
        fig.legend(handles=h, loc="outside lower center", ncols=5)
        style.save(pdf, fig)


def gospa_page(res, pdf):
    from scene_metrics import GOSPA_C_M
    tr = res["tracking"]
    parts = (("loc", "localisation", SERIES[2]), ("missed", "missed objects", SERIES[3]), ("false", "false objects", MUTED))
    fig = plt.figure(figsize=(DOUBLE, 2.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
    ax = fig.add_subplot(gs[0])
    fr = tr["identity"]
    t = np.array([f["t"] for f in fr])
    grid = np.arange(0, t[-1], 0.5)
    stack = []
    for key, _, _ in parts:
        v = np.array([f["gospa"][key] for f in fr])
        stack.append([v[(t >= x - 1.0) & (t < x + 1.0)].mean() if np.any((t >= x - 1.0) & (t < x + 1.0)) else np.nan
                      for x in grid])
    ax.stackplot(grid, *[np.nan_to_num(q) for q in stack], colors=[c for _, _, c in parts], alpha=0.85, lw=0)
    start = 0.0
    for run, dur in tr["runs"]:
        ax.text(start + dur / 2, 0.98, _short(run), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, color=INK)
        ax.axvspan(start + dur, start + dur + 10.0, color="white", lw=0)
        start += dur + 10.0
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"GOSPA$^2$ (m$^2$, 2 s Mean)")
    title(ax, "What the GOSPA Error Is Made Of (Per Look)")
    ax = fig.add_subplot(gs[1])
    for v, view in enumerate(v for v in ("look", "live") if v in tr):
        x = 0.36 * (2 * v - 1)
        b = _box(ax, [[q["GOSPA"] for q in tr[view]["segments"]]], [x], PRIMARY, 0.3)
        if b and view == "live":
            b["boxes"][0].set_facecolor("white")
        ax.plot(x, tr[view]["gospa"]["gospa"], ls="", marker="D", ms=3.2, color=INK,
                mfc=PRIMARY if view == "look" else "white", zorder=4)
        ax.plot(x, tr[view]["gospa"]["ospa"], ls="", marker="o", ms=3.2, color=INK, mfc="none", zorder=4)
    ax.axhline(GOSPA_C_M / np.sqrt(2), color=MUTED, ls=":", lw=0.8)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0, None)
    ax.set_xticks([-0.36, 0.36], ["per look", "real time"])
    ax.set_ylabel("GOSPA per Look (m), 10 s Segments")
    title(ax, f"GOSPA ($c$ = {GOSPA_C_M:g} m, $p$ = 2)")
    g = tr["look"]["gospa"]
    h = [Patch(color=c, alpha=0.85, label=lbl) for _, lbl, c in parts]
    h += [Line2D([], [], marker="D", color=INK, ls="", ms=3, label="whole run, GOSPA"),
          Line2D([], [], marker="o", color=INK, mfc="none", ls="", ms=3, label="whole run, OSPA"),
          Line2D([], [], color=MUTED, ls=":", label="one object missed per look")]
    fig.legend(handles=h, loc="outside lower center", ncols=6)
    style.title(fig, "Tracking Error per Look (GOSPA)")
    style.save(pdf, fig)



def motion(res, pdf):
    mo, dec, w = res["motion"], res["decisions"], res["waits"]
    moved = any(np.ptp(np.array(r["trace"], float)[:, 1]) > 0 for r in mo["runs"] if len(r["trace"]))
    side = any(np.ptp(np.array(r["trace"], float)[:, 2]) > 0 for r in mo["runs"] if len(r["trace"]))
    approach = any(r["approach"] for r in mo["runs"])
    panels = (["rail"] if moved else []) + (["side"] if side else []) + (["approach"] if approach else []) + \
        (["decisions"] if dec["n"] else []) + ["waits"]
    rows = 2 if len(panels) > 3 else 1
    cols = -(-len(panels) // rows)
    fig, axs = plt.subplots(rows, cols, figsize=(DOUBLE if len(panels) > 1 else style.SINGLE, 2.3 * rows),
                            layout="constrained", squeeze=False)
    ax_of = dict(zip(panels, axs.ravel()))
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    T = max([1.0] + [float(np.array(r["trace"], float)[-1, 0]) for r in mo["runs"] if len(r["trace"])])
    if "rail" in ax_of:
        ax = ax_of["rail"]
        for i, r in enumerate(mo["runs"]):
            q = np.array(r["trace"], float)
            ax.plot(q[:, 0], 1e3 * q[:, 1], color=SERIES[i % len(SERIES)], lw=1.0, label=_short(r["take"]))
        ax.set_xlim(0, T)
        for run, truth in res["truth"].items():
            for q in truth:
                if q.kind == "obstacle":
                    ax.axhline(1e3 * q.lo[0], color=GOAL, ls="-.", lw=0.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Antenna $x$ (mm)")
        title(ax, "Rail Position")
    if "side" in ax_of:
        ax = ax_of["side"]
        for i, r in enumerate(mo["runs"]):
            q = np.array(r["trace"], float)
            ax.plot(q[:, 0], 1e3 * q[:, 2], color=SERIES[i % len(SERIES)], lw=1.0)
        ax.set_xlim(0, T)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Antenna $y$ (mm)")
        title(ax, "Sideways Steps")
    if "approach" in ax_of:
        ax = ax_of["approach"]
        for i, r in enumerate(mo["runs"]):
            if r["approach"]:
                a = np.array(r["approach"])
                ax.plot(a[:, 0], 100 * a[:, 1], color=SERIES[i % len(SERIES)], lw=1.0, label=_short(r["take"]))
        ax.set_xlim(0, T)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Antenna to Obstacle Face (cm)")
        title(ax, "Approach")
    if "decisions" in ax_of:
        ax = ax_of["decisions"]
        for i, r in enumerate(dec["runs"]):
            ax.bar(i, 100 * r["clearance_m"], 0.6, color=GOOD if r["clearance_m"] >= 0 else LOSS,
                   hatch="" if r["correct"] else "///", edgecolor=INK, lw=0.3)
        ax.axhline(0, color=NORM, lw=0.6)
        ax.set_xticks(range(dec["n"]), [_short(r["take"]) for r in dec["runs"]], rotation=30, ha="right")
        title(ax, f"Decisions: {dec['correct']}/{dec['n']} Right")
        ax.set_ylabel("Minimum Clearance (cm)")
    ax = ax_of["waits"]
    lat = (("person $\\to$ wait", w["latency_s"]), ("wait $\\to$ still", w["still_s"]))
    top = 0.0
    for i, (_, v) in enumerate(lat):
        if len(v) >= 3:
            _box(ax, [v], [i], PRIMARY)
        ax.plot(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.1, 0.1, len(v)), v, "o", ms=2.5, color=INK, alpha=0.7)
        top = max(top, float(np.max(v)) if len(v) else 0.0)
    ax.set_ylim(0, max(1.0, 1.25 * top))
    ax.set_xticks(range(2), [n for n, _ in lat])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Reaction Time (s)")
    title(ax, f"Waits: {w['justified']}/{w['onsets']} Justified")
    h = [Line2D([], [], color=SERIES[i % len(SERIES)], lw=1.0, label=_short(r["take"])) for i, r in enumerate(mo["runs"])]
    h += [Line2D([], [], color=GOAL, ls="-.", lw=0.6, label="obstacle face")] if "rail" in ax_of else []
    h += [Patch(color=GOOD, label="no contact"), Patch(color=LOSS, label="contact"),
          Patch(facecolor="white", edgecolor=INK, hatch="///", label="not the oracle's decision")] if "decisions" in ax_of else []
    fig.legend(handles=h, loc="outside lower center", ncols=8)
    style.save(pdf, fig)


def stills(res, pdf):
    import cv2
    picks = []
    for run in res["tracks"]:
        video = paths.VIDEOS / f"{run.replace('/', '_')}.mp4"
        if not video.exists():
            continue
        st = res["states"][run]
        T = res["tracks"][run][-1][0]
        ev = next((t for t, s, _ in st if s in ("avoid", "blocked", "wait")), None)
        for t in ([ev] if ev is not None else []) + [0.5 * T]:
            picks.append((run, video, t))
    for k in range(0, len(picks), 2):
        page = picks[k:k + 2]
        fig, axs = plt.subplots(len(page), 1, figsize=(DOUBLE, len(page) * DOUBLE * 720 / 1280 + 0.3), layout="constrained",
                                squeeze=False)
        for ax, (run, video, t) in zip(axs[:, 0], page):
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, img = cap.read()
            cap.release()
            if ok:
                ax.imshow(img[:, :, ::-1])
            ax.axis("off")
            title(ax, f"{_short(run)}, $t$ = {t:.1f} s")
        style.save(pdf, fig)


def crops(res, pdf, per_role: int = 8):
    import cv2
    from experiment import _crop
    roles = [r for r in ("ball", "person", "collider") if any(c["role"] == r for q in res["camera"]["runs"] for c in q["crops"])]
    if not roles:
        return
    fig, axs = plt.subplots(len(roles), per_role, figsize=(DOUBLE, 1.02 * len(roles) * DOUBLE / per_role + 0.35),
                            layout="constrained", squeeze=False)
    for i, role in enumerate(roles):
        pool = [(q, c) for q in res["camera"]["runs"] for c in q["crops"] if c["role"] == role]
        pick = [pool[int(k)] for k in np.linspace(0, len(pool) - 1, min(per_role, len(pool)))]
        for j, ax in enumerate(axs[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            if j >= len(pick):
                ax.axis("off")
                continue
            q, c = pick[j]
            cap = cv2.VideoCapture(q["video"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, c["pos"])
            ok, img = cap.read()
            cap.release()
            crop = _crop(img[:, :, ::-1], c["rect"], 160) if ok else None
            if crop is not None:
                ax.imshow(crop)
            ax.set_xlabel(f"{_short(q['take'])}, {c['t_s']:.0f} s", fontsize=6.5, labelpad=1.5)
        axs[i, 0].set_ylabel(f"{role} ({len(pool)})", fontsize=8)
    style.title(fig, "Attention Crops per Role")
    style.save(pdf, fig)


def camera(res, pdf):
    stills(res, pdf)
    crops(res, pdf)
    cam, li = res["camera"], res["light"]
    lit = any(r["events"].get("light") or r["events"].get("flashing") or np.any(np.asarray(r["light"]) > 0) for r in cam["runs"])
    if lit:
        fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 4.2), layout="constrained")
        axs = axs.ravel()
    else:
        fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.3), layout="constrained")
        axs = np.r_[[None], axs]
    ax = axs[0]
    for i, r in enumerate(cam["runs"] if lit else []):
        for name in ("light", "flashing"):
            for a, b in r["events"].get(name, []):
                ax.fill_between([a, b], i - 0.4, i + 0.4, color=GOAL, alpha=0.35, lw=0)
        on = r["light"] > 0
        ax.fill_between(r["t"], i - 0.25, i + 0.25, where=on, color=PRIMARY, step="post", lw=0)
    if lit:
        ax.set_yticks(range(len(cam["runs"])), [_short(r["take"]) for r in cam["runs"]])
        ax.set_xlim(0, max([float(r["t"][-1]) for r in cam["runs"] if len(r["t"])] + [1.0]))
        ax.set_ylim(-0.6, len(cam["runs"]) - 0.4)
        ax.set_xlabel("Time (s)")
        title(ax, "Light: Annotated (Gold), Detected (Blue)")
    ax = axs[1]
    reads = li["reads"]
    groups = (("no light", [r["share"] for r in reads if not r["light_true"]], PRIMARY),
              ("light, kept", [r["share"] for r in reads if r["light_true"] and r["accepted"]], GOOD),
              ("light, filtered", [r["share"] for r in reads if r["light_true"] and not r["accepted"]], LOSS))
    groups = [g for g in groups if len(g[1])]
    for i, (_, v, c) in enumerate(groups):
        _box(ax, [v], [i], c)
        ax.text(i, 1.02, f"{len(v)}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(groups)), [q[0] for q in groups], rotation=20)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Letters Read Right (Fraction)")
    title(ax, "Reading")
    ax = axs[2]
    for i, r in enumerate(cam["runs"]):
        if r["stable"]:
            q = np.array(r["stable"], float)
            ax.step(q[:, 0], q[:, 1], where="post", color=SERIES[i % len(SERIES)], lw=1.0, label=_short(r["take"]))
            ax.axhline(r["letters"], color=GOAL, ls="-.", lw=0.8)
    ax.set_ylim(0, max([r["letters"] for r in cam["runs"]] + [1]) + 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Correct Letters (Count)")
    title(ax, "Stable Reading")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="center right")
    ax = axs[3]
    sel = li["selective"]
    if len(sel["coverage"]):
        ax.plot(sel["coverage"], sel["accuracy"], color=PRIMARY, lw=1.0, label="most confident first")
        ax.axhline(sel["accuracy"][-1], color=MUTED, ls="--", lw=0.8, label="all reads")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Reads Kept (Fraction)")
    ax.set_ylabel("Letters Read Right (Fraction)")
    title(ax, f"Selective Reading: AUROC {sel['auc']:.2f}")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right")
    style.save(pdf, fig)


def system(res, pdf):
    from tables import STAGES
    c, lat = res["cognition"], res["latency"]
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 4.2), layout="constrained")
    ax = axs[0, 0]
    plans = sorted(c["plans"].items(), key=lambda kv: -kv[1])
    ax.barh(range(len(plans)), [k / max(c["looks"], 1) for _, k in plans], 0.6, color=PRIMARY)
    ax.set_yticks(range(len(plans)), [p.replace("+", " + ") for p, _ in plans])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of Looks (Fraction)")
    title(ax, "Chirps Processed")
    ax = axs[0, 1]
    g = 1e3 * c["gaps_s"]
    if len(g):
        ax.hist(g, bins=np.linspace(0, max(300, g.max() * 1.05), 40), color=PRIMARY)
        ax.axvline(1.5e3 * c["frame_s"], color=MUTED, ls="--", lw=0.8)
        ax.set_yscale("log")
    ax.set_xlabel("Time Between Search Looks (ms)")
    ax.set_ylabel("Looks (Count)")
    title(ax, "Blind Time")
    ax = axs[1, 0]
    keys = [(k, n) for k, n in STAGES if k in lat["stages_raw"] and len(lat["stages_raw"][k])]
    ax.boxplot([np.asarray(lat["stages_raw"][k], float) for k, _ in keys], positions=range(len(keys)), orientation="horizontal",
               widths=0.6, patch_artist=True, showfliers=False, medianprops={"color": "white", "lw": 1.2},
               whiskerprops={"color": PRIMARY}, capprops={"color": PRIMARY}, boxprops={"facecolor": PRIMARY, "edgecolor": PRIMARY})
    ax.axvline(1e3 * c["frame_s"], color=MUTED, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(keys)), [n for _, n in keys])
    ax.invert_yaxis()
    ax.set_xlabel("Time per Look (ms; Dashed: One Frame)")
    title(ax, "Run Time per Stage")
    ax = axs[1, 1]
    for i, (name, v) in enumerate((("processing", lat["processing"]), ("frame $\\to$ result", lat["result"]))):
        v = np.sort(np.asarray(v, float))
        if len(v):
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=SERIES[i], lw=1.0, label=name)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cumulative Share of Looks")
    title(ax, "Latency")
    ax.legend(loc="lower right")
    style.save(pdf, fig)


def _box(ax, data, pos, color, width=0.28):
    data = [np.asarray(d, float)[np.isfinite(np.asarray(d, float))] for d in data]
    keep = [i for i, d in enumerate(data) if len(d)]
    if keep:
        return ax.boxplot([data[i] for i in keep], positions=[pos[i] for i in keep], widths=width, patch_artist=True,
                   showfliers=False, medianprops={"color": "white", "lw": 1.2}, whiskerprops={"color": color},
                   capprops={"color": color}, boxprops={"facecolor": color, "edgecolor": color})


def understanding(res, pdf):
    u, s = res["understanding"], res["separation"]
    kinds = [k for k in ("obstacle", "lamp", "ball", "person") if k in u["kinds"]]
    n = 3 if s["pairs"] else 2
    fig, axs = plt.subplots(1, n, figsize=(DOUBLE, 2.6), layout="constrained", gridspec_kw={"width_ratios": [2.1, 2.1, 1][:n]})
    ax = axs[0]
    steps = (("recall", "found as its kind"), ("sq", "located (SQ)"), ("material_acc", "material"), ("person_acc", "person label"))
    w = 0.8 / len(steps)
    for i, k in enumerate(kinds):
        for j, (key, lbl) in enumerate(steps):
            v = u["kinds"][k][key]
            ax.bar(i + (j - 1.5) * w, v if np.isfinite(v) else 0, w * 0.92, color=SERIES[j], hatch=style.HATCHES[j],
                   edgecolor="white", lw=0.3, label=lbl if i == 0 else None)
        lo, hi = u["kinds"][k]["recall_ci"]
        if np.isfinite(lo):
            ax.errorbar(i - 1.5 * w, u["kinds"][k]["recall"], yerr=[[u["kinds"][k]["recall"] - lo], [hi - u["kinds"][k]["recall"]]],
                        color=NORM, lw=0.6, capsize=1.5)
    ax.set_xticks(range(len(kinds)), kinds)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of Looks (Fraction)")
    title(ax, "What Was Understood, per Object")
    ax.set_ylim(0, 1.05)
    ax = axs[1]
    for i, r in enumerate(res["runs"]):
        ser = r["series"]
        t = np.array([q["t_s"] for q in ser])
        pq = np.array([q["pq"] for q in ser], float)
        k = max(1, len(t) // 40)
        sm = np.convolve(np.nan_to_num(pq, nan=np.nanmean(pq)), np.ones(10) / 10, mode="same")
        ax.plot(t[::k], sm[::k], color=SERIES[i % len(SERIES)], lw=1.0, label=_short(r["name"]))
        full = np.array([q["pq_full"] for q in ser], float)
        ax.plot(t[::k], np.convolve(np.nan_to_num(full, nan=np.nanmean(full)), np.ones(10) / 10, mode="same")[::k],
                color=SERIES[i % len(SERIES)], lw=0.8, ls="--")
    ax.axhline(u["PQ"], color=MUTED, ls=":", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Scene PQ (1 s Mean)")
    title(ax, "Scene Understanding (PQ)")
    h = [Patch(color=SERIES[j], label=lbl) for j, (_, lbl) in enumerate(steps)]
    h += [Line2D([], [], color=SERIES[i % len(SERIES)], lw=1.0, label=_short(r["name"])) for i, r in enumerate(res["runs"])]
    h += [Line2D([], [], color=INK, lw=1, label="PQ"), Line2D([], [], color=INK, lw=0.8, ls="--", label="PQ with classes")]
    fig.legend(handles=h, loc="outside lower center", ncols=9)
    if s["pairs"]:
        ax = axs[2]
        vals = [("separated", s["separated"], GOOD), ("merged", s["merged"], LOSS), ("missed", s["missed"], MUTED)]
        ax.bar(range(3), [v for _, v, _ in vals], 0.6, color=[c for _, _, c in vals])
        ax.set_xticks(range(3), [q for q, _, _ in vals], rotation=20)
        ax.set_ylabel("Looks with a Close Pair (Count)")
        title(ax, f"Close Objects ($<${0.6:.1f} m)")
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
    title(ax, name)
    return im


def classification(res, pdf):
    rows = res["classification"]["rows"]
    if not rows:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), layout="constrained")
    for ax, head in zip(axs, ("material", "motion")):
        true = np.array([r[f"{head}_true"] or "none" for r in rows], object)
        pred = np.array([r[f"{head}_pred"] or "unlabelled" for r in rows], object)
        classes = list(MATERIAL_COLORS if head == "material" else MOTION_COLORS)
        trows = [c for c in classes if np.any(true == c)]
        cols = [c for c in classes + ["unlabelled"] if np.any(pred == c)]
        im = _matrix(ax, true, pred, trows, cols, f"{head.capitalize()}: {res['classification'][head]['acc'] * 100:.0f}\\,% of Labels")
        ax.set_xlabel("Label of the Matched Object")
    axs[0].set_ylabel("True Class (Looks)")
    fig.colorbar(im, ax=axs, shrink=0.8, label="Share of the True Class (Fraction)")
    style.save(pdf, fig)


def readme_figure(res, path):
    style.use()
    run = res["runs"][-1]["name"]
    looks, truth = res["tracks"][run], res["truth"][run]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    known, trails = {}, {}
    for t, pose, objs, tracks, _ in looks:
        for tr in tracks:
            if tr["status"] == "confirmed":
                trails.setdefault(tr["id"], []).append((t, tr["wy"], tr["wx"], tr.get("classes") or {}))
        for o in objs:
            if o.get("known"):
                known[o["id"]] = o
    for o in known.values():
        ax.add_patch(Rectangle((o["lo"][1], o["lo"][0]), o["hi"][1] - o["lo"][1], o["hi"][0] - o["lo"][0], color=MUTED,
                               alpha=0.35, lw=0))
    for q in trails.values():
        if q[-1][0] - q[0][0] >= TRAIL_S:
            lab = (q[-1][3].get("material") or {}).get("label")
            ax.plot([a[1] for a in q], [a[2] for a in q], color=MATERIAL_COLORS.get(lab, MUTED), lw=0.5, alpha=0.7)
    used = {-1: [], 1: []}
    for o in sorted((o for o in truth if o.kind != "person"), key=lambda o: -o.hi[0]):
        ax.add_patch(Rectangle((o.lo[1], o.lo[0]), o.hi[1] - o.lo[1], o.hi[0] - o.lo[0], fill=False, edgecolor=GOAL,
                               ls="-.", lw=0.8))
        side = -1 if (o.lo[1] + o.hi[1]) / 2 < 0 else 1
        y0 = (o.lo[0] + o.hi[0]) / 2
        y = next(y0 + 0.13 * k for k in (0, 1, -1, 2, -2, 3, -3, 4, -4) if all(abs(y0 + 0.13 * k - u) >= 0.13 for u in used[side]))
        used[side].append(y)
        ax.text(o.lo[1] - 0.05 if side < 0 else o.hi[1] + 0.05, y, o.label, fontsize=6, color=INK, va="center",
                ha="right" if side < 0 else "left", bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
    xs = [q[1][0] for q in looks]
    ax.plot([0, 0], [min(xs) - 0.05, max(xs) + 0.05], color=NORM, lw=3, solid_capstyle="butt")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.15, 2.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Sideways $y$ (m)")
    ax.set_ylabel("Forward $x$ (m)")
    title(ax, f"The Scene from Above, {_short(run)}")
    fig.legend(handles=[Line2D([], [], color=MATERIAL_COLORS[m], lw=1, label=f"track: {m}") for m in ("metal", "dielectric", "waterlike")] +
                      [Patch(facecolor="none", edgecolor=GOAL, ls="-.", label="measured object"),
                       Patch(color=MUTED, alpha=0.35, label="known room"), Line2D([], [], color=NORM, lw=3, label="radar on the rail")],
              loc="outside lower center", ncols=6)
    understanding_bars(axs[1], res["understanding"])
    style.save(path, fig)


def understanding_bars(ax, u):
    kinds = [k for k in ("obstacle", "lamp", "ball", "person") if k in u["kinds"]]
    steps = (("recall", "found as its kind"), ("sq", "located (SQ)"), ("material_acc", "material"), ("person_acc", "person label"))
    w = 0.8 / len(steps)
    for i, k in enumerate(kinds):
        for j, (key, lbl) in enumerate(steps):
            v = u["kinds"][k][key]
            ax.bar(i + (j - 1.5) * w, v if np.isfinite(v) else 0, w * 0.92, color=SERIES[j], hatch=style.HATCHES[j],
                   edgecolor="white", lw=0.3, label=lbl if i == 0 else None)
    ax.set_xticks(range(len(kinds)), kinds)
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Share of Radar Looks (Fraction)")
    title(ax, "What It Understood")
    ax.legend(loc="upper center", ncols=2)


GROUPS = (("final", "method (Infini-Rex)", PRIMARY), ("ctrex", "CT-Rex detector", SERIES[2]),
          ("cfar", "OS-CFAR detector", MUTED), ("baseline", "classical chain", NORM))
SHORT = {"final": "Infini-Rex", "ctrex": "CT-Rex", "cfar": "OS-CFAR", "baseline": "classical"}
SCORES = (("Scene PQ", lambda r: r["understanding"]["PQ"]), ("HOTA", lambda r: r["tracking"]["look"]["hota"]["HOTA"]),
          ("IDF1", lambda r: r["tracking"]["look"]["idf1"]["IDF1"]), ("MOTA", lambda r: r["tracking"]["look"]["clear"]["MOTA"]),
          ("Obstacles", lambda r: r["understanding"]["kinds"]["obstacle"]["recall"]),
          ("Ball", lambda r: r["understanding"]["kinds"]["ball"]["recall"]),
          ("Person", lambda r: r["understanding"]["kinds"]["person"]["recall"]),
          ("FDR", lambda r: r["detection"]["fdr"]))
DOA_COLOR = {"beam_peak": NORM, "os_cfar": MUTED, "omp_noise": SERIES[4], "ctrex_tomp": SERIES[2], "infinirex_look": PRIMARY}
DOA_OF = {"final": "infinirex_look", "ctrex": "ctrex_tomp", "cfar": "os_cfar"}


def comparison(res, pdf):
    series = [(k, lbl, c) for k, lbl, c in GROUPS if k in res]
    if len(series) < 2:
        return
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.7), layout="constrained", gridspec_kw={"width_ratios": [4, 1.4, 1]})
    ax = axs[0]
    x, w = np.arange(len(SCORES)), 0.8 / len(series)
    low = 0.0
    for i, (key, lbl, color) in enumerate(series):
        v = np.array([get(res[key]) for _, get in SCORES], float)
        low = min(low, float(np.nanmin(v)))
        ax.bar(x + (i - (len(series) - 1) / 2) * w, np.nan_to_num(v), w * 0.92, color=color, hatch=style.HATCHES[i],
               edgecolor="white", lw=0.3, label=lbl)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xticks(x, [k for k, _ in SCORES])
    ax.set_ylim(min(0.0, low - 0.05), 1.05)
    ax.set_ylabel("Score, Share Found as Its Kind")
    title(ax, "Method Against Baselines, Same Recordings")
    ax = axs[1]
    for i, (key, lbl, color) in enumerate(series):
        d = res[key]["detection"]
        ax.bar(i, d["true_per_look"], 0.7, color=color, edgecolor="white", lw=0.3)
        ax.bar(i, d["false_per_look"], 0.7, bottom=d["true_per_look"], color=LOSS, hatch="///", edgecolor="white", lw=0.3)
    ax.set_xticks(range(len(series)), [SHORT[k] for k, _, _ in series], rotation=30, ha="right")
    ax.set_ylabel("Detections per Look")
    title(ax, "True and False")
    ax = axs[2]
    doa = res.get("doa") or {}
    cost = [(i, k, c) for i, (k, _, c) in enumerate(series) if DOA_OF.get(k) in doa]
    for i, key, color in cost:
        ax.bar(i, doa[DOA_OF[key]]["ms"], 0.7, color=color, hatch=style.HATCHES[i], edgecolor="white", lw=0.3)
    ax.set_yscale("log")
    ax.set_xticks([i for i, _, _ in cost], [SHORT[k] for _, k, _ in cost], rotation=30, ha="right")
    ax.set_ylabel("Detection per Look (ms)")
    title(ax, "Cost, Same Cells")
    h = [Patch(color=c, label=lbl) for _, lbl, c in series]
    h += [Patch(facecolor="white", edgecolor=INK, label="on a measured object"),
          Patch(facecolor=LOSS, hatch="///", edgecolor="white", label="on nothing known")]
    fig.legend(handles=h, loc="outside lower center", ncols=6)
    style.save(pdf, fig)


def doa(res, pdf):
    d = res.get("doa")
    if not d:
        return
    kinds = [k for k in ("obstacle", "ball", "person", "lamp") if any(k in v["recall"] for v in d.values())]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), layout="constrained", gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    x, w = np.arange(len(kinds)), 0.8 / len(d)
    for i, (m, v) in enumerate(d.items()):
        color = DOA_COLOR[m]
        ax.bar(x + (i - (len(d) - 1) / 2) * w, [v["recall"].get(k, np.nan) for k in kinds], w * 0.92, color=color,
               hatch=style.HATCHES[i % len(style.HATCHES)], edgecolor="white", lw=0.3, label=v["label"])
    ax.set_xticks(x, kinds)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Objects in View Found (Fraction)")
    title(ax, "Which DoA Method Finds Which Object")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=6)
    ax = axs[1]
    for i, (m, v) in enumerate(d.items()):
        rec = np.mean([v["recall"][k] for k in kinds if k in v["recall"]])
        ax.scatter(v["fdr"], rec, s=22, color=DOA_COLOR[m],
                   marker="o" if v["classical"] else "s", zorder=3)
        ax.annotate(v["label"], (v["fdr"], rec), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    ax.axvline(res["final"]["detection"]["alpha"] if "final" in res else 0.1, color=GOAL, ls="--", lw=0.8)
    ax.set_xlabel("False Share of the Detections")
    ax.set_ylabel("Mean Share Found")
    title(ax, "Found Against False")
    style.save(pdf, fig)


FIGS = (("scene", scene), ("tracking", tracking), ("understanding", understanding), ("motion", motion),
        ("camera", camera), ("classification", classification), ("system", system))


def draw_group(res, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        path = folder / f"{name}.pdf"
        with style.pdf(path) as pdf:
            fn(res, pdf)
        if path.exists() and path.stat().st_size < 1500:
            path.unlink()
        if path.exists():
            print(f"  {path.relative_to(paths.OUTPUT).as_posix()}", flush=True)


def draw(res):
    style.use()
    if "final" in res:
        draw_group(res["final"], FIGURES)
    with style.pdf(FIGURES / "comparison.pdf") as pdf:
        comparison(res, pdf)
        doa(res, pdf)
    print("  figures/comparison.pdf")
    readme_figure(res["final"], FIGURES / "readme.svg")
    print("  figures/readme.svg")


if __name__ == "__main__":
    import pickle
    draw(pickle.loads((paths.OUTPUT / "cache" / "results.pkl").read_bytes()))
