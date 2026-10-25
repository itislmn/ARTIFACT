"""
common.py - what every figure shares: method colours and markers (filled = FDR-controlled, hollow = no guarantee), the
summaries of the cached results (FDR over every frame, TPR over frames with reflectors) and the recurring plots.

    from estimation.plotting import common
"""
from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from estimation import dataset  # noqa: E402
from estimation.core.frame import process_frame  # noqa: E402
from estimation.core.radar import open_recording  # noqa: E402
from estimation.methods import FDR_CONTROLLED, labels  # noqa: E402
from visualization import style  # noqa: E402

FIGURES = dataset.OUTPUT / "figures"
CACHE = dataset.OUTPUT / "cache"
TEAL, VIOLET, CYAN, OLIVE = style.SERIES[2], style.SERIES[3], style.SERIES[4], style.SERIES[5]
LOOK = {
    "infinirex_radar": (style.PRIMARY, "-", "o"), "infinirex": (style.PRIMARY, "--", "s"),
    "ctrex_tlars": (TEAL, "-", "^"), "ctrex_tomp": (TEAL, "--", "v"), "trex_tlars": (TEAL, ":", "D"),
    "ctrex_tlasso": (TEAL, "-.", "<"), "ctrex_ten": (TEAL, (0, (5, 1)), ">"), "ctrex_tols": (TEAL, (0, (1, 1)), "p"),
    "ctrex_tmp": (TEAL, (0, (3, 1, 1, 1)), "h"), "ctrex_tgp": (TEAL, (0, (6, 2)), "8"), "ctrex_tacgp": (TEAL, (0, (2, 2)), "*"),
    "ctrex_tlars_da": (TEAL, (0, (4, 1, 1, 1, 1, 1)), "P"), "ctrex_tomp_da": (TEAL, (0, (8, 2, 1, 2)), "X"),
    "trex_tomp": (TEAL, (0, (1, 2)), "d"),
    "bh": (VIOLET, "-", "P"), "by": (VIOLET, ":", "p"), "knockoffs": (VIOLET, "--", "X"), "knockoffs_offset0": (VIOLET, (0, (1, 1)), "x"), "knockoffs_lowdim": (VIOLET, "-.", "*"), "knockoffs_lowdim_offset0": (VIOLET, (0, (3, 1, 1, 1)), "h"),
    "os_cfar": (style.ACCENT, "-", "d"), "ca_cfar": (style.ACCENT, "--", "D"),
    "bartlett": (OLIVE, ":", "o"), "music": (OLIVE, "-", "X"), "esprit": (OLIVE, "--", "h"), "capon": (OLIVE, "-.", "s"),
    "dnn": (CYAN, "-", "p"),
}
BASELINES = ["bartlett", "music", "esprit", "os_cfar", "dnn", "knockoffs"]
FDRSET = ["bh", "knockoffs_lowdim", "ctrex_tlars", "ctrex_tomp", "trex_tlars",
          "infinirex", "infinirex_radar"]
PANELS = (("No Error Guarantee", BASELINES), ("FDR-Controlled", FDRSET))
GROUPS = (("Subspace Methods and DNN", ["bartlett", "capon", "music", "esprit", "dnn"]),
          (r"CFAR (Per-Cell $P_\mathrm{fa}$)", ["ca_cfar", "os_cfar"]),
          ("BH, BY, Fixed-X Knockoffs", ["bh", "by", "knockoffs_lowdim", "knockoffs_lowdim_offset0"]),
          ("CT-Rex, LARS Solvers", ["ctrex_tlars", "ctrex_tlasso", "ctrex_ten", "ctrex_tlars_da"]),
          ("CT-Rex, Pursuit Solvers", ["ctrex_tomp", "ctrex_tols", "ctrex_tmp", "ctrex_tgp", "ctrex_tacgp", "ctrex_tomp_da"]),
          ("T-Rex, Knockoffs, Infini-Rex", ["trex_tlars", "trex_tomp", "knockoffs", "knockoffs_offset0",
                                           "infinirex", "infinirex_radar"]))
ALPHA = 0.1
TRUTH = dict(marker="*", color=style.GOAL, ms=11, mec="white", mew=0.7, ls="none", zorder=9)
TRUTH_LINE = dict(color=style.GOAL, lw=1.0, ls="-.", zorder=1)
LABEL = {}


def lab(m):
    if not LABEL:
        LABEL.update(labels())
    return LABEL.get(m, m)


def tt(name):
    return r"$\mathtt{" + str(name).replace("_", r"\_") + "}$"


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load(name):
    p = CACHE / f"{name}.csv"
    return dataset.read_csv(p) if p.exists() else []


def guaranteed(m):
    return m in FDR_CONTROLLED


def summary(recs, keys):
    groups = {}
    for r in recs:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = {}
    for key, g in groups.items():
        live = [r for r in g if num(r["n_gates"]) > 0]
        na = any(int(r["applicable"]) == 0 for r in live)
        ea = np.array([float(x) for r in g for x in r["err_angle_deg"].split(";") if x])
        er = np.array([float(x) for r in g for x in r["err_range_m"].split(";") if x])
        per_gate = [num(r["seconds"]) / num(r["n_gates"]) for r in live]
        fdp = np.array([num(r["FDP"]) for r in g])
        tpp = np.array([num(r["TPP"]) for r in g if int(r["n_true"]) > 0])
        n, nt = len(g), len(tpp)
        out[key] = {"FDR": np.nan if na else fdp.mean(), "TPR": np.nan if na or not nt else tpp.mean(),
                    "FDR_se": fdp.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
                    "TPR_se": tpp.std(ddof=1) / np.sqrt(nt) if nt > 1 else 0.0,
                    "FWER": np.nan if na else np.mean([num(r["V"]) >= 1 for r in g]),
                    "RMSE": float(np.sqrt(np.mean(ea ** 2))) if ea.size and not na else np.nan,
                    "RMSE_R": float(100 * np.sqrt(np.mean(er ** 2))) if er.size else np.nan,
                    "ms": 1e3 * float(np.median(per_gate)) if per_gate else np.nan, "n": n}
    return out


def by_k(recs, scenario=None):
    rows = [r for r in recs if scenario is None and r["scenario"] in ("single", "multi", "null") or r["scenario"] == scenario]
    return {(m, int(k)): v for (m, k), v in summary(rows, ("method", "K")).items()}


def needed_k(table):
    out = {}
    for m in {m for m, _ in table}:
        ks = sorted(k for mm, k in table if mm == m and np.isfinite(table[(mm, k)]["FDR"]))
        ok = [k for k in ks if table[(m, k)]["FDR"] <= ALPHA]
        if ks:
            out[m] = (ok[0], True) if ok else (min(ks, key=lambda k: (table[(m, k)]["FDR"], k)), False)
    return out


def by_snr(recs, scenario):
    return {(m, num(s)): v for (m, s), v in summary([r for r in recs if r["scenario"] == scenario], ("method", "snr_db")).items()}


def sweep_table(rows, K=None, scenario=None):
    g = {}
    for r in rows:
        if K is not None and int(r["K"]) != K or scenario is not None and r.get("scenario") != scenario:
            continue
        g.setdefault((r["method"], num(r["level"])), []).append(r)
    return {k: (np.mean([num(r["FDP"]) for r in v]), np.mean([num(r["TPP"]) for r in v if int(r["n_true"]) > 0]), len(v))
            for k, v in g.items()}


def dodge(x, i, n, log):
    k = i - (n - 1) / 2
    x = np.asarray(x, float)
    return x * 2 ** (0.05 * k) if log else x + 0.3 * k


def line(ax, x, y, m, label=True, band=None, **kw):
    c, ls, mk = LOOK.get(m, (style.NORM, "-", "o"))
    mfc = c if guaranteed(m) else "white"
    ax.plot(x, y, color=c, ls=ls, marker=mk, ms=4.5, mfc=mfc, mec=c, mew=0.9, lw=1.2, label=lab(m) if label else None,
            zorder=4 if guaranteed(m) else 3, **kw)
    if band is not None:
        ax.fill_between(x, np.clip(y - band, 0, 1), np.clip(y + band, 0, 1), color=c, alpha=0.12, lw=0)


def alpha_line(ax):
    ax.axhline(ALPHA, color=style.MUTED, ls="--", lw=0.9, label=r"Nominal $\alpha = 0.1$", zorder=2)


def guarantee_key(ax):
    ax.plot([], [], "o", color=style.NORM, ms=4.5, label="Filled: FDR-Controlled")
    ax.plot([], [], "o", color=style.NORM, mfc="white", ms=4.5, label="Hollow: No FDR Guarantee")


def metric_vs(pdf, table, x_values, metric, xlabel, ylabel, title_text, logx=False, crb=None, bands=False, ylim=None):
    fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.46), sharey=True, layout="constrained")
    for ax, (name, methods) in zip(axs, PANELS):
        for i, m in enumerate(methods):
            pts = [(x, table[(m, x)]) for x in x_values if (m, x) in table]
            if not pts:
                continue
            xs = dodge([p[0] for p in pts], i, len(methods), logx)
            ys = np.array([p[1][metric] for p in pts], float)
            se = np.array([p[1].get(metric + "_se", 0.0) for p in pts], float) * 1.96 if bands and metric in ("FDR", "TPR") else None
            line(ax, xs, ys, m, band=se)
        if metric in ("FDR", "FWER") and name == "FDR-Controlled":
            alpha_line(ax)
        if crb is not None:
            ax.plot(crb[0], crb[1], color=style.NORM, ls=(0, (1, 1)), lw=1.0, label=r"Cram\'er--Rao Bound", zorder=5)
        if logx:
            ax.set_xscale("log", base=2)
        if metric == "RMSE" and bands:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        style.title(ax, name)
    axs[0].set_ylabel(ylabel)
    style.legend_below(fig, axs, ncols=4)
    style.title(fig, title_text)
    style.save(pdf, fig)


def scatter(ax, table, K, methods, legend=True):
    pts = [(m, table[(m, K)]) for m in methods if (m, K) in table and np.isfinite(table[(m, K)]["FDR"])]
    x = np.array([v["FDR"] for _, v in pts])
    y = np.array([v["TPR"] if np.isfinite(v["TPR"]) else 0.0 for _, v in pts])
    xe = 1.96 * np.array([v["FDR_se"] for _, v in pts])
    ye = 1.96 * np.array([v["TPR_se"] for _, v in pts])

    def draw(a, off, label):
        for i, (m, _) in enumerate(pts):
            c, _, mk = LOOK.get(m, (style.NORM, "-", "o"))
            a.errorbar(x[i] + off[i], y[i], xerr=xe[i], yerr=ye[i], color=c, marker=mk, ms=5.5,
                       mfc=c if guaranteed(m) else "white", mec=c, mew=1.0, lw=0.7, capsize=1.5, ls="none",
                       label=lab(m) if label else None, zorder=4 if guaranteed(m) else 3)

    ax.axvspan(-0.03, ALPHA, color=style.GOOD, alpha=0.2, lw=0, label=r"$\widehat{\mathrm{FDR}} \le \alpha = 0.1$", zorder=0)
    draw(ax, np.zeros(len(pts)), legend)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel(r"$\widehat{\mathrm{FDR}}$")
    ax.set_ylabel("TPR")
    near = (x <= 2.5 * ALPHA) & (y >= 0.5)
    if near.sum() < 3:
        return
    zx = (-0.01, float(np.max(x[near] + xe[near])) + 0.02)
    zy = (float(np.min(y[near] - ye[near])) - 0.03, 1.02)
    off = np.zeros(len(pts))
    todo = list(np.flatnonzero(near))
    step = 0.06 * (zx[1] - zx[0])
    while todo:
        i = todo.pop(0)
        grp = [i] + [j for j in todo if abs(x[j] - x[i]) < 0.02 * (zx[1] - zx[0]) and abs(y[j] - y[i]) < 0.02 * (zy[1] - zy[0])]
        todo = [j for j in todo if j not in grp]
        for k, j in enumerate(grp):
            off[j] = (k - (len(grp) - 1) / 2) * step
    zx = (min(zx[0], float(np.min((x + off)[near])) - step), max(zx[1], float(np.max((x + off)[near])) + step))
    win = ax.inset_axes([0.42, 0.1, 0.55, 0.5])
    win.axvspan(zx[0], ALPHA, color=style.GOOD, alpha=0.2, lw=0, zorder=0)
    draw(win, off, False)
    win.set_xlim(*zx)
    win.set_ylim(*zy)
    win.set_xticks([t for t in win.get_xticks() if 0 <= t <= zx[1]])
    win.set_xlim(*zx)
    win.tick_params(labelsize=6)
    win.set_title("Zoom (Equal Values Side by Side)", fontsize=6.5, pad=2)
    ax.indicate_inset_zoom(win, edgecolor=style.MUTED, alpha=1.0, lw=0.8)


def overview(pdf, table, title_text, methods, Ks=(1, 128)):
    fig, axs = plt.subplots(1, len(Ks), figsize=style.size(style.DOUBLE, 0.66), sharey=True, layout="constrained")
    for i, (ax, K) in enumerate(zip(np.atleast_1d(axs), Ks)):
        scatter(ax, table, K, methods, legend=(i == 0))
        style.title(ax, f"$K = {K}$ Chirp{'s' if K > 1 else ' (Single Snapshot)'}")
        if i:
            ax.set_ylabel("")
    guarantee_key(np.atleast_1d(axs)[0])
    style.legend_below(fig, np.atleast_1d(axs), ncols=5)
    style.title(fig, f"{title_text}: Every Method in the FDR-TPR Plane")
    style.save(pdf, fig)


ALPHA_METHODS = ["bh", "by", "knockoffs", "knockoffs_lowdim", "knockoffs_lowdim_offset0", "ctrex_tlars", "ctrex_tomp",
                 "trex_tlars", "infinirex", "infinirex_radar"]


def nominal_axes(ax, metric, top=0.55):
    if metric == "FDP":
        ax.fill_between([0, top], [0, top], 0, color=style.GOOD, alpha=0.15, lw=0, label="Achieved $\\le$ Nominal", zorder=0)
        ax.plot([0, top], [0, top], color=style.NORM, ls="--", lw=1.0, label="Ideal: Achieved = Nominal", zorder=1)
        ax.set_ylim(-0.01, top)
        ax.set_aspect("equal")
    else:
        ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 0.52)


def alpha_diagonal(pdf, rows, title_text, scenario=None, Ks=None):
    Ks = Ks or sorted({int(r["K"]) for r in rows})
    tabs = {K: sweep_table(rows, K, scenario) for K in Ks}
    methods = [m for m in ALPHA_METHODS if any(k[0] == m for t in tabs.values() for k in t)]
    cols = min(len(Ks), 4)
    nrow = int(np.ceil(len(Ks) / cols))
    for metric, ylabel, what in (("FDP", r"Achieved $\widehat{\mathrm{FDR}}$", "Achieved FDR"),
                                ("TPP", "TPR", "TPR")):
        fig, axs = plt.subplots(nrow, cols, figsize=style.size(style.DOUBLE, 0.27 * nrow + 0.12), sharex=True,
                                sharey=True, layout="constrained", squeeze=False)
        top = max([0.55] + [v[0] + 0.05 for t in tabs.values() for v in t.values()])
        for ax, K in zip(axs.flat, Ks):
            nominal_axes(ax, metric, top)
            for m in methods:
                pts = sorted((a, v[0 if metric == "FDP" else 1]) for (mm, a), v in tabs[K].items() if mm == m)
                if pts:
                    line(ax, [p[0] for p in pts], [p[1] for p in pts], m)
            style.title(ax, f"$K = {K}$")
        for ax in axs.flat[len(Ks):]:
            ax.set_visible(False)
        for ax in axs[-1]:
            ax.set_xlabel(r"Nominal Level $\alpha$")
        for ax in axs[:, 0]:
            ax.set_ylabel(ylabel)
        style.legend_below(fig, axs.flat, ncols=5)
        style.title(fig, f"{title_text}: {what} vs.\\ Nominal Level $\\alpha$")
        style.save(pdf, fig)


def all_methods(pdf, table, xs, xl, logx, title_text):
    for metric in ("FDR", "TPR"):
        fig, axs = plt.subplots(2, 3, figsize=style.size(style.DOUBLE, 0.95), sharex=True, sharey=True, layout="constrained")
        for ax, (gname, ms) in zip(axs.flat, GROUPS):
            for i, m in enumerate(ms):
                pts = [(x, table[(m, x)][metric]) for x in xs if (m, x) in table]
                if pts:
                    line(ax, dodge([p[0] for p in pts], i, len(ms), logx), [p[1] for p in pts], m)
            if metric == "FDR" and all(guaranteed(m) for m in ms):
                alpha_line(ax)
            if logx:
                ax.set_xscale("log", base=2)
            ax.set_ylim(-0.03, 1.03)
            style.title(ax, gname)
        for ax in axs[1]:
            ax.set_xlabel(xl)
        for ax in axs[:, 0]:
            ax.set_ylabel(metric)
        guarantee_key(axs[0, 0])
        style.legend_below(fig, axs.flat, ncols=4)
        style.title(fig, f"{title_text}: {metric} of Every Method")
        style.save(pdf, fig)


def frame_result(session, take, K, methods, settings=None):
    t = next(x for x in session.evaluated if x["name"] == take)
    f = open_recording(t["path"], session.cfg)[1](0)[:, :K]
    st = replace(settings or session.settings, keep_info=True)
    return t, process_frame(f, session.cfg, session.array, methods, st, [0, K], session.calibration, session.background)


def range_azimuth(ax, session, res, truth, dets=()):
    a = session.array
    Y = res.rfft[:, :, a.channels].mean(axis=1)
    P = np.abs(Y @ a.steering(a.u_grid).conj()) ** 2
    r = session.cfg.range_axis_m
    k = (r >= 0.3) & (r <= session.settings.r_max_m + 0.3)
    mesh = ax.pcolormesh(a.theta_grid_deg, r[k], 10 * np.log10(P[k] / P[k].max()), cmap=style.SEQUENTIAL, vmin=-40, vmax=0,
                         shading="nearest", rasterized=True)
    if len(truth):
        ax.plot(truth[:, 1], truth[:, 0], label="Truth", **TRUTH)
    for name, d, c, mk in dets:
        if len(d):
            ax.plot(d[:, 1], d[:, 0], ls="none", marker=mk, color="white", mfc="none", ms=14, mew=2.8, zorder=7)
            ax.plot(d[:, 1], d[:, 0], ls="none", marker=mk, color=c, mfc="none", ms=14, mew=1.3, label=name, zorder=8)
    ax.set_xlabel(r"Azimuth $\theta$ [deg]")
    ax.set_ylabel("Range $R$ [m]")
    return mesh
