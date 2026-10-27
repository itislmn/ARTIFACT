"""
measured.py - the figures of the measured session S2: set-up, range-azimuth-elevation maps, spatial spectra, FDR, TPR
and RMSE vs. the number of chirps and vs. SNR (with the Cramer-Rao bound), FDR control, range estimation (precision and
resolution too), all methods,
and the fewest chirps with which each method controls the FDR.

    from estimation.plotting import measured
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from estimation import dataset
from estimation.core.truth import EL_SIGN
from estimation.core.metrics import crb_theta_deg
from estimation.methods import doa_methods
from estimation.methods.range_czt import N_POINTS
from estimation.plotting.common import (LOOK, TRUTH, TRUTH_LINE, all_methods, alpha_diagonal, by_k, dodge, frame_result,
                                        guarantee_key, lab, line, metric_vs, needed_k, num, overview, range_azimuth, tt)
from estimation.ranging import crb_range_m
from visualization import style

TITLE = "Measured (S2)"


def setup(pdf, session):
    cfg, a = session.cfg, session.array
    fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.38), layout="constrained")
    az = np.zeros(cfg.n_virt, bool)
    az[a.channels] = True
    x, z = cfg.x_lam, cfg.z_lam
    axs[0].plot(np.arange(12) * 0.5, np.zeros(12), "|", color=style.MUTED, ms=9, label=r"Uniform $\lambda/2$ Grid")
    axs[0].plot(x[az], z[az], "o", color=style.PRIMARY, ms=6, label="Azimuth Row (12 Channels, TI Layout)")
    axs[0].plot(x[~az], z[~az], "s", color=style.ACCENT, ms=5, label="Elevated TX1 Row (4 Channels)")
    axs[0].set_ylim(-0.4, 1.5)
    axs[0].set_xlabel(r"Position $x$ [$\lambda$]")
    axs[0].set_ylabel(r"Height $z$ [$\lambda$]")
    style.title(axs[0], "Virtual Array")
    du = np.linspace(-0.6, 0.6, 1201)
    D = np.abs(np.exp(2j * np.pi * a.d * np.arange(a.n)[:, None] * du[None, :]).sum(axis=0)) / a.n
    axs[1].plot(du, D, color=style.NORM, lw=1.0, label=r"$|\mathbf{a}^H(u)\mathbf{a}(u + \Delta u)|/N$")
    kk = np.arange(-8, 9) * a.du
    axs[1].plot(kk, np.abs(np.exp(2j * np.pi * a.d * np.arange(a.n)[:, None] * kk[None, :]).sum(axis=0)) / a.n, "o",
                color=style.PRIMARY, ms=3, label=r"Fine Grid, $\delta u = 1/48$ ($p = 83$)")
    axs[1].axvspan(-a.rayleigh_u, a.rayleigh_u, color=style.MUTED, alpha=0.2, lw=0, label="One Rayleigh Cell")
    axs[1].set_xlabel(r"$\Delta u = \Delta \sin\theta$")
    axs[1].set_ylabel("Coherence")
    style.title(axs[1], "Coherence of the Steering Matrix")
    style.legend_below(fig, axs, ncols=3)
    style.title(fig, "AWR2944P: Virtual Array and Steering Matrix")
    style.save(pdf, fig)
    geometry = dataset.plan().get("geometry", {})
    shown = [t for t in session.takes if t["scenario"] != "background"]
    fig, axs = plt.subplots(3, 4, figsize=style.size(style.DOUBLE, 1.08), sharex=True, sharey=True, layout="constrained")
    for ax in axs.flat[len(shown):]:
        ax.set_visible(False)
    marks = {"big CR": ("s", style.SERIES[0]), "small CR 1": ("^", style.SERIES[1]), "small CR 2": ("v", style.SERIES[2])}
    for t, ax in zip(shown, axs.flat):
        room(ax, geometry, session.settings)
        for tg in t["doc"].get("targets", []):
            rh = np.sqrt(max(tg["range_m"] ** 2 - tg.get("z_m", 0.0) ** 2, 0.0))
            m, c = marks.get(tg.get("label", ""), ("o", style.NORM))
            ax.plot(rh * np.sin(np.radians(tg["azimuth_deg"])), rh * np.cos(np.radians(tg["azimuth_deg"])), marker=m,
                    color=c, ms=7, ls="none", label=tg.get("label", ""), zorder=6)
        ax.set_xlim(-1.6, 0.9)
        ax.set_ylim(-0.1, geometry.get("front_wall_x_m", 3.0) + 0.15)
        ax.set_aspect("equal")
        style.title(ax, tt(t["name"]))
    for ax in axs[-1]:
        ax.set_xlabel(r"$y$ [m]")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$x$ [m]")
    style.legend_below(fig, axs.flat, ncols=4)
    style.title(fig, f"{TITLE}: Where Each Take Was Measured")
    style.save(pdf, fig)
    if session.background is not None:
        fig, ax = plt.subplots(figsize=style.size(style.SINGLE, 1.1), layout="constrained")
        Y = session.background["rfft"][:, a.channels]
        P = np.abs(Y @ a.steering(a.u_grid).conj()) ** 2
        r = cfg.range_axis_m
        k = (r > 0.3) & (r < 3.5)
        RR, TT = np.meshgrid(r[k], np.radians(a.theta_grid_deg), indexing="ij")
        ax.pcolormesh(RR * np.sin(TT), RR * np.cos(TT), 10 * np.log10(P[k] / P[k].max()), cmap=style.SEQUENTIAL,
                      vmin=-50, vmax=0, shading="auto", rasterized=True)
        room(ax, geometry, session.settings)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.1, 3.4)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$y$ [m]")
        ax.set_ylabel(r"$x$ [m]")
        style.title(ax, "Empty Room as the Radar Sees It")
        style.legend_below(fig, [ax], ncols=2)
        style.save(pdf, fig)


def room(ax, g, settings):
    ax.plot(0, 0, "^", color=style.NORM, ms=8, ls="none", label="Radar", zorder=7)
    r0, r1, ym = settings.r_min_m, settings.r_max_m, settings.y_max_m
    po = np.radians(np.linspace(-90, np.degrees(np.arcsin(min(ym / r1, 1))), 200))
    pi = np.radians(np.linspace(np.degrees(np.arcsin(min(ym / r0, 1))), -90, 200))
    poly = np.vstack([np.c_[r1 * np.cos(po), r1 * np.sin(po)], np.c_[r0 * np.cos(pi), r0 * np.sin(pi)]])
    ax.plot(np.r_[poly[:, 1], poly[0, 1]], np.r_[poly[:, 0], poly[0, 0]], color=style.PRIMARY, ls="--", lw=0.9,
            label="Evaluation Region")
    if "rail_y_m" in g:
        ax.axvspan(*g["rail_y_m"], color=style.MUTED, alpha=0.35, lw=0, label="Metal Rail")
    if "right_wall_y_m" in g:
        ax.axvline(g["right_wall_y_m"], color=style.NORM, lw=2.0, label="Wall")
    if "front_wall_x_m" in g:
        ax.axhline(g["front_wall_x_m"], color=style.NORM, lw=2.0)


def cube(session, res, th, ph):
    cfg, a = session.cfg, session.array
    rows = np.asarray(cfg.virt_rows)
    up = rows != rows.max()
    A = np.exp(2j * np.pi * a.az_sign * cfg.x_lam[:, None] * np.sin(np.radians(th))[None, :])
    E = np.exp(1j * EL_SIGN * 2 * np.pi * cfg.z_lam[rows != rows.max()].mean() * np.sin(np.radians(ph)))
    r = cfg.range_axis_m
    k = (r >= 0.3) & (r <= session.settings.r_max_m + 0.3)
    Y = res.rfft.mean(axis=1)[k]
    low = Y[:, ~up] @ A[~up].conj() / (~up).sum()
    high = Y[:, up] @ A[up].conj() / up.sum()
    P = np.abs(low[:, None, :] + E.conj()[None, :, None] * high[:, None, :]) ** 2
    return r[k], P / P.max()


def maps(pdf, session, checks):
    show = ["os_cfar", "music", "ctrex_tlars", "infinirex_radar"]
    methods = doa_methods(show)
    th, ph = np.linspace(-60, 60, 241), np.linspace(-40, 40, 161)
    db = lambda x: 10 * np.log10(x / x.max() + 1e-12)
    for t in session.evaluated:
        truth = t["truth"] if t["truth"].size else np.zeros((0, 2))
        elev = np.asarray(t["elev"], float)
        fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.5), layout="constrained")
        _, res = frame_result(session, t["name"], 128, methods)
        _, res1 = frame_result(session, t["name"], 1, methods)
        dets = [(f"{lab(m)}, $K = 1$", res1.detections[m], LOOK[m][0], LOOK[m][2]) for m in show]
        range_azimuth(axs[0], session, res, truth, dets)
        style.title(axs[0], "Range-Azimuth")
        r, P = cube(session, res, th, ph)
        near = np.zeros(r.size, bool)
        for R in truth[:, 0]:
            near |= np.abs(r - R) <= session.cfg.range_res_m
        sel = near if near.any() else np.ones(r.size, bool)
        mesh = axs[1].pcolormesh(th, ph, db(P[sel].max(axis=0)), cmap=style.SEQUENTIAL, vmin=-20, vmax=0,
                                 shading="nearest", rasterized=True)
        if truth.size:
            axs[1].plot(truth[:, 1], elev, **TRUTH)
        est = [(num(c["azimuth_deg"]), num(c["elevation_measured_deg"])) for c in checks if c["take"] == t["name"]]
        if est:
            axs[1].plot([e[0] for e in est], [e[1] for e in est], "s", color="white", mfc="none", ms=12, mew=2.8, zorder=7)
            axs[1].plot([e[0] for e in est], [e[1] for e in est], "s", color=style.NORM, mfc="none", ms=12, mew=1.1,
                        label="Elevation from the TX1 Row (Phase)", zorder=8)
        axs[1].set_xlabel(r"Azimuth $\theta$ [deg]")
        axs[1].set_ylabel(r"Elevation $\varphi$ [deg]")
        style.title(axs[1], "Azimuth-Elevation at the Reflectors")
        fig.colorbar(mesh, ax=axs[1], label="Power [dB]")
        style.legend_below(fig, axs[:2], ncols=3, markerscale=0.6)
        style.title(fig, f"{TITLE}, Take {tt(t['name'])}: Power and Detections")
        style.save(pdf, fig)


def spectra(pdf, session):
    methods = doa_methods(["bartlett", "capon", "music"])
    for take in ("single_m30", "multi_doa", "multi_close_az"):
        fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.4), sharey=True, layout="constrained")
        for ax, K in zip(axs, (1, 128)):
            t, res = frame_result(session, take, K, methods)
            gi = int(np.argmin([abs(g.range_m - t["truth"][0, 0]) for g in res.gates]))
            for i, m in enumerate(methods):
                info = res.results[m.name][gi].info
                P = info["spectrum"]
                ax.plot(np.degrees(np.arcsin(info["u"])), 10 * np.log10(P / P.max()), color=style.SERIES[i],
                        ls=("-", "--", ":")[i], lw=1.2, label=f"{lab(m.name)} ($\\hat{{k}} = {info['k_hat']}$)")
            for th in t["truth"][np.abs(t["truth"][:, 0] - res.gates[gi].range_m) < 0.2, 1]:
                ax.axvline(th, **TRUTH_LINE)
            ax.plot([], [], label="Truth", **TRUTH_LINE)
            ax.set_xlim(-60, 60)
            ax.set_ylim(-45, 3)
            ax.set_xlabel(r"Azimuth $\theta$ [deg]")
            style.title(ax, f"$K = {K}$ Chirp{'s' if K > 1 else ''}")
        axs[0].set_ylabel("Spatial (Pseudo-)Spectrum [dB]")
        style.legend_below(fig, axs, ncols=4)
        style.title(fig, f"{TITLE}, Take {tt(take)}: Spatial Spectra")
        style.save(pdf, fig)


SNR_METHODS = ["music", "os_cfar", "dnn", "bh", "knockoffs_lowdim", "ctrex_tlars", "trex_tlars", "infinirex",
               "infinirex_radar"]
SNR_BIN_DB = 6.0


def reflector_snr(dataset_rows):
    return {(r["take"], int(r["frame"]), int(r["reflector"])): num(r["snr_db"]) for r in dataset_rows}


def snr(pdf, preds, dataset_rows, session):
    snr_of = reflector_snr(dataset_rows)
    K_all = sorted({int(p["K"]) for p in preds})
    scen = {t["name"]: t["scenario"] for t in session.evaluated}
    found = {}
    for p in preds:
        if int(p["truth"]) >= 0:
            found[(p["method"], p["unit"], int(p["frame"]), int(p["K"]), int(p["truth"]))] = \
                num(p["azimuth_deg"]) - num(p["truth_azimuth_deg"])
    reflectors = [(t, f, K, j, snr_of[(t, f, j)] - 10 * np.log10(128 / K)) for (t, f, j) in snr_of if t in scen
                  for K in K_all]
    edges = np.arange(np.floor(min(r[4] for r in reflectors) / SNR_BIN_DB) * SNR_BIN_DB,
                      max(r[4] for r in reflectors) + SNR_BIN_DB, SNR_BIN_DB)
    centre = 0.5 * (edges[1:] + edges[:-1])
    tape = np.degrees(0.01 / np.mean([r for t in session.evaluated for r in t["truth"][:, 0]]))
    for metric, ylabel in (("TPR", "TPR"), ("RMSE", r"RMSE of $\theta$ [deg]")):
        fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.46), sharey=True, layout="constrained")
        for ax, (sc, name) in zip(axs, (("single", "One Reflector"), ("multi", "Two or Three Reflectors"))):
            pool = [r for r in reflectors if scen[r[0]] == sc]
            for i, m in enumerate(SNR_METHODS):
                ys, xs = [], []
                for lo, hi, c in zip(edges[:-1], edges[1:], centre):
                    sel = [r for r in pool if lo <= r[4] < hi]
                    if len(sel) < 20:
                        continue
                    errs = [found[(m, t, f, K, j)] for t, f, K, j, _ in sel if (m, t, f, K, j) in found]
                    v = len(errs) / len(sel) if metric == "TPR" else (np.sqrt(np.mean(np.square(errs))) if len(errs) >= 10 else np.nan)
                    xs.append(c)
                    ys.append(v)
                if xs:
                    line(ax, dodge(xs, i, len(SNR_METHODS), False), np.array(ys), m)
            if metric == "RMSE":
                s = np.linspace(edges[0], edges[-1], 80)
                if sc == "single":
                    ax.plot(s, crb_theta_deg(s), color=style.NORM, ls=(0, (1, 1)), lw=1.0, label=r"Cram\'er--Rao Bound (One Reflector)")
                ax.axhline(tape, color=style.MUTED, ls="-.", lw=1.0, label="Ground-Truth Accuracy ($\\pm 1$ cm Tape)")
                ax.set_yscale("log")
                ax.set_ylim(1e-4, 3)
            else:
                ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("SNR of the Reflector per Channel [dB]")
            style.title(ax, name)
        axs[0].set_ylabel(ylabel)
        guarantee_key(axs[0])
        style.legend_below(fig, axs, ncols=4)
        what = "TPR" if metric == "TPR" else "Angle RMSE"
        style.title(fig, f"{TITLE}: {what} vs.\\ SNR ($K = 1 \\ldots 128$ Chirps)")
        style.save(pdf, fig)


def snapshots(pdf, meas):
    overview(pdf, by_k(meas), TITLE, [m.name for m in doa_methods()])
    Ks = sorted({int(r["K"]) for r in meas})
    for sc, what in (("single", "One Reflector"), ("multi", "Two or Three Reflectors"), (None, "Every Take and the Empty Room")):
        t = by_k(meas, sc)
        for metric, ylabel in (("FDR", r"$\widehat{\mathrm{FDR}}$"), ("TPR", "TPR"), ("RMSE", r"RMSE of $\theta$ [deg]")):
            metric_vs(pdf, t, Ks, metric, r"Number of Chirps $K$", ylabel,
                      f"{TITLE}, {what}: {metric if metric != 'RMSE' else 'RMSE of $\\theta$'} vs.\\ Number of Chirps",
                      logx=True, ylim=(-0.03, 1.03) if metric != "RMSE" else None)


def fdr_control(pdf, sw_m):
    alpha_diagonal(pdf, sw_m, TITLE)


def range_estimation(pdf, all_rows):
    rows = [r for r in all_rows if r["role"] == "evaluated"]
    takes = sorted({r["take"] for r in rows})
    items = [(t, j) for t in takes for j in sorted({int(r["reflector"]) for r in rows if r["take"] == t})]
    looks = (("fft", "FFT (Bin Centre)", style.MUTED, "s"), ("parabolic", "FFT + Parabola", style.SERIES[2], "^"),
             ("czt", "CZT Zoom", style.PRIMARY, "o"))
    fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.62), sharey=True, layout="constrained")
    for ax, K in zip(axs, (1, 128)):
        for k, (m, name, c, mk) in enumerate(looks):
            e = [[num(r["error_cm"]) for r in rows if r["take"] == t and int(r["reflector"]) == j and r["method"] == m
                  and int(r["K"]) == K] for t, j in items]
            y = np.arange(len(items)) + (k - 1) * 0.25
            ax.errorbar([np.mean(v) for v in e], y, xerr=[np.std(v) for v in e], color=c, marker=mk, ms=4.5, ls="none",
                        lw=0.8, capsize=1.5, label=name, mfc=c if m == "czt" else "white")
        ax.axvline(0, **TRUTH_LINE)
        ax.plot([], [], label="Tape Measure", **TRUTH_LINE)
        ax.set_xlabel(r"$\hat{R} - R_\mathrm{tape}$ [cm] (Mean $\pm$ Std over 20 Frames)")
        style.title(ax, f"$K = {K}$ Chirp{'s' if K > 1 else ''}")
    axs[0].set_yticks(np.arange(len(items)), [f"{tt(t)}, Reflector {j + 1}" for t, j in items])
    axs[0].invert_yaxis()
    style.legend_below(fig, axs[:1], ncols=4)
    style.title(fig, f"{TITLE}: Range against the Tape Measure")
    style.save(pdf, fig)
    range_offsets(pdf, all_rows)


def range_precision(pdf, rows, resolution, dataset_rows, cfg):
    rows = [r for r in rows if r["role"] == "evaluated"]
    snr_of = {}
    for r in dataset_rows:
        snr_of.setdefault((r["take"], int(r["reflector"])), []).append(num(r["snr_db"]))
    items = sorted({(r["take"], int(r["reflector"])) for r in rows})
    looks = {"fft": ("FFT (Bin Centre)", style.MUTED, "s"), "parabolic": ("FFT + Parabola", style.SERIES[2], "^"),
             "czt": ("CZT Zoom", style.PRIMARY, "o")}
    fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.42), layout="constrained")
    snrs = []
    for K in (1, 128):
        for m, (name, c, mk) in looks.items():
            x = [np.mean(snr_of[it]) - 10 * np.log10(128 / K) for it in items]
            y = [10 * np.std([num(r["error_cm"]) for r in rows if (r["take"], int(r["reflector"])) == it and r["method"] == m
                              and int(r["K"]) == K]) for it in items]
            axs[0].plot(x, y, color=c, marker=mk, ms=4.5, ls="none", mfc=c if K == 128 else "white",
                        label=name if K == 128 else None)
            snrs += x
    s = np.linspace(min(snrs) - 1, max(snrs) + 1, 60)
    axs[0].plot(s, 1e3 * crb_range_m(s, cfg, cfg.n_virt), color=style.NORM, ls=(0, (1, 1)), lw=1.0, label=r"Cram\'er--Rao Bound")
    axs[0].axhline(1e3 * 2 * cfg.range_res_m / (N_POINTS - 1) / np.sqrt(12), color=style.MUTED, lw=0.8, ls="-.",
                   label=r"CZT Step$/\sqrt{12}$")
    axs[0].plot([], [], color=style.INK, marker="o", ms=4.5, ls="none", label="$K = 128$")
    axs[0].plot([], [], color=style.INK, marker="o", ms=4.5, ls="none", mfc="white", label="$K = 1$")
    axs[0].set_xlabel("SNR of the Reflector per Channel [dB]")
    axs[0].set_ylabel(r"Std of $\hat{R}$ over 20 Frames [mm]")
    style.title(axs[0], "Precision, Every Reflector")
    seps = sorted({(r["take"], r["pair"]) for r in resolution})
    for k, (m, (name, c, mk)) in enumerate(looks.items()):
        sel = [[r for r in resolution if (r["take"], r["pair"]) == sp and r["method"] == m and int(r["K"]) == 128] for sp in seps]
        axs[1].plot([num(v[0]["separation_cells"]) + (k - 1) * 0.06 for v in sel], [np.mean([int(r["resolved"]) for r in v]) for v in sel],
                    color=c, marker=mk, ms=4.5, ls="none")
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].set_xlabel(r"Spacing of Two Reflectors by Tape [Range Cells $\Delta R$]")
    axs[1].set_ylabel("Share Resolved")
    style.title(axs[1], "Resolution, Pairs in the Takes, $K = 128$")
    style.legend_below(fig, axs[:1], ncols=4)
    style.title(fig, f"{TITLE}: Range Estimators")
    style.save(pdf, fig)


def range_offsets(pdf, rows):
    sel = [r for r in rows if r["method"] == "czt" and int(r["K"]) == 128]
    items = sorted({(r["role"] == "calibration", r["take"], int(r["reflector"])) for r in sel})
    off = [np.mean([num(r["error_cm"]) for r in sel if r["take"] == t and int(r["reflector"]) == j]) for _, t, j in items]
    fig, ax = plt.subplots(figsize=style.size(style.DOUBLE, 0.45), layout="constrained")
    y = np.arange(len(items))
    ax.barh(y, off, height=0.6, color=[style.ACCENT if cal else style.PRIMARY for cal, _, _ in items])
    for yi, v in zip(y, off):
        ax.text(v + 0.2 * np.sign(v), yi, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=6.5)
    mean = np.mean([v for (cal, _, _), v in zip(items, off) if not cal])
    ax.axvline(mean, color=style.MUTED, ls="--", lw=0.9, label=f"Mean of the Evaluated Reflectors: {mean:+.1f} cm")
    ax.axvline(0, **TRUTH_LINE)
    ax.margins(x=0.1)
    ax.set_yticks(y, [f"{tt(t)}, Reflector {j + 1}" + (" (Calibration)" if cal else "") for cal, t, j in items])
    ax.invert_yaxis()
    ax.set_xlabel(r"Mean of $\hat{R} - R_\mathrm{tape}$ [cm] (CZT, $K = 128$)")
    ax.bar(0, 0, color=style.PRIMARY, label="Evaluated Reflector")
    ax.bar(0, 0, color=style.ACCENT, label="Large Corner Reflector of the Calibration Take")
    style.title(ax, f"{TITLE}: Range Offset per Reflector")
    style.legend_below(fig, [ax], ncols=3)
    style.save(pdf, fig)


def everything(pdf, meas):
    all_methods(pdf, by_k(meas), sorted({int(r["K"]) for r in meas}), r"Number of Chirps $K$", True, TITLE)



def chirps(pdf, meas):
    table = by_k(meas)
    need = needed_k(table)
    Ks = sorted({k for _, k in table})
    names = sorted(need, key=lambda m: (not need[m][1], need[m][0], table[(m, need[m][0])]["FDR"]))
    F = np.array([[table.get((m, k), {}).get("FDR", np.nan) for k in Ks] for m in names])
    T = np.array([[table.get((m, k), {}).get("TPR", np.nan) for k in Ks] for m in names])
    fig, ax = plt.subplots(figsize=style.size(style.SINGLE * 1.45, 1.75), layout="constrained")
    im = ax.imshow(F, cmap=style.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i, m in enumerate(names):
        for j in range(len(Ks)):
            if np.isfinite(F[i, j]):
                ax.text(j, i, f"{F[i, j]:.2f} / {T[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if F[i, j] > 0.55 else style.INK)
        k, ok = need[m]
        ax.add_patch(plt.Rectangle((Ks.index(k) - 0.5, i - 0.5), 1, 1, fill=False, lw=1.6,
                                   edgecolor=style.GOAL if ok else style.LOSS, ls="-" if ok else "--"))
    ax.set_xticks(range(len(Ks)), [str(k) for k in Ks])
    ax.set_xlabel(r"Number of Chirps $K$")
    ax.set_yticks(range(len(names)), [lab(m) for m in names])
    fig.colorbar(im, ax=ax, label=r"$\widehat{\mathrm{FDR}}$", shrink=0.6)
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor=style.GOAL, lw=1.6,
                               label=r"Fewest $K$ with $\widehat{\mathrm{FDR}} \le \alpha = 0.1$"))
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, fill=False, edgecolor=style.LOSS, lw=1.6, ls="--",
                               label=r"Never $\le \alpha$: $K$ of the Lowest FDR"))
    style.legend_below(fig, [ax], ncols=1)
    style.title(ax, r"FDR / TPR of Every Method vs.\ Number of Chirps")
    style.title(fig, f"{TITLE}: Chirps Needed for FDR Control")
    style.save(pdf, fig)


def model_order(pdf, rows):
    looks = [("measured", 1, style.PRIMARY, ":", style.HATCHES[1]), ("measured", 128, style.PRIMARY, "-", ""),
             ("simulated", 1, style.NORM, ":", style.HATCHES[1]), ("simulated", 128, style.NORM, "-", "")]
    fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.4), layout="constrained")
    names = sorted((k for k in rows[0] if k.startswith("lam")), key=lambda k: int(k[3:-3]))
    over = np.arange(0, 8)
    for i, (src, K, c, ls, hatch) in enumerate(looks):
        sel = [r for r in rows if r["source"] == src and int(r["K"]) == K]
        label = f"{src.capitalize()}, $K = {K}$"
        lam = np.median([[num(r[n]) for n in names] for r in sel], axis=0)
        axs[0].plot(np.arange(1, lam.size + 1), lam, ls=ls, color=c, marker="o", ms=3, label=label)
        d = np.array([int(r["k_hat"]) - int(r["n_true"]) for r in sel])
        share = [np.mean(d == k) for k in over]
        axs[1].bar(over + (i - 1.5) * 0.2, share, width=0.2, color="white" if hatch else c, edgecolor=c, hatch=hatch,
                   lw=0.6, label=label)
    axs[0].set_xlabel("Eigenvalue Index")
    axs[0].set_ylabel("Eigenvalue [dB re Smallest] (Median Gate)")
    style.title(axs[0], "Eigenvalues of the Smoothed Covariance")
    axs[1].set_xticks(over)
    axs[1].set_xlabel(r"MDL Order $-$ Reflectors in the Gate")
    axs[1].set_ylabel("Share of Range Gates")
    style.title(axs[1], "Over-Counting by MDL")
    style.legend_below(fig, axs[:1], ncols=4)
    style.save(pdf, fig)
