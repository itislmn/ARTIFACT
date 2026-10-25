"""
simulation.py - the figures of the simulations: when FDR control holds (on and off the angle grid), FDR, TPR and RMSE
vs. SNR (with the Cramer-Rao bound) and vs. the number of chirps, achieved vs. nominal FDR, range, computation, all methods.

    from estimation.plotting import simulation
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from estimation import dataset
from estimation.core.array import ArrayModel
from estimation.core.interfaces import Gate
from estimation.core.metrics import crb_theta_deg
from estimation.core.radar import RadarConfig
from estimation.methods import CORE, doa_methods
from estimation.methods.infinirex import continuous_path, dummy_win_prob, omp_path, whitening
from estimation.plotting.common import (LOOK, all_methods, alpha_diagonal, alpha_line, by_k, by_snr, dodge,
                                        guarantee_key, lab, line, metric_vs, needed_k, num, overview)
from estimation.ranging import crb_range_m
from visualization import style

TITLE = "Simulation"
GUARANTEE_METHODS = ["os_cfar", "bh", "ctrex_tlars", "ctrex_tomp", "trex_tlars", "infinirex", "infinirex_radar"]
MODEL_NAMES = {"on_grid": "Reflectors on the Angle Grid", "off_grid": "Between Grid Angles",
               "phase_error": "Between Grid Angles, $5^\\circ$ Phase Errors"}


def mechanism(pdf):
    cfg = RadarConfig.from_cfg(dataset.CFG)
    a = ArrayModel(cfg)
    rng = np.random.default_rng(3)
    u0 = a.u_grid[30] + 0.5 * a.du
    y = np.sqrt(1e4) * a.steering(np.array([u0]))[:, 0] + (rng.standard_normal(a.n) + 1j * rng.standard_normal(a.n)) / np.sqrt(2)
    X = a.steering(a.u_grid)
    Xn = X / np.linalg.norm(X, axis=0)
    path_g, rho_g = omp_path(Xn, y, a.n // 2)
    res_g = 10 * np.log10(np.vdot(y, y).real * np.cumprod(np.r_[1.0, 1 - rho_g[:-1]]))
    W = whitening(Gate(0, 1.0, y[:, None], 1.0))
    path_c, rho_c, _, steer = continuous_path(y, W, a, a.n // 2)
    res_c = []
    for t in range(1, path_c.size + 1):
        r = y
        if t > 1:
            A = steer(continuous_path(y, W, a, t - 1)[2])
            r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        res_c.append(10 * np.log10(np.vdot(r, r).real))
    fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.4), layout="constrained")
    for name, path, rho, res, c, ls, mk in (("Grid Path (Infini-Rex, General)", path_g, rho_g, res_g, style.ACCENT, "--", "s"),
                                            ("Continuous Path (Infini-Rex, Radar)", path_c, rho_c, np.array(res_c), style.PRIMARY, "-", "o")):
        t = np.arange(1, path.size + 1)
        axs[0].plot(t, res, color=c, ls=ls, marker=mk, ms=4.5, label=name)
        axs[1].semilogy(t, dummy_win_prob(rho, a.n), color=c, ls=ls, marker=mk, ms=4.5)
    axs[0].axhline(10 * np.log10(a.n), color=style.MUTED, ls=":", lw=1.0, label="Noise Power")
    axs[0].set_xlabel("Greedy Step $t$ (Residual Before the Step)")
    axs[0].set_ylabel("Residual Power [dB]")
    style.title(axs[0], "What Is Left Before Each Step")
    axs[1].axhline(1.0 / a.p, color=style.MUTED, ls=":", lw=1.0, label="$1/p$")
    axs[1].set_xlabel("Greedy Step $t$")
    axs[1].set_ylabel(r"$P_t = (1 - \rho_t)^{n-1}$")
    style.title(axs[1], "Chance That One Dummy Beats the Step")
    style.legend_below(fig, axs, ncols=4)
    style.title(fig, f"{TITLE}: One Reflector off the Grid")
    style.save(pdf, fig)


def guarantee(pdf, rows):
    snrs = sorted({num(r["snr_db"]) for r in rows})
    for metric in ("FDP", "TPP"):
        fig, axs = plt.subplots(1, 3, figsize=style.size(style.DOUBLE, 0.42), sharey=True, layout="constrained")
        for ax, (model, name) in zip(axs, MODEL_NAMES.items()):
            for i, m in enumerate(GUARANTEE_METHODS):
                v = [[num(r[metric]) for r in rows if r["model"] == model and r["method"] == m and num(r["snr_db"]) == s] for s in snrs]
                y = np.array([np.mean(x) for x in v])
                se = 1.96 * np.array([np.std(x, ddof=1) / np.sqrt(len(x)) for x in v])
                line(ax, dodge(snrs, i, len(GUARANTEE_METHODS), False), y, m, band=se)
            if metric == "FDP":
                alpha_line(ax)
                ax.set_ylim(-0.01, 0.3)
            else:
                ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("SNR per Channel [dB]")
            style.title(ax, name)
        axs[0].set_ylabel(r"$\widehat{\mathrm{FDR}}$" if metric == "FDP" else "TPR")
        guarantee_key(axs[0])
        style.legend_below(fig, axs, ncols=5)
        what = "FDR" if metric == "FDP" else "TPR"
        style.title(fig, f"{TITLE}, Two Reflectors: {what}")
        style.save(pdf, fig)


def snr(pdf, recs):
    grid = sorted({num(r["snr_db"]) for r in recs})
    for scen, what in (("single", "One Reflector"), ("multi", "Three Reflectors")):
        t = by_snr(recs, scen)
        s = np.linspace(min(grid), max(grid), 60)
        for metric, ylabel in (("FDR", r"$\widehat{\mathrm{FDR}}$"), ("TPR", "TPR"), ("RMSE", r"RMSE of $\theta$ [deg]")):
            metric_vs(pdf, t, grid, metric, "SNR per Channel, Chirp-Averaged [dB]", ylabel,
                      f"{TITLE}, {what}: {metric if metric != 'RMSE' else 'RMSE of $\\theta$'} vs.\\ SNR",
                      crb=(s, crb_theta_deg(s)) if metric == "RMSE" and scen == "single" else None, bands=True,
                      ylim=(-0.03, 1.03) if metric != "RMSE" else None)


def snapshots(pdf, recs):
    Ks = sorted({int(r["K"]) for r in recs})
    for scen, what in (("single", "One Reflector"), ("multi", "Three Reflectors")):
        t = by_k(recs, scen)
        overview(pdf, t, f"{TITLE}, {what}, 15 dB per Chirp", [m.name for m in doa_methods()])
        kk = np.array(Ks, float)
        for metric, ylabel in (("FDR", r"$\widehat{\mathrm{FDR}}$"), ("TPR", "TPR"), ("RMSE", r"RMSE of $\theta$ [deg]")):
            metric_vs(pdf, t, Ks, metric, r"Number of Chirps $K$ (15 dB per Chirp)", ylabel,
                      f"{TITLE}, {what}: {metric if metric != 'RMSE' else 'RMSE of $\\theta$'} vs.\\ Number of Chirps",
                      logx=True, crb=(kk, crb_theta_deg(15.0 + 10 * np.log10(kk))) if metric == "RMSE" and scen == "single" else None,
                      bands=True, ylim=(-0.03, 1.03) if metric != "RMSE" else None)


def fdr_control(pdf, sw_s):
    for sc, what in (("single", "One Reflector"), ("multi", "Three Reflectors")):
        alpha_diagonal(pdf, sw_s, f"{TITLE}, {what}", sc)


def range_estimation(pdf, precision, resolution):
    cfg = RadarConfig.from_cfg(dataset.CFG)
    fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.42), layout="constrained")
    snrs = sorted({num(r["snr_db"]) for r in precision})
    looks = {"fft": ("FFT (Bin Centre)", style.MUTED, ":", "s"), "parabolic": ("FFT + Parabola", style.SERIES[2], "--", "^"),
             "czt": ("CZT Zoom", style.PRIMARY, "-", "o")}
    for m, (name, c, ls, mk) in looks.items():
        rm = [np.sqrt(np.mean([num(r["error_cm"]) ** 2 for r in precision if r["method"] == m and num(r["snr_db"]) == s]))
              for s in snrs]
        axs[0].plot(snrs, rm, color=c, ls=ls, marker=mk, ms=4.5, label=name)
    s = np.linspace(min(snrs), max(snrs), 60)
    axs[0].plot(s, 100 * crb_range_m(s, cfg, cfg.n_virt), color=style.NORM, ls=(0, (1, 1)), lw=1.0, label=r"Cram\'er--Rao Bound")
    axs[0].axhline(100 * cfg.range_res_m / np.sqrt(12), color=style.MUTED, lw=0.8, ls="-.", label=r"$\Delta R/\sqrt{12}$")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("SNR per Channel, Chirp-Averaged [dB]")
    axs[0].set_ylabel("RMSE of $R$ [cm]")
    style.title(axs[0], "Precision, One Reflector, $K = 128$")
    seps = sorted({num(r["separation_cells"]) for r in resolution})
    for m in looks:
        name, c, ls, mk = looks[m]
        v = [[int(r["resolved"]) for r in resolution if r["method"] == m and num(r["separation_cells"]) == x] for x in seps]
        p = np.array([np.mean(x) for x in v])
        se = 1.96 * np.sqrt(p * (1 - p) / np.array([len(x) for x in v]))
        axs[1].plot(seps, p, color=c, ls=ls, marker=mk, ms=4.5)
        axs[1].fill_between(seps, np.clip(p - se, 0, 1), np.clip(p + se, 0, 1), color=c, alpha=0.12, lw=0)
    axs[1].set_ylim(-0.02, 1.02)
    axs[1].set_xlabel(r"Spacing of Two Equal Reflectors [Range Cells $\Delta R$]")
    axs[1].set_ylabel("Share Resolved")
    style.title(axs[1], "Resolution, 30 dB, $K = 128$")
    style.legend_below(fig, axs[:1], ncols=5)
    style.title(fig, f"{TITLE}: Range Estimators")
    style.save(pdf, fig)


def computation(pdf, comp, meas):
    table = by_k(meas)
    need = needed_k(table)
    cost = {(r["method"], int(r["K"])): (1e3 * num(r["seconds_per_gate"]), num(r["peak_mb_per_gate"]))
            for r in comp if r["sweep"] == "K"}
    Kmax = max(k for _, k in cost)
    names = sorted((m for m in need if (m, need[m][0]) in cost), key=lambda m: cost[(m, need[m][0])][0])
    y = np.arange(len(names))
    fig, axs = plt.subplots(1, 2, figsize=style.size(style.DOUBLE, 0.8), sharey=True, layout="constrained")
    for ax, idx, xl in ((axs[0], 0, "Median Time per Range Gate [ms]"), (axs[1], 1, "Peak Memory per Range Gate [MB]")):
        v = np.array([cost[(m, need[m][0])][idx] for m in names])
        ax.barh(y, v, color=[LOOK.get(m, (style.NORM,))[0] for m in names], height=0.65,
                hatch=[None if need[m][1] else "///" for m in names], edgecolor="white", lw=0)
        ax.plot([cost[(m, Kmax)][idx] for m in names], y, "o", color=style.INK, mfc="white", ms=3.5, mew=0.8, zorder=4)
        for i, x in enumerate(v):
            ax.text(x * 1.12, i, f"{x:.3g}", va="center", fontsize=6.5)
        ax.set_xscale("log")
        ax.set_xlim(right=max(v.max(), max(cost[(m, Kmax)][idx] for m in names)) * 8)
        ax.set_xlabel(xl)
    axs[0].set_yticks(y, [f"{lab(m)} ($K = {need[m][0]}$)" for m in names])
    axs[0].invert_yaxis()
    style.title(axs[0], "Time")
    style.title(axs[1], "Memory")
    handles = [Patch(color=style.MUTED, label=r"At the Fewest $K$ with $\widehat{\mathrm{FDR}} \le \alpha$ (Measured S2)"),
               Patch(facecolor=style.MUTED, hatch="///", edgecolor="white", label=r"Never $\le \alpha$: at the $K$ of the Lowest FDR"),
               Line2D([], [], marker="o", ls="", color=style.INK, mfc="white", ms=3.5, label=f"Same Method at $K = {Kmax}$")]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    style.title(fig, f"{TITLE}: Cost per Range Gate")
    style.save(pdf, fig)

    fig, ax = plt.subplots(figsize=style.size(style.DOUBLE, 0.62), layout="constrained")
    for m in names:
        c, _, mk = LOOK.get(m, (style.NORM, "-", "o"))
        k, ok = need[m]
        x, t = cost[(m, k)][0], table[(m, k)]["TPR"]
        ax.plot(x, t, mk, color=c, mfc=c if ok else "white", mec=c, ms=6, label=lab(m), zorder=4)
        if k != min(kk for _, kk in cost):
            ax.annotate(f"$K = {k}$", (x, t), xytext=(4, 2), textcoords="offset points", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Median Time per Range Gate at That $K$ [ms]")
    ax.set_ylabel("TPR on S2 at That $K$")
    ax.plot([], [], "o", color=style.NORM, ms=5, label=r"Filled: $\widehat{\mathrm{FDR}} \le \alpha$")
    ax.plot([], [], "o", color=style.NORM, mfc="white", ms=5, label=r"Hollow: $\widehat{\mathrm{FDR}} > \alpha$ at Every $K$")
    style.title(ax, "TPR against Time")
    style.legend_below(fig, [ax], ncols=5)
    style.save(pdf, fig)

    names = [m.name for m in doa_methods() if m.name in CORE]
    sweeps = [(s, k, xl) for s, k, xl in (("p", "p", r"Grid Size $p$"), ("n", "n", r"Regression Rows $n$"),
                                         ("K", "K", r"Number of Chirps $K$")) if any(r["sweep"] == s for r in comp)]
    fig, axs = plt.subplots(1, len(sweeps), figsize=style.size(style.DOUBLE, 0.46), sharey=True, layout="constrained")
    for ax, (sweep, key, xl) in zip(np.atleast_1d(axs), sweeps):
        sub = [r for r in comp if r["sweep"] == sweep]
        for m in names:
            pts = sorted((num(r[key]), 1e3 * num(r["seconds_per_gate"])) for r in sub if r["method"] == m)
            if len(pts) > 1:
                line(ax, [p[0] for p in pts], [p[1] for p in pts], m)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        style.title(ax, f"Time vs.\\ {xl.split(' $')[0]}")
    np.atleast_1d(axs)[0].set_ylabel("Median Time per Gate [ms]")
    style.legend_below(fig, np.atleast_1d(axs), ncols=4)
    style.title(fig, f"{TITLE}: Growth of the Time")
    style.save(pdf, fig)


def everything(pdf, recs):
    all_methods(pdf, by_snr(recs, "multi"), sorted({num(r["snr_db"]) for r in recs}), "SNR [dB]", False,
                f"{TITLE}, Three Reflectors")
