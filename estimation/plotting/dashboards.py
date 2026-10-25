"""
dashboards.py - one PDF per method: how it decides in one range gate of one measured frame (one chirp and 128 chirps)
and its result in the range-azimuth map, then FDR, TPR and angle RMSE vs. K (measured and simulated) and, for the
FDR-controlled methods of the alpha sweep (fdr_control.py), the achieved vs. nominal FDR.

    from estimation.plotting import dashboards
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from estimation.methods import doa_methods
from estimation.plotting.common import (FIGURES, TRUTH_LINE, alpha_line, by_k, frame_result, guaranteed, lab,
                                        nominal_axes, range_azimuth, sweep_table, tt)
from visualization import style

TAKE = "multi_doa"


def process_panels(axs, m, gate_res, truth_near, K):
    info, est = gate_res.info, np.atleast_1d(gate_res.angles_deg)
    kind = info.get("kind")
    deg = lambda u: np.degrees(np.arcsin(np.clip(u, -1, 1)))

    def truth_lines(ax):
        for th in truth_near:
            ax.axvline(th, **TRUTH_LINE)

    def estimates(ax, y):
        if est.size:
            ax.plot(est, np.full(est.size, y), "v", color=style.PRIMARY, ms=7, mec="white", mew=0.6, ls="none",
                    label=r"Estimate $\hat\theta$", zorder=6)

    a, b = axs
    if kind == "spectrum":
        P = info["spectrum"]
        a.plot(deg(info["u"]), 10 * np.log10(P / P.max()), color=style.NORM, lw=1.0, label="Spectrum")
        truth_lines(a)
        estimates(a, 1.5)
        a.set_ylim(-45, 4)
        a.set_ylabel("Spectrum [dB]")
        lam = info["eigvals"]
        b.semilogy(np.arange(lam.size), lam, "o-", color=style.NORM, ms=3, lw=0.8, label=r"Eigenvalues $\lambda_i$")
        b.axvspan(-0.5, info["k_hat"] - 0.5, color=style.GOOD, alpha=0.3, lw=0, label=f"Signal Subspace (MDL $\\hat{{k}} = {info['k_hat']}$)")
        b.set_xlabel("Index $i$")
        b.set_ylabel(r"$\lambda_i$")
    elif kind == "esprit":
        tt = np.linspace(0, 2 * np.pi, 361)
        a.plot(np.cos(tt), np.sin(tt), color=style.MUTED, lw=0.8)
        phi = info["phi"]
        if phi.size:
            a.plot(phi.real, phi.imag, "o", color=style.PRIMARY, ms=5, ls="none", label=r"Eigenvalues $\phi_i$")
        for th in truth_near:
            w = 2 * np.pi * info["d"] * np.sin(np.radians(th))
            a.plot([0, np.cos(w)], [0, np.sin(w)], **TRUTH_LINE)
        a.set_aspect("equal")
        a.set_xlim(-1.3, 1.3)
        a.set_ylim(-1.3, 1.3)
        a.set_xlabel(r"$\Re\,\phi$")
        a.set_ylabel(r"$\Im\,\phi$")
        mdl = info["mdl"]
        b.plot(np.arange(mdl.size), mdl, "o-", color=style.NORM, ms=3, lw=0.8, label=r"$\mathrm{MDL}(k)$")
        b.axvline(info["k_hat"], color=style.PRIMARY, lw=1.1, label=f"$\\hat{{k}} = {info['k_hat']}$")
        b.set_xlabel("Model Order $k$")
        b.set_ylabel("MDL")
    elif kind == "cfar":
        u, P, thr = info["u"], info["power"], info["threshold"]
        ref = np.nanmax(P)
        a.plot(deg(u), 10 * np.log10(P / ref), color=style.NORM, lw=1.0, label="Bartlett Power")
        a.plot(deg(u), 10 * np.log10(np.where(np.isfinite(thr), thr, np.nan) / ref), color=style.ACCENT, lw=1.0,
               label=f"{info['variant'].upper()}-CFAR Threshold")
        truth_lines(a)
        estimates(a, 2.0)
        a.set_ylim(-40, 5)
        a.set_ylabel("Power [dB]")
        ratio = 10 * np.log10(P / np.where(np.isfinite(thr), thr, np.nan))
        b.plot(deg(u), ratio, color=style.NORM, lw=1.0, label="Power / Threshold")
        b.axhline(0, color=style.ACCENT, ls="--", lw=0.9, label="Detection Above 0 dB")
        truth_lines(b)
        b.set_ylim(-40, 15)
        b.set_ylabel("Cell Test [dB]")
    elif kind == "pvalues":
        u, T = info["u"], np.maximum(np.asarray(info["T"], float), 1e-3)
        a.semilogy(deg(u), T, "o-", color=style.NORM, ms=3, lw=0.8, label=r"$T_k = -\ln p_k$ per Nyquist Cell")
        rej = info["rejected"]
        if rej.size:
            a.semilogy(deg(u[rej]), T[rej], "o", color=style.PRIMARY, ms=5, ls="none", label="Rejected")
        truth_lines(a)
        estimates(a, T.max() * 8)
        a.set_ylim(None, T.max() * 30)
        a.set_ylabel(r"$-\ln p_k$")
        i = np.arange(1, T.size + 1)
        b.semilogy(i, np.sort(T)[::-1], "o", color=style.NORM, ms=3, label=r"Sorted $-\ln p_{(i)}$")
        b.semilogy(i, -np.log(i * info["level"] / info["m"]), color=style.ACCENT, lw=1.0,
                   label=r"Step-Up Line $-\ln(i\alpha/p)$")
        b.set_xlabel("Rank $i$")
        b.set_ylabel(r"$-\ln p_{(i)}$")
    elif kind == "knockoffs":
        u, W = info["u"], info["W"]
        a.vlines(deg(u), 0, W, color=style.NORM, lw=0.9, label="$W_j$")
        if np.isfinite(info["tau"]):
            a.axhline(info["tau"], color=style.ACCENT, lw=1.0, label=r"Knockoff+ Threshold $\tau$")
        else:
            a.plot([], [], color=style.ACCENT, label=r"$\tau = \infty$ (Fewer Than $1/\alpha$ Candidates)")
        truth_lines(a)
        a.set_ylabel("$W_j$")
        b.plot(info["Zk"], info["Z"], "o", color=style.PRIMARY, ms=3, ls="none", label=r"$(\tilde Z_j, Z_j)$")
        lim = max(np.max(info["Z"]), np.max(info["Zk"]), 1e-12) * 1.05
        b.plot([0, lim], [0, lim], color=style.MUTED, ls="--", lw=0.8, label=r"$Z_j = \tilde Z_j$")
        b.set_xlabel(r"Knockoff Entry $\tilde Z_j$")
        b.set_ylabel("Original Entry $Z_j$")
    elif kind == "trex":
        r = info["result"]
        u = info["u"]
        Phi = r.Phi_mat[max(r.T_star - 1, 0)]
        if info["domain"] == "real":
            Phi = np.maximum(Phi[:info["p"]], Phi[info["p"]:])
        a.vlines(deg(u), 0, Phi, color=style.NORM, lw=0.9, label=f"$\\Phi_{{T^*,L}}(j)$, $T^* = {r.T_star}$")
        if r.v_star is not None:
            a.axhline(r.v_star, color=style.ACCENT, lw=1.0, label=f"Voting Level $v^* = {r.v_star:.2f}$")
        truth_lines(a)
        estimates(a, 1.06)
        a.set_ylim(0, 1.15)
        a.set_ylabel("Relative Occurrence")
        n_t = r.FDP_hat_mat.shape[0]
        for k in range(n_t):
            b.plot(r.V, r.FDP_hat_mat[k], color=style.SEQUENTIAL(0.35 + 0.6 * k / max(n_t - 1, 1)), lw=0.9,
                   label=f"$\\widehat{{\\mathrm{{FDP}}}}(v, T)$, $T = 1 \\ldots {n_t}$" if k == n_t - 1 else None)
        if r.v_star is not None:
            b.plot(r.v_star, r.FDP_hat_mat[r.T_star - 1][np.argmin(np.abs(r.V - r.v_star))], "o", color=style.ACCENT,
                   ms=5, label="$(v^*, T^*)$")
        b.axhline(info["alpha"], color=style.MUTED, ls="--", lw=0.9, label=r"$\alpha$")
        fin = r.FDP_hat_mat[np.isfinite(r.FDP_hat_mat)]
        b.set_ylim(-0.02, max(3 * info["alpha"], 1.2 * fin.max() if fin.size else 1.0))
        b.set_xlabel("Voting Level $v$")
        b.set_ylabel(r"$\widehat{\mathrm{FDP}}$")
    elif kind == "infinirex":
        u, e = info["u"], np.asarray(info["e"], float)
        pos = e > 0
        a.vlines(deg(u)[pos], 1e-3, e[pos], color=style.NORM, lw=0.9, label="e-Value $e_j$")
        a.axhline(info["threshold"], color=style.ACCENT, lw=1.0, label=f"e-BH Threshold, $\\hat{{k}} = {info['k_hat']}$")
        truth_lines(a)
        estimates(a, info["threshold"] * 4)
        a.set_yscale("log")
        a.set_ylim(1e-1, max(info["threshold"] * 1e3, e.max() * 1e2 if e.size else 10))
        a.set_ylabel("e-Value")
        S = np.asarray(info["S"], float)
        t = np.arange(1, S.size + 1)
        b.plot(t, np.clip(1 - S, 0, 1), "o-", color=style.NORM, ms=3, lw=0.9, label=r"$1 - S_t$: No Dummy Before Step $t$")
        for i, j in enumerate(np.asarray(info["path"], int)[:S.size]):
            b.annotate(f"${deg(u[j]):+.0f}^\\circ$", (t[i], max(1 - S[i], 0)), textcoords="offset points", xytext=(0, 4),
                       ha="center", fontsize=5.5)
        b.set_ylim(-0.05, 1.15)
        b.set_xlabel("Greedy Step $t$")
        b.set_ylabel("$1 - S_t$")
    elif kind == "dnn":
        a.plot(deg(info["u"]), info["p"], color=style.NORM, lw=1.0, label="Network Output")
        a.axhline(info["threshold"], color=style.MUTED, ls="--", lw=0.9, label="Threshold")
        truth_lines(a)
        estimates(a, 1.05)
        a.set_ylim(-0.03, 1.12)
        a.set_ylabel("Output $p_k$")
        b.set_visible(False)
    else:
        a.text(0.5, 0.5, "No Intermediate Values", ha="center", va="center", transform=a.transAxes)
        b.set_visible(False)
    if kind not in ("esprit",):
        a.set_xlim(-62, 62)
        a.set_xlabel(r"Azimuth $\theta$ [deg]")
        if kind == "cfar":
            b.set_xlim(-62, 62)
            b.set_xlabel(r"Azimuth $\theta$ [deg]")
    style.title(a, f"Decision, $K = {K}$")
    if b.get_visible():
        style.title(b, f"Decision Rule, $K = {K}$")
    for ax in axs:
        if ax.get_visible():
            ax.legend(loc="upper right" if (ax is a and kind == "infinirex") else "best", fontsize=6.5)


def dashboards(session, meas, snap, sw_m):
    folder = FIGURES / "dashboards"
    tm, ts = by_k(meas), by_k(snap, "multi")
    sweeps = {K: sweep_table(sw_m, K) for K in (1, 128)} if sw_m else {}
    for m in doa_methods():
        with style.pdf(folder / f"{m.name}.pdf") as pdf:
            fig, axs = plt.subplots(2, 3, figsize=style.size(style.DOUBLE, 0.7), layout="constrained")
            for row, K in enumerate((1, 128)):
                t, res = frame_result(session, TAKE, K, [m])
                gi = int(np.argmin([abs(g.range_m - t["truth"][0, 0]) for g in res.gates]))
                near = t["truth"][np.abs(t["truth"][:, 0] - res.gates[gi].range_m) < 0.2, 1]
                process_panels(axs[row, :2], m.name, res.results[m.name][gi], near, K)
                range_azimuth(axs[row, 2], session, res, t["truth"], [(lab(m.name), res.detections[m.name], style.PRIMARY, "o")])
                style.title(axs[row, 2], f"Result, $K = {K}$")
                axs[row, 2].legend(loc="upper left", fontsize=6.5)
            style.title(fig, f"{lab(m.name)}: Measured Take {tt(TAKE)}")
            style.save(pdf, fig)
            curves = {K: sorted((a, v[0]) for (mm, a), v in tab.items() if mm == m.name) for K, tab in sweeps.items()}
            nominal = guaranteed(m.name) and any(curves.values())
            if nominal:
                fig, axs = plt.subplots(2, 2, figsize=style.size(style.DOUBLE, 0.62), layout="constrained")
            else:
                fig, axs = plt.subplots(1, 3, figsize=style.size(style.DOUBLE, 0.34), layout="constrained")
            for ax, metric in zip(axs.flat, ("FDR", "TPR", "RMSE")):
                for t, name, c, ls, mk in ((tm, "Measured (S2)", style.PRIMARY, "-", "o"),
                                           (ts, "Simulation, Three Reflectors", style.SERIES[2], "--", "s")):
                    Ks = sorted(k for (mm, k) in t if mm == m.name)
                    ax.plot(Ks, [t[(m.name, k)][metric] for k in Ks], color=c, ls=ls, marker=mk, ms=4.5, label=name)
                if metric == "FDR" and guaranteed(m.name):
                    alpha_line(ax)
                ax.set_xscale("log", base=2)
                ax.set_xlabel(r"Number of Chirps $K$")
                ax.set_ylabel({"FDR": r"$\widehat{\mathrm{FDR}}$", "TPR": "TPR", "RMSE": r"RMSE of $\theta$ [deg]"}[metric])
                if metric != "RMSE":
                    ax.set_ylim(-0.03, 1.03)
                style.title(ax, {"FDR": "FDR vs.\\ $K$", "TPR": "TPR vs.\\ $K$", "RMSE": r"Angle RMSE vs.\ $K$"}[metric])
            if nominal:
                ax = axs[1, 1]
                nominal_axes(ax, "FDP")
                for K, pts in curves.items():
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-" if K == 1 else "D:", ms=4,
                            color=style.PRIMARY if K == 1 else style.ACCENT, label=f"Measured, $K = {K}$")
                ax.set_xlabel(r"Nominal Level $\alpha$")
                ax.set_ylabel(r"Achieved $\widehat{\mathrm{FDR}}$")
                style.title(ax, r"Achieved vs.\ Nominal FDR")
            style.legend_below(fig, axs.flat, ncols=4 if nominal else 3)
            style.title(fig, f"{lab(m.name)}: FDR, TPR and Angle RMSE vs.\\ $K$")
            style.save(pdf, fig)
