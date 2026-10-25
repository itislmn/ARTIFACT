"""
scorecard.py - the README picture: every method in one table, measured and simulated, one chirp and 128 chirps, FDR
(green at or below alpha, red above), TPR, angle RMSE and time per range gate.

    from estimation.plotting import scorecard
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from estimation.plotting.common import ALPHA, GROUPS, by_k, guaranteed, lab, num
from visualization import style


def cell_colour(kind, v):
    if not np.isfinite(v):
        return (1, 1, 1)
    if kind == "FDR":
        base = np.array(to_rgb(style.GOOD if v <= ALPHA else style.LOSS))
        strength = 0.55 if v <= ALPHA else min(0.25 + v, 0.9)
        return tuple(1 - strength * (1 - base))
    if kind == "TPR":
        return style.SEQUENTIAL(0.08 + 0.6 * v)
    return (0.96, 0.96, 0.96)


def scorecard(path, meas, snap, comp):
    tm, ts = by_k(meas), by_k(snap, "multi")
    ms = {r["method"]: 1e3 * num(r["seconds_per_gate"]) for r in comp if r["sweep"] == "p" and int(r["p"]) == 83}
    cols = [("Measured", 1, "FDR", tm), ("Measured", 1, "TPR", tm), ("Measured", 128, "FDR", tm), ("Measured", 128, "TPR", tm),
            ("Simulated", 1, "FDR", ts), ("Simulated", 1, "TPR", ts), ("Simulated", 128, "FDR", ts), ("Simulated", 128, "TPR", ts),
            ("Measured", 128, "RMSE", tm)]
    rows = [(g, m) for g, members in GROUPS for m in members]
    n_r, n_c = len(rows), len(cols) + 2
    fig, ax = plt.subplots(figsize=(style.DOUBLE, 0.215 * (n_r + len(GROUPS)) + 0.9), layout="constrained")
    y, ticks, labels_ = 0, [], []
    for g, members in GROUPS:
        ax.text(0.05, y + 0.55, style.bold(g), ha="left", va="center", fontsize=7)
        y += 1
        for m in members:
            for j, (src, K, kind, tab) in enumerate(cols):
                v = tab.get((m, K), {}).get(kind, np.nan)
                ax.add_patch(plt.Rectangle((j, y), 1, 1, color=cell_colour(kind, v), lw=0))
                txt = "--" if not np.isfinite(v) else (f"{v:.2f}" if kind != "RMSE" else f"{v:.2f}")
                ax.text(j + 0.5, y + 0.5, txt, ha="center", va="center", fontsize=6.5,
                        color="white" if kind == "TPR" and np.isfinite(v) and v > 0.75 else style.NORM)
            t = ms.get(m, np.nan)
            ax.add_patch(plt.Rectangle((len(cols), y), 1, 1, color=(0.96, 0.96, 0.96), lw=0))
            ax.text(len(cols) + 0.5, y + 0.5, "--" if not np.isfinite(t) else f"{t:.3g}", ha="center", va="center", fontsize=6.5)
            ax.text(len(cols) + 1.5, y + 0.5, "Yes" if guaranteed(m) else "No", ha="center", va="center", fontsize=6.5,
                    color=style.NORM if guaranteed(m) else style.MUTED)
            for x in (4, 8, 9):
                ax.plot([x, x], [y, y + 1], color=style.NORM, lw=0.8)
            ticks.append(y + 0.5)
            labels_.append(lab(m))
            y += 1
    for x in range(0, n_c + 1):
        ax.axvline(x, color="white", lw=1.2)
    head = [f"{kind}\n$K = {K}$" if kind != "RMSE" else "RMSE $\\theta$\n[deg]" for _, K, kind, _ in cols] + ["ms per\nGate", "FDR\nGuarantee"]
    ax.set_xticks(np.arange(n_c) + 0.5, head, fontsize=6.5)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=2, top=True, bottom=False)
    ax.set_yticks(ticks, labels_, fontsize=6.5)
    ax.tick_params(axis="y", length=0, left=False, right=False)
    for x0, name in ((0, "Measured (S2)"), (4, "Simulated, Three Reflectors, 15 dB per Chirp")):
        ax.text(x0 + 2, -1.6, style.bold(name), ha="center", va="bottom", fontsize=7.5)
    ax.set_xlim(0, n_c)
    ax.set_ylim(y, -0.2)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    style.save(path, fig)
