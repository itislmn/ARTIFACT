"""
plots.py - the pattern figures from output/cache/results.pkl: every reading of all 16 channels per data set (marker
= TX, colour = RX, 0 dB = its own largest reading, noise floor dashed), the radomes against the open antenna on one
common 0 dB with their difference, and readme.svg.

    python plots.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pattern_measurements import dataset  # noqa: E402
from pattern_measurements.evaluate import BIN_DEG, CACHE  # noqa: E402
from visualization import style  # noqa: E402

FIGURES = dataset.OUTPUT / "figures"
TX_MARKERS = ["o", "s", "^", "D"]
COLOR = {"air": style.NORM, "pla": style.SERIES[1], "petg": style.SERIES[2]}


def channels(res, pdf):
    for tag, label in dataset.TAGS.items():
        d = res["data"][tag]
        db = 20 * np.log10(d["gain_linear"] / d["gain_linear"].max())
        floor = 20 * np.log10(np.nanmedian(d["noise_linear"]) / d["gain_linear"].max())
        fig, ax = plt.subplots(figsize=style.size(style.DOUBLE, 0.5), layout="constrained")
        for t in range(4):
            for r in range(4):
                sel = (d["tx"] == t) & (d["rx"] == r)
                ax.plot(d["angle"][sel], db[sel], TX_MARKERS[t], ms=2.2, mfc="none", mew=0.5,
                        color=style.SERIES[[0, 2, 3, 4][r]], label=f"TX{t}-RX{r}")
        ax.axhline(floor, color=style.MUTED, ls="--", lw=1.0, label="Noise Floor (Median)")
        ax.set_xlim(-92, 92)
        ax.set_ylim(5 * np.floor((min(np.nanmin(db), floor) - 2) / 5), 2)
        ax.set_xlabel(r"Azimuth Angle ($^\circ$)")
        ax.set_ylabel("Gain (dB re Largest Reading)")
        style.title(ax, f"{label}: All 16 Channels")
        style.legend_below(fig, [ax], ncols=6)
        style.save(pdf, fig)


def compare_axes(ax, ax2, res, points=True):
    c = res["centres"]
    for tag, label in dataset.TAGS.items():
        d = res["data"][tag]
        curve = res["curves"][tag][0] - res["ref_db"]
        if points:
            ax.plot(d["angle"], 20 * np.log10(d["gain_linear"]) - res["ref_db"], ".", ms=1, alpha=0.12,
                    color=COLOR[tag], rasterized=True)
        ax.plot(c, curve, "-" if tag == dataset.REFERENCE else "-o", ms=2.5, lw=1.4, color=COLOR[tag], label=label)
        if ax2 is not None and tag != dataset.REFERENCE:
            ax2.plot(c, res["curves"][tag][0] - res["curves"][dataset.REFERENCE][0], "-o", ms=2.5, lw=1.2,
                     color=COLOR[tag], label=f"{label} $-$ Open Antenna")
    ax.set_ylim(-45, 3)
    ax.set_ylabel("Gain (dB re Open Antenna Peak)")
    if ax2 is not None:
        ax2.axhline(0, color=style.NORM, lw=0.8)
        for lim in (-60, 60):
            ax2.axvline(lim, color=style.MUTED, ls=":", lw=0.8)
        ax2.set_ylabel(r"$\Delta$ (dB)")
        ax2.set_xlabel(r"Azimuth Angle ($^\circ$)")
    else:
        ax.set_xlabel(r"Azimuth Angle ($^\circ$)")


def compare(res, pdf):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=style.size(style.DOUBLE, 0.72), sharex=True, layout="constrained",
                                  gridspec_kw={"height_ratios": [3, 1.6]})
    compare_axes(ax, ax2, res)
    style.title(ax, "Radomes against the Open Antenna")
    style.title(ax2, "Difference to the Open Antenna")
    style.legend_below(fig, [ax, ax2], ncols=5)
    style.save(pdf, fig)


def readme_figure(res, path):
    fig, ax = plt.subplots(figsize=style.size(style.DOUBLE, 0.38), layout="constrained")
    compare_axes(ax, None, res, points=False)
    style.title(ax, "Azimuth Pattern with and without Radome")
    style.legend_below(fig, [ax], ncols=3)
    style.save(path, fig)


def draw(res):
    style.use()
    for name, fn in (("channels", channels), ("compare", compare)):
        with style.pdf(FIGURES / f"{name}.pdf") as pdf:
            fn(res, pdf)
        print(f"  figures/{name}.pdf")
    readme_figure(res, FIGURES / "readme.svg")
    print("  figures/readme.svg")


if __name__ == "__main__":
    draw(pickle.loads(CACHE.read_bytes()))
