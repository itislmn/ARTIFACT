"""
style.py - the ARTIFACT-TeX figure style shared by every experiment: real LaTeX text, journal sizes, Title Case and
one palette in which every colour has a meaning.

    from visualization import style; style.use()        (python style.py draws palette.svg for the README)
"""
import re
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

SINGLE, DOUBLE = 3.5, 7.16

PRIMARY = "#2F4B7C"
ACCENT = "#B23A48"
GOOD = "#66BB6A"
LOSS = ACCENT
GOAL = "#B8860B"
NORM = "#222222"
MUTED = "#9A9A9A"
INK = NORM
GRID = "#E6E6E6"
SERIES = ["#2F4B7C", "#B23A48", "#2A9D8F", "#9C6ADE", "#17BECF", "#BCBD22"]
HATCHES = ["", "///", "\\\\\\", "...", "xxx", "+++"]

MATERIAL_COLORS = {"metal": "#0072B2", "dielectric": "#E69F00", "waterlike": "#009E73", "none": MUTED}
MOTION_COLORS = {"static": "#000000", "rigid": "#D55E00", "articulated": "#CC79A7", "distributed": "#56B4E9",
                 "none": MUTED}
MARKERS = {"metal": "o", "dielectric": "s", "waterlike": "^", "static": "o", "rigid": "s",
           "articulated": "^", "distributed": "D", "none": "x"}

SEQUENTIAL = LinearSegmentedColormap.from_list("artifact", ["#FFFFFF", "#C9D3E6", "#6C84B0", PRIMARY, "#14213D"])

RC = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": matplotlib.cycler(color=[PRIMARY, ACCENT, MUTED]),
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.5,
    "legend.frameon": False,
    "legend.markerscale": 1.3,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.6,
    "legend.labelspacing": 0.5,
    "legend.columnspacing": 1.6,
    "legend.borderaxespad": 0.6,
    "legend.borderpad": 0.4,
    "image.cmap": "artifact",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}


def use():
    if SEQUENTIAL.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(SEQUENTIAL)
    plt.style.use("default")
    plt.rcParams.update(RC)


def size(width=DOUBLE, ratio=0.62):
    return width, width * ratio


def tex(text):
    parts = str(text).split("$")
    for k in range(0, len(parts), 2):
        parts[k] = re.sub(r"(?<!\\)([_%&])", r"\\\1", parts[k])
    return "$".join(parts)


def bold(text):
    return r"\textbf{" + tex(text) + "}"


def title(target, text):
    (target.suptitle if hasattr(target, "suptitle") else target.set_title)(bold(text))


SMALL = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "of", "on", "or", "over", "per",
         "the", "to", "via", "vs", "with"}
UNITS = {"ms", "deg", "db", "hz", "ghz", "mps", "cm", "mm", "it"}
WORD = re.compile(r"(?<![\\A-Za-z'`^\"~])[A-Za-z](?:[A-Za-z]|\\_)*")
LEGEND_MIN_PT = 7.5


def caps(text):
    first = [True]

    def fix(match):
        word, is_first = match.group(0), first[0]
        first[0] = False
        if "\\_" in word or not word.islower() or len(word) == 1 or word in UNITS:
            return word
        if word in SMALL and not is_first:
            return word
        return word[0].upper() + word[1:]

    parts = str(text).split("$")
    for k in range(0, len(parts), 2):
        parts[k] = WORD.sub(fix, parts[k])
    return "$".join(parts)


def _texts(fig):
    texts = [fig._suptitle] + fig.texts
    legends = list(fig.legends)
    for ax in fig.axes:
        texts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + ax.texts
        legends.append(ax.get_legend())
        for axis in (ax.xaxis, ax.yaxis):
            formatter = axis.get_major_formatter()
            if isinstance(formatter, (matplotlib.ticker.FixedFormatter, matplotlib.ticker.FuncFormatter)):
                locs = axis.get_majorticklocs()
                axis.set_ticks(locs, [caps(formatter(x, i)) for i, x in enumerate(locs)])
    for legend in legends:
        if legend is not None:
            texts += legend.get_texts() + [legend.get_title()]
    return [t for t in texts if t is not None]


def legend_below(fig, axs, ncols=None, **kw):
    seen = {}
    for ax in np.ravel(np.asarray(axs, dtype=object)):
        for h, label in zip(*ax.get_legend_handles_labels()):
            if not label.startswith("_"):
                seen.setdefault(label, h)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    if seen:
        n = ncols or min(len(seen), max(1, int(fig.get_figwidth() / 1.45)))
        fig.legend(list(seen.values()), list(seen.keys()), loc="outside lower center", ncols=n, **kw)


def finish(fig):
    for t in _texts(fig):
        t.set_text(caps(t.get_text()))
    for legend in list(fig.legends) + [ax.get_legend() for ax in fig.axes]:
        if legend is not None:
            for t in legend.get_texts():
                t.set_fontsize(max(t.get_fontsize(), LEGEND_MIN_PT))


def save(target, fig, close=True):
    finish(fig)
    if isinstance(target, PdfPages):
        target.savefig(fig)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target)
    if close:
        plt.close(fig)


@contextmanager
def pdf(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pages:
        yield pages


def palette(path):
    use()
    rows = (("Material", MATERIAL_COLORS), ("Motion", MOTION_COLORS),
            ("Meaning", {"primary": PRIMARY, "accent / loss": ACCENT, "good": GOOD, "goal": GOAL, "norm": NORM,
                         "neutral": MUTED}),
            ("Series, in this order", {f"{k + 1}": c for k, c in enumerate(SERIES)}))
    fig, axs = plt.subplots(len(rows), 1, figsize=(DOUBLE, 2.5), layout="constrained")
    for ax, (name, colors) in zip(axs, rows):
        for k, (label, color) in enumerate(colors.items()):
            ax.barh(0, 1, left=k, color=color, edgecolor="white", lw=2)
            ax.text(k + 0.5, -0.62, label, ha="center", va="top", fontsize=7)
        ax.set_xlim(0, 6)
        ax.set_ylim(-1.1, 0.5)
        ax.axis("off")
        ax.set_title(bold(name), loc="left", fontsize=8)
    save(path, fig)


if __name__ == "__main__":
    from pathlib import Path
    palette(Path(__file__).resolve().parent / "palette.svg")
