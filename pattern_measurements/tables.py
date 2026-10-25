"""
tables.py - the pattern tables as booktabs LaTeX in output/tables.tex: the data sets and the comparison of every
radome with the open antenna (peak, beamwidths, differences within +-30 and +-60 degrees and beyond, spread, noise).

    python tables.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pattern_measurements import dataset  # noqa: E402
from pattern_measurements.evaluate import BIN_DEG, CACHE  # noqa: E402


def table(name, cols, header, rows) -> str:
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    return (f"\\newcommand{{\\{name}}}{{%\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}\n}}\n")


def num(v, f="{:+.1f}"):
    return f.format(v) if np.isfinite(v) else "--"


def write(res, path=dataset.OUTPUT / "tables.tex"):
    data = [(dataset.TAGS[r["tag"]], r["readings"], r["frames"], f"{r['angle_min_deg']:+.0f} to {r['angle_max_deg']:+.0f}",
             f"{r['target_range_m']:.2f}") for r in dataset.rows()]
    t1 = table("pmData", "lrrrr", r"Data set & Readings & Frames & Angles ($^\circ$) & Target range (m)", data)
    rows = []
    for tag, s in res["summary"].items():
        rows.append((dataset.TAGS[tag], num(s["peak_db"]), num(s["bw3_deg"], "{:.0f}"), num(s["bw10_deg"], "{:.0f}"),
                     f"{num(s['diff30_mean_db'])} / {num(s['diff30_max_db'], '{:.1f}')}",
                     f"{num(s['diff60_mean_db'])} / {num(s['diff60_max_db'], '{:.1f}')}",
                     num(s["diff_beyond60_mean_db"]), num(s["spread_db"], "{:.1f}"), num(s["noise_db"], "{:.0f}")))
    t2 = table("pmCompare", "lrrrrrrrr",
               r"Data set & Peak (dB) & BW$_{-3}$ ($^\circ$) & BW$_{-10}$ ($^\circ$) & $\Delta$ $|\theta|\le 30^\circ$ "
               r"& $\Delta$ $|\theta|\le 60^\circ$ & $\Delta$ $|\theta| > 60^\circ$ & Spread (dB) & Noise (dB) \\"
               rf" & & & & mean / max & mean / max & mean & & \\ \multicolumn{{9}}{{l}}{{\footnotesize 0 dB: highest "
               rf"{BIN_DEG:g}$^\circ$ bin of the open antenna; $\Delta$: difference to it per bin}}", rows)
    path.write_text("% \\input{tables.tex} once, then \\pmData, \\pmCompare\n" + t1 + "\n" + t2, encoding="utf-8")


if __name__ == "__main__":
    write(pickle.loads(CACHE.read_bytes()))
    print("output/tables.tex")
