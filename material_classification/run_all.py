"""
run_all.py - the whole experiment in one command: feature table, evaluation, final model, tables and every figure
(output/figures/*.pdf).

    python run_all.py            (python run_all.py --again recomputes every feature)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from material_classification import dataset, evaluate, plots  # noqa: E402
from visualization import style  # noqa: E402

FIGURES = HERE / "output" / "figures"


def example_windows(table, radar, session=None):
    out = {}
    rows = table if session is None else evaluate.subset(table, table["session"] == session)
    for label in sorted(set(rows["joint"])):
        m = rows["joint"] == label
        if label != "none":
            m &= rows["detected"]
        takes = sorted(set(rows["take"][m]))
        strength = [np.median(rows["change_db"][m & (rows["take"] == t)]) for t in takes]
        take = takes[int(np.argmin(strength) if label == "none" else np.argmax(strength))]
        idx = np.flatnonzero(m & (rows["take"] == take))
        i = idx[len(idx) // 2]
        name, rel = take.split("/", 1)
        out[label] = radar.window(name, rel, rows["t_start_s"][i])
    return out


def main():
    dataset.build(again="--again" in sys.argv)
    res = evaluate.run()
    style.use()
    radar = dataset.Radar()
    table = dataset.read_table(dataset.TABLE)
    figures = (("confusion", plots.confusion), ("scores", plots.scores), ("roc", plots.roc),
               ("margin", plots.margin), ("confidence", plots.confidence), ("features", plots.features),
               ("embedding", plots.embedding), ("learning", plots.learning), ("optimization", plots.optimization),
               ("timing", plots.timing), ("takes", plots.takes))
    for name, draw in figures:
        with style.pdf(FIGURES / f"{name}.pdf") as pdf:
            draw(res, pdf)
        print(f"  figures/{name}.pdf")
    with style.pdf(FIGURES / "microdoppler.pdf") as pdf:
        plots.microdoppler(example_windows(table, radar), pdf)
    session = evaluate.TEST_SESSION if res["test"] is not None else None
    plots.dashboards(example_windows(table, radar, session), res["model"], radar.fx, FIGURES / "dashboards")
    plots.readme_figure(res, FIGURES / "readme.svg")
    print(f"figures -> {FIGURES.relative_to(HERE)}, tables -> output/tables.tex, windows -> output/predictions.csv")


if __name__ == "__main__":
    main()
