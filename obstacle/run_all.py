"""
run_all.py - the whole evaluation in one command: the runs and their ground truth, every metric, dataset.csv,
predictions.csv, detections.csv, decisions.csv, tables.tex and the figures in output/.

    python run_all.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import evaluate  # noqa: E402
import plots  # noqa: E402


def main():
    res = evaluate.run()
    plots.draw(res)
    print("figures -> output/figures, tables -> output/tables.tex, rows -> output/dataset.csv, predictions.csv, "
          "detections.csv, decisions.csv")


if __name__ == "__main__":
    main()
