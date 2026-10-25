"""
run_all.py - the whole tutorial offline, no hardware: dataset.csv, every step on the recordings and the synthetic
targets (predictions.csv), tables.tex and the figures in output/.

    python run_all.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from radar_tutorial import evaluate, plots, tables  # noqa: E402


def main():
    res = evaluate.run()
    tables.write(res)
    plots.draw(res)
    print("output/dataset.csv, predictions.csv, tables.tex, figures/")


if __name__ == "__main__":
    main()
