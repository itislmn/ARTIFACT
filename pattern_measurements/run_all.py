"""
run_all.py - every result of the pattern measurements from the recorded sweeps in data/: dataset.csv, the binned
patterns (predictions.csv), tables.tex and the figures in output/.

    python run_all.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pattern_measurements import evaluate, plots, tables  # noqa: E402


def main():
    res = evaluate.run()
    tables.write(res)
    plots.draw(res)
    print("output/dataset.csv, predictions.csv, tables.tex, figures/")


if __name__ == "__main__":
    main()
