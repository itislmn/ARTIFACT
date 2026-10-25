"""
run_all.py - the whole evaluation in one command: the camera ground truth, the method replayed on every recorded take,
every metric, predictions.csv, detections.csv, tables.tex and the figures in output/; with --videos also the videos
of the live view.

    python run_all.py            (--again replays every take, --videos renders the videos, ~10 min)
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

import dataset  # noqa: E402
import evaluate  # noqa: E402
import plots  # noqa: E402


def main():
    res = evaluate.run(again="--again" in sys.argv)
    plots.draw(res)
    if "--videos" in sys.argv:
        import make_video
        for take in dataset.TAKES:
            make_video.render(take)
    print("figures -> output/figures, tables -> output/tables.tex, "
          "rows -> output/dataset.csv, output/predictions.csv, output/detections.csv")


if __name__ == "__main__":
    main()
