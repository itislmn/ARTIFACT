"""
run_all.py - the whole evaluation in one command: dataset, both readers on every frame, tables and every figure in
output/; with --videos also the two thesis videos (reading animation, light and reading).

    python run_all.py            (--again reads every take again, --videos renders the videos, ~30 min)
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
from backend.settings import FIGURES  # noqa: E402
from visualization import style  # noqa: E402

VIDEOS = (("make_video.py", "text_085c"), ("light_video.py", "light_3"))


def main():
    res = evaluate.run(again="--again" in sys.argv)
    style.use()
    for name, draw in plots.FIGURES:
        with style.pdf(FIGURES / f"{name}.pdf") as pdf:
            draw(res, pdf)
        print(f"  figures/{name}.pdf")
    plots.readme_figure(res, FIGURES / "readme.svg")
    if "--videos" in sys.argv:
        import subprocess
        for script, take in VIDEOS:
            subprocess.run([sys.executable, str(HERE / script), str(HERE / "recordings" / "CV1" / take)], check=True)
    print("figures -> output/figures, tables -> output/tables.tex, frames -> output/dataset.csv, output/predictions.csv")


if __name__ == "__main__":
    main()
