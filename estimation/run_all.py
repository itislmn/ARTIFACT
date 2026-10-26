"""
run_all.py - the whole experiment in one command: dataset, every method on the measured session and the
simulations, the FDR-control study, the guarantee study, the range estimators, the complexity, Model-X under its own
assumption, tables and every figure;
recomputes only missing results.

    python run_all.py            (python run_all.py --again recomputes everything, about 3 h)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from estimation import dataset, evaluate  # noqa: E402

CACHE = dataset.OUTPUT / "cache"


def step(script, *args):
    print(f"python {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)


def main():
    again = "--again" in sys.argv
    if again or not (dataset.OUTPUT / "dataset.csv").exists():
        step("dataset.py")
    todo = [s for s in evaluate.STUDIES if again or not (CACHE / f"records_{s}.csv").exists()]
    if todo:
        step("evaluate.py", "--only", *todo)
    if again or not (CACHE / "sweep_simulation.csv").exists():
        step("fdr_control.py")
    if again or not (CACHE / "guarantee.csv").exists():
        step("guarantee.py")
    if again or not (CACHE / "range_resolution.csv").exists():
        step("ranging.py")
    if again or not (CACHE / "complexity.csv").exists():
        step("complexity.py")
    if again or not (CACHE / "model_order.csv").exists():
        step("model_order.py")
    if again or not (CACHE / "model_x_check.csv").exists():
        step("model_x_check.py", "--workers", "1")
    step("tables.py")
    step("plots.py")


if __name__ == "__main__":
    main()
