"""
plots.py - every figure from the cached results, in reading order: output/figures/measured/ (the session S2),
output/figures/simulation/ (the Monte Carlo studies), output/figures/dashboards/ (one PDF per method) and readme.svg.

    python plots.py            (python plots.py maps simulation dashboards readme: only these)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from estimation import dataset  # noqa: E402
from estimation.plotting import measured, simulation  # noqa: E402
from estimation.plotting.common import FIGURES, load  # noqa: E402
from estimation.plotting.dashboards import dashboards  # noqa: E402
from estimation.plotting.scorecard import scorecard  # noqa: E402
from visualization import style  # noqa: E402


def write(folder, name, draw):
    with style.pdf(FIGURES / folder / f"{name}.pdf") as pdf:
        draw(pdf)
    print(f"  figures/{folder}/{name}.pdf", flush=True)


def main():
    style.use()
    meas, snr, snap = (load(f"records_{s}") for s in ("measured", "snr", "snapshots"))
    sw_m, sw_s, comp = load("sweep_measured"), load("sweep_simulation"), load("complexity")
    guar, r_meas, r_prec, r_res = load("guarantee"), load("range_measured"), load("range_precision"), load("range_resolution")
    orders = load("model_order")
    preds, rows = load("predictions_measured"), dataset.read_csv(dataset.OUTPUT / "dataset.csv")
    session = dataset.load(dataset.MEASURED)
    checks = load("checks_measured")
    jobs = [("measured", "setup", lambda pdf: measured.setup(pdf, session)),
            ("measured", "maps", lambda pdf: measured.maps(pdf, session, checks)),
            ("measured", "spectra", lambda pdf: measured.spectra(pdf, session)),
            ("measured", "snapshots", lambda pdf: measured.snapshots(pdf, meas)) if meas else None,
            ("measured", "chirps", lambda pdf: measured.chirps(pdf, meas)) if meas else None,
            ("measured", "snr", lambda pdf: measured.snr(pdf, preds, rows, session)) if preds else None,
            ("measured", "fdr_control", lambda pdf: measured.fdr_control(pdf, sw_m)) if sw_m else None,
            ("measured", "range", lambda pdf: measured.range_estimation(pdf, r_meas)) if r_meas else None,
            ("measured", "model_order", lambda pdf: measured.model_order(pdf, orders)) if orders else None,
            ("measured", "all_methods", lambda pdf: measured.everything(pdf, meas)) if meas else None,
            ("simulation", "guarantee", lambda pdf: (simulation.mechanism(pdf), simulation.guarantee(pdf, guar))) if guar else None,
            ("simulation", "snr", lambda pdf: simulation.snr(pdf, snr)) if snr else None,
            ("simulation", "snapshots", lambda pdf: simulation.snapshots(pdf, snap)) if snap else None,
            ("simulation", "fdr_control", lambda pdf: simulation.fdr_control(pdf, sw_s)) if sw_s else None,
            ("simulation", "range", lambda pdf: simulation.range_estimation(pdf, r_prec, r_res)) if r_prec and r_res else None,
            ("simulation", "computation", lambda pdf: simulation.computation(pdf, comp, meas)) if comp and meas else None,
            ("simulation", "all_methods", lambda pdf: simulation.everything(pdf, snr)) if snr else None]
    only = sys.argv[1:]
    for folder, name, draw in filter(None, jobs):
        if not only or name in only or folder in only:
            write(folder, name, draw)
    if (not only or "dashboards" in only) and meas:
        dashboards(session, meas, snap, sw_m)
        print("  figures/dashboards/*.pdf")
    if (not only or "readme" in only) and meas and snap and comp:
        scorecard(FIGURES / "readme.svg", meas, snap, comp)
        print("  figures/readme.svg")


if __name__ == "__main__":
    main()
