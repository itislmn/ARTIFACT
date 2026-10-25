"""
tables.py - the booktabs tables of the thesis in output/tables.tex, one \\newcommand each (takes, measured results,
simulation results, complexity, every method): the thesis does \\input{tables.tex} once and uses the commands.

    python tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from estimation import complexity, dataset  # noqa: E402
from estimation.plotting import common  # noqa: E402
from estimation.methods import CORE, doa_methods, labels  # noqa: E402

ORDER = {
    "bartlett": r"$O(N^2K + Np)$", "capon": r"$O(N^2K + m^3 + mp)$", "music": r"$O(N^2K + m^3 + mp)$",
    "esprit": r"$O(N^2K + m^3)$", "ca_cfar": r"$O(Np + pN_T)$", "os_cfar": r"$O(Np + pN_T\log N_T)$",
    "bh": r"$O(Np_0 + p_0\log p_0)$", "by": r"$O(Np_0 + p_0\log p_0)$", "knockoffs": r"$O(p^{3.5} + F n_\lambda n p)$", "knockoffs_offset0": r"$O(p^{3.5} + F n_\lambda n p)$",
    "knockoffs_lowdim": r"$O(p_0^{3.5} + F n_\lambda n p_0)$", "knockoffs_lowdim_offset0": r"$O(p_0^{3.5} + F n_\lambda n p_0)$", "infinirex": r"$O(n^2p + snp)$", "infinirex_radar": r"$O(n^2p + snp)$",
    "dnn": r"$O(\sum_l d_l d_{l+1})$",
}
TREX_ORDER = r"$O(K_\mathrm{rex}\,T\,n\,(p + L))$"


def f(v, nd=2):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def table(name, cols, rows, align):
    head = " & ".join(cols) + r" \\"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (f"\\newcommand{{\\{name}}}{{%\n\\begin{{tabular}}{{{align}}}\n\\toprule\n{head}\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}}}\n")


def takes_table(session):
    rows = []
    for t in session.takes:
        refl = "; ".join(f"{tg.get('label', '')}: {tg['range_m']:.2f} m, ${tg['azimuth_deg']:+.1f}^\\circ$, "
                         f"${tg.get('elevation_deg', 0.0):+.1f}^\\circ$" for tg in t["doc"].get("targets", [])) or "--"
        rows.append([t["name"].replace("_", r"\_"), t["scenario"], refl])
    return table("estTakes", ["Take", "Role", r"Reflectors ($R$, $\theta$, $\varphi$)"], rows, "llp{0.62\\linewidth}")


def measured_table(meas, comp):
    t = common.by_k(meas)
    ms = {r["method"]: 1e3 * common.num(r["seconds_per_gate"]) for r in comp if r["sweep"] == "p" and int(r["p"]) == 83}
    lab = labels()
    rows = []
    for m in CORE:
        a, b = t.get((m, 1), {}), t.get((m, 128), {})
        rows.append([lab[m], f(a.get("FDR")), f(a.get("TPR")), f(a.get("RMSE")), f(b.get("FDR")), f(b.get("TPR")),
                     f(b.get("RMSE")), f(ms.get(m, np.nan), 1)])
    cols = ["Method", "FDR", "TPR", r"RMSE$_\theta$", "FDR", "TPR", r"RMSE$_\theta$", "ms/gate"]
    return table("estMeasured", cols, rows, "lrrrrrrr").replace(r"\toprule" + "\n",
            r"\toprule" + "\n" + r" & \multicolumn{3}{c}{$K = 1$} & \multicolumn{3}{c}{$K = 128$} & \\" + "\n"
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n")


def simulation_table(snr, snap):
    tm, ts, tk = common.by_snr(snr, "multi"), common.by_snr(snr, "single"), common.by_k(snap, "multi")
    lab = labels()
    rows = []
    for m in CORE:
        rows.append([lab[m], f(tm.get((m, 10.0), {}).get("FDR")), f(tm.get((m, 10.0), {}).get("TPR")),
                     f(tm.get((m, 30.0), {}).get("FDR")), f(tm.get((m, 30.0), {}).get("TPR")),
                     f(ts.get((m, 20.0), {}).get("RMSE")), f(tk.get((m, 1), {}).get("FDR")), f(tk.get((m, 1), {}).get("TPR"))])
    cols = ["Method", "FDR", "TPR", "FDR", "TPR", r"RMSE$_\theta$", "FDR", "TPR"]
    return table("estSimulation", cols, rows, "lrrrrrrr").replace(r"\toprule" + "\n",
            r"\toprule" + "\n" + r" & \multicolumn{2}{c}{10 dB} & \multicolumn{2}{c}{30 dB} & 20 dB & \multicolumn{2}{c}{$K = 1$} \\" + "\n"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}\cmidrule(lr){7-8}" + "\n")


def complexity_table(comp):
    lab = labels()
    rows = []
    for m in doa_methods():
        r = next((r for r in comp if r["method"] == m.name and r["sweep"] == "p" and int(r["p"]) == 83), None)
        if r is None:
            continue
        ap = complexity.exponent(comp, m.name, "p", "p")
        an = complexity.exponent(comp, m.name, "n", "n")
        order = ORDER.get(m.name, TREX_ORDER if m.family in ("ctrex", "trex") else "--")
        rows.append([lab[m.name], f(1e3 * common.num(r["seconds_per_gate"]), 2), f(common.num(r["peak_mb_per_gate"]), 2),
                     f(ap), f(an), order])
    return table("estComplexity", ["Method", "ms/gate", "MB/gate", r"$a_p$", r"$a_n$", "Order"], rows, "lrrrrl")


def appendix_table(meas):
    t = common.by_k(meas)
    lab = labels()
    rows = [[lab[m.name]] + [f(t.get((m.name, K), {}).get(k)) for K in (1, 4, 16, 128) for k in ("FDR", "TPR")]
            for m in doa_methods()]
    cols = ["Method"] + [f"{k}" for K in (1, 4, 16, 128) for k in ("FDR", "TPR")]
    return table("estAllMethods", cols, rows, "l" + "r" * 8).replace(r"\toprule" + "\n",
            r"\toprule" + "\n" + " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$K = {K}$}}" for K in (1, 4, 16, 128)) + r" \\" + "\n")


def range_table(meas, prec, res):
    rows = []
    for m, name in (("fft", "FFT (bin centre)"), ("parabolic", "FFT + parabola"), ("czt", "CZT zoom")):
        e = np.array([common.num(r["error_cm"]) for r in meas if r["method"] == m and int(r["K"]) == 128
                      and r["role"] == "evaluated"])
        rm = [np.sqrt(np.mean([common.num(r["error_cm"]) ** 2 for r in prec if r["method"] == m and common.num(r["snr_db"]) == s]))
              for s in (10.0, 30.0)]
        rs = [np.mean([int(r["resolved"]) for r in res if r["method"] == m and common.num(r["separation_cells"]) == x])
              if any(r["method"] == m for r in res) else np.nan for x in (1.5, 2.0)]
        rows.append([name, f(e.mean(), 1), f(e.std(), 1), f(rm[0], 2), f(rm[1], 2), f(rs[0]), f(rs[1])])
    cols = ["Estimator", "Bias [cm]", "Std [cm]", "10 dB", "30 dB", r"$1.5\,\Delta R$", r"$2\,\Delta R$"]
    head = (r" & \multicolumn{2}{c}{Measured, $K = 128$} & \multicolumn{2}{c}{RMSE [cm]} & \multicolumn{2}{c}{Resolved} \\"
            + "\n" + r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" + "\n")
    return table("estRange", cols, rows, "lrrrrrr").replace("\\toprule\n", "\\toprule\n" + head)


def main():
    session = dataset.load(dataset.MEASURED)
    meas, snr, snap, comp = (common.load(n) for n in ("records_measured", "records_snr", "records_snapshots", "complexity"))
    parts = ["% generated by estimation/tables.py; \\usepackage{booktabs}", takes_table(session)]
    if meas:
        parts += [measured_table(meas, comp), appendix_table(meas)]
    if snr and snap:
        parts.append(simulation_table(snr, snap))
    if comp:
        parts.append(complexity_table(comp))
    rng = [common.load(n) for n in ("range_measured", "range_precision", "range_resolution")]
    if all(rng):
        parts.append(range_table(*rng))
    (dataset.OUTPUT / "tables.tex").write_text("\n".join(parts), encoding="utf-8")
    print("output/tables.tex")


if __name__ == "__main__":
    main()
