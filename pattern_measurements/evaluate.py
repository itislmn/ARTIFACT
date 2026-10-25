"""
evaluate.py - the pattern of every data set against one common 0 dB (the reference's highest bin): power mean of all
16 channels in 5-degree bins, beamwidths 3 and 10 dB below the own peak, the difference to the reference within +-30 and
+-60 degrees and the spread of one channel inside a bin; writes output/predictions.csv and output/cache/results.pkl.

    python evaluate.py
"""
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pattern_measurements import dataset  # noqa: E402

CACHE = dataset.OUTPUT / "cache" / "results.pkl"
BIN_DEG = 5.0


def binned(d, edges):
    idx = np.digitize(d["angle"], edges) - 1
    n = len(edges) - 1
    curve, spread, count = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, int)
    for b in range(n):
        sel = idx == b
        count[b] = sel.sum()
        if count[b] >= 3:
            curve[b] = 10 * np.log10(np.mean(d["gain_linear"][sel] ** 2))
            sds = [np.std(20 * np.log10(d["gain_linear"][sel & (d["channel"] == c)]))
                   for c in np.unique(d["channel"][sel]) if (sel & (d["channel"] == c)).sum() >= 3]
            spread[b] = np.median(sds) if sds else np.nan
    return curve, spread, count


def width(x, y, drop):
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    ip = int(np.argmax(y))
    level = y[ip] - drop
    left = right = np.nan
    for i in range(ip, 0, -1):
        if y[i - 1] < level:
            left = x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
            break
    for i in range(ip, len(y) - 1):
        if y[i + 1] < level:
            right = x[i] + (y[i] - level) * (x[i + 1] - x[i]) / (y[i] - y[i + 1])
            break
    return right - left


def run():
    dataset.write()
    data = {t: dataset.load(t) for t in dataset.TAGS}
    lo = min(d["angle"].min() for d in data.values())
    hi = max(d["angle"].max() for d in data.values())
    edges = np.arange(np.floor(lo / BIN_DEG) * BIN_DEG, hi + BIN_DEG, BIN_DEG)
    centres = 0.5 * (edges[:-1] + edges[1:])
    curves = {t: binned(d, edges) for t, d in data.items()}
    ref = np.nanmax(curves[dataset.REFERENCE][0])
    summary, rows = {}, []
    for t, (curve, spread, count) in curves.items():
        diff = curve - curves[dataset.REFERENCE][0]
        s = {"peak_db": float(np.nanmax(curve) - ref), "bw3_deg": width(centres, curve, 3),
             "bw10_deg": width(centres, curve, 10), "spread_db": float(np.nanmedian(spread)),
             "noise_db": float(20 * np.log10(np.nanmedian(data[t]["noise_linear"])) - ref)}
        for lim in (30, 60, 90):
            m = (np.abs(centres) <= lim) & np.isfinite(diff)
            s[f"diff{lim}_mean_db"] = float(np.mean(diff[m])) if m.any() else np.nan
            s[f"diff{lim}_max_db"] = float(np.max(np.abs(diff[m]))) if m.any() else np.nan
        outer = (np.abs(centres) > 60) & np.isfinite(diff)
        s["diff_beyond60_mean_db"] = float(np.mean(diff[outer])) if outer.any() else np.nan
        summary[t] = s
        for c, g, dd, sp, n in zip(centres, curve - ref, diff, spread, count):
            rows.append({"tag": t, "angle_deg": c, "gain_db": g, "diff_to_reference_db": dd, "spread_db": sp,
                         "readings": n})
    with open(dataset.OUTPUT / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = {"data": data, "centres": centres, "curves": curves, "ref_db": ref, "summary": summary}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(res))
    return res


if __name__ == "__main__":
    for tag, s in run()["summary"].items():
        print(tag, {k: round(v, 1) for k, v in s.items()})
