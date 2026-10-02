"""
compare_patterns.py - compare several data sets (e.g. air / PLA radome / PETG radome)
against ONE common 0 dB reference, with numbers.

    python compare_patterns.py --cut azimuth --tags air pla petg --ref air

Data sets are output/<cut>_<tag>_measurements.json (made with --tag, or by split_runs.py).

WHY THIS EXISTS
  plot_pattern.py normalises every plot to its OWN largest reading, so a radome's insertion
  loss is invisible there. Here the 0 dB reference is taken from the --ref data set only, so
  a lossy radome shows up as a curve that sits lower.
  NOTE: that is only meaningful if everything else was the same in all runs (radar gain
  settings, target, distance, background handling). Insertion-loss numbers inherit every
  difference in the setup.

WHAT IS COMPUTED (nothing is removed from the data)
  For every data set all readings of all 16 channels are sorted into angle bins (--bin, default
  5 deg). The curve of a bin is the POWER mean of its readings (mean of amplitude^2, shown in
  dB). This is only a summary line for the comparison - the individual readings are drawn
  faintly behind it. All channels are used for every data set alike, so differences between
  channels (which are the same in every run) cancel in the comparison.
  Printed per data set:
    peak     highest bin, dB re the reference's highest bin ("insertion loss" if negative)
    BW-3/-10 angular width where the curve is within 3 / 10 dB of its OWN peak
    diff     mean and largest |difference| to the reference curve, within +-30 and +-60 deg
    spread   typical scatter of ONE channel's readings inside a bin (std of the dB values) - if the
             differences between data sets are smaller than this, they are not significant.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "output"
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def load(cut, tag):
    path = os.path.join(OUT_DIR, f"{cut}_{tag}_measurements.json")
    if not os.path.exists(path):
        sys.exit(f"Missing {path}")
    log = json.load(open(path))
    ang = np.array([r["angle"] for r in log], float)
    amp = np.array([r["gain_linear"] for r in log], float)
    ch = np.array([4 * r["tx"] + r["rx"] for r in log], int)
    ok = np.isfinite(amp) & (amp > 0)
    return ang[ok], amp[ok], ch[ok]


def binned(ang, amp, ch, edges):
    """power-mean curve (dB, absolute units), per-bin spread (dB), and counts."""
    idx = np.digitize(ang, edges) - 1
    n = len(edges) - 1
    curve = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    cnt = np.zeros(n, int)
    for b in range(n):
        sel = idx == b
        cnt[b] = sel.sum()
        if cnt[b] >= 3:
            curve[b] = 10 * np.log10(np.mean(amp[sel] ** 2))
            # scatter of ONE channel inside the bin (channels differ systematically, which
            # is not noise), median over the channels
            sds = [np.std(20 * np.log10(amp[sel & (ch == c)])) for c in np.unique(ch[sel])
                   if (sel & (ch == c)).sum() >= 3]
            spread[b] = np.median(sds) if sds else np.nan
    return curve, spread, cnt


def width_below_peak(x, y, drop):
    """Angular width over which y >= max(y) - drop, by linear interpolation; nan if the
    curve never falls that far on one side."""
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
    return right - left, left, right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", required=True, choices=["azimuth", "elevation"])
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--ref", default=None, help="tag that defines 0 dB (default: first tag)")
    ap.add_argument("--bin", type=float, default=5.0, help="angle bin width in degrees")
    ap.add_argument("--no-points", action="store_true", help="draw only the bin curves")
    args = ap.parse_args()
    ref = args.ref or args.tags[0]
    if ref not in args.tags:
        sys.exit("--ref must be one of --tags")

    data = {t: load(args.cut, t) for t in args.tags}
    lo = min(d[0].min() for d in data.values())
    hi = max(d[0].max() for d in data.values())
    edges = np.arange(np.floor(lo / args.bin) * args.bin, hi + args.bin, args.bin)
    centers = 0.5 * (edges[:-1] + edges[1:])
    res = {t: binned(*data[t], edges) for t in args.tags}
    ref_peak = np.nanmax(res[ref][0])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
    print(f"0 dB = highest {args.bin:g}-degree bin of '{ref}' (power mean of all channels).\n")
    print(f"{'set':8s} {'readings':>8s} {'peak dB':>8s} {'BW-3':>7s} {'BW-10':>7s} "
          f"{'diff<=30 mean/max':>19s} {'diff<=60 mean/max':>19s} {'spread':>7s}")
    for k, t in enumerate(args.tags):
        c = COLORS[k % len(COLORS)]
        ang, amp, _ = data[t]
        curve, spread, cnt = res[t]
        if not args.no_points:
            ax.plot(ang, 20 * np.log10(amp) - ref_peak, ".", color=c, alpha=0.10, markersize=2, zorder=1)
        ax.plot(centers, curve - ref_peak, "-o", color=c, linewidth=2, markersize=3.5, label=t, zorder=3)
        d = curve - res[ref][0]
        ax2.plot(centers, d, "-o", color=c, linewidth=1.5, markersize=3, label=f"{t} - {ref}")
        row = [f"{t:8s} {len(ang):8d} {np.nanmax(curve) - ref_peak:8.1f}"]
        for drop in (3, 10):
            w, _, _ = width_below_peak(centers, curve, drop)
            row.append(f"{w:7.0f}" if np.isfinite(w) else f"{'n/a':>7s}")
        for lim in (30, 60):
            m = np.abs(centers) <= lim
            dd = d[m][np.isfinite(d[m])]
            row.append(f"{np.mean(dd):+8.1f} /{np.max(np.abs(dd)):6.1f}" if dd.size and t != ref else f"{'-':>19s}")
        row.append(f"{np.nanmedian(spread):7.1f}")
        print(" ".join(row))
    print("\nBW = width in degrees of the curve within 3 / 10 dB of its own peak. spread = median "
          "std (dB) of the readings inside a bin: differences below it are not significant.")

    ax.set_ylabel(f"Gain re peak of '{ref}' (dB)")
    ax.set_title(f"{args.cut.capitalize()} pattern comparison (common 0 dB reference)", fontweight="bold")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="lower center", ncol=len(args.tags))
    lo_y = np.nanmin([np.nanmin(res[t][0]) for t in args.tags]) - ref_peak
    ax.set_ylim(max(lo_y - 3, -75), 3)
    ax2.axhline(0, color="0.3", linewidth=0.8)
    ax2.set_ylabel("difference to reference (dB)")
    ax2.set_xlabel("Angle (Degree)")
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"compare_{args.cut}_{'_'.join(args.tags)}")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=200)
    print(f"Saved {out}.pdf (+ .png)")


if __name__ == "__main__":
    main()
