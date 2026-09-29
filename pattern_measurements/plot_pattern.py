"""
plot_pattern.py - reference-style antenna pattern plot: every reading of all 16
channels, one marker SHAPE per TX, one COLOR per RX, one shared axes. Nothing is
filtered, averaged across channels, smoothed or rejected.

    python plot_pattern.py --cut azimuth
    python plot_pattern.py --cut elevation

Options
  --use clean|raw   clean (default) = the numbers as stored in "gain_linear"
                    (empty-room reference subtracted if the measurement used
                    --background); raw = "gain_raw_linear", without subtraction
                    (only present if the measurement stored it).
  --noise-correct   subtract the measured noise POWER from every reading
                    (P_signal = P_measured - P_noise). Near the noise floor a
                    plain amplitude reads too high (noise adds); this removes that
                    bias. Readings at/below the floor are clamped 3 dB under it.
  --connect         thin line through each channel's points.
  --no-floor        do not draw the noise-floor line.
  --size N          marker size (default 4; 3 for logs above 3000 readings).

dB calculation
  gain_dB = 20*log10( amplitude / A_ref ), A_ref = the single largest amplitude
  in the data (0 dB there, like the reference figure). Amplitudes come from
  pattern_lib.py: coherent chirp average, 3-bin energy, see its header. The dashed
  line is the measured noise floor on the same scale: readings close to it are
  the limit of what this setup can resolve, not antenna response.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "output"
TX_MARKERS = ["*", "o", "x", "s"]                 # TX1..TX4
RX_COLORS = ["red", "green", "blue", "cyan"]      # RX1..RX4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", required=True, choices=["azimuth", "elevation"])
    ap.add_argument("--use", choices=["clean", "raw"], default="clean")
    ap.add_argument("--noise-correct", action="store_true")
    ap.add_argument("--connect", action="store_true")
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--size", type=float, default=None,
                    help="marker size (default 4, or 3 for very dense logs)")
    args = ap.parse_args()

    log_path = os.path.join(OUT_DIR, f"{args.cut}_measurements.json")
    if not os.path.exists(log_path):
        print(f"No measurements yet at {log_path} - run measure_interactive.py or "
              "measure_continuous.py first.")
        return
    log = json.load(open(log_path))
    if not log:
        print(f"{log_path} is empty.")
        return

    key = "gain_raw_linear" if args.use == "raw" else "gain_linear"
    missing = [r for r in log if key not in r]
    if missing:
        print(f"{len(missing)} of {len(log)} readings have no '{key}' (older log format); "
              "using their 'gain_linear' instead.")

    ang = np.array([float(r["angle"]) for r in log])
    tx = np.array([int(r["tx"]) for r in log])
    rx = np.array([int(r["rx"]) for r in log])
    amp = np.array([float(r.get(key, r["gain_linear"])) for r in log])
    noise = np.array([float(r["noise_linear"]) if r.get("noise_linear") is not None else np.nan
                      for r in log])

    if args.noise_correct:
        if not np.isfinite(noise).any():
            print("--noise-correct needs 'noise_linear' in the log (new measurement scripts "
                  "store it); ignoring.")
        else:
            nz = np.where(np.isfinite(noise), noise, 0.0)
            p_sig = amp ** 2 - nz ** 2
            floor_p = (nz / 10 ** (3 / 20)) ** 2            # clamp 3 dB below the noise level
            amp = np.sqrt(np.where(p_sig > floor_p, p_sig, np.maximum(floor_p, 1e-30)))

    if not np.all(np.isfinite(amp)) or np.any(amp <= 0):
        bad = ~np.isfinite(amp) | (amp <= 0)
        print(f"{bad.sum()} readings have a zero/invalid amplitude and cannot be shown on a dB axis.")
        amp = np.where(bad, np.nan, amp)

    a_ref = np.nanmax(amp)
    db = 20 * np.log10(amp / a_ref)
    floor_db = None
    if not args.no_floor and np.isfinite(noise).any():
        floor_db = 20 * np.log10(np.nanmedian(noise) / a_ref)

    txs = sorted(set(tx.tolist()))
    rxs = sorted(set(rx.tolist()))
    print(f"{len(log)} readings, {len(set(np.round(ang, 3)))} distinct angles, "
          f"{ang.min():+.1f} to {ang.max():+.1f} deg, {len(txs)} TX x {len(rxs)} RX.")
    print(f"0 dB reference = largest amplitude in the data ({args.use}"
          f"{', noise-corrected' if args.noise_correct else ''}).")
    if floor_db is not None:
        print(f"Measured noise floor: {floor_db:.1f} dB (dashed line). Readings near it are "
              "noise-limited, not antenna response.")

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9, "axes.edgecolor": "0.25",
                         "grid.color": "0.85", "grid.linewidth": 0.7, "legend.frameon": True})
    msize = args.size if args.size is not None else (4.0 if len(log) < 3000 else 3.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    for t in txs:
        for r in rxs:
            sel = (tx == t) & (rx == r)
            if not sel.any():
                continue
            order = np.argsort(ang[sel], kind="stable")
            ax.plot(ang[sel][order], db[sel][order], marker=TX_MARKERS[t % 4],
                    linestyle="-" if args.connect else "none", color=RX_COLORS[r % 4],
                    markersize=msize, markerfacecolor="none", markeredgewidth=0.9,
                    linewidth=0.5, label=f"TX{t+1}-RX{r+1}")
    if floor_db is not None and np.isfinite(floor_db):
        ax.axhline(floor_db, color="0.35", linestyle="--", linewidth=1.0, zorder=0,
                   label="noise floor")

    lo_db = np.nanmin(db)
    if floor_db is not None and np.isfinite(floor_db):
        lo_db = min(lo_db, floor_db)
    ax.set_ylim(5 * np.floor((lo_db - 2) / 5), 2)
    span = max(ang.max() - ang.min(), 10)
    ax.set_xlim(ang.min() - 0.03 * span, ang.max() + 0.03 * span)
    ax.set_xlabel("Angle (Degree)")
    ax.set_ylabel("Antenna Gain (dB)")
    ax.set_title(f"{'Azimuth' if args.cut == 'azimuth' else 'Elevation'} Angle Sweep",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="lower center", ncol=4, fontsize=7, framealpha=0.9, columnspacing=1.0,
              handletextpad=0.3, borderpad=0.5)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUT_DIR, f"pattern_{args.cut}.pdf")
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=200)
    print(f"Saved {out_pdf} (+ .png)")


if __name__ == "__main__":
    main()
