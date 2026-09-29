"""
diagnose_channels.py — one capture, boresight, answers one question fast:

    "Is one RX (or TX) channel systematically weak, independent of angle?"

That's a hardware/cable/calibration signature, not a directional-pattern
effect, and it's exactly what your azimuth plot showed (all "RX2" points
sitting ~15 dB below everything else at EVERY angle, including 0 deg).
A real antenna pattern effect would vary smoothly with angle; a flat
per-channel offset that doesn't is a wiring/connector/gain problem.

Point the radar at your target at 0 deg (boresight) and run:

    python diagnose_channels.py --cli COM4 --cfg pattern_awr2944P.cfg --range-m 1.5

It prints each of the 16 TX-RX channels' peak amplitude, sorted, and flags
any channel that's an outlier vs. the group median.
"""
import argparse

import numpy as np

from capture_lib import preflight_check, capture_once, load_complex_cube, CaptureError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--range-m", type=float, default=None,
                     help="Expected target range; narrows the peak search "
                          "window. Omit to just take the global peak.")
    ap.add_argument("--seconds", type=float, default=4.0)
    args = ap.parse_args()

    print("Preflight...")
    preflight_check()
    print("OK. Capture ONE boresight shot (target centered, 0 deg)...")

    bin_path = "output/_diagnose.bin"
    import os
    os.makedirs("output", exist_ok=True)
    n_bytes, p = capture_once(args.cli, args.cfg, args.seconds, bin_path)
    print(f"Captured {n_bytes} bytes.\n")

    cube = load_complex_cube(bin_path, p)          # (frames, chirps, 16, samples)
    window = np.hanning(cube.shape[-1])
    range_fft = np.fft.fft(cube * window, axis=-1)
    n_half = cube.shape[-1] // 2
    per_channel = np.abs(range_fft[:, :, :, :n_half]).mean(axis=(0, 1))  # (16, n_half)

    if args.range_m is not None:
        center = int(np.argmin(np.abs(p["range_axis"] - args.range_m)))
        tol = max(3, int(0.3 / (p["range_axis"][1] - p["range_axis"][0])))
        lo, hi = max(0, center - tol), min(n_half, center + tol)
        peak_bin = lo + int(np.argmax(per_channel.mean(axis=0)[lo:hi]))
    else:
        peak_bin = int(np.argmax(per_channel.mean(axis=0)))

    amps = per_channel[:, peak_bin]
    amps_db = 20 * np.log10(amps / amps.max() + 1e-12)

    rows = []
    for ch in range(amps.shape[0]):
        tx, rx = ch // p["numRx"], ch % p["numRx"]
        rows.append((f"TX{tx+1}-RX{rx+1}", amps_db[ch]))

    rows.sort(key=lambda r: -r[1])
    med = np.median(amps_db)
    print(f"Peak range bin: {p['range_axis'][peak_bin]:.2f} m")
    print(f"Median channel level: {med:.1f} dB (relative to strongest channel)\n")
    print(f"{'channel':10s}  {'dB':>7s}   vs median")
    print("-" * 40)
    for name, db in rows:
        delta = db - med
        flag = "  <-- WEAK, check this channel's connector/calibration" if delta < -8 else ""
        print(f"{name:10s}  {db:7.1f}   {delta:+6.1f} dB{flag}")

    weak = [name for name, db in rows if db - med < -8]
    print()
    if weak:
        by_rx = {}
        for name in weak:
            rx = name.split("-")[1]
            by_rx[rx] = by_rx.get(rx, 0) + 1
        by_tx = {}
        for name in weak:
            tx = name.split("-")[0]
            by_tx[tx] = by_tx.get(tx, 0) + 1
        if any(c == p["numTx"] for c in by_rx.values()):
            print("=> Every TX paired with the same RX is weak. That RX element,")
            print("   its cable, or its ADC lane is the problem - not the antenna")
            print("   pattern. Re-check that RX's connector/seating before")
            print("   trusting any azimuth/elevation numbers from it.")
        elif any(c == p["numRx"] for c in by_tx.values()):
            print("=> Every RX paired with the same TX is weak. Check that TX's")
            print("   connector/cable, and that its chirp is actually enabled.")
        else:
            print("=> Weak channels don't line up on one clean TX or RX - could")
            print("   be a couple of individual bad connections, or normal")
            print("   multipath/near-field effects if the target is very close.")
    else:
        print("=> No channel is a big outlier vs the others. Good - the array")
        print("   itself looks healthy; angle-to-angle noise in your sweep is")
        print("   more likely SNR/averaging/multipath, not a bad channel.")


if __name__ == "__main__":
    main()
