"""
Quick DCA1000 capture health check - answers "is this real data or garbage?"
without requiring you to understand radar cubes first.

    python check_dca_capture.py captures/adc_Raw_0.bin

What this looks at, and why:

  1. FILE SIZE - an empty or tiny file (a few KB) almost always means the
     sensor never actually streamed over LVDS - usually a missing
     lvdsStreamCfg line in the .cfg, or sensorStart was never issued while
     the DCA1000 was armed and listening.

  2. IS IT ALL ZEROS / ALL ONE VALUE? - if the DCA1000 opened a file but the
     Ethernet link never received real packets (wrong IP, cable issue,
     firewall), you get either an empty file or a file padded with a
     constant filler value. Real ADC noise is never constant.

  3. AMPLITUDE STATISTICS - real ADC samples from an antenna sitting in a
     normal room look like noise: roughly random values in a modest range
     (typically small compared to the full int16 range of +-32768). If a
     corner reflector or wall is nearby, you'll see periodic structure once
     you FFT it (that's what step3_process_adc.py does properly with real
     units) - but even without a proper .cfg, a raw FFT here will show SOME
     peak if the RF front end is actually alive and receiving.

  4. A rough range profile - not calibrated (we don't have your exact
     numTx/numRx/numAdcSamples confirmed here), but if you see a clear peak
     rather than flat noise, that's strong evidence the whole chain (chip ->
     LVDS -> DCA1000 -> Ethernet -> this file) is carrying a real, live RF
     signal - the single most convincing sign the DCA1000 setup works.
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin_path")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    if not os.path.exists(args.bin_path):
        print(f"!! File does not exist: {args.bin_path}")
        print("   The capture never produced output. Check step2's console log")
        print("   for the fileBasePath the DCA1000 CLI actually used.")
        return

    size = os.path.getsize(args.bin_path)
    print(f"File: {args.bin_path}")
    print(f"Size: {size:,} bytes  ({size/1e6:.2f} MB)")

    if size == 0:
        print("\n!! ZERO BYTES. The DCA1000 never received any data.")
        print("   Most likely: lvdsStreamCfg missing from the .cfg, or")
        print("   sensorStart was sent before the DCA1000 was armed to record.")
        return
    if size < 10_000:
        print("\n!! Suspiciously small (<10 KB). This is not a real multi-frame")
        print("   capture - likely the link opened briefly then stopped.")

    raw = np.fromfile(args.bin_path, dtype=np.int16)
    print(f"\nTotal int16 samples: {raw.size:,}")

    # sanity on value distribution
    uniq = np.unique(raw[: min(len(raw), 2_000_000)])
    print(f"Distinct values in first ~2M samples: {len(uniq)}")
    if len(uniq) < 5:
        print("!! Almost no variation in the data - this looks like a constant")
        print("   filler value, not real ADC noise. The Ethernet link is likely")
        print("   not carrying real samples (check cabling / IP config / that")
        print("   the sensor was actually running while this was captured).")
    else:
        print("   Good - real ADC data has many distinct values (this does).")

    mean, std = raw.astype(np.float64).mean(), raw.astype(np.float64).std()
    absmax = np.abs(raw).max()
    print(f"\nAmplitude stats: mean={mean:.1f}  std={std:.1f}  max|x|={absmax}")
    if std < 1.0:
        print("!! Standard deviation is ~0 - flat signal, not real RF noise.")
    elif absmax >= 32760:
        print("!! Values are hitting the int16 ceiling (~32767) - the receiver")
        print("   may be saturated (too much reflected power, e.g. sensor")
        print("   pointed at something very close/reflective).")
    else:
        print("   Good - looks like real noisy ADC data, not saturated, not flat.")

    # rough uncalibrated FFT on a chunk, just to look for ANY structure
    os.makedirs(args.outdir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chunk = raw[: min(raw.size, 8192)].astype(np.float64)
    chunk = chunk - chunk.mean()
    spec = np.abs(np.fft.fft(chunk * np.hanning(len(chunk))))
    spec = spec[: len(spec) // 2]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(raw[:2000])
    a1.set_title("Raw samples (first 2000)")
    a1.set_xlabel("sample index"); a1.set_ylabel("ADC value"); a1.grid(alpha=0.3)

    a2.plot(20 * np.log10(spec + 1))
    a2.set_title("Rough uncalibrated FFT (look for any peak)")
    a2.set_xlabel("bin"); a2.set_ylabel("dB"); a2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = f"{args.outdir}/dca_healthcheck.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot -> {out_path}")
    print("Open it and look at the right-hand FFT plot. A clear spike above")
    print("the noise floor = real RF signal reaching the DCA1000. Flat noise")
    print("everywhere = likely no real signal (but file structure may still")
    print("be fine - point the sensor at a large flat object like a wall")
    print("about 1-2m away and re-capture to get an obvious peak).")

    print("\n" + "=" * 60)
    print("VERDICT:")
    if size > 10_000 and len(uniq) > 5 and std > 1.0 and absmax < 32760:
        print("  Looks like a real, healthy raw capture. Proceed to")
        print("  process_adc.py with the matching .cfg for calibrated")
        print("  range/Doppler plots with real units.")
    else:
        print("  Something is off - see the !! warnings above before trusting")
        print("  this as a working DCA1000 capture.")


if __name__ == "__main__":
    main()