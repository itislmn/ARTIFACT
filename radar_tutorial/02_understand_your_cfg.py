"""
LESSON 02 — What does your .cfg file actually say?

No hardware needed for this one. A .cfg file is just a list of plain-text
commands, sent one per line to the sensor's CLI. Each line configures one
aspect of how the radar senses. This lesson reads your .cfg and, instead of
just parsing numbers, PRINTS what each one means in plain English, plus the
physical quantities (range resolution, max range, velocity resolution) that
fall out of them.

Run it:
    python 02_understand_your_cfg.py --cfg path\\to\\your_profile.cfg

Read the printed explanation, then open your .cfg file side by side and
match each line to what got printed. That side-by-side reading is the
actual lesson - this script is just a translator.
"""
import argparse
import os
import re

C = 299792458.0  # speed of light, m/s - the one physical constant everything else derives from
OUT_DIR = "output/02_understand_your_cfg"


def parse_and_explain(path):
    p = {"chirpTx": {}}
    print(f"Reading {path}\n" + "=" * 60)
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        cmd, args = parts[0], parts[1:]

        if cmd == "channelCfg":
            p["rxMask"] = int(args[0])
            p["numRx"] = bin(p["rxMask"]).count("1")
            print(f"channelCfg   -> {p['numRx']} RX antennas turned on "
                  f"(mask {p['rxMask']:04b}). More RX = better angle estimation later.")
        elif cmd == "profileCfg":
            p["startFreq_GHz"] = float(args[1])
            p["idleTime_us"] = float(args[2])
            p["rampEndTime_us"] = float(args[5])
            p["slope_MHz_us"] = float(args[7])
            p["numAdcSamples"] = int(args[9])
            p["sampleRate_ksps"] = float(args[10])
            print(f"profileCfg   -> each chirp starts at {p['startFreq_GHz']} GHz, "
                  f"sweeps at {p['slope_MHz_us']} MHz/us for {p['rampEndTime_us']} us,")
            print(f"                and the ADC grabs {p['numAdcSamples']} samples at "
                  f"{p['sampleRate_ksps']} ksps during that sweep.")
            print("                This ONE line sets your bandwidth, and therefore your")
            print("                range resolution and max range (computed below).")
        elif cmd == "chirpCfg":
            idx, tx_mask = int(args[0]), int(args[7])
            p["chirpTx"][idx] = tx_mask
            print(f"chirpCfg {idx}  -> this chirp slot fires TX antenna mask {tx_mask:04b}")
        elif cmd == "frameCfg":
            p["chirpStartIdx"], p["chirpEndIdx"] = int(args[0]), int(args[1])
            p["numLoops"] = int(args[2])
            p["framePeriod_ms"] = float(args[4])
            chirps_per_loop = p["chirpEndIdx"] - p["chirpStartIdx"] + 1
            p["chirpsPerFrame"] = chirps_per_loop * p["numLoops"]
            print(f"frameCfg     -> {chirps_per_loop} chirp(s) repeated {p['numLoops']} times "
                  f"= {p['chirpsPerFrame']} chirps per frame,")
            print(f"                a new frame every {p['framePeriod_ms']} ms "
                  f"({1000/p['framePeriod_ms']:.1f} frames/sec).")
        elif cmd == "adcCfg":
            fmt = int(args[1])
            p["isComplex"] = fmt != 0
            kind = "complex (real IQ pairs)" if p["isComplex"] else "real-only (no Q)"
            print(f"adcCfg       -> ADC output format is {kind}.")
        elif cmd == "lvdsStreamCfg":
            print("lvdsStreamCfg-> raw ADC data WILL stream to the DCA1000 over LVDS.")
            print("                Without this exact line, DCA1000 raw capture gets nothing.")
        elif cmd == "sensorStart":
            print("sensorStart  -> the command that actually turns the radar on and running.")

    if p.get("chirpTx"):
        p["numTx"] = len({m for m in p["chirpTx"].values() if m})

    print("\n" + "=" * 60)
    print("DERIVED PHYSICAL QUANTITIES (this is the point of the whole file):")
    if all(k in p for k in ("slope_MHz_us", "numAdcSamples", "sampleRate_ksps")):
        # slope_MHz_us is "MHz per microsecond" -> convert to Hz per second:
        # 1 MHz/us = 1e6 Hz / 1e-6 s = 1e12 Hz/s. (This exact unit conversion
        # is the easiest place to introduce an off-by-a-million bug, so it's
        # worth staring at until it's obvious.)
        slope_hz_per_s = p["slope_MHz_us"] * 1e12
        fs_hz = p["sampleRate_ksps"] * 1e3
        bw_hz = slope_hz_per_s * (p["numAdcSamples"] / fs_hz)
        range_res = C / (2 * bw_hz)
        range_axis = [i * (C * fs_hz) / (2 * slope_hz_per_s * p["numAdcSamples"])
                      for i in range(p["numAdcSamples"])]
        print(f"  bandwidth used     : {bw_hz/1e9:.3f} GHz")
        print(f"  -> range resolution: {range_res*100:.1f} cm  "
              "(two objects closer than this look like one)")
        print(f"  -> max range       : ~{range_axis[-1]:.1f} m")
    if "numTx" in p and "chirpsPerFrame" in p:
        print(f"  virtual antennas   : {p['numTx']} TX x {p.get('numRx','?')} RX "
              f"= {p['numTx']*p.get('numRx',0)}")
        print(f"  doppler bins/frame : {p['chirpsPerFrame']//p['numTx']} "
              "(chirps per frame, divided across TX antennas - see lesson 09 for why)")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "explained.txt"), "w") as f:
        for k, v in p.items():
            f.write(f"{k}: {v}\n")
    print(f"\nAlso saved a plain dump of every parsed value to "
          f"{OUT_DIR}/explained.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    parse_and_explain(args.cfg)


if __name__ == "__main__":
    main()
