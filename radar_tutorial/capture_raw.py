"""
capture_raw.py - records raw ADC frames through the DCA1000 into recordings/<name>.bin (+ .json with packets, lost
packets and the .cfg time); the .cfg must contain lvdsStreamCfg -1 0 1 0 and adcCfg 2 0.

    python capture_raw.py --cli COM4 --name corner_1m --seconds 3
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from radar_tutorial.capture import record  # noqa: E402
from radar_tutorial.cfgfile import Radar, read_lines  # noqa: E402


def finite(lines, seconds, period_s):
    out = []
    for line in lines:
        if line.startswith("frameCfg"):
            v = line.split()
            v[4] = str(int(seconds / period_s) + 2)
            line = " ".join(v)
        out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--cfg", default=str(HERE / "config" / "awr2944p.cfg"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=3.0)
    a = ap.parse_args()
    radar = Radar.from_cfg(a.cfg)
    out = HERE / "recordings" / f"{a.name}.bin"
    stats = record(a.cli, finite(read_lines(a.cfg), a.seconds, radar.frame_period_s), a.seconds, out)
    frames = stats["bytes"] / radar.bytes_per_frame
    print(f"{stats['packets']} packets, {stats['lost_packets']} lost, {stats['bytes']} bytes = {frames:.1f} frames "
          f"-> {out}")
    if stats["bytes"] == 0:
        print("no data: check lvdsStreamCfg, adcCfg 2 0, the DCA1000 switch SW_CONFIG and check_setup.py")


if __name__ == "__main__":
    main()
