"""
probe_cli.py - asks the radar's CLI which form of a command it accepts: sends each candidate line after the given set-up
lines and prints the firmware's full reply (nothing is started).

    python probe_cli.py --cli COM4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_tutorial import uart  # noqa: E402

SETUP = ["sensorStop", "flushCfg", "dfeDataOutputMode 3"]
CANDIDATES = ["help", "advFrameCfg", "advFrameCfg 3 0 0 1 0", "advFrameCfg 3 0 0 1", "advFrameCfg 3 0 0 1 0 0",
              "advFrameCfg 3 0 0 1 0 0 0", "advFrameCfg 3 0 0 1 0 0 0 0", "subFrameCfg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    a = ap.parse_args()
    with uart.open_cli(a.cli) as port:
        uart.send_cfg(port, SETUP)
        for line in CANDIDATES:
            port.write((line + "\n").encode())
            reply = uart.quiet_read(port, quiet_s=0.4, max_s=4.0).decode(errors="ignore").strip()
            print(f"\n>>> {line}\n{reply}")
        uart.send_cfg(port, ["sensorStop"], log=None)


if __name__ == "__main__":
    main()
