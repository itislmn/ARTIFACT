"""
check_setup.py - checks the set-up before a measurement: the serial ports of the EVM, the CLI answering sensorStop,
the PC's address 192.168.33.30, the DCA1000 answering SYSTEM_CONNECT and the data port 4098 being free (pass or fail).

    python check_setup.py --cli COM4
"""
import argparse
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_tutorial import dca1000, uart  # noqa: E402


def ports():
    from serial.tools import list_ports
    found = list(list_ports.comports())
    for p in found:
        print(f"    {p.device:8s} {p.description}")
    return bool(found)


def cli(name):
    with uart.open_cli(name) as port:
        return uart.send_cfg(port, ["sensorStop"], log=None)[0][1]


def address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((dca1000.DCA_IP, dca1000.CONFIG_PORT))
    ip = s.getsockname()[0]
    s.close()
    print(f"    route to the DCA1000 leaves from {ip}")
    return ip == dca1000.PC_IP


def card():
    c = dca1000.Dca1000()
    try:
        r = c.send("SYSTEM_CONNECT")
        v = c.send("READ_FPGA_VERSION")
        print(f"    SYSTEM_CONNECT status {r['status']}, FPGA version field {v['status']}")
        return r["ok"]
    finally:
        c.close()


def data_port():
    dca1000.data_socket().close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=None, help="CLI port of the EVM, e.g. COM4")
    a = ap.parse_args()
    checks = [("serial ports", ports), ("PC address 192.168.33.30", address), ("DCA1000 answers", card),
              ("data port 4098 free", data_port)]
    if a.cli:
        checks.insert(1, (f"CLI on {a.cli} answers", lambda: cli(a.cli)))
    ok = True
    for name, fn in checks:
        try:
            passed = bool(fn())
        except (OSError, ImportError, RuntimeError) as e:
            print(f"    {e}")
            passed = False
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
