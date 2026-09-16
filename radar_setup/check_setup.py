"""
AWR2944PEVM and DCA1000 connection setup verification
"""
import socket
import struct
import sys

import serial.tools.list_ports

DCA_IP = "192.168.33.180"
DCA_CFG_PORT = 4096
PC_IP = "192.168.33.30"
PC_CFG_PORT = 4096
PC_DATA_PORT = 4098


def check_com_ports():
    print("\n=== 1. USB / COM ports ===")
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("  !! No serial ports at all. Check the micro-USB cable into J10,")
        print("     and install the FTDI + XDS110 drivers.")
        return
    for p in ports:
        print(f"  {p.device:10s}  {p.description}")
    print("\n  On the AWR2944P EVM you should see FOUR FTDI ports (from J10).")
    print("  Convention: the 3rd is the CLI/application port (115200),")
    print("  the 4th is the data port. If you only see two, you are looking")
    print("  at the XDS110 ports instead - that is a different USB connector.")


def check_dca1000():
    print("\n=== 2. DCA1000 over Ethernet ===")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(3.0)
    try:
        sock.bind((PC_IP, PC_CFG_PORT))
    except OSError as e:
        print(f"  !! Could not bind {PC_IP}:{PC_CFG_PORT} -> {e}")
        print("     Set your Ethernet adapter to STATIC IP 192.168.33.30 / 255.255.255.0")
        print("     (no gateway, no DNS). Then re-run.")
        return False

    # SYSTEM_CONNECT then READ_FPGA_VERSION.
    # Frame layout: 0xA55A | cmd(2) | datalen(2) | data | 0xEEAA  (all little-endian)
    def cmd(code, data=b""):
        return struct.pack("<HHH", 0xA55A, code, len(data)) + data + struct.pack("<H", 0xEEAA)

    for name, code in (("SYSTEM_CONNECT", 0x09), ("READ_FPGA_VERSION", 0x0E)):
        try:
            sock.sendto(cmd(code), (DCA_IP, DCA_CFG_PORT))
            resp, addr = sock.recvfrom(2048)
            print(f"  {name:20s} -> reply from {addr[0]}: {resp.hex()}")
        except socket.timeout:
            print(f"  {name:20s} -> NO REPLY (timeout)")
            print("     Checklist: DCA1000 5V barrel jack in; Ethernet straight into the PC;")
            print("     DCA1000 config switch in SW_CONFIG (software) position; firewall off")
            print("     for this adapter; you pressed the DCA1000 reset button after power-up.")
            sock.close()
            return False
    sock.close()
    return True


def check_data_port():
    print("\n=== 3. Data port availability ===")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((PC_IP, PC_DATA_PORT))
        print(f"  {PC_IP}:{PC_DATA_PORT} is free and bindable. Good.")
        s.close()
        return True
    except OSError as e:
        print(f"  !! {e}  - something else already owns port 4098 (mmWave Studio? CLI?)")
        return False


if __name__ == "__main__":
    check_com_ports()
    ok = check_dca1000()
    check_data_port()
    print("\n" + "=" * 60)
    if ok:
        print("DCA1000 communication going well.")
    else:
        print("DCA1000 not responding.")
    sys.exit(0)