"""
Pure-Python DCA1000 raw ADC capture. No mmWave Studio, no CLI .exe.

Based on TI's documented DCA1000EVM command/data protocol (DCA1000EVM CLI
Software Developer Guide). Every command prints its raw response so you can
see exactly what's happening at each step - same diagnostic approach we used
last night for the ROM bootloader handshake.

    python dca_raw_capture.py --cli COM4 --cfg profile_raw_capture.cfg --seconds 5

This will:
  1. Configure the DCA1000 FPGA directly over UDP (no external tool)
  2. Arm it to record and listen on the data port
  3. Send your sensor .cfg over UART (must include lvdsStreamCfg, which
     yours already does)
  4. Save everything that streams in to a .bin file
  5. Tell you plainly whether real data arrived

If ANY step here prints something unexpected, STOP and paste it back before
continuing - each step builds on the last, so a silent wrong assumption
early on wastes the whole run.
"""
import argparse
import os
import socket
import struct
import threading
import time

import serial

DCA_IP = "192.168.33.180"
DCA_CFG_PORT = 4096
PC_IP = "192.168.33.30"
PC_CFG_PORT = 4096
PC_DATA_PORT = 4098

# Command codes, from TI's DCA1000EVM command protocol
RESET_FPGA = 0x01
CONFIG_FPGA_GEN = 0x03
RECORD_START = 0x05
RECORD_STOP = 0x06
SYSTEM_CONNECT = 0x09
CONFIG_PACKET_DATA = 0x0B
READ_FPGA_VERSION = 0x0E


def frame(cmd, data=b""):
    return struct.pack("<HHH", 0xA55A, cmd, len(data)) + data + struct.pack("<H", 0xEEAA)


class DCA1000:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(3.0)
        self.sock.bind((PC_IP, PC_CFG_PORT))

    def send(self, name, cmd, data=b""):
        self.sock.sendto(frame(cmd, data), (DCA_IP, DCA_CFG_PORT))
        try:
            resp, _ = self.sock.recvfrom(2048)
            print(f"  {name:20s} -> {resp.hex()}")
            return resp
        except socket.timeout:
            print(f"  {name:20s} -> TIMEOUT (no response)")
            return None

    def close(self):
        self.sock.close()


def config_and_arm(lvds_lanes=2, data_format_bits=16, timer_s=30):
    dca = DCA1000()
    print("Step 1: SYSTEM_CONNECT")
    dca.send("SYSTEM_CONNECT", SYSTEM_CONNECT)

    print("\nStep 2: READ_FPGA_VERSION")
    dca.send("READ_FPGA_VERSION", READ_FPGA_VERSION)

    print("\nStep 3: RESET_FPGA")
    dca.send("RESET_FPGA", RESET_FPGA)
    time.sleep(0.5)

    print("\nStep 4: CONFIG_FPGA_GEN (raw mode, ethernet stream)")
    log_mode = 1        # 1 = raw
    lvds_mode = 1 if lvds_lanes == 4 else 2   # 1=4lane, 2=2lane
    transfer_mode = 1   # 1 = LVDS capture
    capture_mode = 2    # 2 = Ethernet stream
    fmt_map = {12: 1, 14: 2, 16: 3}
    data_format = fmt_map.get(data_format_bits, 3)
    payload = bytes([log_mode, lvds_mode, transfer_mode, capture_mode, data_format, timer_s])
    dca.send("CONFIG_FPGA_GEN", CONFIG_FPGA_GEN, payload)

    print("\nStep 5: CONFIG_PACKET_DATA (packet size / delay)")
    # packet size (bytes), delay (us), reserved - widely-used community values
    pkt_payload = struct.pack("<HHH", 1466, 25, 0)
    dca.send("CONFIG_PACKET_DATA", CONFIG_PACKET_DATA, pkt_payload)

    print("\nStep 6: RECORD_START (arming - no radar data yet)")
    dca.send("RECORD_START", RECORD_START)

    return dca


def record_stop(dca):
    print("\nStopping: RECORD_STOP")
    dca.send("RECORD_STOP", RECORD_STOP)


def send_sensor_config(cli_port, cfg_path, baud=115200):
    print(f"\nSending sensor config from {cfg_path} over {cli_port}...")
    with serial.Serial(cli_port, baud, timeout=1) as ser:
        ser.write(b"\n")
        time.sleep(0.2)
        ser.reset_input_buffer()
        for raw in open(cfg_path):
            line = raw.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            ser.write((line + "\n").encode())
            time.sleep(0.05)
            reply = ser.read(512).decode(errors="ignore")
            status = "Done" if "Done" in reply else reply.strip()[:50]
            print(f"  > {line[:50]:50s} | {status}")
            if "Error" in reply or "not recognized" in reply:
                print(f"  !! Sensor rejected: {line}")


def receive_data(seconds, out_path):
    """Listen on the data port and dump raw payload bytes to a file.

    Each DCA1000 UDP packet starts with a 4-byte sequence number and a
    6-byte byte-count field (both little-endian), followed by payload data.
    We strip that 10-byte header and concatenate payloads in order.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind((PC_IP, PC_DATA_PORT))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    total_bytes = 0
    packet_count = 0
    last_seq = None
    dropped = 0
    t0 = time.time()

    print(f"\nListening on {PC_IP}:{PC_DATA_PORT} for {seconds}s...")
    with open(out_path, "wb") as f:
        while time.time() - t0 < seconds:
            try:
                pkt, addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            if len(pkt) <= 10:
                continue
            seq = struct.unpack("<I", pkt[0:4])[0]
            payload = pkt[10:]
            f.write(payload)
            total_bytes += len(payload)
            packet_count += 1
            if last_seq is not None and seq != last_seq + 1:
                dropped += (seq - last_seq - 1)
            last_seq = seq
            if packet_count % 200 == 0:
                print(f"  {packet_count} packets, {total_bytes/1e6:.2f} MB, "
                      f"~{dropped} dropped so far")

    sock.close()
    print(f"\nReceived {packet_count} packets, {total_bytes:,} bytes total, "
          f"{dropped} packets appear dropped.")
    return total_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, help="Sensor CLI COM port, e.g. COM4")
    ap.add_argument("--cfg", required=True, help="Sensor .cfg WITH lvdsStreamCfg enabled")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default="captures/dca_raw_python.bin")
    ap.add_argument("--lvds-lanes", type=int, default=2, choices=[2, 4])
    ap.add_argument("--bits", type=int, default=16, choices=[12, 14, 16])
    args = ap.parse_args()

    print("=" * 60)
    print("PHASE 1: Configure and arm the DCA1000")
    print("=" * 60)
    dca = config_and_arm(lvds_lanes=args.lvds_lanes, data_format_bits=args.bits)

    # start the data receiver in the background BEFORE starting the sensor,
    # so we don't miss the first frames
    result = {}
    def _recv():
        result["bytes"] = receive_data(args.seconds + 3, args.out)
    t = threading.Thread(target=_recv, daemon=True)
    t.start()
    time.sleep(0.5)

    print("\n" + "=" * 60)
    print("PHASE 2: Configure and start the sensor over UART")
    print("=" * 60)
    send_sensor_config(args.cli, args.cfg)

    print(f"\nWaiting {args.seconds}s for data to stream in...")
    t.join()

    record_stop(dca)
    dca.close()

    print("\n" + "=" * 60)
    size = result.get("bytes", 0)
    if size > 10_000:
        print(f"SUCCESS-LOOKING: {size:,} bytes captured -> {args.out}")
        print("Now run: python check_dca_capture.py " + args.out)
    else:
        print(f"Only {size} bytes captured - something in the chain didn't")
        print("connect. Check the per-step responses above for a TIMEOUT or")
        print("unexpected hex value first.")


if __name__ == "__main__":
    main()