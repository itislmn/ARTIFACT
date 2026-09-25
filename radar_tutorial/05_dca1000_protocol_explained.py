"""
LESSON 05 — Talking to the DCA1000 directly, no vendor tool

The DCA1000 is a separate little computer (an FPGA) that just captures raw
ADC bytes and forwards them to your PC over Ethernet. You control it with
short UDP messages. No special library required - it's genuinely just
`socket.sendto()` with carefully formatted bytes.

Run it:
    python 05_dca1000_protocol_explained.py --dca-ip 192.168.33.180

THE MESSAGE FORMAT, explained once, used forever:
    0xA55A  |  command (2 bytes)  |  data length (2 bytes)  |  data  |  0xEEAA
    (start)                                                          (end)

Every command below is sent this way. We print the hex of what we SEND and
what we RECEIVE, so you can see the actual bytes crossing the wire - not an
abstraction, literally the bytes.
"""
import argparse
import socket
import struct
import time

PC_IP = "192.168.33.30"
PC_PORT = 4096
DCA_PORT = 4096

# --- the command codes themselves, from TI's DCA1000 command protocol ---
RESET_FPGA = 0x01
CONFIG_FPGA_GEN = 0x03
RECORD_START = 0x05
RECORD_STOP = 0x06
SYSTEM_CONNECT = 0x09
CONFIG_PACKET_DATA = 0x0B
READ_FPGA_VERSION = 0x0E


def frame(cmd, data=b""):
    return struct.pack("<HHH", 0xA55A, cmd, len(data)) + data + struct.pack("<H", 0xEEAA)


def send(sock, dca_ip, name, cmd, data=b""):
    pkt = frame(cmd, data)
    print(f"\nSending {name} (code 0x{cmd:02x}):")
    print(f"  bytes out -> {pkt.hex()}")
    sock.sendto(pkt, (dca_ip, DCA_PORT))
    try:
        resp, _ = sock.recvfrom(2048)
        print(f"  bytes in  <- {resp.hex()}")
        # the response repeats the header, then a 2-byte status code (0=success)
        status = struct.unpack("<H", resp[6:8])[0]
        print(f"  status field = {status}  ({'success' if status == 0 else 'ERROR'})")
        return resp
    except socket.timeout:
        print("  no reply - TIMEOUT")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dca-ip", default="192.168.33.180")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(3.0)
    sock.bind((PC_IP, PC_PORT))

    print("STEP 1: say hello.")
    send(sock, args.dca_ip, "SYSTEM_CONNECT", SYSTEM_CONNECT)

    print("\nSTEP 2: ask what firmware version it's running (pure curiosity, harmless).")
    send(sock, args.dca_ip, "READ_FPGA_VERSION", READ_FPGA_VERSION)

    print("\nSTEP 3: reset it to a known clean state before configuring.")
    send(sock, args.dca_ip, "RESET_FPGA", RESET_FPGA)
    time.sleep(0.5)

    print("\nSTEP 4: CONFIG_FPGA_GEN - the important one. Six data bytes, each")
    print("  with a specific meaning (this is where 'raw mode', 'how many LVDS")
    print("  lanes', 'stream over Ethernet vs save to SD card' get decided):")
    log_mode = 1       # 1 = raw ADC data, 2 = pre-processed
    lvds_lanes_code = 2  # 1 = 4 lanes, 2 = 2 lanes -- depends on YOUR board wiring
    transfer_mode = 1  # 1 = capture over LVDS (what we want)
    capture_mode = 2   # 1 = save to SD card, 2 = stream over Ethernet (what we want)
    data_format = 3    # 1=12bit, 2=14bit, 3=16bit - must match your adcCfg
    timeout_s = 30      # give up waiting for data after this many seconds
    payload = bytes([log_mode, lvds_lanes_code, transfer_mode,
                      capture_mode, data_format, timeout_s])
    print(f"  payload bytes = {payload.hex()}  "
          f"(logMode={log_mode}, lvdsLanes={lvds_lanes_code}, "
          f"transfer={transfer_mode}, capture={capture_mode}, "
          f"format={data_format}, timeout={timeout_s}s)")
    send(sock, args.dca_ip, "CONFIG_FPGA_GEN", CONFIG_FPGA_GEN, payload)

    print("\nSTEP 5: CONFIG_PACKET_DATA - how big each Ethernet packet should be,")
    print("  and how many microseconds to wait between sending packets (too fast")
    print("  and your PC's network stack starts dropping them).")
    pkt_payload = struct.pack("<HHH", 1466, 25, 0)   # packet size, delay_us, reserved
    send(sock, args.dca_ip, "CONFIG_PACKET_DATA", CONFIG_PACKET_DATA, pkt_payload)

    print("\nSTEP 6: RECORD_START - arm it. It's now waiting for the SENSOR CHIP")
    print("  (separately, over UART - see lesson 06) to actually start sending")
    print("  chirps. The DCA1000 itself generates no radar signal at all.")
    send(sock, args.dca_ip, "RECORD_START", RECORD_START)

    print("\nSTEP 7: since we never actually started the sensor in this lesson,")
    print("  immediately stop recording again - nothing was captured, and")
    print("  that's fine, this lesson was about the CONTROL protocol only.")
    send(sock, args.dca_ip, "RECORD_STOP", RECORD_STOP)

    sock.close()
    print("\nNEXT: lesson 06 does this same sequence, but also starts the real")
    print("sensor over UART afterward, and actually listens for data.")


if __name__ == "__main__":
    main()
