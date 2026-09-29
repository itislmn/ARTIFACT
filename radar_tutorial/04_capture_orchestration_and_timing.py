r"""
LESSON 04 - Putting UART + UDP together, and the timing bug that ate a day

    python 04_capture_orchestration_and_timing.py --demo
    python 04_capture_orchestration_and_timing.py --cli COM4

This lesson has no new protocol in it - it's lessons 02 and 03 RUN AT THE
SAME TIME, which is where a real, costly bug in this project actually
lived. Reading lessons 02/03 in isolation, the bug is invisible; it only
appears once you look at the two channels together on the same timeline.

------------------------------------------------------------------------
THE SEQUENCE THAT HAS TO HAPPEN, IN ORDER, ACROSS TWO CHANNELS
------------------------------------------------------------------------
   UDP (control, port 4096)     UART (CLI port)         UDP (data, port 4098)
   ------------------------     ----------------         ---------------------
1. arm the DCA1000
   (SYSTEM_CONNECT, RESET_FPGA,
   CONFIG_FPGA_GEN,
   CONFIG_PACKET_DATA,
   RECORD_START)
                                                     2. START LISTENING here,
                                                        for a fixed window
                                 3. send the whole
                                    .cfg file, line
                                    by line
                                 4. sensorStart -----> 5. chip starts
                                    (last line)           chirping, DCA1000
                                                           forwards samples
                                                           as UDP packets

Step 2 (open the receive window) and step 3 (send the config) MUST run
CONCURRENTLY - in this project's code, via a background thread - because
you don't know exactly when step 4/5 will happen, and you can't afford
to start listening only AFTER sensorStart, or you'd race the very first
packets. So the receive window opens generously early and stays open
for `capture_seconds + padding`.

THE BUG: if step 3 (sending ~30 config lines) takes much longer than
expected - lesson 02's naive UART reader turned a ~5 second job into a
~30 second one - and the receive window's padding was only a couple of
seconds, then the window in step 2 CLOSES before step 4/5 ever happens.
The receiver was listening at exactly the wrong 4-second slice of time,
completely missing a data stream that only started at second 30. The
diagnostic output made this genuinely confusing to debug: "received 0
UDP packets" would print in the middle of the UART send loop, seemingly
unrelated to it, because the receive thread's timeout simply expired
while step 3 was still slowly grinding through config lines.

Fixed two ways at once, matching what actually shipped in this project:
  1. Lesson 02's fast UART reader - config send drops from ~30s to ~5-7s
     for a real ~30-line .cfg.
  2. Generous receive-window padding (this project settled on
     `seconds + 10`, not `+2`) as a safety margin on top of that.

Watch it fail, then watch it succeed, below - same code, only the UART
reader and the window padding change.
"""
import argparse
import socket
import struct
import threading
import time

from common import (DCA_IP, PC_IP, PC_DATA_PORT, CONFIG_PORT, dca_frame,
                     RESET_FPGA, CONFIG_FPGA_GEN, RECORD_START,
                     SYSTEM_CONNECT, CONFIG_PACKET_DATA,
                     FakeSerial, FakeDCASocket)

CFG_LINES = [  # same 14-line shape as lesson 02, trimmed for a fast demo
    "sensorStop", "flushCfg", "channelCfg 15 15 0 0 0", "adcCfg 2 0",
    "adcbufCfg -1 1 1 1 1", "profileCfg 0 77 186 7 57.14 0 0 70 1 656 13349 0 0 158",
    "chirpCfg 0 0 0 0 0 0 0 1", "chirpCfg 1 1 0 0 0 0 0 4",
    "chirpCfg 2 2 0 0 0 0 0 8", "chirpCfg 3 3 0 0 0 0 0 2",
    "frameCfg 0 3 64 5 656 100 1 0", "lvdsStreamCfg -1 0 1 0", "sensorStart",
]


# ---- lesson 02's two readers, reused here (kept local so this file runs
# standalone - Python module names can't start with a digit, so
# "import 02_uart_cli_protocol" isn't valid syntax; see the README for
# why each lesson is self-contained rather than importing its numbered
# neighbors) ----
def naive_read(ser):
    return ser.read(400)


def quiet_period_read(ser, quiet_after=0.15, max_wait=2.0):
    buf = b""
    t0 = time.time()
    last_data_t = t0
    while time.time() - t0 < max_wait:
        chunk = ser.read(256)
        if chunk:
            buf += chunk
            last_data_t = time.time()
        elif time.time() - last_data_t > quiet_after:
            break
    return buf


def send_config(ser, lines, reader):
    for line in lines:
        ser.write((line + "\n").encode())
        reader(ser)


# ---- the DCA1000 side: arm it, then receive for a fixed window ----
def arm_dca1000(send_recv, lvds_lanes=4):
    def do(code, data=b""):
        resp = send_recv(dca_frame(code, data))
        status = struct.unpack("<H", resp[4:6])[0]
        if status != 0:
            raise RuntimeError(f"command 0x{code:02X} failed, status={status}")
    do(SYSTEM_CONNECT)
    do(RESET_FPGA)
    lvds_code = 1 if lvds_lanes == 4 else 2
    do(CONFIG_FPGA_GEN, bytes([1, lvds_code, 1, 2, 3, 30]))
    do(CONFIG_PACKET_DATA, struct.pack("<HHH", 1466, 25, 0))
    do(RECORD_START)


def receive_window_fake(fake_dca, window_s):
    """Mirrors capture_lib.py's receive_raw(): open the window, collect
    packets until it closes, strip each packet's 10-byte header (4-byte
    sequence number + 6-byte byte-count - the DCA1000's OWN framing,
    layered on top of the UDP packet it's carried in) before keeping the
    payload. Lesson 06 picks up right where this leaves off: what those
    kept payload bytes actually mean."""
    t0 = time.time()
    n_packets, total_bytes = 0, 0
    while time.time() - t0 < window_s:
        try:
            pkt = fake_dca.recv_data(timeout=0.05)
        except socket.timeout:
            continue
        n_packets += 1
        if len(pkt) > 10:
            total_bytes += len(pkt) - 10   # drop the 10-byte DCA1000 header
    return n_packets, total_bytes


def run_scenario(label, reader, window_padding_s, capture_seconds=2.0):
    print(f"\n--- {label} ---")
    fake = FakeDCASocket(adc_bytes_provider=lambda: b"\x00" * 200_000,
                          stream_start_delay_s=0.0)  # set once we know timing

    def send_recv(frame):
        fake.sendto(frame, (DCA_IP, CONFIG_PORT))
        return fake.recvfrom(2048)[0]

    # Time how long config-send WOULD take with this reader, using the
    # same fake-serial machinery as lesson 02, so the streaming delay
    # below reflects a REAL measurement, not a made-up number.
    probe_ser = FakeSerial("FAKE", timeout=(1.0 if reader is naive_read else 0.05),
                            reply_delay_s=0.01)
    with probe_ser:
        t0 = time.time()
        send_config(probe_ser, CFG_LINES, reader)
        uart_send_s = time.time() - t0

    window_s = capture_seconds + window_padding_s
    # Data only starts flowing once sensorStart (the LAST cfg line) has
    # actually been sent - i.e. after the full UART send completes, plus
    # the ~0.5s startup slack this project's real code leaves before
    # starting to send.
    fake.stream_start_delay_s = 0.5 + uart_send_s

    arm_dca1000(send_recv)
    t_receive_start = time.time()
    n_packets, total_bytes = receive_window_fake(fake, window_s)

    print(f"  UART config-send time      : {uart_send_s:.2f}s")
    print(f"  Receive window open for    : {window_s:.2f}s "
          f"(capture_seconds={capture_seconds} + padding={window_padding_s})")
    print(f"  Data would start flowing at: {fake.stream_start_delay_s:.2f}s "
          "after the window opened")
    print(f"  Result: {n_packets} packets, {total_bytes} payload bytes")
    if n_packets == 0:
        print("  -> ZERO packets. The window closed BEFORE data ever "
              "started - this is precisely the '0 packets received' "
              "failure this project hit repeatedly.")
    else:
        print("  -> Data arrived comfortably inside the window. Success.")


def run_demo():
    run_scenario("BEFORE THE FIX (naive UART reader, +2s padding - what "
                 "this project's code actually did at first)",
                 naive_read, window_padding_s=2.0)
    run_scenario("AFTER THE FIX (quiet-period UART reader, +10s padding - "
                 "what this project's code does now)",
                 quiet_period_read, window_padding_s=10.0)
    print("\nSame arm/receive logic both times. Only the UART reader and")
    print("the window padding changed - and that alone is the entire")
    print("difference between 0 packets and a successful capture.")


def run_real(cli_port, capture_seconds=4.0):
    """The real thing: capture_lib.py's capture_once(), condensed. Needs
    both boards powered, cabled, and the .cfg's adcCfg already set to
    '2 0' (lesson 01/05)."""
    import serial

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(3.0)
    sock.bind((PC_IP, CONFIG_PORT))

    def send_recv(frame):
        sock.sendto(frame, (DCA_IP, CONFIG_PORT))
        return sock.recvfrom(2048)[0]

    print("Arming DCA1000...")
    arm_dca1000(send_recv)

    result = {}

    def _receive():
        data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        data_sock.settimeout(1.0)
        data_sock.bind((PC_IP, PC_DATA_PORT))
        window_s = capture_seconds + 10
        t0 = time.time()
        n_packets, total = 0, 0
        while time.time() - t0 < window_s:
            try:
                pkt, _ = data_sock.recvfrom(65536)
            except socket.timeout:
                continue
            n_packets += 1
            total += max(0, len(pkt) - 10)
        data_sock.close()
        result["packets"], result["bytes"] = n_packets, total

    t = threading.Thread(target=_receive, daemon=True)
    t.start()
    time.sleep(0.5)
    print(f"Sending {len(CFG_LINES)} config lines over {cli_port}...")
    with serial.Serial(cli_port, 115200, timeout=0.05) as ser:
        ser.write(b"\n")
        time.sleep(0.2)
        ser.reset_input_buffer()
        send_config(ser, CFG_LINES, quiet_period_read)
    t.join()
    print(f"Received {result['packets']} packets, {result['bytes']} bytes.")
    sock.sendto(dca_frame(0x06), (DCA_IP, CONFIG_PORT))  # RECORD_STOP
    sock.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=None)
    ap.add_argument("--demo", action="store_true",
                     help="(default when --cli is omitted)")
    args = ap.parse_args()
    if args.cli:
        run_real(args.cli)
    else:
        run_demo()
    print("\nNext: 05_adc_sample_format_and_iq.py - what those payload "
          "bytes you just received actually ARE.")
