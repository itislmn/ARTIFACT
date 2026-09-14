# dca_streamer.py
import socket
import time

DCA_IP = "192.168.33.180"
CONTROL_PORT = 5032
PC_IP = "192.168.33.30"
DATA_PORT = 4096

# Standard DCA1000 command byte sequences
CONFIG_FPGA_CMD = bytes([0x5A, 0xA5, 0x03, 0x00, 0x03, 0x00, 0x01, 0x02, 0x01])
CONFIG_ETH_CMD = bytes([
    0x5A,
    0xA5,
    0x0E,
    0x00,
    0x05,
    0x00,
    0xC0,
    0xA8,
    0x21,
    0x1E,
    0x00,
    0x01,
    0x00,
    0x01,
    0x01,
    0x00,
    0x00,
    0x00,
])
START_REC_CMD = bytes([0x5A, 0xA5, 0x05, 0x00, 0x01, 0x00])


def arm_and_capture():
  # Control socket for arming the DCA1000 FPGA
  control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  control_sock.bind((PC_IP, 0))

  print("Arming DCA1000 FPGA via control port 5032...")
  try:
    control_sock.sendto(CONFIG_FPGA_CMD, (DCA_IP, CONTROL_PORT))
    time.sleep(0.1)
    control_sock.sendto(CONFIG_ETH_CMD, (DCA_IP, CONTROL_PORT))
    time.sleep(0.1)
    control_sock.sendto(START_REC_CMD, (DCA_IP, CONTROL_PORT))
    print("[SUCCESS] DCA1000 armed and record command sent.")
  except Exception as e:
    print(f"[ERROR] Failed to arm DCA1000: {e}")
  finally:
    control_sock.close()

  # Data socket for capturing UDP stream on port 4096
  data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  data_sock.bind((PC_IP, DATA_PORT))
  data_sock.settimeout(5.0)

  print(f"Listening for raw UDP data packets on {PC_IP}:{DATA_PORT}...")
  try:
    packet_count = 0
    total_bytes = 0
    while packet_count < 50:  # Capture first 50 packets as validation
      data, _ = data_sock.recvfrom(65535)
      packet_count += 1
      total_bytes += len(data)

    print(
        f"[SUCCESS] Captured {packet_count} packets! Total bytes received:"
        f" {total_bytes}"
    )
  except socket.timeout:
    print(
        "[WARNING] Timed out waiting for UDP packets. Ensure physical link and"
        " switches are correct."
    )
  finally:
    data_sock.close()


if __name__ == "__main__":
  arm_and_cascade_capture = arm_and_capture()