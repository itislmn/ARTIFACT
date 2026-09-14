# radar_config.py
import os
import time
import serial

DEFAULT_CONFIG = [
    "sensorStop",
    "flushCfg",
    "dfeDataOutputMode 1",
    "channelCfg 15 7 0",
    "adcCfg 2 1",
    "adcbufCfg -1 0 1 1 1",
    "profileCfg 0 77.0 7.0 40 62.0 0 0 52.17 1 256 10000 0 0 30",
    "chirpCfg 0 0 0 0 0 0 0 1",
    "chirpCfg 1 1 0 0 0 0 0 2",
    "chirpCfg 2 2 0 0 0 0 0 4",
    "frameCfg 0 2 16 0 100.0 1 0",
    "lvdsStreamCfg -1 0 1 0",
    "sensorStart",
]


def load_config(filepath="config/awr2944_profile.cfg"):
  if os.path.exists(filepath):
    print(f"Loading custom configuration from {filepath}...")
    with open(filepath, "r") as f:
      lines = [
          line.strip()
          for line in f
          if line.strip() and not line.startswith("%")
      ]
    return lines
  else:
    print(
        "[INFO] Config file not found. Falling back to default mmWave Studio"
        " profile."
    )
    return DEFAULT_CONFIG


def configure_radar(port="COM4", baudrate=115200, config_path=None):
  config_lines = load_config(
      config_path
  ) if config_path else DEFAULT_CONFIG

  print(f"Connecting to AWR2944P on {port} at {baudrate} baud...")
  ser = serial.Serial(port, baudrate, timeout=1.0)
  time.sleep(1)
  ser.flushInput()
  ser.flushOutput()

  for cmd in config_lines:
    ser.write((cmd + "\n").encode("utf-8"))
    time.sleep(0.05)
    response = ser.readline().decode("utf-8", errors="ignore")
    print(f"Sent: {cmd} | Response: {response.strip()}")

  print("[SUCCESS] Radar configuration completed and sensor started via UART.")
  ser.close()


if __name__ == "__main__":
  configure_radar()