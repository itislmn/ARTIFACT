r"""
LESSON 10 - Capstone: raw bytes in, range + velocity + bearing out

    python 10_end_to_end_capstone.py --demo            (synthetic target, always works)
    python 10_end_to_end_capstone.py --cli COM4 --cfg example_awr2944P.cfg --seconds 3

Every lesson from 01-08 in one compact pipeline, on either a synthetic
target (demo) or a real capture (--cli). This file leans on the SAME
functions this tutorial already built and tested lesson by lesson -
nothing new is introduced here except wiring them together in order.

------------------------------------------------------------------------
PIPELINE
------------------------------------------------------------------------
  .cfg parameters (01)  ->  UART config-send + UDP receive (02/03/04)
    ->  raw int16 bytes (05)  ->  TX-reorder-aware cube reshape (06)
    ->  range FFT (07)  ->  Doppler FFT (07)  ->  CA-CFAR (07)
    ->  angle-of-arrival on the detected bin (08)  ->  printed result

------------------------------------------------------------------------
ON THE ANGLE ESTIMATE SPECIFICALLY
------------------------------------------------------------------------
This capstone uses only the 8 virtual channels formed by TX0 and TX1
(the pair lesson 06 observed sitting on the main horizontal/azimuth
line) crossed with all 4 RX - deliberately EXCLUDING TX2/TX3, which the
same observation suggested sit on a different (elevation-offset) row.
Naively feeding elements from two different physical rows into a single
1D azimuth beamformer would corrupt the result, so excluding them here
is the technically correct choice given the genuine uncertainty lesson
06 flagged, not a shortcut. A confirmed, complete 2D (az+el) geometry
would let you use all 16 channels and estimate both angles at once.
"""
import argparse
import socket
import struct
import time

import numpy as np

from common import (DCA_IP, PC_IP, PC_DATA_PORT, CONFIG_PORT, dca_frame,
                     RESET_FPGA, CONFIG_FPGA_GEN, RECORD_START, RECORD_STOP,
                     SYSTEM_CONNECT, CONFIG_PACKET_DATA, synthesize_raw_adc)

C = 2.99792458e8


def load_cfg_numbers():
    return {
        "numAdcSamples": 656, "sampleRate_ksps": 13349.0,
        "slope_MHz_us": 70.0, "startFreq_GHz": 77.0,
        "idleTime_us": 186.0, "rampEndTime_us": 57.14,
        "numRx": 4, "numTx": 4, "numLoops": 64, "numFrames": 5,
        "chirpsPerFrame": 4 * 64, "isComplex": False,
        "chirpTx": {0: 1, 1: 4, 2: 8, 3: 2},   # this project's real firing order
        # Illustrative positions (lesson 06/08's honesty note applies):
        # TX0/TX1 on the horizontal (azimuth) line at the standard
        # gap-free spacing; TX2/TX3 offset in elevation by the EVM's
        # confirmed 0.8-lambda vertical spacing, kept OUT of the angle
        # estimate below for exactly that reason.
        "tx_positions_lambda": [(0.0, 0.0), (2.0, 0.0), (0.0, 0.8), (2.0, 0.8)],
        "rx_positions_lambda": [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0)],
    }


def compute_tx_reorder(chirp_tx, num_tx):
    perm = [None] * num_tx
    for slot, mask in chirp_tx.items():
        if mask:
            tx_id = mask.bit_length() - 1
            if tx_id < num_tx:
                perm[tx_id] = slot
    assert None not in perm, f"unresolved TX id in {chirp_tx}"
    return perm


def bytes_to_cube(raw_int16, p):
    per_frame = p["numAdcSamples"] * p["numRx"] * p["chirpsPerFrame"]
    n_frames = raw_int16.size // per_frame
    cube = raw_int16.astype(np.complex64)[:n_frames * per_frame].reshape(
        n_frames, p["chirpsPerFrame"], p["numRx"], p["numAdcSamples"])
    cube = cube.reshape(n_frames, p["numLoops"], p["numTx"], p["numRx"],
                         p["numAdcSamples"])
    perm = compute_tx_reorder(p["chirpTx"], p["numTx"])
    cube = cube[:, :, perm, :, :]
    cube = cube.reshape(n_frames, p["numLoops"], p["numTx"] * p["numRx"],
                         p["numAdcSamples"])
    return cube    # (frame, loop, channel[=tx*numRx+rx], sample)


def range_doppler_cfar(cube, p):
    window = np.hanning(cube.shape[-1])
    rfft = np.fft.fft(cube * window, axis=-1)
    n_half = p["numAdcSamples"] // 2
    r_axis = (np.arange(p["numAdcSamples"]) * (C * p["sampleRate_ksps"] * 1e3)
              / (2 * p["slope_MHz_us"] * 1e12 * p["numAdcSamples"]))[:n_half]

    # Doppler FFT MUST run on the still-COMPLEX range-FFT output (phase
    # across loops is exactly what encodes velocity) - doing it on
    # already-squared magnitude, as an earlier version of this exact
    # function did, throws the phase away first and the Doppler FFT then
    # has nothing but a near-constant (DC) signal to find, which always
    # "detects" zero velocity regardless of the truth. Doppler-FFT each
    # channel separately (per-channel phase must stay intact), THEN
    # combine magnitude non-coherently across channels/frames just to
    # find a robust peak bin. The angle estimate afterward goes back to
    # the individual channels' COMPLEX values at that bin (lesson 08).
    rfft_half = rfft[..., :n_half]                                  # (frame, loop, ch, range)
    dfft_per_channel = np.fft.fftshift(np.fft.fft(rfft_half, axis=1), axes=1)
    combined_magnitude = np.abs(dfft_per_channel).mean(axis=(0, 2))  # (doppler, range)

    peak_flat = np.argmax(combined_magnitude)
    v_bin, r_bin = np.unravel_index(peak_flat, combined_magnitude.shape)

    lam = C / (p["startFreq_GHz"] * 1e9)
    chirp_period_s = (p["idleTime_us"] + p["rampEndTime_us"]) * 1e-6
    doppler_pri_s = p["numTx"] * chirp_period_s
    v_max = lam / (4 * doppler_pri_s)
    v_axis = np.linspace(-v_max, v_max, p["numLoops"], endpoint=False)

    return {
        "range_m": r_axis[r_bin], "velocity_mps": v_axis[v_bin],
        "range_bin": r_bin, "doppler_bin_idx": v_bin,
        "rfft": rfft[..., :n_half], "n_half": n_half,
    }


def estimate_angle(rfft, r_bin, v_bin_idx, p):
    # Doppler-FFT each of the AZIMUTH-LINE channels (TX0,TX1 x RX0-3)
    # individually, at the detected range bin, then read off each one's
    # complex value at the detected Doppler bin - that's the per-element
    # snapshot the beamformer (lesson 08) needs.
    az_channels = [tx * p["numRx"] + rx for tx in (0, 1) for rx in range(p["numRx"])]
    positions = []
    for tx in (0, 1):
        tx_x, tx_y = p["tx_positions_lambda"][tx]
        for rx in range(p["numRx"]):
            rx_x, rx_y = p["rx_positions_lambda"][rx]
            positions.append(tx_x + rx_x)   # y should match (0.0) for both - same row

    snapshot = np.zeros(len(az_channels), dtype=np.complex64)
    for i, ch in enumerate(az_channels):
        chirp_dfft = np.fft.fftshift(np.fft.fft(rfft[0, :, ch, r_bin]))
        snapshot[i] = chirp_dfft[v_bin_idx]

    positions = np.array(positions)
    angles_deg = np.arange(-90, 90.05, 0.2)
    spectrum = np.zeros(len(angles_deg))
    for i, ang in enumerate(angles_deg):
        theta = np.radians(ang)
        compensating_phase = np.exp(-1j * 2 * np.pi * positions * np.sin(theta))
        spectrum[i] = np.abs(np.sum(compensating_phase * snapshot)) ** 2
    return angles_deg[int(np.argmax(spectrum))]


def run_demo():
    p = load_cfg_numbers()
    true_range, true_velocity, true_angle = 1.5, 0.2, -15.0
    print(f"Synthesizing: range={true_range} m, velocity={true_velocity} "
          f"m/s, angle={true_angle} deg\n")
    raw_bytes = synthesize_raw_adc(p, target_range_m=true_range,
                                    target_velocity_mps=true_velocity,
                                    target_angle_deg=true_angle, snr_db=18.0)
    raw = np.frombuffer(raw_bytes, dtype=np.int16)
    pipeline(raw, p, ground_truth=(true_range, true_velocity, true_angle))


def run_real(cli_port, cfg_path, seconds):
    import serial
    p = parse_cfg_for_capstone(cfg_path)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(3.0)
    sock.bind((PC_IP, CONFIG_PORT))

    def send_recv(frame):
        sock.sendto(frame, (DCA_IP, CONFIG_PORT))
        return sock.recvfrom(2048)[0]

    def do(code, data=b""):
        resp = send_recv(dca_frame(code, data))
        status = struct.unpack("<H", resp[4:6])[0]
        if status != 0:
            raise RuntimeError(f"command 0x{code:02X} failed, status={status}")

    print("Arming DCA1000...")
    do(SYSTEM_CONNECT)
    do(RESET_FPGA)
    do(CONFIG_FPGA_GEN, bytes([1, 1, 1, 2, 3, 30]))
    do(CONFIG_PACKET_DATA, struct.pack("<HHH", 1466, 25, 0))
    do(RECORD_START)

    import threading
    result = {}

    def _receive():
        data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        data_sock.settimeout(1.0)
        data_sock.bind((PC_IP, PC_DATA_PORT))
        window_s = seconds + 10
        t0 = time.time()
        buf = bytearray()
        while time.time() - t0 < window_s:
            try:
                pkt, _ = data_sock.recvfrom(65536)
            except socket.timeout:
                continue
            if len(pkt) > 10:
                buf.extend(pkt[10:])
        data_sock.close()
        result["bytes"] = bytes(buf)

    t = threading.Thread(target=_receive, daemon=True)
    t.start()
    time.sleep(0.5)
    print(f"Sending {cfg_path} over {cli_port}...")
    with serial.Serial(cli_port, 115200, timeout=0.05) as ser:
        ser.write(b"\n")
        time.sleep(0.2)
        ser.reset_input_buffer()
        for raw_line in open(cfg_path):
            line = raw_line.strip()
            if not line or line.startswith(("%", "#")):
                continue
            ser.write((line + "\n").encode())
            _quiet_read(ser)
    t.join()
    sock.sendto(dca_frame(RECORD_STOP), (DCA_IP, CONFIG_PORT))
    sock.close()

    raw = np.frombuffer(result.get("bytes", b""), dtype=np.int16)
    print(f"Captured {len(raw)} int16 samples.")
    if len(raw) == 0:
        print("Zero samples - check lesson 04 if this happens.")
        return
    pipeline(raw, p, ground_truth=None)


def _quiet_read(ser, quiet_after=0.15, max_wait=2.0):
    buf = b""
    t0 = time.time()
    last = t0
    while time.time() - t0 < max_wait:
        chunk = ser.read(256)
        if chunk:
            buf += chunk
            last = time.time()
        elif time.time() - last > quiet_after:
            break
    return buf


def parse_cfg_for_capstone(cfg_path):
    """Minimal parse for real-hardware mode - lesson 01 has the fully
    annotated version; this just extracts what pipeline() needs."""
    p = load_cfg_numbers()   # start from the known-good defaults/geometry
    p["chirpTx"] = {}
    for raw_line in open(cfg_path):
        line = raw_line.strip()
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "profileCfg":
            v = parts[1:]
            p["startFreq_GHz"], p["idleTime_us"] = float(v[1]), float(v[2])
            p["rampEndTime_us"], p["slope_MHz_us"] = float(v[4]), float(v[7])
            p["numAdcSamples"], p["sampleRate_ksps"] = int(v[9]), float(v[10])
        elif parts[0] == "channelCfg":
            p["numRx"] = bin(int(parts[1])).count("1")
        elif parts[0] == "chirpCfg":
            v = parts[1:]
            p["chirpTx"][int(v[0])] = int(v[7])
        elif parts[0] == "frameCfg":
            v = parts[1:]
            p["numLoops"], p["numFrames"] = int(v[2]), int(v[3])
    p["numTx"] = len({m for m in p["chirpTx"].values() if m}) or 1
    chirps_per_loop = len(p["chirpTx"])
    p["chirpsPerFrame"] = chirps_per_loop * p["numLoops"]
    return p


def pipeline(raw, p, ground_truth):
    cube = bytes_to_cube(raw, p)
    result = range_doppler_cfar(cube, p)
    angle = estimate_angle(result["rfft"], result["range_bin"],
                            result["doppler_bin_idx"], p)

    print("=" * 50)
    print("DETECTION")
    print("=" * 50)
    print(f"  range:    {result['range_m']:.3f} m")
    print(f"  velocity: {result['velocity_mps']:+.3f} m/s")
    print(f"  angle:    {angle:+.1f} deg (azimuth, TX0/TX1 x RX0-3 only)")

    if ground_truth:
        tr, tv, ta = ground_truth
        print(f"\nGround truth was: range={tr} m, velocity={tv:+.3f} m/s, "
              f"angle={ta:+.1f} deg")
        assert abs(result["range_m"] - tr) < 0.10
        assert abs(result["velocity_mps"] - tv) < 0.05
        assert abs(angle - ta) < 2.0
        print("All three within tolerance - the full pipeline, lessons "
              "01 through 08, checks out end to end.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=None)
    ap.add_argument("--cfg", default="example_awr2944P.cfg")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.cli:
        run_real(args.cli, args.cfg, args.seconds)
    else:
        run_demo()

    print("\nThat's the whole pipeline. From here: swap the synthetic")
    print("target for your real pattern-measurement or DOA project code,")
    print("or build lesson 09's active loop around this exact pipeline.")
