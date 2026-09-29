"""
capture_lib.py — rebuilt, with the status-byte bug fixed.

BUG THAT WAS HERE: the DCA1000's reply to every command is exactly 8 bytes:
    0xA55A (2B header) | cmd echo (2B) | status (2B) | 0xEEAA (2B footer)
The previous version read status from resp[6:8] - that's actually the
FOOTER (0xEEAA = 61098 as an int), not the status field. So every
successful command was being misreported as "ERROR status=61098". The
DCA1000 was working the entire time; only the status check was wrong.
Fixed below: status is resp[4:6].

Uses REAL ADC sampling (matches pattern_awr2944P.cfg's adcCfg 2 0) - this
is the TI-confirmed correct format for AWR2944-family raw DCA1000 capture
in TDM mode. The range FFT of this real data is still fully complex.
"""
import re
import socket
import struct
import time

import numpy as np

DCA_IP = "192.168.33.180"
PC_IP = "192.168.33.30"
PC_DATA_PORT = 4098
RESET_FPGA, CONFIG_FPGA_GEN, RECORD_START, RECORD_STOP = 0x01, 0x03, 0x05, 0x06
SYSTEM_CONNECT, CONFIG_PACKET_DATA, READ_FPGA_VERSION = 0x09, 0x0B, 0x0E


class CaptureError(Exception):
    pass


def _frame(cmd, data=b""):
    return struct.pack("<HHH", 0xA55A, cmd, len(data)) + data + struct.pack("<H", 0xEEAA)


def _log(msg):
    print(f"  [capture_lib] {msg}", flush=True)


def _parse_response_status(resp, name):
    """resp layout: 0xA55A(2) | cmd_echo(2) | status(2) | 0xEEAA(2) = 8 bytes
    total for a plain ack. READ_FPGA_VERSION and similar queries return
    extra data BEFORE the footer, so we read status right after the cmd
    echo, not by counting back from the end."""
    if len(resp) < 8:
        raise CaptureError(f"{name}: reply too short ({len(resp)} bytes) to parse")
    status = struct.unpack("<H", resp[4:6])[0]
    return status


def preflight_check():
    """Cheap, fast check BEFORE attempting a real capture: is the DCA1000
    even reachable right now? Catches network/power problems in under a
    second instead of after a full failed capture attempt."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(2.0)
    try:
        sock.bind((PC_IP, 4096))
    except OSError as e:
        sock.close()
        raise CaptureError(
            f"Cannot bind {PC_IP}:4096 -> {e}\n"
            "  Your PC's Ethernet adapter is not set to static IP 192.168.33.30.")
    sock.sendto(_frame(SYSTEM_CONNECT), (DCA_IP, 4096))
    try:
        resp, _ = sock.recvfrom(2048)
        status = _parse_response_status(resp, "SYSTEM_CONNECT")
        _log(f"preflight: DCA1000 replied {resp.hex()} -> status={status} "
             f"({'OK' if status == 0 else 'ERROR'})")
        if status != 0:
            sock.close()
            raise CaptureError(f"SYSTEM_CONNECT returned error status {status}")
    except socket.timeout:
        sock.close()
        raise CaptureError(
            "DCA1000 did not respond to SYSTEM_CONNECT.\n"
            "  Check: 5V power on the DCA1000, Ethernet cable, and that the\n"
            "  DCA1000's config switch is in SW_CONFIG mode. Press its reset\n"
            "  button and try again.")
    sock.close()


def parse_cfg(path):
    p = {"chirpTx": {}}
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith(("%", "#")):
            continue
        v = re.split(r"\s+", line)[1:]
        if line.startswith("profileCfg"):
            p["startFreq_GHz"], p["idleTime_us"] = float(v[1]), float(v[2])
            p["rampEndTime_us"], p["slope_MHz_us"] = float(v[5]), float(v[7])
            p["numAdcSamples"], p["sampleRate_ksps"] = int(v[9]), float(v[10])
        elif line.startswith("channelCfg"):
            p["numRx"] = bin(int(v[0])).count("1")
        elif line.startswith("adcCfg"):
            p["isComplex"] = int(v[1]) != 0
            p["_adcOutputFmt"] = int(v[1])
        elif line.startswith("chirpCfg"):
            p["chirpTx"][int(v[0])] = int(v[7])
        elif line.startswith("frameCfg"):
            p["chirpStartIdx"], p["chirpEndIdx"] = int(v[0]), int(v[1])
            p["numLoops"], p["numFrames"] = int(v[2]), int(v[3])
            p["framePeriod_ms"] = float(v[4])
    p["numTx"] = len({m for m in p["chirpTx"].values() if m}) or 1
    chirps_per_loop = p["chirpEndIdx"] - p["chirpStartIdx"] + 1
    p["chirpsPerFrame"] = chirps_per_loop * p["numLoops"]
    N, fs = p["numAdcSamples"], p["sampleRate_ksps"] * 1e3
    slope = p["slope_MHz_us"] * 1e12
    p["range_axis"] = np.arange(N) * (2.99792458e8 * fs) / (2 * slope * N)

    # BUG FOUND IN YOUR ACTUAL .cfg: the chirps fire in this order within
    # each loop -> chirp0=TX0(mask 1), chirp1=TX2(mask 4), chirp2=TX3(mask 8),
    # chirp3=TX1(mask 2). That is NOT ascending TX order. The old code just
    # took the chirp-firing-slot index (0,1,2,3) and called it "tx" directly,
    # so everything labeled "TX2" in earlier plots was actually physical TX3,
    # and "TX3" was actually physical TX1, etc. tx_reorder below is the
    # permutation that un-scrambles this: tx_reorder[physical_tx_id] =
    # firing_slot_index. load_complex_cube() applies it before labeling.
    perm = [None] * p["numTx"]
    for chirp_idx, mask in p["chirpTx"].items():
        if not mask:
            continue
        tx_id = mask.bit_length() - 1
        if tx_id < p["numTx"]:
            perm[tx_id] = chirp_idx
    if None in perm:
        _log(f"WARNING: could not resolve a firing slot for every TX id from "
             f"chirpCfg (chirpTx={p['chirpTx']}, numTx={p['numTx']}). Falling "
             "back to identity order - TX labels may be wrong.")
        perm = list(range(p["numTx"]))
    p["tx_reorder"] = perm
    if perm != list(range(p["numTx"])):
        _log(f"chirp firing order -> physical TX id map: {p['chirpTx']}")
        _log(f"tx_reorder = {perm} (i.e. TX{{i}} data lives in firing slot "
             "tx_reorder[i]) - correcting for this automatically.")

    if p["_adcOutputFmt"] != 0:
        _log(f"WARNING: cfg has adcCfg format={p['_adcOutputFmt']} (complex).")
        _log("This is NOT the TI-confirmed working format for AWR2944-family")
        _log("raw DCA1000 capture in TDM mode. Expect 0 packets. Use adcCfg 2 0.")
    return p


def arm_dca1000(lvds_lanes=4, verbose=True, timer=30):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(3.0)
    sock.bind((PC_IP, 4096))

    def cmd(name, code, data=b""):
        sock.sendto(_frame(code, data), (DCA_IP, 4096))
        try:
            resp, _ = sock.recvfrom(2048)
            status = _parse_response_status(resp, name)
            ok = status == 0
            if verbose:
                _log(f"{name:20s} -> {'OK' if ok else 'ERROR status=' + str(status)}"
                     f"  ({resp.hex()})")
            if not ok:
                raise CaptureError(f"{name} returned error status {status}")
            return resp
        except socket.timeout:
            raise CaptureError(f"{name} timed out - no reply from DCA1000")

    cmd("SYSTEM_CONNECT", SYSTEM_CONNECT)
    cmd("RESET_FPGA", RESET_FPGA)
    time.sleep(0.5)
    lvds_code = 1 if lvds_lanes == 4 else 2
    # raw(1), lvds_lanes, LVDS capture(1), Ethernet stream(2), 16-bit(3), timer (default 30, as before)
    cmd("CONFIG_FPGA_GEN", CONFIG_FPGA_GEN, bytes([1, lvds_code, 1, 2, 3, timer]))
    cmd("CONFIG_PACKET_DATA", CONFIG_PACKET_DATA, struct.pack("<HHH", 1466, 25, 0))
    cmd("RECORD_START", RECORD_START)
    return sock


def stop_dca1000(sock):
    try:
        sock.sendto(_frame(RECORD_STOP), (DCA_IP, 4096))
        sock.settimeout(1.0)
        try:
            sock.recvfrom(2048)
        except socket.timeout:
            pass
    finally:
        sock.close()


def force_sensor_stop(cli_port, baud=115200):
    import serial
    try:
        with serial.Serial(cli_port, baud, timeout=1) as ser:
            ser.write(b"\nsensorStop\n")
            time.sleep(0.3)
            ser.read(300)
    except serial.SerialException as e:
        _log(f"could not open {cli_port} to force-stop: {e}")


def _read_reply(ser, quiet_after=0.15, max_wait=2.0):
    """Read until the line goes quiet for `quiet_after` seconds, instead of
    always blocking for the full port timeout.

    BUG THIS REPLACES: ser.read(400) with timeout=1 does NOT return as soon
    as the device's short reply ("Done\\r\\n") arrives - pyserial keeps
    trying to fill all 400 bytes and only gives up once the FULL 1-second
    timeout elapses. With ~30 .cfg lines that turned "send the config" into
    a ~30 second operation, while the DCA1000 receive window (a few
    seconds) had already closed and stopped listening long before
    sensorStart was ever sent. That silent mismatch, not the sensor or the
    network, was why captures kept coming back with 0 packets.
    """
    buf = b""
    t0 = time.time()
    last_data_t = t0
    while time.time() - t0 < max_wait:
        chunk = ser.read(256)   # ser's own per-call timeout should be SHORT (see below)
        if chunk:
            buf += chunk
            last_data_t = time.time()
        elif time.time() - last_data_t > quiet_after:
            break
    return buf.decode(errors="ignore")


def send_sensor_config(cli_port, cfg_path, baud=115200, strict=True):
    """strict=True: ABORT immediately if any line errors, instead of
    plowing ahead with a chip left in an unknown, possibly stale state."""
    import serial
    # NOTE: port timeout is now short (0.05s per low-level read attempt) -
    # _read_reply() does its own accumulate-until-quiet loop on top of that,
    # so a line that replies in 10ms finishes in ~150ms, not ~1000ms.
    with serial.Serial(cli_port, baud, timeout=0.05) as ser:
        ser.write(b"\n")
        time.sleep(0.2)
        ser.reset_input_buffer()
        t_start = time.time()
        for raw in open(cfg_path):
            line = raw.strip()
            if not line or line.startswith(("%", "#")):
                continue
            ser.write((line + "\n").encode())
            reply = _read_reply(ser)
            ok = "Done" in reply
            _log(f"UART {line[:40]:40s} -> {'Done' if ok else reply.strip()[:60] or 'NO REPLY'}")
            if not ok and strict:
                raise CaptureError(
                    f"Sensor rejected '{line}'. Aborting rather than continuing "
                    "with an unknown chip state. Fix this line and retry.")
        _log(f"config send took {time.time() - t_start:.2f}s total "
             "(used to silently take ~30s and blow past the receive window)")


def receive_raw(seconds, out_path, min_expected_bytes=10_000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    try:
        sock.bind((PC_IP, PC_DATA_PORT))
    except OSError as e:
        raise CaptureError(f"Cannot bind data port {PC_IP}:{PC_DATA_PORT} -> {e}")

    total = 0
    n_packets = 0
    t0 = time.time()
    try:
        with open(out_path, "wb") as f:
            while time.time() - t0 < seconds:
                try:
                    pkt, addr = sock.recvfrom(65536)
                except socket.timeout:
                    continue
                n_packets += 1
                if len(pkt) > 10:
                    f.write(pkt[10:])
                    total += len(pkt) - 10
    finally:
        sock.close()

    _log(f"received {n_packets} UDP packets, {total} payload bytes")
    if n_packets == 0:
        _log("ZERO packets arrived at the network layer at all. This means the")
        _log("chip never sent anything over LVDS to the DCA1000, OR the DCA1000")
        _log("never forwarded anything over Ethernet. Most common cause: adcCfg")
        _log("is not '2 0' (real), or the sensor .cfg errored before sensorStart")
        _log("(check the UART log above for anything other than 'Done').")
    elif total < min_expected_bytes:
        _log(f"Packets arrived but total bytes ({total}) is small - capture")
        _log("duration may be too short, or numFrames in the .cfg finished early.")
    return total


def capture_once(cli_port, cfg_path, seconds, out_bin, lvds_lanes=4, strict=True):
    """The full sequence, in order, each step logged. Raises CaptureError
    with a specific, actionable message the moment anything goes wrong,
    rather than silently producing 0 bytes at the very end."""
    p = parse_cfg(cfg_path)
    force_sensor_stop(cli_port)
    dca_sock = arm_dca1000(lvds_lanes)
    try:
        import threading
        result = {}
        def _go():
            try:
                # Padding covers config-send time (~5s measured for a full
                # 30-line .cfg after the timing fix above) plus slack, so the
                # receive window comfortably outlasts sensorStart actually
                # being sent, instead of closing before it as it did before.
                result["bytes"] = receive_raw(seconds + 10, out_bin)
            except CaptureError as e:
                result["error"] = e
        t = threading.Thread(target=_go, daemon=True)
        t.start()
        time.sleep(0.5)
        send_sensor_config(cli_port, cfg_path, strict=strict)
        t.join()
        if "error" in result:
            raise result["error"]
        return result.get("bytes", 0), p
    finally:
        stop_dca1000(dca_sock)
        force_sensor_stop(cli_port)


def load_complex_cube(bin_path, p):
    """Reshape into (frames, chirps_per_tx, virtual_rx, samples), complex.
    p['isComplex'] should be False for AWR2944-family raw capture - the
    real branch below is the one that actually gets used."""
    raw = np.fromfile(bin_path, dtype=np.int16)
    if p["isComplex"]:
        quads = raw.reshape(-1, 4)
        data = (quads[:, 0:2] + 1j * quads[:, 2:4]).reshape(-1)
    else:
        data = raw.astype(np.complex64)
    per_frame = p["numAdcSamples"] * p["numRx"] * p["chirpsPerFrame"]
    n_frames = data.size // per_frame
    if n_frames == 0:
        raise CaptureError(
            f"Zero complete frames: file has {data.size} samples, need "
            f"{per_frame} per frame. Capture too short, or numTx/numRx/"
            f"numAdcSamples in the .cfg don't match what was captured.")
    cube = data[:n_frames * per_frame].reshape(
        n_frames, p["chirpsPerFrame"], p["numRx"], p["numAdcSamples"])
    if p["numTx"] > 1:
        cube = cube.reshape(n_frames, -1, p["numTx"], p["numRx"], p["numAdcSamples"])
        # Un-scramble firing order -> physical TX id (see parse_cfg). Without
        # this, axis index 0..3 here is "which chirp slot fired", not "which
        # TX antenna" - they only coincide by accident when chirpCfg happens
        # to list TX in ascending order, which this .cfg does NOT do.
        cube = cube[:, :, p["tx_reorder"], :, :]
        cube = cube.reshape(n_frames, -1, p["numTx"] * p["numRx"], p["numAdcSamples"])
    return cube
