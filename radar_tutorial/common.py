"""
common.py — shared TEST SCAFFOLDING for this tutorial series.

Nothing in this file is a radar concept. It is fake hardware: a pretend
serial port and a pretend DCA1000 UDP socket, plus a function that
synthesizes a realistic raw-ADC byte stream for a target at a chosen
range/velocity/angle. Every lesson imports from here ONLY so that its
"--demo" mode can run on a laptop with no radar attached at all — on a
train, at home, wherever. All of the actual protocol/DSP logic you're here
to learn lives inline in each lesson file, not in here.

If you skip straight to reading a real capture, you never need to open
this file. It's plumbing, not curriculum.

------------------------------------------------------------------------
WHY A "DEMO MODE" AT ALL?
------------------------------------------------------------------------
Across this whole project, almost every real bug you hit (status bytes
misread, 0-packet captures, scrambled TX labels) was invisible until data
was flowing. Being able to run every lesson's logic against a fake but
byte-accurate stand-in for the real hardware means you can read AND RUN
each concept immediately, then flip one flag (--cli COM4 --demo=false, or
just pass real args) to point the exact same code at the real EVM.
"""
import struct
import threading
import time

import numpy as np

# ---------------------------------------------------------------------
# Radar/board constants used across lessons (matches the user's real
# AWR2944P + DCA1000EVM setup, from pattern_measurement/pattern_awr2944P.cfg)
# ---------------------------------------------------------------------
DCA_IP = "192.168.33.180"
PC_IP = "192.168.33.30"
PC_DATA_PORT = 4098
CONFIG_PORT = 4096

RESET_FPGA, CONFIG_FPGA_GEN, RECORD_START, RECORD_STOP = 0x01, 0x03, 0x05, 0x06
SYSTEM_CONNECT, CONFIG_PACKET_DATA, READ_FPGA_VERSION = 0x09, 0x0B, 0x0E

CMD_NAMES = {
    RESET_FPGA: "RESET_FPGA", CONFIG_FPGA_GEN: "CONFIG_FPGA_GEN",
    RECORD_START: "RECORD_START", RECORD_STOP: "RECORD_STOP",
    SYSTEM_CONNECT: "SYSTEM_CONNECT", CONFIG_PACKET_DATA: "CONFIG_PACKET_DATA",
    READ_FPGA_VERSION: "READ_FPGA_VERSION",
}


def dca_frame(cmd, data=b""):
    """Build one OUTGOING DCA1000 command frame: header | cmd | LENGTH of
    data | data | footer. See lesson 03 for the full byte layout
    explanation."""
    return struct.pack("<HHH", 0xA55A, cmd, len(data)) + data + struct.pack("<H", 0xEEAA)


def dca_ack_frame(cmd, status=0):
    """Build a REPLY frame for a plain (no-payload) command: header | cmd
    | STATUS | footer - always exactly 8 bytes, no separate data section.
    This is a DIFFERENT shape from dca_frame() above even though both
    commands share the same 8-byte skeleton - see lesson 03. Used by
    FakeDCASocket to answer command frames the way a real DCA1000 does."""
    return struct.pack("<HHH", 0xA55A, cmd, status) + struct.pack("<H", 0xEEAA)


# =======================================================================
# FAKE SERIAL PORT  (stands in for pyserial's serial.Serial)
# =======================================================================
class FakeSerial:
    """Enough of pyserial's Serial interface for these lessons: write(),
    read(n), a context manager, reset_input_buffer(). Behaves like the
    mmWave CLI firmware: echoes "Done\\r\\n" for any line it doesn't
    specifically recognize as bad, after a configurable delay.

    reply_delay_s: how long the fake firmware "thinks" before answering.
    Set this to something large (e.g. 0.1-0.3) in lesson 04 to reproduce
    the real timing bug you hit with 30 config lines.
    """

    def __init__(self, port, baud=115200, timeout=0.05, reply_delay_s=0.01,
                 bad_lines=()):
        self.port = port
        self.timeout = timeout
        self.reply_delay_s = reply_delay_s
        self.bad_lines = set(bad_lines)
        self._pending = b""
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        self._closed = True

    def reset_input_buffer(self):
        self._pending = b""

    def write(self, data):
        text = data.decode(errors="ignore").strip()
        if not text:
            return len(data)
        # Simulate the firmware "thinking" before it replies - this is the
        # exact mechanism behind the real config-send timing bug.
        threading.Thread(target=self._reply_after_delay, args=(text,),
                          daemon=True).start()
        return len(data)

    def _reply_after_delay(self, line):
        time.sleep(self.reply_delay_s)
        if line in self.bad_lines:
            self._pending += f"{line}\r\nError: command not recognized\r\n".encode()
        else:
            self._pending += f"{line}\r\nDone\r\n".encode()

    def read(self, n):
        # REAL pyserial semantics (this is the part that's easy to get
        # wrong, including in an earlier version of this file): read(n)
        # blocks until EITHER n bytes have accumulated OR `timeout`
        # seconds have elapsed since THIS call started - NOT until "at
        # least something" arrives. A short reply that arrives in 10ms
        # does not make read(400) return in 10ms; it still waits out the
        # full timeout budget before giving up and returning the few
        # bytes it actually has. That is the exact mechanism behind the
        # real bug lesson 02 demonstrates - get this fake wrong (e.g.
        # return as soon as ANY byte is pending) and the demo shows the
        # opposite lesson from reality.
        t0 = time.time()
        while len(self._pending) < n and time.time() - t0 < self.timeout:
            time.sleep(0.002)
        chunk, self._pending = self._pending[:n], self._pending[n:]
        return chunk


# =======================================================================
# FAKE DCA1000 UDP ENDPOINT
# =======================================================================
class FakeDCASocket:
    """Stands in for a real UDP socket talking to a DCA1000. Answers the
    command frames from dca_frame() with correct 8-byte ack replies, and,
    once RECORD_START has been sent, starts "streaming" a synthetic raw
    ADC capture as properly-headered UDP packets in a background thread -
    with a configurable startup delay so you can reproduce the classic
    "receiver opened before the sensor actually started" race.
    """

    def __init__(self, adc_bytes_provider, stream_start_delay_s=0.0,
                 packet_payload=1466):
        self.adc_bytes_provider = adc_bytes_provider
        self.stream_start_delay_s = stream_start_delay_s
        self.packet_payload = packet_payload
        self._recording = False
        self._config_acks = []
        self._out_queue = []
        self._lock = threading.Lock()

    # --- the "control channel" (config port 4096) ---
    def sendto(self, frame, addr):
        cmd = struct.unpack("<H", frame[2:4])[0]
        if cmd == RECORD_START:
            self._recording = True
            threading.Thread(target=self._stream, daemon=True).start()
        elif cmd == RECORD_STOP:
            self._recording = False
        with self._lock:
            self._config_acks.append(dca_ack_frame(cmd, status=0))

    def recvfrom(self, bufsize):
        t0 = time.time()
        while time.time() - t0 < 3.0:
            with self._lock:
                if self._config_acks:
                    return self._config_acks.pop(0), (DCA_IP, CONFIG_PORT)
            time.sleep(0.005)
        import socket
        raise socket.timeout()

    # --- the "data channel" (data port 4098), used by a SEPARATE fake
    # socket instance in real code, but we fold both into one class here
    # for simplicity; recv_data() is polled by receive_raw() in lesson 04.
    def _stream(self):
        time.sleep(self.stream_start_delay_s)
        raw = self.adc_bytes_provider()
        seq = 1
        for i in range(0, len(raw), self.packet_payload):
            if not self._recording:
                break
            chunk = raw[i:i + self.packet_payload]
            header = struct.pack("<I", seq) + struct.pack("<Q", i)[:6]
            with self._lock:
                self._out_queue.append(header + chunk)
            seq += 1
            time.sleep(0.001)

    def recv_data(self, timeout=0.05):
        """Non-blocking-ish poll used by the fake data-port socket."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._out_queue:
                    return self._out_queue.pop(0)
            time.sleep(0.002)
        import socket
        raise socket.timeout()


# =======================================================================
# SYNTHETIC RAW-ADC DATA (what a real capture's .bin file contains)
# =======================================================================
def synthesize_raw_adc(p, target_range_m=1.5, target_velocity_mps=0.0,
                        target_angle_deg=0.0, snr_db=20.0, n_frames=None,
                        seed=0):
    """Build a fake raw int16 ADC stream, in the SAME byte layout a real
    AWR2944P capture produces (real/non-complex samples, see lesson 05),
    for one point target. Good enough to make range FFT / Doppler FFT /
    CFAR / angle-estimation lessons show a believable, correct peak -
    not a physically perfect radar simulation, but the bin-level math is
    right.

    p: a parsed-cfg dict as produced by parse_cfg() in lesson 01 (needs
       numAdcSamples, numRx, numTx, chirpsPerFrame, numLoops, range_axis,
       slope_MHz_us, startFreq_GHz, idleTime_us, rampEndTime_us,
       tx_positions_lambda / rx_positions_lambda if angle matters).
    """
    rng = np.random.default_rng(seed)
    n_frames = n_frames or p["numFrames"]
    N = p["numAdcSamples"]
    numRx = p["numRx"]
    numTx = p["numTx"]
    chirpsPerFrame = p["chirpsPerFrame"]
    fs = p["sampleRate_ksps"] * 1e3
    slope = p["slope_MHz_us"] * 1e12
    f0 = p["startFreq_GHz"] * 1e9
    c = 2.99792458e8

    t = np.arange(N) / fs
    beat_freq = 2 * slope * target_range_m / c
    n_loops = p["numLoops"]

    # Element positions in units of lambda at f0, used only to add a
    # realistic inter-channel phase slope for the angle lessons. Falls
    # back to a plain ULA guess if the cfg lesson didn't attach real
    # geometry.
    tx_pos = p.get("tx_positions_lambda", [(i * 2.0, 0.0) for i in range(numTx)])
    rx_pos = p.get("rx_positions_lambda", [(i * 0.5, 0.0) for i in range(numRx)])
    lam = c / f0

    cube = np.zeros((n_frames, chirpsPerFrame, numRx, N), dtype=np.complex128)
    chirp_period_s = (p["idleTime_us"] + p["rampEndTime_us"]) * 1e-6

    for fr in range(n_frames):
        for loop in range(n_loops):
            for tx in range(numTx):
                chirp_idx = loop * numTx + tx
                if chirp_idx >= chirpsPerFrame:
                    continue
                slow_time = (fr * n_loops + loop) * chirp_period_s * numTx
                doppler_phase = 2 * np.pi * (2 * target_velocity_mps / lam) * slow_time
                for rx in range(numRx):
                    dx = tx_pos[tx][0] + rx_pos[rx][0]  # virtual element x, in lambda
                    aoa_phase = 2 * np.pi * dx * np.sin(np.radians(target_angle_deg))
                    sig = np.exp(1j * (2 * np.pi * beat_freq * t + doppler_phase + aoa_phase))
                    cube[fr, chirp_idx, rx, :] += sig

    sig_power = np.mean(np.abs(cube) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = (rng.normal(0, np.sqrt(noise_power / 2), cube.shape)
             + 1j * rng.normal(0, np.sqrt(noise_power / 2), cube.shape))
    cube = cube + noise

    # Real ADC hardware only outputs the REAL part (see lesson 05) -
    # scale so it looks like a plausible 16-bit ADC code, then quantize.
    real_samples = cube.real
    real_samples = real_samples / (np.abs(real_samples).max() + 1e-12) * 8000
    int16_samples = real_samples.astype(np.int16)
    return int16_samples.tobytes()
