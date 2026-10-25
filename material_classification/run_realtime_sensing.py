r"""
run_realtime_sensing.py - receives the DCA1000 stream, records takes (--record) or classifies every new window live
with the dashboard (--background; press s to save it as a PDF).

    python run_realtime_sensing.py --cli COM4 --background data/raw/sessionB/background
"""
import argparse
import queue
import socket
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE.parent))

from material_classification.features import FeatureExtractor, background_map  # noqa: E402
from material_classification.utils.micro_doppler import (FramePreprocessor, RadarParams, load_channel_cal,  # noqa: E402
                                 padding_matches, raw_frame_to_cube)
from material_classification.utils.recording import load_take, meta_path, save_meta  # noqa: E402

PC_IP = "192.168.33.30"
DATA_PORT = 4098
HEADER_BYTES = 10
PAYLOAD_BYTES = 1456
MODEL = HERE / "models" / "material_motion_model.json"


class FrameAssembler:

    def __init__(self, frame_bytes):
        self.fb = frame_bytes
        self.buf = bytearray(frame_bytes)
        self.cur, self.filled = None, 0
        self.complete = self.incomplete = 0
        self.origin = 0

    def restart(self, byte_count):
        self.origin, self.cur, self.filled = byte_count, None, 0

    def push(self, byte_count, payload):
        out = []
        byte_count -= self.origin
        pos, end, src = byte_count, byte_count + len(payload), 0
        while pos < end:
            fidx, off = divmod(pos, self.fb)
            n = min(end - pos, self.fb - off)
            if self.cur is None or fidx > self.cur:
                if self.cur is not None and self.filled:
                    self.incomplete += 1
                self.cur, self.filled = fidx, 0
            if fidx == self.cur:
                self.buf[off:off + n] = payload[src:src + n]
                self.filled += n
                if self.filled >= self.fb:
                    out.append(bytes(self.buf))
                    self.complete += 1
                    self.cur, self.filled = fidx + 1, 0
            pos += n
            src += n
        return out


class UdpListener(threading.Thread):

    def __init__(self, bind_ip, port, frame_bytes, clock, maxq=32, record=None):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        self.sock.settimeout(0.5)
        try:
            self.sock.bind((bind_ip, port))
        except OSError as e:
            raise SystemExit(f"Cannot bind {bind_ip}:{port} -> {e}\n  Is the Ethernet adapter on "
                             "static 192.168.33.30? Is another tool holding the port?")
        self.asm = FrameAssembler(frame_bytes)
        self.clock = clock
        self.q = queue.Queue(maxsize=maxq)
        self.stop_evt = threading.Event()
        self.packets = self.lost_packets = self.dropped_frames = 0
        self._last_seq = None
        self.t_first = self.t_last = None
        self.n_run = 0
        self._t_pkt = 0.0
        self.record = record

    def run(self):
        pkt = bytearray(PAYLOAD_BYTES + HEADER_BYTES + 512)
        view = memoryview(pkt)
        while not self.stop_evt.is_set():
            try:
                n = self.sock.recv_into(pkt)
            except socket.timeout:
                continue
            except OSError:
                break
            if n <= HEADER_BYTES:
                continue
            self.packets += 1
            now = self.clock()
            if self.packets == 1 or now - self._t_pkt > 1.0:
                self.asm.restart(int.from_bytes(pkt[4:10], "little"))
                self.t_first, self.n_run = None, 0
                if self.record is not None:
                    self.record.clear()
                while not self.q.empty():
                    self.q.get_nowait()
            self._t_pkt = now
            seq = int.from_bytes(pkt[0:4], "little")
            if self._last_seq is not None and seq > self._last_seq + 1:
                self.lost_packets += seq - self._last_seq - 1
            self._last_seq = seq
            for frame in self.asm.push(int.from_bytes(pkt[4:10], "little"), view[HEADER_BYTES:n]):
                item = (frame, self.clock())
                if self.t_first is None:
                    self.t_first, self.n_run = item[1], 0
                else:
                    self.n_run += 1
                self.t_last = item[1]
                if self.record is not None:
                    self.record.append(item)
                if self.q.full():
                    self.q.get_nowait()
                    self.dropped_frames += 1
                self.q.put_nowait(item)
        self.sock.close()

    def stop(self):
        self.stop_evt.set()


def load_background(folder, params, pre):
    p = Path(folder)
    files = sorted(p.glob("*.npy")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"[bg] no background takes in {folder}")

    def frames():
        for f in files:
            for c in load_take(f):
                yield pre.process(c)
    print(f"[bg] background from {len(files)} take(s)")
    return background_map(frames())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default=str(HERE / "config" / "awr2944_material_spec.cfg"))
    ap.add_argument("--cli", default=None, help="UART CLI port (e.g. COM4): configure the radar + arm the DCA1000")
    ap.add_argument("--lvds-lanes", type=int, default=4)
    ap.add_argument("--bind-ip", default=PC_IP)
    ap.add_argument("--data-port", type=int, default=DATA_PORT)
    ap.add_argument("--background", default=None, help="folder of empty-room takes: enables live classification")
    ap.add_argument("--hop", type=int, default=10, help="classify every N new frames")
    ap.add_argument("--record", default=None, help="save all frames to this .npy (+ .meta.json)")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds of streaming")
    ap.add_argument("--countdown", type=int, default=0,
                    help="N-second countdown (with beeps) before the radar is configured; the radar then "
                         "needs about 8 s more before the 'RECORDING' marker appears")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    params, cal = RadarParams.from_cfg(args.cfg), load_channel_cal()
    print(f"[radar] {params.describe()}")
    pre = FramePreprocessor(params, max_range_m=2.6, channel_cal=cal)
    fx, model, bg, window = FeatureExtractor(params, pre), None, None, 40
    if args.background:
        if not MODEL.exists():
            sys.exit(f"[model] {MODEL.name} not found - run run_all.py first")
        from material_classification.classifier import MaterialMotionModel
        model = MaterialMotionModel.load(MODEL)
        window = model.settings.get("window", 40)
        print(f"[model] trained on {model.settings.get('trained_on')}")
        bg = load_background(args.background, params, pre)
    elif not args.record:
        print("[run] no --background: frames are received but not classified")

    t_zero = time.perf_counter()
    clock = lambda: time.perf_counter() - t_zero  # noqa: E731

    if args.countdown > 0:
        for i in range(args.countdown, 0, -1):
            print(f"\r[countdown] radar starts configuring in {i:2d} s ... get into position   ", end="", flush=True)
            if i <= 3:
                print("\a", end="", flush=True)
            time.sleep(1)
        print("\r[countdown] configuring the radar now (about 8 s) - start pouring when 'RECORDING' appears   ")

    recorded = [] if args.record else None
    listener = UdpListener(args.bind_ip, args.data_port, params.frame_bytes, clock, record=recorded)
    listener.start()
    dca_sock, cl = None, None
    if args.cli:
        sys.path.insert(0, str(REPO / "pattern_measurements"))
        import capture_lib as cl
        cl.preflight_check()
        cl.force_sensor_stop(args.cli)
        dca_sock = cl.arm_dca1000(args.lvds_lanes)
        cl.send_sensor_config(args.cli, args.cfg)
    else:
        print(f"[udp] no --cli: assuming the radar already streams to {args.bind_ip}:{args.data_port}")

    view = None
    if model is not None and not args.no_plot:
        from material_classification.plots import Dashboard
        view = Dashboard(fx, model)

    frames, last_labels, n_new, checked = deque(maxlen=window), deque(maxlen=5), 0, False
    t_run0 = clock()
    print("\a[RECORDING] ===== radar is streaming NOW =====" +
          (f"  ({args.duration:.0f} s)" if args.duration else ""), flush=True)
    next_tick = 5.0
    try:
        while True:
            if args.duration and clock() - t_run0 > args.duration:
                break
            if args.duration and clock() - t_run0 >= next_tick:
                print(f"[RECORDING] {next_tick:.0f} s of {args.duration:.0f} s", flush=True)
                next_tick += 5.0
            if view is not None and view.closed:
                break
            try:
                raw, t_arr = listener.q.get(timeout=1.0)
            except queue.Empty:
                print(f"  ... waiting for data ({listener.packets} packets so far)")
                continue
            if not checked:
                checked = True
                seen, expected = padding_matches(raw, params)
                if seen != expected:
                    print(f"[udp] WARNING: the stream {'has' if seen else 'has no'} [sample, sample, -1, 0] padding "
                          f"but adc_padded={expected} - set \"adc_padded\": {str(seen).lower()} "
                          "in config/array_conventions.json")
            if model is None:
                continue
            if frames and t_arr - frames[-1].t > 1.0:
                frames.clear()
            frames.append(pre.process(raw_frame_to_cube(raw, params), t_arr))
            n_new += 1
            if len(frames) < window or n_new < args.hop:
                continue
            n_new, t0 = 0, time.perf_counter()
            w = fx.window(list(frames), bg)
            if w.detected:
                p = model.predict({k: np.array([v]) for k, v in w.features.items()})
                label = f"{p['material'][0]}-{p['motion'][0]}"
                conf = f"motion {p['motion_conf'][0] * 100:3.0f} %, material {p['material_conf'][0] * 100:3.0f} %"
            else:
                label, conf = "none", ""
            last_labels.append(label)
            smooth = Counter(last_labels).most_common(1)[0][0]
            ms = (time.perf_counter() - t0) * 1e3
            print(f"  {label:24s} | last 5: {smooth:24s} | R={w.range_m:4.2f} m az={w.azimuth_deg:+4.0f} "
                  f"change={w.features['change_db']:4.1f} dB | {conf} | {ms:3.0f} ms")
            if view is not None:
                view.update(w, label, smooth)
    except KeyboardInterrupt:
        print("\n[run] stopping")
    finally:
        listener.stop()
        if dca_sock is not None:
            cl.stop_dca1000(dca_sock)
            cl.force_sensor_stop(args.cli)
        listener.join(timeout=2)
        print("\a[DONE] ===== recording finished - you can stop now =====", flush=True)
        print(f"[udp] packets={listener.packets} lost={listener.lost_packets} frames complete="
              f"{listener.asm.complete} incomplete={listener.asm.incomplete} dropped={listener.dropped_frames}")
        if listener.t_first is not None and listener.n_run >= 10 and listener.t_last > listener.t_first:
            fps = listener.n_run / (listener.t_last - listener.t_first)
            want = 1.0 / params.frame_period_s
            print(f"[udp] measured frame rate {fps:.1f} fps (cfg: {want:.1f} fps)"
                  + ("" if abs(fps - want) < 0.1 * want else
                     "  <-- MISMATCH: frame size or frame period in the cfg does not match the stream"))
        if recorded:
            out = Path(args.record)
            out.parent.mkdir(parents=True, exist_ok=True)
            stack = np.stack([raw_frame_to_cube(f, params) for f, _ in recorded])
            np.save(out, stack.astype(np.int16 if params.real_adc else np.complex64))
            save_meta(meta_path(out), [t for _, t in recorded], extra={"cfg": Path(args.cfg).name})
            print(f"[rec] {stack.shape} -> {out} (+ {meta_path(out).name})")


if __name__ == "__main__":
    main()
