"""
sim_stream.py - stand-ins for the radar and the camera (simulated scenes in real time), so that track_live.py runs
end to end without hardware.

    python track_live.py --take dry --simulate ball_person --seconds 30
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

import paths  # noqa: F401  (import paths of ../estimation, ../computer_vision)

ADC_SCALE = 4.0


def to_wire(cube: np.ndarray, rc) -> bytes:
    x = np.clip(np.round(cube * ADC_SCALE), -32768, 32767).astype("<i2")
    x = x.transpose(1, 2, 0).reshape(rc.n_loops, rc.n_tx, rc.n_rx, rc.n_samples)
    out = np.empty_like(x)
    out[:, list(rc.tx_reorder)] = x
    w = out.reshape(-1, 2)
    g = np.empty((w.shape[0], 4), "<i2")
    g[:, :2], g[:, 2], g[:, 3] = w, -1, 0
    return g.tobytes()


class _Rx:
    def __init__(self):
        self.frames = self.packets = self.lost = 0
        self.newest = None
        self.waveform = ""
        self.cond = threading.Condition()
        self.stop_evt = threading.Event()

        class _A:
            incomplete = 0
        self.asm = _A()


class SimStream:
    def __init__(self, scenario: str, waveform, t_origin: float, gap_s: float, log=print, seed: int = 1, queues=None):
        from hardware.simulate import Renderer, scenario as make
        objs, static, pose = make(scenario, seed)
        self.sim = Renderer(objs, static, pose, seed=seed)
        self.scenario, self.seed, self.queues = scenario, seed, queues
        self.wf, self.t_origin, self.gap_s, self.log = waveform, t_origin, gap_s, log
        self.rx = _Rx()
        self.changes = []
        self._blind_until = 0.0

    def background(self, wf, n_frames: int = 8):
        from estimation.core.frame import background_from_recording
        from estimation.core.radar import frame_from_words
        from hardware.simulate import RadarPose
        saved, self.sim.objects = self.sim.objects, []
        pose, self.sim.pose = self.sim.pose, RadarPose()
        frames = [frame_from_words(np.frombuffer(to_wire(self.sim.frame(wf, -10.0 - i * wf.frame_period_s)[0], wf.cfg), "<i2"),
                                   wf.cfg, True) for i in range(n_frames)]
        self.sim.objects, self.sim.pose = saved, pose
        return background_from_recording(n_frames, lambda i: frames[i], wf.cfg)

    def arm(self, mode: str = "single"):
        self.mode, self.fallback_reason = "single", None
        self._q, self._ctl = self.queues
        self.rx.waveform = self.wf.name
        self.th = threading.Thread(target=self._run, daemon=True, name="sim-radar")
        self.th.start()
        return self

    def _run(self):
        import queue as _queue
        while not self.rx.stop_evt.is_set():
            try:
                buf, name = self._q.get(timeout=0.2)
            except _queue.Empty:
                continue
            t = time.perf_counter()
            with self.rx.cond:
                self.rx.newest = (self.rx.frames, t, buf, name)
                self.rx.frames += 1
                self.rx.cond.notify_all()

    def newest(self, after: int, timeout: float = 0.5):
        with self.rx.cond:
            self.rx.cond.wait_for(lambda: self.rx.newest is not None and self.rx.newest[0] > after, timeout)
            return self.rx.newest if self.rx.newest is not None and self.rx.newest[0] > after else None

    def reconfigure(self, wf) -> float:
        self._ctl.put((wf.name, time.perf_counter() + self.gap_s))
        n = self.rx.frames
        self.wf = wf
        t0 = time.perf_counter()
        while self.rx.frames == n and time.perf_counter() - t0 < self.gap_s + 2:
            time.sleep(0.005)
        gap = time.perf_counter() - t0
        self.changes.append({"to": wf.name, "gap_s": round(gap, 3)})
        self.log(f"  [sim radar] waveform -> '{wf.name}': blind for {gap:.2f} s")
        return gap

    def disarm(self):
        self.rx.stop_evt.set()
        if hasattr(self, "_ctl"):
            self._ctl.put(None)


def _render_loop(scenario: str, seed: int, wf_name: str, t_origin: float, q, ctl):
    import queue as _queue
    import paths as _paths
    from methods.radar_model import waveform_library
    from hardware.simulate import Renderer, scenario as make
    lib = waveform_library(_paths.load_config())
    objs, static, pose = make(scenario, seed)
    sim = Renderer(objs, static, pose, seed=seed)
    wf, blind_until, t_next = lib[wf_name], 0.0, time.perf_counter()
    while True:
        try:
            msg = ctl.get_nowait()
            if msg is None:
                return
            wf, blind_until = lib[msg[0]], msg[1]
        except _queue.Empty:
            pass
        now = time.perf_counter()
        wait = max(t_next, blind_until) - now
        if wait > 0:
            time.sleep(min(wait, 0.01))
            continue
        x, _ = sim.frame(wf, now - t_origin)
        try:
            q.put_nowait((to_wire(x, wf.cfg), wf.name))
        except _queue.Full:
            pass
        t_next = max(t_next + wf.frame_period_s, time.perf_counter() - wf.frame_period_s)


def start_renderer(scenario: str, wf_name: str, t_origin: float, seed: int = 1):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q, ctl = ctx.Queue(maxsize=4), ctx.Queue()
    proc = ctx.Process(target=_render_loop, args=(scenario, seed, wf_name, t_origin, q, ctl), daemon=True, name="sim-render")
    proc.start()
    return proc, (q, ctl)


@dataclass
class _Frame:
    image: np.ndarray
    index: int
    t_capture: float


class SimCamera:

    def __init__(self, cfg: dict, scenario: str, t_origin: float, seed: int = 1):
        from hardware.simulate import scenario as make
        self.objs, _, self.pose = make(scenario, seed)
        self.cfg, self.t_origin = cfg, t_origin
        self.w, self.h = cfg["camera"]["width"], cfg["camera"]["height"]
        from backend.alignment import Projector
        self.proj = Projector()
        self.index, self.t_capture, self.fps_nominal = 0, [], 30.0
        self._next = time.perf_counter()

    def start(self):
        return self

    def set_exposure(self, v):
        pass

    def describe(self):
        return {"name": "simulated camera", "width": self.w, "height": self.h}

    def read(self) -> _Frame:
        now = time.perf_counter()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + 1 / 30, time.perf_counter())
        from hardware.simulate import to_radar
        t = time.perf_counter()
        img = np.full((self.h, self.w, 3), 120, np.uint8)
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        for o in self.objs:
            P = to_radar(o.points(np.array([t - self.t_origin]))[0], self.pose, np.array([t - self.t_origin]))[0]
            for k, p in enumerate(P[:1] if o.name == "ball" else P):
                r = float(np.linalg.norm(p))
                uv = self.proj.point(r, float(np.degrees(np.arcsin(p[1] / r))), float(p[2]))
                if uv is None:
                    continue
                rad = self.proj.f * (0.10 if o.name == "ball" else 0.07) / r
                m = (xx - uv[0]) ** 2 + (yy - uv[1]) ** 2 <= rad ** 2
                img[m] = (240, 240, 240) if o.name == "ball" else (200, 120, 90)
                if o.name == "ball":
                    img[m & (((yy - uv[1]) // max(rad / 4, 1)) % 2 == 0) & (np.abs(xx - uv[0]) < rad * 0.6)] = 30
        self.index += 1
        self.t_capture.append(t)
        return _Frame(img, self.index, t)

    def recent_fps(self, n: int = 30) -> float:
        t = self.t_capture[-n:]
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else float("nan")

    def stats(self):
        dt = np.diff(self.t_capture) * 1e3
        return {"frames_grabbed": len(self.t_capture), "frames_skipped": 0,
                "fps_measured": round(1e3 / float(np.median(dt)), 2) if dt.size else float("nan")}

    def stop(self):
        pass
