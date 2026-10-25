"""
session.py - one live take: camera, window, live or finite radar, video, snapshots and the frames.csv / summary.json
logs.

    from backend.session import Live
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from backend import overlay as ov  # noqa: E402
from backend import sync  # noqa: E402
from backend.camera import Camera  # noqa: E402
from backend.display import Window  # noqa: E402
from backend.imgio import VideoWriter, write_png  # noqa: E402
from backend.settings import ROOT, radar_dir, take_dir  # noqa: E402


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


class RadarJob:

    def __init__(self, session: str, name: str, args: list[str]):
        cmd = [sys.executable, "-m", "backend.radar_capture", "--session", session, "--name", name, *args]
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.lines, self.streaming = [], False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip()
            self.lines.append(line)
            print("  [radar] " + line, flush=True)
            self.streaming |= "streaming" in line

    @property
    def done(self) -> bool:
        return self.proc.poll() is not None


class Live:
    def __init__(self, title: str, cfg: dict, session: str, take: str, exposure=None, camera: str | None = None):
        self.cfg, self.session, self.take = cfg, session, take
        self.dir = take_dir(session, take)
        self.dir.mkdir(parents=True, exist_ok=True)
        if camera:
            cfg["camera"]["name"] = camera
        w, h = cfg["camera"]["width"], cfg["camera"]["height"]
        self.win = Window(f"{title}  -  {session}/{take}   (s = snapshot, r = video, q = quit)")
        self.win.show(np.full((h // 2, w, 3), 255, np.uint8))
        self.proc = psutil.Process()
        self.proc.cpu_percent()
        self.cam = Camera(cfg["camera"]).start()
        exposure = cfg["camera"]["exposure"] if exposure is None else exposure
        self.cam.set_exposure(None if str(exposure) == "auto" else int(exposure))
        self.camera = self.cam.describe()
        print("camera:", self.camera)
        self.rows, self.writer, self.show_t, self.last_index, self.radar = [], None, -1e9, -1, None

    def frame(self):
        f = self.cam.read()
        if self.writer is not None:
            self.writer.write(f.image, max(1, f.index - self.last_index) if self.last_index >= 0 else 1)
        self.last_index = f.index
        return f

    def show(self, view: np.ndarray) -> list[str]:
        t = time.perf_counter()
        if t - self.show_t >= 1 / self.cfg["display"]["max_hz"]:
            self.win.show(view, scale=view.shape[0] / self.cfg["camera"]["height"])
            self.show_t = t
            return self.win.keys()
        return []

    def due(self) -> bool:
        return time.perf_counter() - self.show_t >= 1 / self.cfg["display"]["max_hz"]

    def idle(self, label: str):
        f = self.frame()
        view = np.array(f.image[::2, ::2])
        ov.text(view, label[:90], 6, 6, ov.RADAR, 2)
        return self.show(view)

    @property
    def alive(self) -> bool:
        return self.win.alive

    def run_while(self, fn, label: str):
        out = {}

        def work():
            try:
                out["value"] = fn()
            except BaseException as e:                      # noqa: BLE001 - re-raised in the main thread
                out["error"] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()
        while th.is_alive():
            self.idle(label)
        if "error" in out:
            raise out["error"]
        return out["value"]

    def live_radar(self, port: str | None = None, raw: bool = False):
        from backend.radar_live import LiveRadar
        self.radar = LiveRadar(self.cfg, self.session, port, log_path=self.dir / "radar_live.jsonl",
                               raw_path=self.dir / "radar_raw.bin" if raw else None).start()
        self.radar.wait_armed(tick=lambda: self.idle("RADAR: ARMING (" + self.radar.status + ")"))
        return self.radar

    def radar_take(self, port: str | None, scenario: str, targets: list[str], overwrite: bool = False) -> dict:
        args = ["--cli", port or self.cfg["radar"]["cli_port"], "--scenario", scenario] + (["--overwrite"] if overwrite else [])
        for t in targets:
            args += ["--target-xy", t]
        job = RadarJob(self.session, self.take, args)
        kept, n = [], 0
        while not job.done:
            f = self.frame()
            if n % 3 == 0:
                kept = kept[-119:] + [(f.t_capture, f.index, np.array(f.image))]
            n += 1
            self.idle("RADAR TAKE: " + (job.lines[-1] if job.lines else "STARTING"))
        path = radar_dir(self.session) / f"{self.take}.bin"
        if job.proc.returncode != 0 or not path.with_suffix(".json").exists():
            raise SystemExit(f"radar take failed (exit {job.proc.returncode}): " + " | ".join(job.lines[-4:]))
        doc = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        times = doc.get("frame_time_perf_s", [])
        pairs = sync.match(times, [k[0] for k in kept])
        out = {"path": path, "doc": doc, "snapshot": None, "pairs": pairs}
        if not pairs and kept:
            out["snapshot"] = self.snapshot({"raw": kept[-1][2]}, {"radar_take": str(path.relative_to(ROOT)),
                                                                   "camera_frame": kept[-1][1], "dt_ms": None})
            print(f"radar take {self.take}: {len(times)} frames; no frame times - snapshot of the last camera frame")
        if pairs:
            ri, ci, dt = pairs[len(pairs) // 2]
            out["snapshot"] = self.snapshot({"raw": kept[ci][2]}, {
                "radar_take": str(path.relative_to(ROOT)), "radar_frame": ri, "camera_frame": kept[ci][1], "dt_ms": dt,
                "pairs_radar_camera_dt_ms": [[a, kept[b][1], d] for a, b, d in pairs]})
            print(f"radar take {self.take}: {len(times)} frames, {doc['lost_packets']} packets lost; camera frame "
                  f"{kept[ci][1]} paired with radar frame {ri} (dt {dt:+.1f} ms; all |dt| <= {max(abs(p[2]) for p in pairs):.0f} ms)")
        return out

    def snapshot(self, images: dict, meta: dict) -> Path:
        base = self.dir / f"snapshot_{stamp()}"
        rec = {**{k: f"{base.name}_{k}.png" for k in images}, "session": self.session, "take": self.take,
               "camera": self.camera, **meta}
        imgs = {k: np.array(v) for k, v in images.items()}

        def write():
            for k, v in imgs.items():
                write_png(f"{base}_{k}.png", v)
            Path(f"{base}.json").write_text(json.dumps(rec, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))

        threading.Thread(target=write, name="snapshot-writer").start()
        print(f"snapshot {base.name}: {', '.join(images)}")
        return Path(f"{base}.json")

    def video(self, on: bool):
        if on and self.writer is None:
            path = self.dir / "camera.mp4"
            if path.exists():
                path = self.dir / f"camera_{stamp()}.mp4"
            self.writer = VideoWriter(path, (self.cfg["camera"]["width"], self.cfg["camera"]["height"]), self.cam.fps_nominal)
            self.video_path = path
            print(f"video {path.relative_to(ROOT)} (every camera frame; the frame times are in frames.csv)")
        elif not on and self.writer is not None:
            self.writer.close()
            self.writer = None

    def log(self, row: dict):
        self.rows.append(row)

    def close(self, summary: dict | None = None) -> dict:
        if self.radar is not None:
            self.radar.stop()
        self.video(False)
        self.cam.stop()
        self.win.close()
        cpu = self.proc.cpu_percent()
        if self.rows:
            cols = list(dict.fromkeys(k for r in self.rows for k in r))
            with open(self.dir / "frames.csv", "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=cols)
                wr.writeheader()
                wr.writerows(self.rows)
        s = {"session": self.session, "take": self.take, "camera": self.camera, "stream": self.cam.stats(),
             "cpu_percent_of_one_core": round(cpu, 1), "ram_peak_mb": round(self.proc.memory_info().peak_wset / 2 ** 20, 1),
             **({"radar_last": self.radar.last, "radar_errors": self.radar.errors} if self.radar else {}), **(summary or {})}
        (self.dir / "summary.json").write_text(json.dumps(s, indent=2, default=str))
        st = s["stream"]
        print(f"\ncamera {st['fps_measured']} fps, {st['frames_skipped']} frames skipped; camera program CPU {cpu:.0f} % of "
              f"one core, RAM peak {s['ram_peak_mb']} MB; folder {self.dir.relative_to(ROOT)}")
        return s


def add_common_args(ap):
    ap.add_argument("--take", required=True, help="name of this take (folder recordings/<session>/<take>/)")
    ap.add_argument("--session", default=None, help="default: session in config.toml")
    ap.add_argument("--cli", default=None, help="radar CLI port (default [radar] cli_port)")
    ap.add_argument("--exposure", default=None, help="'auto' or log2 seconds, e.g. -7 (default [camera] exposure)")
    ap.add_argument("--camera", default=None, help="part of another camera's name (tests only)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this many seconds (0 = until q)")
