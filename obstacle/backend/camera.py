"""
camera.py - the camera through DirectShow without OpenCV, with fixed exposure (computer_vision's camera).

    from backend.camera import Camera
"""
from __future__ import annotations

import argparse
import threading
import time
from ctypes import POINTER, c_long
from dataclasses import dataclass

import numpy as np
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

from paths import load_config

EXPOSURE, AUTO, MANUAL = 4, 1, 2


class IAMCameraControl(IUnknown):
    _iid_ = GUID("{C6E13370-30AC-11d0-A18C-00A0C9118956}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetRange", (["in"], c_long, "Property"), (["out"], POINTER(c_long), "pMin"),
                  (["out"], POINTER(c_long), "pMax"), (["out"], POINTER(c_long), "pStep"),
                  (["out"], POINTER(c_long), "pDefault"), (["out"], POINTER(c_long), "pCaps")),
        COMMETHOD([], HRESULT, "Set", (["in"], c_long, "Property"), (["in"], c_long, "lValue"),
                  (["in"], c_long, "Flags")),
        COMMETHOD([], HRESULT, "Get", (["in"], c_long, "Property"), (["out"], POINTER(c_long), "lValue"),
                  (["out"], POINTER(c_long), "Flags")),
    ]


def camera_names() -> list[str]:
    from pygrabber.dshow_graph import FilterGraph
    return FilterGraph().get_input_devices()


def find_camera(name: str) -> tuple[int, str]:
    names = camera_names()
    hits = [i for i, n in enumerate(names) if name.lower() in n.lower()]
    if len(hits) != 1:
        problem = "no connected camera" if not hits else f"{len(hits)} cameras"
        raise RuntimeError(f'{problem} named "{name}". Connected: {names}. Plug in the camera on the radar '
                           f"or change [camera] name in config.toml (python -m backend.camera --list).")
    return hits[0], names[hits[0]]


@dataclass
class Frame:
    image: np.ndarray
    index: int
    t_capture: float


class Camera:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.error = None
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._last_returned = -1
        self.t_capture = []
        self.age = {}

    def start(self) -> "Camera":
        from pygrabber.dshow_graph import FilterGraph, FilterType
        self.index, self.name = find_camera(self.cfg["name"])
        w, h = self.cfg["width"], self.cfg["height"]
        g = FilterGraph()
        g.add_video_input_device(self.index)
        fmts = [f for f in g.get_input_device().get_formats()
                if f["media_type_str"] == "MJPG" and (f["width"], f["height"]) == (w, h)]
        if not fmts:
            raise RuntimeError(f'"{self.name}" offers no MJPG {w}x{h}: {g.get_input_device().get_formats()}')
        g.get_input_device().set_format(fmts[0]["index"])
        self.fps_nominal = round(fmts[0]["max_framerate"], 2)
        g.add_sample_grabber(self._on_frame)
        g.add_null_render()
        g.prepare_preview_graph()
        self._grabber = g.filters[FilterType.sample_grabber].callback
        self._control = g.filters[FilterType.video_input].instance.QueryInterface(IAMCameraControl)
        self._graph = g
        g.run()
        self._grabber.keep_photo = True
        with self._cond:
            if not self._cond.wait_for(lambda: self._latest is not None, timeout=5.0):
                raise RuntimeError(f'"{self.name}" started but sent no frame within 5 s')
        return self

    def stop(self):
        self._graph.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _on_frame(self, img: np.ndarray):
        t = time.perf_counter()
        self._grabber.keep_photo = True
        with self._cond:
            self._latest = Frame(img[:, :, ::-1], len(self.t_capture), t)
            self.t_capture.append(t)
            self._cond.notify_all()

    def read(self, timeout: float = 2.0) -> Frame:
        with self._cond:
            ok = self._cond.wait_for(lambda: self._latest.index > self._last_returned, timeout)
            if not ok:
                raise RuntimeError(f"no new camera frame within {timeout} s (camera unplugged?)")
            f = self._latest
            self._last_returned = f.index
        self.age[f.index] = time.perf_counter() - f.t_capture
        return f

    def exposure(self) -> dict:
        value, flags = self._control.Get(EXPOSURE)
        lo, hi, _, default, _ = self._control.GetRange(EXPOSURE)
        return {"value": value, "mode": "auto" if flags & AUTO else "manual", "range": [lo, hi], "default": default}

    def set_exposure(self, value: int | None):
        if value is None:
            self._control.Set(EXPOSURE, self.exposure()["value"], AUTO)
        else:
            self._control.Set(EXPOSURE, int(value), MANUAL)

    def describe(self) -> dict:
        return {"name": self.name, "device_index": self.index, "api": "DirectShow (pygrabber)",
                "width": int(self._latest.image.shape[1]), "height": int(self._latest.image.shape[0]),
                "format": "MJPG", "fps_nominal": self.fps_nominal, "exposure": self.exposure()}

    def recent_fps(self, n: int = 30) -> float:
        t = self.t_capture[-n:]
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else float("nan")

    def stats(self) -> dict:
        dt = np.diff(self.t_capture) * 1e3
        age = np.array(list(self.age.values())) * 1e3
        last = max(self.age) if self.age else -1

        def q(x, p):
            return round(float(np.percentile(x, p)), 2) if x.size else float("nan")

        return {"frames_grabbed": len(self.t_capture), "frames_delivered": len(self.age),
                "frames_skipped": last + 1 - len(self.age),
                "fps_measured": round(1e3 / float(np.median(dt)), 2) if dt.size else float("nan"),
                "interval_ms_median": q(dt, 50), "interval_ms_p95": q(dt, 95),
                "age_ms_median": q(age, 50), "age_ms_p95": q(age, 95)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", help="part of the camera name (overrides the config)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--exposure", help="'auto' or an integer (log2 seconds), e.g. -7")
    args = ap.parse_args()
    cfg = load_config()["camera"]
    if args.camera:
        cfg["name"] = args.camera
    if args.list:
        for i, n in enumerate(camera_names()):
            print(f"  {i}: {n}{'  <- selected' if cfg['name'].lower() in n.lower() else ''}")
        return
    import psutil
    proc = psutil.Process()
    with Camera(cfg) as cam:
        if args.exposure is not None:
            cam.set_exposure(None if args.exposure == "auto" else int(args.exposure))
        print("camera:", cam.describe())
        proc.cpu_percent()
        t_end = time.perf_counter() + args.seconds
        while time.perf_counter() < t_end:
            cam.read()
        cpu = proc.cpu_percent()
    print("timing:", cam.stats())
    print(f"CPU {cpu:.0f} % of one core, peak RAM {proc.memory_info().peak_wset / 2 ** 20:.0f} MB")


if __name__ == "__main__":
    main()
