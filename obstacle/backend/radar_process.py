"""
radar_process.py - the radar and the method in their own process: arm, receive every sub-frame, run one look of
methods/scene.py, choose the next chirps, log, publish.

    from backend.radar_process import RadarProcess
"""
from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import queue
import time
from collections import deque
from pathlib import Path

import numpy as np


def channel_cal(cfg: dict):
    import paths
    for p in (paths.reference_dir(cfg["session"]) / "channel_cal.npy", paths.CALIB / "channel_cal.npy"):
        if p.exists():
            return np.load(p), p
    return None, None


class Poses:

    def __init__(self):
        self.buf = deque(maxlen=400)

    def add(self, t, p):
        self.buf.append((t, np.asarray(p, float)))

    def at(self, t: float):
        if not self.buf:
            return np.zeros(3), np.zeros(3)
        ts = np.array([b[0] for b in self.buf])
        P = np.array([b[1] for b in self.buf])
        recent = ts >= ts[-1] - 0.25
        v = (P[recent][-1] - P[recent][0]) / max(ts[recent][-1] - ts[recent][0], 1e-3) if recent.sum() > 1 else np.zeros(3)
        if t > ts[-1]:
            return P[-1] + v * min(t - ts[-1], 0.1), v
        return np.array([np.interp(t, ts, P[:, a]) for a in range(3)]), v


def first_response_time(cfg: dict) -> float:
    from methods.planner import response_time
    from methods.waveforms import frame_ms_of
    frame = frame_ms_of(cfg) / 1e3
    rc = cfg["rail"]
    return response_time(frame, frame, 1 / 30, 2 * rc["stop_mm"] / max(rc["expected_speed_mm_s"], 1e-3))


def compact(rec: dict) -> dict:
    keep = ("t", "chirp", "frame", "pose", "v_rail", "revisit_s", "lambda", "alpha", "objects", "looks", "look_reason",
            "doppler_sign", "coverage", "noise_db", "material_paused", "mode", "received", "processed", "skipped",
            "not_looked", "lost", "dropped")
    out = {k: rec[k] for k in keep if k in rec}
    out["ms"] = round(rec["ms"]["total"], 1)
    out["dets"] = [{k: d[k] for k in ("world", "label", "track", "still", "change_db", "range_m", "theta_deg")}
                   for d in rec["dets"]]
    out["tracks"] = [{k: t[k] for k in ("id", "status", "wx", "wy", "vx", "vy", "range_m", "theta_deg", "classes", "region",
                                        "updated")} for t in rec["tracks"]]
    return out


def _js(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def _worker(cfg: dict, take_dir: str, mode: str, raw: bool, q, qin, stop, port: str):
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import psutil
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS if os.name == "nt" else -10)
    except Exception:                                          # noqa: BLE001
        pass
    import paths
    from backend.radar_stream import RadarStream
    from methods.chirp_choice import pick_looks
    from methods.material.track_material import MaterialSRC, TrackMaterial
    from methods.path_reference import PathReference, ReferenceRecorder
    from methods.radar_config import frame_from_words, range_fft
    from methods.range_doppler import FrontEnd
    from methods.scene import ObstacleScene, StaticDetector
    from methods.waveforms import Library, chirp_choice

    def say(kind, **kw):
        q.put({"type": kind, **kw})

    rs, log, take = None, None, Path(take_dir)
    try:
        lib = Library(cfg)
        cal, cal_path = channel_cal(cfg)
        say("status", text=f"channel calibration: {cal_path if cal_path else 'NONE (angles uncalibrated)'}")
        ref_path = paths.reference_dir(cfg["session"]) / "path_reference.npz"
        main = lib.names[0]
        if mode == "reference":
            fe = FrontEnd(lib[main], cfg["radar"]["r_min_m"], cfg["radar"]["r_max_m"], cal, None)
            msrc = MaterialSRC(lib[main].path, extent_m=cfg["tracking"]["object_extent_m"])
            recorder = ReferenceRecorder(fe, {"session": cfg["session"], "chirp": main, "date": time.strftime("%Y-%m-%d %H:%M"),
                                              "calibration": str(cal_path)}, StaticDetector(cfg, fe, lib[main]))
        else:
            reference = PathReference.load(ref_path) if ref_path.exists() else None
            say("status", text=f"empty-path reference: {ref_path.name + ' (x ' + '%.3f-%.3f m' % reference.x_span + ')' if reference else 'NONE - every static echo counts as new (python experiment.py reference)'}")
            material = TrackMaterial(lib[main].path, reference, cfg["tracking"]["object_extent_m"])
            c = lib[main].cfg
            tone = 2000 * np.cos(2 * np.pi * 40 * np.arange(c.n_samples) / c.n_samples)
            dummy = (tone[:, None, None] + np.random.default_rng(0).normal(0, 5, (c.n_samples, c.n_loops, c.n_virt))).astype(np.float32)
            warm = ObstacleScene(cfg, lib, cal, reference, None)
            for k in range(3):
                for i, name in enumerate(lib.names):
                    warm.step(dummy, -10.0 + 0.1 * k + 0.03 * i, name, [0, 0, 0], [0, 0, 0])
            scene = ObstacleScene(cfg, lib, cal, reference, material)
        poses = Poses()
        rail = {"stop_m": cfg["rail"]["stop_mm"] / 1e3, "speed": cfg["rail"]["expected_speed_mm_s"] / 1e3, "state": "idle",
                "t_resp": first_response_time(cfg)}
        if cfg.get("_radar_file"):
            from backend.radar_stream import FileStream
            rs = FileStream(Path(cfg["_radar_file"]), lib, log=lambda s: say("status", text=s))
        else:
            rs = RadarStream(port, lib, take / "radar_raw.bin" if raw else None, tuple(lib.names),
                             log=lambda s: say("status", text=s))
        rs.arm(cfg["radar"]["mode"])
        say("armed", mode=rs.mode, chirps=rs.names, fallback=rs.fallback_reason, library=lib.describe(),
            frame_period_s=lib.frame_ms / 1e3)
        chooser = None if mode == "reference" else chirp_choice(cfg, lib, rs.names)
        if mode != "reference":
            log = gzip.open(take / "radar.jsonl.gz", "wt", encoding="utf-8")
        last, processed, not_looked, t_wait = -1, 0, 0, time.perf_counter()
        looks = [main]
        while not stop.is_set():
            while True:
                try:
                    m = qin.get_nowait()
                except queue.Empty:
                    break
                if m[0] == "pose":
                    poses.add(m[1], m[2])
                elif m[0] == "rail":
                    rail.update(m[1])
                elif m[0] == "save_reference" and mode == "reference":
                    ref = recorder.result()
                    ref_path.parent.mkdir(parents=True, exist_ok=True)
                    ref.save(ref_path)
                    say("reference_saved", path=str(ref_path), frames=len(recorder.rows), x_span=ref.x_span)
            items = rs.take(last, 0.5)
            if not items:
                if time.perf_counter() - t_wait > 3.0:
                    say("error", text="no radar frame for 3 s (DCA1000 / Ethernet / cfg?)")
                    t_wait = time.perf_counter()
                continue
            t_wait = time.perf_counter()
            pick = pick_looks(items, main, looks, lambda f: f[0], lambda f: f[3])
            not_looked += sum(1 for f in items if f[3] != main and f[3] not in looks)
            last = max(f[0] for f in items)
            for idx, t_arr, buf, name in pick:
                wf = lib[name]
                t = t_arr - wf.active_s / 2
                pose, v = poses.at(t)
                frame = frame_from_words(np.frombuffer(buf, "<i2"), wf.cfg, True)
                if mode == "reference":
                    if name == main:
                        recorder.add(fe.process(frame), pose, msrc.pre.process(frame, t).ra)
                        if np.max(np.abs(pose)) <= 5e-5 and np.linalg.norm(v) <= 1e-4:
                            recorder.add_still(range_fft(frame, lib[main].cfg).mean(axis=1), pose)
                        processed += 1
                        if processed % 10 == 0:
                            say("frame", t=t, frame=idx, pose=pose.tolist(), recorded=len(recorder.rows),
                                received=rs.rx.frames, lost=rs.rx.lost, objects=[], dets=[], tracks=[], ms=0.0)
                    continue
                rec = scene.step(frame, t, name, pose, v)
                processed += 1
                t_c = time.perf_counter()
                looks, why, table = chooser.decide(scene.tracker.tracks, scene.tracker.doppler_sign, after=name)
                rec["ms"]["choice"] = 1e3 * (time.perf_counter() - t_c)
                rec["ms"]["total"] += rec["ms"]["choice"]
                chooser.observe(name, rec["ms"]["total"] / 1e3)
                rec.update(frame=idx, t_arrival=t_arr, t_ready=time.perf_counter(), looks=looks, look_reason=why,
                           look_table=table, mode=rs.mode, received=rs.rx.frames,
                           processed=processed, skipped=rs.rx.frames - processed - not_looked, not_looked=not_looked,
                           lost=rs.rx.lost, dropped=rs.rx.dropped, packets=rs.rx.packets,
                           incomplete=rs.rx.asm.incomplete, rail_state=rail["state"])
                log.write(json.dumps(rec, default=_js) + "\n")
                msg = compact(rec)
                msg["type"] = "frame"
                q.put(msg)
        if mode == "reference" and recorder.rows:
            ref = recorder.result()
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref.save(ref_path)
            say("reference_saved", path=str(ref_path), frames=len(recorder.rows), x_span=ref.x_span)
    except BaseException as e:                                 # noqa: BLE001 - reported to the main program
        import traceback
        say("error", text=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
    finally:
        if log is not None:
            log.close()
        if rs is not None:
            rs.disarm()
            say("stopped", received=rs.rx.frames if rs.rx else 0, lost=rs.rx.lost if rs.rx else 0,
                dropped=rs.rx.dropped if rs.rx else 0, mode=rs.mode)


class RadarProcess:
    def __init__(self, cfg: dict, take_dir: Path, mode: str = "run", raw: bool = False, port: str | None = None):
        self.cfg, self.take_dir, self.mode, self.raw = cfg, Path(take_dir), mode, raw
        self.port = port or cfg["radar"]["cli_port"]
        self.last, self.armed, self.errors, self.status, self.events = None, False, [], "starting", []
        self.armed_info = {}

    def start(self) -> "RadarProcess":
        ctx = mp.get_context("spawn")
        self.q, self.qin, self.stop_evt = ctx.Queue(), ctx.Queue(), ctx.Event()
        self.proc = ctx.Process(target=_worker, args=(self.cfg, str(self.take_dir), self.mode, self.raw, self.q, self.qin,
                                                      self.stop_evt, self.port), daemon=True, name="radar")
        self.proc.start()
        return self

    def send_pose(self, t: float, pose):
        self.qin.put(("pose", float(t), [float(c) for c in pose]))

    def send_rail(self, stop_m: float, speed: float, state: str, t_resp: float):
        self.qin.put(("rail", {"stop_m": float(stop_m), "speed": float(speed), "state": state, "t_resp": float(t_resp)}))

    def save_reference(self):
        self.qin.put(("save_reference",))

    def poll(self) -> list[dict]:
        msgs = []
        while True:
            try:
                m = self.q.get_nowait()
            except queue.Empty:
                break
            msgs.append(m)
            if m["type"] == "frame":
                self.last = m
            elif m["type"] == "armed":
                self.armed, self.status, self.armed_info = True, "armed", m
                print(f"  [radar] armed: {m['mode']} {m['chirps']}" + (f" (fallback: {m['fallback']})" if m["fallback"] else ""),
                      flush=True)
            elif m["type"] == "error":
                self.errors.append(m["text"])
                self.status = "ERROR " + m["text"]
                print("  [radar] ERROR " + m["text"] + ("\n" + m.get("trace", "") if m.get("trace") else ""), flush=True)
            elif m["type"] in ("status", "stopped", "reference_saved"):
                self.events.append(m)
                self.status = m.get("text", m["type"])
                print("  [radar] " + (m.get("text") or json.dumps({k: v for k, v in m.items() if k != 'type'})), flush=True)
        return msgs

    def wait_armed(self, timeout: float = 90.0, tick=None):
        t0 = time.perf_counter()
        while not self.armed and time.perf_counter() - t0 < timeout:
            self.poll()
            if self.errors or not self.proc.is_alive():
                raise SystemExit("radar could not be armed: " + " | ".join(self.errors or ["process ended"]))
            if tick:
                tick()
            else:
                time.sleep(0.05)
        if not self.armed:
            raise SystemExit(f"radar not armed within {timeout:.0f} s")

    def stop(self):
        if hasattr(self, "stop_evt"):
            self.stop_evt.set()
            self.proc.join(timeout=10)
            self.poll()
            if self.proc.is_alive():
                self.proc.terminate()
