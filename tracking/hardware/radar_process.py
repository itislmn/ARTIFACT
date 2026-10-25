"""
radar_process.py - the radar and the method in their own process: arm (sub-frames or one chirp), receive, run the
method on the newest looks, choose the chirps, log every result.

    RadarProcess(cfg, take).start()      (used by track_live.py)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import queue
import time
from pathlib import Path

import numpy as np


def compact(rec: dict) -> dict:
    keep = ("t", "waveform", "looks", "look_reason", "look_table", "alpha", "gates", "beam_gates", "lambda", "info_bits", "doppler_sign", "ego", "coverage",
            "alpha_t", "attention_bits", "switch_to", "ms", "noise_db", "births", "deleted")
    out = {k: rec[k] for k in keep if k in rec}
    out["tracks"] = rec["tracks"]
    out["dets"] = [{k: d[k] for k in ("range_m", "theta_deg", "v_mps", "snr_db", "label", "track")} for d in rec["dets"]]
    out["gains"] = {k: round(v["gain_bits"], 2) for k, v in rec["waveform_table"].items()}
    return out


def _worker(cfg: dict, take_dir: str, raw: bool, q, stop, port: str):
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import psutil
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS if os.name == "nt" else -10)
    except Exception:                                          # noqa: BLE001
        pass
    import paths  # noqa: F401  (import paths)
    from estimation.core.radar import frame_from_words
    from hardware.radar_stream import RadarStream
    from hardware.references import load_references
    from methods.cognitive_tracking import CognitiveTracking
    from methods.chirp_choice import pick_looks
    from methods.material import MaterialHead
    from methods.material_src import for_session
    from methods.radar_model import subframe_library, waveform_library

    def say(kind, **kw):
        q.put({"type": kind, **kw})

    rs = None
    take = Path(take_dir)
    try:
        sim = cfg.get("_simulate")
        subframes = cfg["radar"].get("mode") == "subframes" and bool(cfg.get("subframe")) and not sim
        lib = subframe_library(cfg) if subframes else waveform_library(cfg)
        if sim:
            from hardware.sim_stream import SimStream
            sim_rs = SimStream(sim["scenario"], lib[cfg["radar"]["start_waveform"]], sim["t_origin"],
                               cfg["cognitive"]["gap_s"], log=lambda s: say("status", text=s), queues=sim["queues"])
            refs, unusable = {n: {"bg": sim_rs.background(w)} for n, w in lib.items()}, []
        else:
            refs, unusable = load_references(cfg, lib, log=lambda s: say("status", text=s))
        for n in unusable:
            lib.pop(n)
        start = sorted(lib, key=lambda n: lib[n].subframe)[0] if subframes else cfg["radar"]["start_waveform"]
        if start not in lib:
            raise RuntimeError(f"no background for the start waveform '{start}': python radar_setup.py background")
        mat = MaterialHead(lib[start]) if sim else (for_session(cfg, lib, only=start if subframes else None)
                                                    or MaterialHead(lib[start]))
        ct = CognitiveTracking(cfg, lib, refs, start=start, doppler_sign=cfg["tracking"].get("doppler_sign", 1),
                               material=mat, subframes=subframes)
        say("status", text="material classifier: " + ("trained model loaded" if ct.material.models else
                                                       "NO model (python train_material.py)"))
        c = lib[start].cfg
        tone = 2000 * np.cos(2 * np.pi * 40 * np.arange(c.n_samples) / c.n_samples)
        dummy = (tone[:, None, None] + np.random.default_rng(0).normal(0, 5, (c.n_samples, c.n_loops, c.n_virt))).astype(np.float32)
        for k in range(3):
            for name in lib if subframes else [start]:
                ct.set_waveform(name, revisit_s=0.05 if subframes else None)
                ct.step(dummy, -10.0 + 0.1 * k)
        ct.set_waveform(start)
        ct.tracker.tracks.clear()
        if sim:
            rs = sim_rs
        else:
            rs = RadarStream(port, lib[start], take / "radar_raw.bin" if raw else None, log=lambda s: say("status", text=s),
                             library=lib if subframes else None)
        rs.arm("subframes" if subframes else "single")
        say("armed", waveform=start, frame_period_s=lib[start].frame_period_s, mode=getattr(rs, "mode", "single"),
            fallback=getattr(rs, "fallback_reason", ""), library={k: w.describe() for k, w in lib.items()})
        log = open(take / "radar_frames.jsonl", "w", encoding="utf-8")
        last, processed, t_wait = -1, 0, time.perf_counter()
        force, t_armed = cfg.get("_force_switch"), None
        if subframes:
            looks, not_looked, t_prev = [start], 0, None
            while not stop.is_set():
                items = rs.take(last, 0.5)
                if not items:
                    if time.perf_counter() - t_wait > 3.0:
                        say("error", text="no radar frame for 3 s (DCA1000 / Ethernet / cfg?)")
                        t_wait = time.perf_counter()
                    continue
                t_wait = time.perf_counter()
                pick = pick_looks(items, start, looks, lambda f: f[0], lambda f: f[3])
                not_looked += sum(1 for f in items if f[3] != start and f[3] not in looks)
                last = max(f[0] for f in items)
                for idx, t, buf, wname in pick:
                    ct.set_waveform(wname, revisit_s=(t - t_prev) if t_prev is not None else lib[wname].frame_period_s)
                    t_prev = t
                    rec = ct.step(np.asarray(frame_from_words(np.frombuffer(buf, "<i2"), lib[wname].cfg, True)), t)
                    processed += 1
                    looks = rec["looks"] or [start]
                    rec.update(frame=idx, received=rs.rx.frames, processed=processed, not_looked=not_looked,
                               skipped=rs.rx.frames - processed - not_looked, packets=rs.rx.packets, lost=rs.rx.lost,
                               incomplete=rs.rx.asm.incomplete, mode=rs.mode)
                    log.write(json.dumps(rec, default=_js) + "\n")
                    msg = compact(rec)
                    msg.update(type="frame", frame=idx, received=rs.rx.frames, skipped=rec["skipped"], lost=rs.rx.lost)
                    q.put(msg)
        while not subframes and not stop.is_set():
            item = rs.newest(last, 0.5)
            if item is None:
                if time.perf_counter() - t_wait > 3.0:
                    say("error", text="no radar frame for 3 s (DCA1000 / Ethernet / cfg?)")
                    t_wait = time.perf_counter()
                continue
            idx, t, buf, wname = item
            t_wait = time.perf_counter()
            if wname != ct.waveform:
                ct.set_waveform(wname)
            frame = frame_from_words(np.frombuffer(buf, "<i2"), lib[wname].cfg, True)
            rec = ct.step(frame, t)
            processed += 1
            t_armed = t if t_armed is None else t_armed
            if force and t - t_armed >= force[1] and force[0] in lib and force[0] != ct.waveform:
                rec["switch_to"], force = force[0], None
            rec.update(frame=idx, received=rs.rx.frames, processed=processed, skipped=rs.rx.frames - processed,
                       packets=rs.rx.packets, lost=rs.rx.lost, incomplete=rs.rx.asm.incomplete)
            log.write(json.dumps(rec, default=_js) + "\n")
            msg = compact(rec)
            msg.update(type="frame", frame=idx, received=rs.rx.frames, skipped=rs.rx.frames - processed, lost=rs.rx.lost)
            q.put(msg)
            last = idx
            if rec["switch_to"] and rec["switch_to"] in lib:
                if cfg["radar"].get("reconfigure", True):
                    try:
                        gap = rs.reconfigure(lib[rec["switch_to"]])
                        ct.set_waveform(rec["switch_to"], t, gap)
                        say("switched", to=rec["switch_to"], gap_s=gap, t=t)
                    except Exception as e:                     # noqa: BLE001 - the old chirps were restored
                        ct.controller.switched(ct.waveform, t)
                        say("error", text=f"waveform change failed ({e}); continuing with '{ct.waveform}'")
                else:
                    ct.controller.switched(ct.waveform, t)
        log.close()
    except BaseException as e:                                 # noqa: BLE001 - reported to the camera program
        say("error", text=f"{type(e).__name__}: {e}")
    finally:
        if rs is not None:
            rs.disarm()
            say("stopped", received=rs.rx.frames if rs.rx else 0, lost=rs.rx.lost if rs.rx else 0,
                changes=rs.changes)


def _js(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class RadarProcess:
    def __init__(self, cfg: dict, take_dir: Path, raw: bool = False, port: str | None = None):
        self.cfg, self.take_dir, self.raw = cfg, Path(take_dir), raw
        self.port = port or cfg["radar"]["cli_port"]
        self.last, self.armed, self.errors, self.status, self.events = None, False, [], "starting", []

    def start(self) -> "RadarProcess":
        ctx = mp.get_context("spawn")
        self.q, self.stop_evt = ctx.Queue(), ctx.Event()
        self.proc = ctx.Process(target=_worker, args=(self.cfg, str(self.take_dir), self.raw, self.q, self.stop_evt, self.port),
                                daemon=True, name="radar")
        self.proc.start()
        return self

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
                self.armed, self.status = True, "armed"
                self.armed_info = m
            elif m["type"] == "error":
                self.errors.append(m["text"])
                self.status = "ERROR " + m["text"]
                print("  [radar] ERROR " + m["text"], flush=True)
            elif m["type"] in ("status", "switched", "stopped"):
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
