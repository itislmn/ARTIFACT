"""
experiment.py - the experiment in real time: the radar and the camera on the rail find, track and classify
everything in front of them and the rail stops, waits or steps aside; every look, decision and camera frame is
recorded.

    python experiment.py run --take run_1 --read      (check | calibrate | railscale | reference | replay | compare)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np

import paths
from backend.reader_process import Stable, correct_chars
from methods import attention as att
from methods import glare
from methods import light as lt
from methods.planner import Planner, response_time
from methods.waveforms import frame_ms_of

BANNER = {"wait": "WAIT", "blocked": "BLOCKED", "avoid": "AVOID", "verify": "LOOK AGAIN", "return": "RETURN",
          "done": "END OF TRAVEL"}


class Experiment:

    def __init__(self, cfg: dict, take: Path, proj, reader=None):
        from backend.view import ObstacleView
        self.cfg, self.take, self.proj, self.reader = cfg, take, proj, reader
        self.planner = Planner(cfg)
        self.view = ObstacleView(cfg, self.planner.pf)
        self.monitor = lt.LightMonitor(cfg, proj.f) if proj is not None else None
        self.truth = cfg["text"]["truth"].upper()
        self.stable = Stable()
        self.m, self.cmd, self.t0 = None, None, None
        self.observe_s = float(cfg["planner"]["observe_s"])
        self.lib_frame_ms = float(cfg["radar"].get("frame_ms", 100.0))
        self.paused = False
        self.trail = deque(maxlen=3000)
        self.dec, self.cam_rows, self.reads = [], [], []
        self.glare, self.glare_src, self.t_glare = None, None, -1e9
        self.t_read, self.t_crop, self.n_crops, self.last_target = -1e9, -1e9, 0, None
        self.n_looks, self.ev = 0, None
        self.trails, self.wait_cause, self.light_at = {}, None, None
        self.det_hist = deque()
        self.measured = []
        (take / "crops").mkdir(parents=True, exist_ok=True)

    def on_radar(self, m: dict, pose, speed: float, stop_m: float, moving: bool, t_now: float, t_resp: float):
        self.m = m
        self.n_looks += 1
        for tr in m["tracks"]:
            if tr["status"] == "confirmed":
                self.trails.setdefault(tr["id"], []).append((m["t"], tr["wy"], tr["wx"]))
        self.trails = {i: [q for q in v if m["t"] - q[0] <= 3.0] for i, v in self.trails.items()}
        self.trails = {i: v for i, v in self.trails.items() if v}
        self.det_hist.extend((m["t"], d["track"], d["world"][1], d["world"][0]) for d in m["dets"]
                             if d.get("world") and d.get("label") in ("core", "part", "birth") and d.get("track", -1) >= 0)
        while self.det_hist and m["t"] - self.det_hist[0][0] > 2.0:
            self.det_hist.popleft()
        self.t0 = m["t"] if self.t0 is None else self.t0
        if not self.paused and not self.planner.started and m["t"] - self.t0 >= self.observe_s:
            self.planner.started = True
            self.planner.t_state = t_now
        cmd = self.planner.step(m["objects"], pose, t_now, speed, stop_m, moving, t_resp)
        self.cmd = cmd
        if cmd.state != "wait":
            self.wait_cause = None
        elif cmd.info.get("cause") is not None:
            self.wait_cause = cmd.info["cause"]
        self.trail.append(np.asarray(pose, float).copy())
        self.dec.append({"t": round(t_now, 5), "look_t": round(m["t"], 5), "frame": m.get("frame"), "chirp": m.get("chirp"),
                         "state": cmd.state, "kind": cmd.kind, "reason": cmd.reason,
                         "target": None if cmd.target is None else [round(float(v), 5) for v in cmd.target],
                         "pose": [round(float(v), 5) for v in pose], "speed": round(speed, 5), "stop_m": round(stop_m, 5),
                         "looks": m.get("looks"), "look_reason": m.get("look_reason"), **cmd.info})
        return cmd

    def cause(self):
        if self.cmd is None:
            return None
        if self.cmd.state == "wait":
            return self.cmd.info.get("cause", self.wait_cause)
        if self.cmd.state in ("approach", "blocked", "avoid", "verify", "decide"):
            return self.cmd.info.get("limit_obj")
        return None

    def on_camera(self, rgb: np.ndarray, t_cap: float, index: int, pose, compose: bool):
        cfg, proj = self.cfg, self.proj
        pose = np.asarray(pose, float)
        ev = self.ev = self.monitor.update(rgb, t_cap) if self.monitor else None
        objects = [] if self.m is None else [att.moved(o, t_cap - self.m["t"]) for o in self.m["objects"]]
        size = (rgb.shape[1], rgb.shape[0])
        visible = (lambda o: att.in_picture(proj, o, pose, cfg["text"]["centre_z_m"], size)) if proj is not None \
            else (lambda o: True)
        target, role = att.choose(objects, self.cause(), pose, visible, self.last_target)
        bar_z = cfg["alignment"]["bar_z_m"]
        rect = att.crop_rect(proj, target, pose, cfg["text"]["centre_z_m"], cfg["tracking"]["object_extent_m"]) \
            if (target is not None and proj is not None) else None
        light_on = bool(ev and ev.active)
        if light_on and ev.onset:
            self.light_at = ev.core_uv or ev.blob_uv
        if light_on and proj is not None and (ev.onset or t_cap - self.t_glare >= cfg["light"]["refit_s"]):
            self.t_glare = t_cap
            bars, hw = [], []
            for o in objects:
                c = np.asarray(o["centre"], float) - pose
                R = float(np.linalg.norm(c[:2]))
                az = float(np.degrees(np.arctan2(c[1], c[0])))
                bars.append(proj.bar(R, az, bar_z))
                hw.append(proj.f * cfg["light"]["source_size_m"] / 2 / max(R, 0.1))
            clipped = glare.luma(rgb) >= cfg["camera"]["clip_level"]
            self.glare = glare.analyse(rgb, clipped, bars, hw) if objects else None
            self.glare_src = objects[self.glare.source]["id"] if self.glare is not None else None
            if self.glare is None and self.light_at is not None:
                self.glare = _glare_at(rgb, clipped, self.light_at)
        if not light_on:
            self.glare, self.glare_src, self.light_at = None, None, None
        if self.reader is not None and proj is not None:
            for oid, rd, info in self.reader.poll():
                k = correct_chars(rd.text, self.truth) if self.truth else None
                accepted = not (info["light"] or info["flashing"])
                if accepted:
                    self.stable.add(rd.text)
                self.reads.append({"t": round(info["t"], 5), "object": oid, "role": info["role"], "text": rd.text,
                                   "correct": k, "of": len(self.truth), "light": int(info["light"]),
                                   "flashing": int(info["flashing"]), "accepted": int(accepted),
                                   "min_margin": round(float(rd.margins.min()), 3) if len(rd.margins) else "",
                                   "range_m": round(info["range_m"], 4), "ms": round(rd.ms, 1)})
            if (target is not None and not target.get("person") and target["id"] != self.glare_src
                    and not self.reader.busy and t_cap - self.t_read >= 1.0 / cfg["text"]["read_hz"]):
                uv, R = att.centre_uv(proj, target, pose, cfg["text"]["centre_z_m"])
                if uv is not None and 0 <= uv[0] < rgb.shape[1] and 0 <= uv[1] < rgb.shape[0]:
                    seen = rgb if self.glare is None else glare.clean(rgb, self.glare, glare.luma(rgb) >= cfg["camera"]["clip_level"])
                    if self.reader.submit(target["id"], R, seen, uv, proj.f, {"t": t_cap, "light": light_on, "range_m": R,
                                                                              "flashing": bool(ev and ev.flashing), "role": role}):
                        self.t_read = t_cap
        tid = target["id"] if target is not None else None
        hz = self.cfg["record"]["crops_hz"]
        if hz > 0 and rect is not None and role in ("collider", "person", "ball") and (
                tid != self.last_target or t_cap - self.t_crop >= 1.0 / hz):
            from backend.imgio import write_png
            crop = _crop(rgb, rect, self.cfg["camera"]["crop_px"])
            if crop is not None:
                self.t_crop = t_cap
                self.n_crops += 1
                write_png(self.take / "crops" / f"crop_{self.n_crops:05d}_{role}_{tid}_cam{index}.png", crop)
        self.last_target = tid
        row = {"camera_frame": index, "t": round(t_cap, 6), "radar_frame": self.m.get("frame") if self.m else "",
               "radar_age_ms": round(1e3 * (t_cap - self.m["t"]), 1) if self.m else "",
               "light": int(light_on), "onset": int(bool(ev and ev.onset)), "offset": int(bool(ev and ev.offset)),
               "flashing": int(bool(ev and ev.flashing)), "lit_px": ev.n_up * lt.SUB ** 2 if ev else "",
               "clipped_frac": round(ev.clipped_frac, 5) if ev else "", "light_source": self.glare_src if self.glare_src is not None else "",
               "attention": tid if tid is not None else "", "role": role,
               "crop_px": ";".join(f"{v:.0f}" for v in rect) if rect is not None else "",
               "state": self.cmd.state if self.cmd else "", "x": round(float(pose[0]), 5), "y": round(float(pose[1]), 5),
               "z": round(float(pose[2]), 5)}
        self.cam_rows.append(row)
        if not compose:
            return None
        shown = rgb if self.glare is None else _filtered(rgb, self.glare, cfg["camera"]["clip_level"])
        return self._picture(shown, t_cap, pose, objects, target, role, rect, ev)

    def focus(self, objects: list[dict]) -> dict | None:
        cmd, oid = self.cmd, self.cause()
        o = next((q for q in objects if q["id"] == oid), None)
        if cmd is None or o is None:
            return None
        what = "person" if o.get("person") else ("mover" if o["kind"] == "moving" else "object")
        verb = {"wait": "waiting for", "blocked": "stopped by", "avoid": "stepping around", "verify": "looking again at",
                "decide": "looking again at", "approach": "stopping before"}[cmd.state]
        if cmd.state == "wait" and cmd.info.get("cause") is None:
            verb = "keeping clear of"
        c = np.asarray(o["centre"], float)
        return {"id": oid, "lat": float(c[1]), "fwd": float(c[0]), "text": f"{verb} {what} {oid}"}

    def _picture(self, rgb, t_cap, pose, objects, target, role, rect, ev):
        from backend.view import STATE_COL, style
        cfg, proj = self.cfg, self.proj
        m, cmd = self.m, self.cmd
        fo = self.focus(objects)
        dist = lambda c: float(np.hypot(c[0] - pose[0], c[1] - pose[1]))
        size = (rgb.shape[1], rgb.shape[0])
        by_id = {o["id"]: o for o in objects}
        dt = t_cap - m["t"] if m else 0.0
        boxes, tracks, live = [], [], {}
        for tr in sorted(m["tracks"] if m else [], key=lambda q: dist((q["wx"], q["wy"]))):
            confirmed = tr["status"] == "confirmed"
            o = by_id.get(tr["id"])
            st = style(o) if o is not None else style(tr)
            mover = o is not None and o["kind"] == "moving"
            calibrated = bool(tr["region"].get("calibrated")) and tr.get("updated", True)
            trail = [(y, x) for _, y, x in self.trails.get(tr["id"], [])]
            tracks.append(dict(id=tr["id"], lat=tr["wy"], fwd=tr["wx"], confirmed=confirmed, dist=dist((tr["wx"], tr["wy"])),
                               trail=trail if mover and confirmed and len(trail) > 1 else None,
                               poly=_sector(tr, pose) if confirmed and calibrated else None, **st))
            if confirmed and calibrated:
                live[tr["id"]] = tr
        for o in sorted((q for q in objects if not q.get("known")), key=lambda q: dist(q["centre"])):
            if proj is None or not att.in_picture(proj, o, pose, cfg["text"]["centre_z_m"], size):
                continue
            r = None
            if o["kind"] == "moving" and o["id"] in live:
                z_lo, z_hi = (o["lo"][2] - pose[2], o["hi"][2] - pose[2]) if np.isfinite(o["lo"][2]) and np.isfinite(o["hi"][2]) \
                    else cfg["alignment"]["bar_z_m"]
                r = att.region_rect(proj, live[o["id"]], z_lo, z_hi, dt)
                if r is not None and (r[2] - r[0] > size[0] or r[3] - r[1] > size[1]):
                    r = None
            r = r or att.rect(proj, o, pose, cfg["alignment"]["bar_z_m"])
            if r is not None and r[2] - r[0] <= size[0]:
                boxes.append(dict(rect=r, thick=target is not None and o["id"] == target["id"],
                                  focus=fo is not None and o["id"] == fo["id"], **style(o)))
        measured = [q for q in self.measured if q.active(t_cap)]
        cam_measured = []
        for q in [q for q in measured if q.kind != "lamp"] if proj is not None else []:
            o = {"lo": q.lo, "hi": q.hi, "centre": (q.lo + q.hi) / 2, "z_known": True}
            r = att.rect(proj, o, pose, cfg["alignment"]["bar_z_m"]) \
                if att.in_picture(proj, o, pose, cfg["text"]["centre_z_m"], size) else None
            if r is not None:
                cam_measured.append((q.label, r))
        light_uv = None
        if ev is not None and ev.active:
            light_uv = self.glare.centre_uv if self.glare is not None else (ev.core_uv or ev.blob_uv)
        state = cmd.state if cmd else ("observe" if m else "waiting for the radar")
        t = (m["t"] - self.t0) if m else 0.0
        reason = cmd.reason if cmd else ""
        if cmd and not self.planner.started:
            state, reason = "observe", f"the radar looks before the rail moves ({self.observe_s:.0f} s)"
        if self.paused:
            state, reason = "pause", "stopped by the operator (g = go on)"
        looks = " + ".join(m.get("looks") or [m.get("chirp", "")]) if m else "-"
        light = "no light"
        if ev is not None and ev.active:
            light = ("FLASHING light" if ev.flashing else "light on") + " - readings filtered"
        read = self.stable.text
        k = correct_chars(read, self.truth) if (self.truth and read) else None
        rail = f"look {m['ms'] if m else 0:.0f} ms"
        lines = [f"$t$ = {t:.1f} s     {state}: {reason}",
                 f"radar: {len(m['dets'])} detections at FDR $α$ = {cfg['detection']['alpha']:.2f}, "
                 f"{sum(q["confirmed"] for q in tracks)} tracks     chirps: {looks}" if m else "radar: -",
                 f"camera: {light}     attention: {role or '-'} {target['id'] if target else ''}     "
                 + (f"read: {read}" + (f" ({k}/{len(self.truth)})" if k is not None else "") if read else "read: -"),
                 f"rail: $x$ = {1e3 * pose[0]:.0f} mm, $y$ = {1e3 * pose[1]:+.0f} mm, $z$ = {1e3 * pose[2]:+.0f} mm     {rail}"]
        banner, col = None, None
        if self.paused:
            banner, col = "PAUSED", STATE_COL["idle"]
        elif cmd and cmd.state in ("wait", "blocked"):
            banner, col = f"{BANNER[cmd.state]} - {cmd.reason}", STATE_COL[cmd.state]
        elif ev is not None and ev.active:
            banner, col = ("FLASHING LIGHT" if ev.flashing else "LIGHT") + " - camera readings filtered", (230, 159, 0)
        elif cmd and cmd.state in BANNER:
            banner, col = f"{BANNER[cmd.state]} - {cmd.reason}", STATE_COL.get(cmd.state)
        plan = dict(cmd.info) if cmd else {}
        plan["target"] = None if (cmd is None or cmd.target is None) else list(cmd.target)
        scene = {"pose": pose, "objects": m["objects"] if m else [], "dets": m["dets"] if m else [], "plan": plan,
                 "trail": list(self.trail)[-600:], "tracks": tracks, "focus": fo,
                 "det_hist": [(y, x, tid) for _, tid, y, x in self.det_hist], "measured": [(q.lo, q.hi) for q in measured]}
        light_tag = ("flashing light detected" if ev.flashing else "light detected") if ev is not None and ev.active else None
        return self.view.frame(rgb, boxes, rect, light_uv, scene, lines, banner, col,
                               crop_title=f"attention: {role} {target['id']}" if target else "attention: -", light_tag=light_tag,
                               light_r=_light_r(self.glare) if self.glare is not None else 0.0, measured=cam_measured)

    def save(self, rail=None, extra: dict | None = None):
        take = self.take
        with gzip.open(take / "decisions.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in self.dec:
                f.write(json.dumps(r, default=_js) + "\n")
        _csv(take / "camera.csv", self.cam_rows)
        _csv(take / "readings.csv", self.reads)
        (take / "light.json").write_text(json.dumps(self.monitor.events if self.monitor else [], indent=1, default=_js))
        if rail is not None:
            _csv(take / "rail.csv", [{"t": round(r[0], 6), "x": round(r[1], 6), "y": round(r[2], 6), "z": round(r[3], 6),
                                      "moving": r[4]} for r in rail.log])
            (take / "rail.json").write_text(json.dumps({"commands": rail.commands, "stops": rail.stops,
                                                         "speed_mps": rail.speed_mps, "stop_m": rail.stop_m,
                                                         "limits_m": rail.limits.tolist(), "params": rail.params, "error": rail.error,
                                                         "dry": rail.dry}, indent=1, default=_js))
        acc = [r for r in self.reads if r["correct"] is not None]
        summary = {"take": take.name, "session": self.cfg["session"], "looks": self.n_looks,
                   "planner_events": self.planner.events, "final_state": self.planner.state,
                   "readings": {"all": len(acc), "accepted": sum(r["accepted"] for r in acc),
                                "chars_correct_accepted": sum(r["correct"] for r in acc if r["accepted"]),
                                "chars_correct_filtered": sum(r["correct"] for r in acc if not r["accepted"]),
                                "stable": self.stable.text},
                   "light_events": len(self.monitor.events) if self.monitor else 0, "crops": self.n_crops, **(extra or {})}
        (take / "summary.json").write_text(json.dumps(summary, indent=1, default=_js))
        scene = take / "scene.json"
        if not scene.exists():
            scene.write_text(json.dumps(SCENE_TEMPLATE, indent=1))
        return summary


SCENE_TEMPLATE = {
    "measured": False,
    "_how": "Ground truth of this run, measured with a tape after the run; the method never reads this file. World frame: "
            "antenna centre with the rail at home, x forward along the rail, y right, z up (metres); boxes are the objects' "
            "extents. List every object that was not there during the reference run (the ball's stand too). When things "
            "happened (light, flashing, a person in view, an obstacle placed) is marked on the camera video with "
            "python annotate.py <session>/<take>; obstacle i of this list is key i there. Delete the examples you do not "
            "need, then set measured to true.",
    "obstacles": [{"label": "metal box", "material": "metal", "motion": "static",
                   "x_m": [0.90, 1.10], "y_m": [-0.10, 0.10], "z_m": [-0.38, -0.10]}],
    "ball": {"label": "ball with text", "material": "dielectric", "motion": "rigid", "centre_xyz_m": [1.60, 0.30, 0.00],
             "diameter_m": 0.20, "swing_m": 0.25},
    "persons": [{"label": "person", "material": "waterlike", "motion": "articulated",
                 "walk_zone_x_m": [0.6, 2.4], "walk_zone_y_m": [-1.0, 1.0]}],
    "light": {"label": "flashlight", "x_m": 1.5, "y_m": -0.4, "z_m": 0.2},
    "text": "ARTIFACT",
    "notes": ""}


def _crop(rgb, rect, size):
    h, w = rgb.shape[:2]
    u0, v0, u1, v1 = rect
    side = max(u1 - u0, v1 - v0, 8)
    cu, cv_ = (u0 + u1) / 2, (v0 + v1) / 2
    xs = np.clip((cu - side / 2 + np.arange(size) * side / size).astype(int), 0, w - 1)
    ys = np.clip((cv_ - side / 2 + np.arange(size) * side / size).astype(int), 0, h - 1)
    if xs.max() == xs.min() or ys.max() == ys.min():
        return None
    return rgb[ys][:, xs]


def _cut(rgb, clip_level: int):
    import cv2
    cut = cv2.dilate((cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) >= clip_level).astype(np.uint8), np.ones((7, 7), np.uint8))
    out = rgb.copy()
    out[cut > 0] = 255
    return out


def _light_r(g) -> int:
    return int(max(g.core_px, 8) + 20)


def _filtered(rgb, g, clip_level: int):
    from dataclasses import replace
    out = _cut(rgb, clip_level)
    r, (cu, cv_) = _light_r(g), g.centre_uv
    u0, v0 = max(int(cu) - r, 0), max(int(cv_) - r, 0)
    u1, v1 = min(int(cu) + r + 1, rgb.shape[1]), min(int(cv_) + r + 1, rgb.shape[0])
    if u1 <= u0 or v1 <= v0:
        return out
    crop = rgb[v0:v1, u0:u1]
    local = glare.clean(crop, replace(g, centre_uv=(cu - u0, cv_ - v0), cache={}), glare.luma(crop) >= clip_level)
    yy, xx = np.ogrid[v0:v1, u0:u1]
    disc = (xx - cu) ** 2 + (yy - cv_) ** 2 <= r * r
    out[v0:v1, u0:u1][disc] = local[disc]
    return out


def _glare_at(rgb, clipped, uv):
    from scipy import ndimage
    if not clipped.any():
        return None
    lab, _ = ndimage.label(clipped)
    ys, xs = np.nonzero(clipped)
    i = int(np.argmin((xs - uv[0]) ** 2 + (ys - uv[1]) ** 2))
    blob = lab == lab[ys[i], xs[i]]
    by, bx = np.nonzero(blob)
    return glare.fit_at(rgb, clipped, -1, (float(bx.mean()), float(by.mean())), float(np.sqrt(blob.sum() / np.pi)),
                        np.zeros_like(clipped))


def _sector(tr: dict, pose) -> list | None:
    lo, hi = tr["region"]["lo"], tr["region"]["hi"]
    if lo is None or hi is None or not (np.all(np.isfinite(lo[:2])) and np.all(np.isfinite(hi[:2]))):
        return None
    th = np.linspace(lo[1], hi[1], 9)
    rs = np.r_[np.full(9, max(lo[0], 0.0)), np.full(9, hi[0])]
    ts = np.r_[th, th[::-1]]
    return [(float(pose[1] + r * np.sin(t)), float(pose[0] + r * np.cos(t))) for r, t in zip(rs, ts)]


def _csv(path: Path, rows: list[dict]):
    cols = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


def _js(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class NoWindow:
    alive = True

    def show(self, *a, **k):
        pass

    def keys(self):
        return []

    def close(self):
        pass


class GreyCamera:
    fps_nominal = 30.0

    def __init__(self, cfg):
        from backend.camera import Frame
        self.Frame, self.i, self.img = Frame, 0, np.full((cfg["height"], cfg["width"], 3), 200, np.uint8)

    def read(self):
        time.sleep(1 / 30)
        self.i += 1
        return self.Frame(self.img, self.i, time.perf_counter())

    def set_exposure(self, v):
        pass

    def stop(self):
        pass

    def stats(self):
        return {"fps_measured": 30.0, "frames_skipped": 0}

    def describe(self):
        return {"name": "grey (no camera)"}


def run(cfg: dict, args) -> None:
    from backend.alignment import Projector
    from backend.imgio import VideoWriter, write_png
    from backend.radar_process import RadarProcess
    from backend.rail import Rail
    take = paths.session_dir(cfg["session"]) / args.take
    if take.exists() and any(take.iterdir()) and not args.overwrite:
        raise SystemExit(f"{take} exists - another --take name or --overwrite")
    take.mkdir(parents=True, exist_ok=True)
    shutil.copy(paths.CONFIG, take / "config.toml")
    proj = Projector()
    if args.no_camera:
        cam = GreyCamera(cfg["camera"])
    else:
        from backend.camera import Camera
        cam = Camera(cfg["camera"]).start()
    cam.set_exposure(None if str(cfg["camera"]["exposure"]) == "auto" else int(cfg["camera"]["exposure"]))
    if str(cfg["camera"]["exposure"]) == "auto":
        print("WARNING: automatic exposure - the light detection needs a fixed exposure ([camera] exposure)")
    rail = Rail(cfg, dry="virtual" if args.virtual_rail else ("frozen" if args.no_rail else "")).start()
    if np.linalg.norm(rail.pose) > 0.002:
        print(f"rail at {np.round(1e3 * rail.pose, 1)} mm: driving home first")
        rail.goto(np.zeros(3))
        t0 = time.perf_counter()
        while (rail.moving or np.linalg.norm(rail.pose) > 0.001) and time.perf_counter() - t0 < 300 and not rail.error:
            rail.feed()
            time.sleep(0.05)
    radar = RadarProcess(cfg, take, "run", raw=args.raw).start()
    rail.listeners.append(radar.send_pose)
    reader = None
    if args.read:
        from backend.reader_process import LiveReader
        reader = LiveReader(cfg, cfg["text"]["object_radius_m"]).start()
    exp = Experiment(cfg, take, proj, reader)
    if args.headless:
        win = NoWindow()
    else:
        from backend.display import Window
        win = Window(f"COGNITIVE RADAR-VISION PLATFORM - {cfg['session']}/{args.take}   (SPACE stop, g go, s snapshot, q quit)")
    paths.VIDEOS.mkdir(parents=True, exist_ok=True)
    video_path = paths.VIDEOS / f"{cfg['session']}_{args.take}.mp4"
    writer = VideoWriter(video_path, (1280, 720), cfg["record"]["video_fps"], crf=cfg["record"]["video_crf"])
    cam_writer = None
    if not args.no_camera_video and not args.no_camera:
        cam_writer = VideoWriter(take / "camera.mp4", (cfg["camera"]["width"], cfg["camera"]["height"]), cam.fps_nominal, crf=23)
    t_video, t_show, t_console, last_index = -1e9, -1e9, -1e9, -1
    t_start = time.perf_counter()
    frame_s = frame_ms_of(cfg) / 1e3
    proc = deque([frame_s], maxlen=50)

    def t_resp() -> float:
        stops = [s["time_s"] for s in rail.stops]
        stop_s = float(np.median(stops)) if stops else 2 * rail.stop_m / max(rail.speed_mps, 1e-3)
        fps = cam.recent_fps() if hasattr(cam, "recent_fps") else 30.0
        return response_time(frame_s, float(np.median(proc)), 1.0 / max(fps, 1.0), stop_s)
    try:
        def idle():
            f = cam.read()
            win.show(exp.on_camera(f.image, f.t_capture, f.index, rail.pose.copy(), True) if exp.m else
                     np.full((720, 1280, 3), 255, np.uint8))
            rail.feed()
        radar.wait_armed(tick=idle)
        t_last_radar = time.perf_counter()
        while win.alive and (args.seconds <= 0 or time.perf_counter() - t_start < args.seconds):
            f = cam.read()
            if cam_writer is not None:
                cam_writer.write(f.image, max(1, f.index - last_index) if last_index >= 0 else 1)
            last_index = f.index
            for msg in radar.poll():
                if msg["type"] != "frame":
                    continue
                t_last_radar = time.perf_counter()
                proc.append(msg["ms"] / 1e3)
                tr = t_resp()
                cmd = exp.on_radar(msg, rail.pose.copy(), rail.speed_mps, rail.stop_m, rail.moving, t_last_radar, tr)
                if cmd.kind == "goto":
                    rail.goto(cmd.target)
                elif cmd.kind == "stop":
                    rail.stop()
                rail.feed()
                radar.send_rail(rail.stop_m, rail.speed_mps, exp.planner.state, tr)
            if time.perf_counter() - t_last_radar > 0.5 and rail.moving:
                rail.stop()
                print("  [run] no radar look for 0.5 s - rail stopped")
            if rail.error and not exp.paused:
                exp.paused, exp.planner.started = True, False
                print(f"  [run] RAIL ERROR ({rail.error}) - planner paused; check the controller, then restart the run")
            now = time.perf_counter()
            compose = now - t_video >= 1.0 / cfg["record"]["video_fps"] or now - t_show >= 1.0 / cfg["display"]["max_hz"]
            img = exp.on_camera(f.image, f.t_capture, f.index, rail.pose.copy(), compose)
            keys = []
            if img is not None:
                if now - t_video >= 1.0 / cfg["record"]["video_fps"]:
                    t_video = now
                    writer.write(img)
                if now - t_show >= 1.0 / cfg["display"]["max_hz"]:
                    t_show = now
                    win.show(img)
                    keys = win.keys()
            if now - t_console >= 5.0:
                t_console = now
                m = exp.m
                print(f"  [run] {exp.cmd.state if exp.cmd else '-'}: {exp.cmd.reason if exp.cmd else ''} | rail x "
                      f"{1e3 * rail.pose[0]:.0f} mm, {1e3 * rail.speed_mps:.1f} mm/s | radar {m.get('mode') if m else '-'} "
                      f"looks {m.get('looks') if m else '-'}, {m['ms'] if m else 0:.0f} ms, lost {m.get('lost') if m else 0}, "
                      f"skipped {m.get('skipped') if m else 0} | camera {cam.recent_fps() if hasattr(cam, 'recent_fps') else 30:.0f} fps")
            if " " in keys or "space" in keys:
                exp.paused, exp.planner.started = True, False
                rail.stop()
                print("  [run] PAUSED by the operator (g = go on)")
            if "g" in keys and exp.paused:
                exp.paused, exp.planner.started = False, True
                print("  [run] go on")
            if "s" in keys:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                write_png(take / f"snapshot_{stamp}_camera.png", f.image)
                if img is not None:
                    write_png(take / f"snapshot_{stamp}_view.png", img)
                print(f"  [run] snapshot {stamp}")
            if "q" in keys or "escape" in keys:
                break
    finally:
        rail.stop()
        time.sleep(0.3)
        rail.close()
        radar.stop()
        if reader is not None:
            reader.stop()
        writer.close()
        if cam_writer is not None:
            cam_writer.close()
        cam.stop()
        win.close()
        s = exp.save(rail, {"radar": radar.armed_info, "radar_errors": radar.errors,
                            "radar_events": [e for e in radar.events if e["type"] != "status"],
                            "camera": cam.stats(), "live_video": str(video_path), "rail_error": rail.error})
        print(f"\nrun saved: {take}\nlive video: {video_path}\nfinal state: {s['final_state']}; planner events: "
              f"{len(s['planner_events'])}; looks {s['looks']}; readings {s['readings']}")
        print(f"NEXT: measure the scene into {take / 'scene.json'}, mark the events: python annotate.py {cfg['session']}/{args.take}, "
              "add the run to dataset.TAKES, then: python run_all.py")


class SyncReader:
    busy = False

    def __init__(self, cfg):
        self.cfg, self.out = cfg, []

    def submit(self, oid, range_m, rgb, uv, f_px, info=None):
        from methods.reader import metricsight
        from methods.text_reading import read_object
        self.out.append((oid, read_object(metricsight, rgb, uv, range_m, f_px, self.cfg, self.cfg["text"]["object_radius_m"]),
                         info or {}))
        return True

    def poll(self):
        out, self.out = self.out, []
        return out

    def stop(self):
        pass


def replay(cfg: dict, args) -> None:
    from backend.alignment import Projector
    from backend.imgio import VideoWriter, read_video
    from methods.material.track_material import TrackMaterial
    from methods.path_reference import PathReference
    from methods.radar_config import frame_from_words
    from methods.scene import ObstacleScene
    from methods.waveforms import Library, chirp_choice
    from backend.radar_process import channel_cal, compact
    src = Path(args.folder).resolve()
    out = paths.session_dir("replay") / (args.name or src.name)
    out.mkdir(parents=True, exist_ok=True)
    lib = Library(cfg)
    cal, cal_path = channel_cal(cfg)
    reference = None
    if args.reference_from_background:
        reference = reference_from_background(cfg, lib, cal, Path(args.reference_from_background))
    elif (src.parent / "reference" / "path_reference.npz").exists():
        reference = PathReference.load(src.parent / "reference" / "path_reference.npz")
    main = lib.names[0]
    method = getattr(args, "method", "final") or "final"
    if method == "baseline":
        from methods.baseline import BaselineScene
        scene, args.policy = BaselineScene(cfg, lib, cal, reference), "search"
    else:
        material = TrackMaterial(lib[main].path, reference, cfg["tracking"]["object_extent_m"])
        scene = ObstacleScene(cfg, lib, cal, reference, material, "infinirex" if method == "final" else method)
    index = [json.loads(line) for line in open(src / "radar_raw.jsonl")]
    words = np.memmap(src / "radar_raw.bin", dtype="<i2", mode="r")
    full = lib[main].cfg.wire_bytes_per_frame(True) // 2
    padded = words.size >= full * len(index) - 1
    per = full if padded else full // 2
    names = sorted({r.get("chirp", r.get("waveform", main)) for r in index}, key=lambda n: lib.names.index(n) if n in lib.names else 0)
    names = [n if n in lib.names else main for n in names]
    chooser = chirp_choice(cfg, lib, names, getattr(args, "policy", None))
    cost = getattr(args, "cost_s", None)
    if cost is None and (src / "radar.jsonl.gz").exists():
        live = {}
        for line in gzip.open(src / "radar.jsonl.gz", "rt", encoding="utf-8"):
            x = json.loads(line)
            live.setdefault(x.get("chirp"), []).append(x["ms"]["total"] / 1e3)
        cost = {k: float(np.median(v)) for k, v in live.items()}
    if cost:
        cost = {n: cost.get(n, float(np.median(list(cost.values())))) for n in lib.names}
    for n, v in (cost or {}).items():
        chooser.observe(n, v)
    rail_log = None
    if (src / "rail.csv").exists():
        R = np.genfromtxt(src / "rail.csv", delimiter=",", names=True)
        rail_log = (R["t"], np.c_[R["x"], R["y"], R["z"]])

    def pose_at(t):
        if rail_log is None:
            return np.zeros(3), np.zeros(3)
        ts, P = rail_log
        p = np.array([np.interp(t, ts, P[:, a]) for a in range(3)])
        p2 = np.array([np.interp(t + 0.1, ts, P[:, a]) for a in range(3)])
        return p, (p2 - p) / 0.1

    cams = []
    cam_csv = src / ("camera.csv" if (src / "camera.csv").exists() else "frames.csv")
    if (src / "camera.mp4").exists() and cam_csv.exists() and not getattr(args, "no_video", False):
        rows = list(csv.DictReader(open(cam_csv)))
        tcol = "t" if "t" in rows[0] else "t_capture_s"
        idx = np.array([int(r["camera_frame"]) for r in rows])
        tt = np.array([float(r[tcol]) for r in rows])
        cams = (idx, tt)
    proj = Projector()
    reader = SyncReader(cfg) if args.read else None
    exp = Experiment(cfg, out, proj, reader)
    measured = json.loads((src / "scene.json").read_text()) if (src / "scene.json").exists() else {}
    if measured.get("measured"):
        import dataset
        exp.measured = [q for q in dataset.truth_objects(measured, cfg, dataset.events_of(src, measured, index[0]["t"]))
                        if q.kind != "person"]
    video_path = paths.VIDEOS / f"replay_{out.name}.mp4"
    paths.VIDEOS.mkdir(parents=True, exist_ok=True)
    writer = None if getattr(args, "no_video", False) else VideoWriter(video_path, (1280, 720), cfg["record"]["video_fps"],
                                                                         crf=cfg["record"]["video_crf"])
    video = read_video(src / "camera.mp4") if cams else None
    grey = np.full((cfg["camera"]["height"], cfg["camera"]["width"], 3), 200, np.uint8)
    log = gzip.open(out / "radar.jsonl.gz", "wt", encoding="utf-8")
    t_video, n_proc, t_wall = -1e9, 0, time.perf_counter()
    k_cam, cam_idx0 = 0, (cams[0][0] if cams else 0)
    looks = [main]
    limit = args.seconds if args.seconds > 0 else 1e9
    rc = cfg["rail"]
    frame_s = frame_ms_of(cfg) / 1e3
    proc = deque([frame_s], maxlen=50)
    stop_s = 2 * rc["stop_mm"] / max(rc["expected_speed_mm_s"], 1e-3)
    t_first = index[0]["t"]
    busy, n_skipped, n_not_looked = -np.inf, 0, 0
    chirp_of = [(lambda nm: nm if nm in lib.names else main)(q.get("chirp", q.get("waveform", main))) for q in index]
    t_of = [q["t"] for q in index]
    try:
        for i, r in enumerate(index):
            if r["t"] - t_first > limit:
                break
            name = r.get("chirp", r.get("waveform", main))
            name = name if name in lib.names else main
            while video is not None and cams and k_cam < 10 ** 7:
                cam_index = cam_idx0 + k_cam
                tc = float(np.interp(cam_index, cams[0], cams[1]))
                if tc > r["t"]:
                    break
                try:
                    img = next(video)
                except StopIteration:
                    video = None
                    break
                k_cam += 1
                pose, _ = pose_at(tc)
                compose = tc - t_video >= 1.0 / cfg["record"]["video_fps"]
                pic = exp.on_camera(img, tc, cam_index, pose, compose)
                if pic is not None:
                    t_video = tc
                    writer.write(pic)
            if name != main and name not in looks:
                n_not_looked += 1
                continue
            if r["t"] < busy:
                newer = [j for j in range(i + 1, len(index)) if t_of[j] <= busy and
                         (chirp_of[j] == name or (name != main and chirp_of[j] == main))]
                if newer:
                    n_skipped += 1
                    continue
            wf = lib[name]
            t = r["t"] - wf.active_s / 2
            pose, v = pose_at(t)
            frame = frame_from_words(np.asarray(words[i * per:(i + 1) * per]), wf.cfg, padded)
            rec = scene.step(frame, t, name, pose, v)
            t_c = time.perf_counter()
            looks, why, table = chooser.decide(scene.tracker.tracks, scene.tracker.doppler_sign, after=name)
            rec["ms"]["choice"] = 1e3 * (time.perf_counter() - t_c)
            rec["ms"]["total"] += rec["ms"]["choice"]
            p_s = cost[name] if cost else rec["ms"]["total"] / 1e3
            busy = max(busy, r["t"]) + p_s
            rec.update(t_arrival=r["t"], t_ready=busy, skipped=n_skipped, not_looked=n_not_looked)
            proc.append(p_s)
            chooser.observe(name, p_s)
            if cost and name == chooser.main:
                looks, why, table = chooser.decide(scene.tracker.tracks, scene.tracker.doppler_sign)
            tr = response_time(frame_s, float(np.median(proc)), 1 / 30, stop_s)
            rec.update(frame=r["frame"], looks=looks, look_reason=why, look_table=table, mode="replay", received=i + 1,
                       processed=n_proc + 1)
            log.write(json.dumps(rec, default=_js) + "\n")
            msg = compact(rec)
            exp.on_radar(msg, pose, float(np.linalg.norm(v)), rc["stop_mm"] / 1e3, bool(np.linalg.norm(v) > 1e-3), t, tr)
            n_proc += 1
            if writer is not None and video is None and t - t_video >= 1.0 / cfg["record"]["video_fps"]:
                pic = exp.on_camera(grey, t, i, pose, True)
                t_video = t
                writer.write(pic)
            if n_proc % 50 == 0:
                print(f"  replay: {n_proc} looks, t = {t - t_first:.1f} s, state {exp.cmd.state}: {exp.cmd.reason}  "
                      f"({time.perf_counter() - t_wall:.0f} s)")
    finally:
        log.close()
        if writer is not None:
            writer.close()
        s = exp.save(None, {"replay_of": str(src), "method": method, "padded": padded, "reference": bool(reference),
                            "calibration": str(cal_path), "live_video": str(video_path)})
    print(f"replay saved: {out}  ({n_proc} looks, policy {chooser.policy}, {n_skipped} skipped, {n_not_looked} not looked); "
          f"video {video_path if writer is not None else '-'}; final state {s['final_state']}")
    return out


def compare(cfg: dict, args) -> None:
    import copy
    import gzip as gz
    from methods.chirp_choice import POLICIES
    src = Path(args.folder).resolve()
    a = copy.copy(args)
    a.policy, a.no_video, a.read, a.name, a.cost_s = "all", True, False, f"{src.name}_calibration", None
    out = replay(copy.deepcopy(cfg), a)
    cal = [json.loads(line) for line in gz.open(out / "radar.jsonl.gz", "rt", encoding="utf-8")]
    cost_s = {}
    for x in cal:
        cost_s.setdefault(x.get("chirp", x.get("waveform")), []).append(x["ms"]["total"] / 1e3)
    cost_s = {k: float(np.median(v)) for k, v in cost_s.items()}
    print("processing per look (median, fixed for every policy): " + ", ".join(f"{k} {1e3 * v:.0f} ms" for k, v in cost_s.items()))
    results = {}
    for policy in args.policies:
        if policy not in POLICIES:
            raise SystemExit(f"policy {policy!r}: one of {POLICIES}")
        a = copy.copy(args)
        a.policy, a.no_video, a.read, a.name, a.cost_s = policy, True, False, f"{src.name}_{policy}", cost_s
        out = replay(copy.deepcopy(cfg), a)
        recs = [json.loads(line) for line in gz.open(out / "radar.jsonl.gz", "rt", encoding="utf-8")]
        if not recs:
            continue
        dur = recs[-1]["t_arrival"] - recs[0]["t_arrival"]
        ent, w = [], []
        for x, y in zip(recs[:-1], recs[1:]):
            e = [tr["entropy_bits"] for tr in x.get("tracks", []) if tr.get("status") == "confirmed"]
            if e:
                ent.append(np.mean(e))
                w.append(y["t_arrival"] - x["t_arrival"])
        states = {}
        for line in gz.open(out / "decisions.jsonl.gz", "rt", encoding="utf-8") if (out / "decisions.jsonl.gz").exists() else []:
            d = json.loads(line)
            states[d.get("state", "?")] = states.get(d.get("state", "?"), 0) + 1
        results[policy] = {"looks_per_s": len(recs) / dur, "skipped": recs[-1]["skipped"], "not_looked": recs[-1]["not_looked"],
                           "latency_ms_median": float(1e3 * np.median([x["t_ready"] - x["t_arrival"] for x in recs])),
                           "info_bits_per_s": float(sum(x["info_bits"] for x in recs) / dur),
                           "mean_entropy_bits": float(np.average(ent, weights=w)) if ent else None,
                           "coverage": recs[-1].get("coverage"), "extra_look_share": float(np.mean([x.get("chirp", x.get("waveform")) != recs[0].get("chirp", recs[0].get("waveform")) for x in recs])),
                           "planner_states": states}
        r = results[policy]
        print(f"{policy:12s} {r['looks_per_s']:5.1f} looks/s  skipped {r['skipped']}  latency {r['latency_ms_median']:5.0f} ms  "
              f"{r['info_bits_per_s']:6.1f} bit/s  entropy {r['mean_entropy_bits'] or float('nan'):6.2f} bit")
    path = paths.OUTPUT / f"policies_{src.name}.json"
    path.write_text(json.dumps({"take": str(src), "cost_ms": {k: 1e3 * v for k, v in cost_s.items()}, "results": results},
                               indent=2, default=float))
    print(f"-> {path}")


def reference_from_background(cfg, lib, cal, bg_bin: Path):
    from methods.material.track_material import MaterialSRC
    from methods.path_reference import ReferenceRecorder
    from methods.radar_config import open_recording
    from methods.range_doppler import FrontEnd
    wf = lib.first
    fe = FrontEnd(wf, cfg["radar"]["r_min_m"], cfg["radar"]["r_max_m"], cal, None)
    msrc = MaterialSRC(wf.path, extent_m=cfg["tracking"]["object_extent_m"])
    from methods.scene import StaticDetector
    rec = ReferenceRecorder(fe, {"from_background": str(bg_bin)}, StaticDetector(cfg, fe, wf))
    from methods.radar_config import range_fft
    n, get = open_recording(bg_bin, wf.cfg)
    for i in range(n):
        fr = get(i)
        rec.add(fe.process(fr), np.zeros(3), msrc.pre.process(fr, 0.0).ra)
        rec.add_still(range_fft(fr, wf.cfg).mean(axis=1), np.zeros(3))
    print(f"empty path from the background take {bg_bin.name}: {n} frames at x = 0")
    return rec.result()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="rail, radar (both chirps), camera")
    c.add_argument("--move", action="store_true", help="small test moves of every axis (asks first)")
    for k in ("rail", "radar", "camera"):
        c.add_argument(f"--no-{k}", action="store_true")
    c = sub.add_parser("calibrate", help="channel calibration with one corner reflector, rail at home")
    c.add_argument("--target-xyz", required=True, help="reflector apex x:y:z relative to the antenna (m), e.g. 1.00:0.00:0.00")
    sub.add_parser("reference", help="the empty path: the rail drives the whole forward travel")
    c = sub.add_parser("railscale", help="the rail's true forward travel, measured by the radar on a corner reflector")
    c.add_argument("--reflector-m", type=float, required=True, help="reflector range from the antenna, rail at home (m)")
    c = sub.add_parser("run", help="THE RUN")
    c.add_argument("--take", required=True)
    c.add_argument("--seconds", type=float, default=0.0)
    c.add_argument("--read", action="store_true", help="read the text on the attention target (reader process)")
    c.add_argument("--no-camera-video", action="store_true", help="do not record the camera (it is the ground truth and annotate.py's video)")
    c.add_argument("--virtual-rail", action="store_true", help="dry runs only: a simulated rail that follows the commands")
    c.add_argument("--raw", action="store_true", help="also record the raw radar frames of EVERY sub-frame (1 MB each, 20 per "
                                                       "second): replays with any look policy (compare)")
    c.add_argument("--no-rail", action="store_true", help="rail not connected: it stays at home (decisions logged)")
    c.add_argument("--no-camera", action="store_true", help="grey picture instead of the camera")
    c.add_argument("--headless", action="store_true")
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--radar-file", default=None, help="dry run without the board: a recorded take's radar_raw.bin at its pace")
    c = sub.add_parser("replay", help="the method on a recorded take (no hardware)")
    c.add_argument("folder")
    c.add_argument("--name", default=None)
    c.add_argument("--seconds", type=float, default=0.0)
    c.add_argument("--read", action="store_true")
    c.add_argument("--reference-from-background", default=None, help="a still radar's empty-room .bin as the empty path")
    c.add_argument("--policy", default=None, help="look policy for this replay (default [radar] look_policy)")
    c.add_argument("--no-video", action="store_true", help="no live picture (faster)")
    c.add_argument("--method", default="final", choices=["final", "baseline", "ctrex", "cfar"],
                   help="final: the method; baseline: the classical chain of methods/baseline.py; ctrex / cfar: the method "
                        "with CT-Rex t-OMP or spatial OS-CFAR deciding every cell instead of Infini-Rex + e-BH")
    c = sub.add_parser("compare", help="every look policy on the same recording, replayed as live (fair comparison)")
    c.add_argument("folder")
    c.add_argument("--seconds", type=float, default=0.0)
    c.add_argument("--policies", nargs="*", default=["search", "all", "cognitive"], help="cognitive / search / all")
    c.add_argument("--reference-from-background", default=None)
    args = ap.parse_args()
    cfg = paths.load_config()
    from backend import setup_steps as st
    if args.cmd == "check":
        st.check(cfg, not args.no_rail, not args.no_radar, not args.no_camera, args.move)
    elif args.cmd == "calibrate":
        st.calibrate(cfg, [float(v) for v in args.target_xyz.split(":")])
    elif args.cmd == "reference":
        st.reference(cfg)
    elif args.cmd == "railscale":
        st.rail_scale(cfg, args.reflector_m)
    elif args.cmd == "run":
        if args.radar_file:
            cfg["_radar_file"] = str(Path(args.radar_file).resolve())
        run(cfg, args)
    elif args.cmd == "replay":
        replay(cfg, args)
    elif args.cmd == "compare":
        compare(cfg, args)


if __name__ == "__main__":
    main()
