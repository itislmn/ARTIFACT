"""
track_live.py - the experiment in real time: the radar tracks and classifies every moving object in its own process,
the camera shows the tracks in the classifier's colours and a crop of the ball.

    python track_live.py --take run_1 --record --raw      (keys: q quit, s snapshot, r video, TAB ball, a automatic)
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time

import numpy as np

import paths
from hardware import camera_view as cv
from hardware.radar_process import RadarProcess
from hardware.scene_view import CHIRP, SceneView, classifier_style, wedge
from methods.radar_model import frame_ms_of

CROP_HZ = 5.0


class NoWindow:
    alive = True

    def show(self, *a, **k):
        pass

    def keys(self):
        return []

    def close(self):
        pass


def choose_ball(tracks: list[dict], manual: int | None) -> int | None:
    conf = [t for t in tracks if t["status"] == "confirmed"]
    if manual is not None and any(t["id"] == manual for t in conf):
        return manual
    b = cv.ball_of(conf)
    return b["id"] if b is not None else None


def update_trails(trails: dict, m: dict | None, last_frame):
    if m is None or m.get("frame") == last_frame:
        return trails, last_frame
    for t in m["tracks"]:
        if t["status"] == "confirmed":
            trails.setdefault(t["id"], []).append((m["t"], t["y"], t["x"]))
    trails = {i: [q for q in v if m["t"] - q[0] <= 3.0] for i, v in trails.items()}
    return {i: v for i, v in trails.items() if v}, m.get("frame")


def camera_step(f, m, manual, proj, cfg, W, H, frame_period):
    tracks = [cv.moved(t, f.t_capture - m["t"]) for t in m["tracks"] if t["status"] == "confirmed"] if m else []
    ball = choose_ball(tracks, manual)
    row = {"camera_frame": f.index, "t_capture_s": round(f.t_capture, 6), "radar_frame": m.get("frame", "") if m else "",
           "radar_age_ms": round(1e3 * (f.t_capture - m["t"]), 1) if m else "", "ball_track": ball if ball else "",
           "tracks": ";".join(f"{t['id']}@{t['range_m']:.3f}/{t['theta_deg']:+.2f}" for t in tracks)}
    rect, crop_img, bt = None, None, None
    if ball is not None and proj is not None:
        bt = next(t for t in tracks if t["id"] == ball)
        z_lo, z_hi = cv.heights(bt, cfg, True)
        rect = cv.region_rect(proj, bt, z_lo, z_hi, pad_m=max(cfg["ball"]["diameter_m"], cfg["ball"]["label_m"][0]) / 2)
        crop_img = cv.crop(f.image, rect, cfg["camera"]["crop_px"])
        age = f.t_capture - m["t"] - m["ms"]["total"] / 1e3
        c = proj.point(bt["range_m"], bt["theta_deg"], cfg["ball"]["centre_z_m"])
        seen = c is not None and 0 <= c[0] < W and 0 <= c[1] < H
        row["sure"] = int(bool(bt.get("updated") and age <= frame_period and seen and bt["region"].get("calibrated")))
        if rect is not None:
            row["crop_px"] = ";".join(f"{x:.0f}" for x in rect)
            side = max(rect[2] - rect[0], rect[3] - rect[1])
            frac = min(side * side / (W * H), 1.0)
            row["crop_fraction"] = round(frac, 5)
            row["gating_kl_bits"] = round(float(np.log2(1 / frac)), 3)
    return tracks, ball, row, rect, crop_img, bt


def live_frame(view, f, m, tracks, ball, row, proj, cfg, trails, ball_rect, reading, recording, radar, t0):
    boxes = []
    if proj is not None:
        for tr in tracks:
            is_ball = ball is not None and tr["id"] == ball
            z_lo, z_hi = cv.heights(tr, cfg, is_ball)
            rr = ball_rect if is_ball and ball_rect is not None else cv.region_rect(proj, tr, z_lo, z_hi)
            if rr is not None:
                boxes.append(dict(rect=rr, thick=is_ball, **classifier_style(tr)))
    top = {"dets": [], "regions": [], "tracks": []}
    banner = None
    if m is None:
        banner = "waiting for the radar"
    elif f.t_capture - m["t"] > max(0.5, 3 * radar.armed_info["frame_period_s"]):
        banner = "radar blind: chirp change" if "waveform" in (radar.status or "") else "no radar data"
    if m is not None and banner is None:
        sty = {t["id"]: classifier_style(t) for t in m["tracks"] if t["status"] == "confirmed"}
        for d in m["dets"]:
            th = np.radians(d["theta_deg"])
            c = sty[d["track"]]["color"] if d.get("label") in ("core", "part") and d.get("track") in sty else None
            top["dets"].append((d["range_m"] * np.sin(th), d["range_m"] * np.cos(th), c))
        for tr in m["tracks"]:
            if tr["status"] != "confirmed":
                continue
            lo, hi = tr["region"]["lo"], tr["region"]["hi"]
            if np.all(np.isfinite(lo[:2])) and np.all(np.isfinite(hi[:2])):
                top["regions"].append((wedge(lo, hi), sty[tr["id"]]["color"]))
            tr_hist = np.array([(y, x) for _, y, x in trails.get(tr["id"], [])])
            top["tracks"].append(dict(lat=tr["y"], rng=tr["x"], trail=tr_hist if len(tr_hist) > 1 else None,
                                      **sty[tr["id"]]))
    sure = bool(row.get("sure"))
    rec = "     REC" if recording else ""
    if m is None:
        lines = [f"$t$ = {f.t_capture - t0:.1f} s{rec}", "", ""]
    else:
        chirp = (f"{' + '.join(m['looks'])} ({1e3 * len(m['looks']) / frame_ms_of(cfg):.0f} looks/s), now: {m['waveform']}"
                 if m.get("looks") else CHIRP.get(m['waveform'], m['waveform']))
        lines = [f"$t$ = {f.t_capture - t0:.1f} s     {chirp}{rec}",
                 f"{len(m['dets'])} detections (FDR level $α$ = {m.get('alpha', cfg['detection']['alpha']):.2f})     "
                 f"{len(tracks)} tracks",
                 (f"crop: {100 * row['crop_fraction']:.0f} % of the image ({row['gating_kl_bits']:.1f} bit)"
                  if sure and "crop_fraction" in row else "crop: not sure")
                 + (f"     read: {reading}" if reading and sure else "")]
    return view.frame(f.image, boxes, ball_rect if sure else None, None, top, lines, banner)


def status_lines(m: dict | None, radar, fps: float) -> list[str]:
    if m is None:
        return [f"RADAR: {radar.status[:80]}"]
    conf = [t for t in m["tracks"] if t["status"] == "confirmed"]
    looks = "+".join(m.get("looks") or [m["waveform"]]).upper()
    return [f"LOOKS {looks}  TRACKS {len(conf)}  RADAR {m['ms']['total']:.0f} MS/FRAME  "
            f"SKIPPED {m.get('skipped', 0)}  LOST PKT {m.get('lost', 0)}  CAMERA {fps:.0f} FPS",
            f"FDR ALPHA 0.10  NEW-OBJECT SHARE {m['lambda']:.3f}  INFO {m['info_bits']:.1f} BIT  "
            f"COVERAGE {m['coverage'].get('step', float('nan')):.2f}  DOPPLER SIGN {m['doppler_sign']:+d}",
            f"WHY {m.get('look_reason', '')[:90]}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", required=True)
    ap.add_argument("--record", action="store_true", help="camera video from the start")
    ap.add_argument("--raw", action="store_true", help="every radar frame to radar_raw.bin (2 MB per frame!)")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--exposure", default=None)
    ap.add_argument("--cli", default=None)
    ap.add_argument("--camera", default=None, help="part of another camera's name")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--simulate", default=None, help="no hardware: simulated radar AND camera (scenario of simulate.py)")
    ap.add_argument("--headless", action="store_true", help="no window (tests)")
    ap.add_argument("--no-camera", action="store_true", help="radar only (grey camera image): isolates radar problems")
    ap.add_argument("--policy", default=None, help="sub-frames: look policy (cognitive / search / all), default config")
    ap.add_argument("--read", action="store_true", help="read the text on the ball live (computer_vision reader, own process)")
    ap.add_argument("--truth", default="", help="the written text, e.g. ARTIFACT: readings are scored")
    ap.add_argument("--fixed-chirps", action="store_true", help="the controller decides but never changes the chirps")
    ap.add_argument("--force-switch", nargs=2, metavar=("WAVEFORM", "SECONDS"), default=None,
                    help="test the live chirp change: switch to WAVEFORM once, SECONDS after arming (gap is printed)")
    args = ap.parse_args()
    cfg = paths.load_config()
    take = paths.session_dir(cfg["session"]) / args.take
    if take.exists() and any(take.iterdir()) and not args.overwrite:
        raise SystemExit(f"{take} exists - another --take name or --overwrite")
    take.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(take / "crops", ignore_errors=True)
    (take / "crops").mkdir()
    shutil.copy(paths.CONFIG, take / "config.toml")
    if args.camera:
        cfg["camera"]["name"] = args.camera
    free_gb = shutil.disk_usage(take).free / 2 ** 30
    if args.raw:
        print(f"raw radar recording: ~{2 * 20 * 60 / 1024:.1f} GB per minute at 20 frames/s; {free_gb:.0f} GB free")

    from backend.alignment import Projector
    from backend.camera import Camera
    from backend.display import Window
    from backend.imgio import VideoWriter, write_png
    try:
        proj = Projector(paths.alignment_dir(cfg["session"]))
    except SystemExit as e:
        proj = None
        print(f"no radar->camera alignment ({e}): tracks only in the top view")
    W, H = cfg["camera"]["width"], cfg["camera"]["height"]
    win = NoWindow() if args.headless else Window(
        f"COGNITIVE RADAR TRACKING - {cfg['session']}/{args.take}   (q quit, s snapshot, r video, TAB ball)")
    win.show(np.full((720, 1280, 3), 255, np.uint8))
    if args.policy:
        cfg["radar"]["look_policy"] = args.policy
    if args.fixed_chirps:
        cfg["radar"]["reconfigure"] = False
    if args.force_switch:
        cfg["_force_switch"] = [args.force_switch[0], float(args.force_switch[1])]
    if args.simulate:
        from hardware.sim_stream import SimCamera, start_renderer
        t_origin = time.perf_counter() + 25.0
        sim_proc, queues = start_renderer(args.simulate, cfg["radar"]["start_waveform"], t_origin)
        cfg["_simulate"] = {"scenario": args.simulate, "t_origin": t_origin, "queues": queues}
        cam = SimCamera(cfg, args.simulate, t_origin)
    elif args.no_camera:
        from hardware.sim_stream import SimCamera
        cam = SimCamera(cfg, "empty", time.perf_counter())
    else:
        cam = Camera(cfg["camera"]).start()
    exposure = cfg["camera"]["exposure"] if args.exposure is None else args.exposure
    cam.set_exposure(None if str(exposure) == "auto" else int(exposure))
    print("camera:", cam.describe())
    radar = RadarProcess(cfg, take, raw=args.raw, port=args.cli).start()

    def idle(text):
        f = cam.read()
        win.show(view.frame(f.image, [], None, None, {}, ["radar: " + text[:70], "", ""]))

    rows, writer, manual, t_crop, t_show, n_crop = [], None, None, -1e9, -1e9, 0
    view = SceneView(note="colour of the dot: material head, ring: motion head")
    trails, last_frame, t_console = {}, None, -1e9
    reader, reading, t_read, n_ok, n_all, n_reads, n_hit = None, None, -1e9, 0, 0, 0, 0
    if args.read and proj is not None:
        from backend.settings import load_config as cv_config
        from hardware.reader_process import LiveReader
        from backend.pipeline import Stable, correct_chars
        cvcfg = cv_config()
        rcfg = {**cvcfg, "text": {**cvcfg["text"], "region_m": list(cfg["ball"]["label_m"])}}
        reader, stable = LiveReader(rcfg, cfg["ball"]["diameter_m"] / 2).start(), Stable()
    snaps = []
    try:
        radar.wait_armed(tick=lambda: idle("RADAR: " + radar.status))
        if args.record:
            writer = VideoWriter(take / "camera.mp4", (W, H), cam.fps_nominal)
        t0, last_index = time.perf_counter(), -1
        while win.alive and (args.seconds <= 0 or time.perf_counter() - t0 < args.seconds):
            f = cam.read()
            if writer is not None:
                writer.write(f.image, max(1, f.index - last_index) if last_index >= 0 else 1)
            last_index = f.index
            radar.poll()
            m = radar.last
            trails, last_frame = update_trails(trails, m, last_frame)
            tracks, ball, row, rect, crop_img, bt = camera_step(f, m, manual, proj, cfg, W, H,
                                                                radar.armed_info["frame_period_s"])
            row["recording"] = int(writer is not None)
            sure = row.get("sure")
            if ball is not None and proj is not None:
                if crop_img is not None and sure and f.t_capture - t_crop >= 1 / CROP_HZ:
                    t_crop = f.t_capture
                    n_crop += 1
                    name = f"crop_{n_crop:05d}_cam{f.index}.png"
                    write_png(take / "crops" / name, crop_img)
                    row["crop_file"] = name
            if reader is not None and ball is not None and proj is not None:
                for _, rd, _ in reader.poll():
                    reading = stable.add(rd.text)
                    row["read"], row["read_stable"] = rd.text, reading
                    if args.truth:
                        k = correct_chars(rd.text, args.truth.upper())
                        n_ok, n_all = n_ok + k, n_all + len(args.truth)
                        row["correct"] = k
                        n_reads, n_hit = n_reads + 1, n_hit + (k >= 1)
                if row.get("sure") and not reader.busy and f.t_capture - t_read >= 1 / cvcfg["text"]["read_hz"]:
                    uv = proj.point(bt["range_m"], bt["theta_deg"], cfg["ball"]["centre_z_m"])
                    if reader.submit(bt["id"], bt["range_m"], f.image, uv, proj.f):
                        t_read = f.t_capture
            rows.append(row)
            keys = []
            if time.perf_counter() - t_show >= 1 / cfg["display"]["max_hz"]:
                t_show = time.perf_counter()
                win.show(live_frame(view, f, m, tracks, ball, row, proj, cfg, trails, rect if ball is not None and proj
                                    is not None else None, reading, writer is not None, radar, t0))
                keys = win.keys()
            if time.perf_counter() - t_console >= 5.0:
                t_console = time.perf_counter()
                for line in status_lines(m, radar, cam.recent_fps()):
                    print("  " + line)
            if "tab" in keys and tracks:
                ids = sorted(t["id"] for t in tracks)
                manual = ids[(ids.index(ball) + 1) % len(ids)] if ball in ids else ids[0]
                print(f"ball = track {manual} (by hand)")
            if "a" in keys:
                manual = None
            if "r" in keys:
                if writer is None:
                    writer = VideoWriter(take / f"camera_{int(time.time())}.mp4", (W, H), cam.fps_nominal)
                else:
                    writer.close()
                    writer = None
            if "s" in keys:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                write_png(take / f"snapshot_{stamp}_camera.png", f.image)
                if crop_img is not None:
                    write_png(take / f"snapshot_{stamp}_ball.png", crop_img)
                (take / f"snapshot_{stamp}.json").write_text(json.dumps({"radar": m, "camera_frame": f.index, "ball": ball}, default=str))
                snaps.append(stamp)
                print(f"snapshot {stamp}")
            if "q" in keys or "escape" in keys:
                break
    finally:
        radar.stop()
        if reader is not None:
            reader.stop()
            print(f"text: final stable reading {stable.text!r}" + (f", {100 * n_ok / n_all:.0f} % characters correct, "
                  f"{n_hit}/{n_reads} reads with at least one correct letter" if n_all else ""))
        if writer is not None:
            writer.close()
        cam.stop()
        win.close()
        if rows:
            cols = list(dict.fromkeys(k for r in rows for k in r))
            with open(take / "frames.csv", "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=cols)
                wr.writeheader()
                wr.writerows(rows)
        summary = {"take": args.take, "session": cfg["session"], "camera": cam.stats(), "radar_last": radar.last,
                   "radar_errors": radar.errors, "radar_events": radar.events, "crops": n_crop, "snapshots": snaps}
        (take / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        st = summary["camera"]
        print(f"\ncamera {st['fps_measured']} fps ({st['frames_skipped']} skipped); radar: "
              f"{radar.last['received'] if radar.last else 0} frames received, "
              f"{radar.last['lost'] if radar.last else 0} packets lost; {n_crop} ball crops; folder {take}")
        print(f"next: python replay.py {take.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
