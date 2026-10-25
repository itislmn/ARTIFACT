"""
setup_steps.py - the steps before the runs of a day: hardware check, channel calibration, the rail's true travel
measured by the radar, and the empty-path reference.

    python experiment.py check --move | calibrate | railscale | reference
"""
from __future__ import annotations

import time

import numpy as np

import paths


def ask(question: str) -> bool:
    try:
        return input(question + " [y/N] ").strip().lower() in ("y", "yes", "j", "ja")
    except EOFError:
        return False


def check(cfg: dict, rail: bool, radar: bool, camera: bool, move: bool) -> bool:
    ok = True
    if rail:
        ok &= check_rail(cfg, move)
    if radar:
        ok &= check_radar(cfg)
    if camera:
        ok &= check_camera(cfg)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED (see above)")
    return ok


def check_rail(cfg: dict, move: bool) -> bool:
    from backend.rail import Rail, RailError
    print("\n== rail ==")
    try:
        r = Rail(cfg).start()
    except (RailError, Exception) as e:                         # noqa: BLE001
        print(f"  FAIL  {type(e).__name__}: {e}")
        return False
    lim = r.limits * 1e3
    print(f"  limits (mm): x {lim[0]}, y {lim[1]}, z {lim[2]} - are they MEASURED? (config.toml [rail])")
    ok = True
    if move and ask("  drive every axis that may move (x 100 mm forward, y 40 mm right) and back?"):
        home = r.pose.copy()

        def drive(p):
            r.goto(p)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 30:
                r.feed()
                time.sleep(0.05)
                if not r.moving and np.linalg.norm(r.pose - p) < 0.0005:
                    break
            return t0, time.perf_counter()

        for a, d, word in ((0, 0.100, "FORWARD, toward the scene"), (1, 0.040, "to the RIGHT, seen from behind the radar"),
                           (2, 0.040, "UP")):
            if r.limits[a, 1] - r.limits[a, 0] < 0.001:
                continue
            tgt = home.copy()
            tgt[a] = np.clip(home[a] + d, r.limits[a, 0], r.limits[a, 1])
            t0, t1 = drive(tgt)
            L = np.array([q for q in r.log if t0 <= q[0] <= t1])
            v = np.abs(np.diff(L[:, 1 + a]) / np.diff(L[:, 0])) if len(L) > 2 else np.zeros(1)
            print(f"  axis {'xyz'[a]}: target {1e3 * tgt[a]:+.1f} mm -> at {1e3 * r.pose[a]:+.1f} mm in {t1 - t0:.1f} s, "
                  f"top speed {1e3 * np.percentile(v, 90):.1f} mm/s")
            if not ask(f"  did the radar move {word}?"):
                print(f"  FAIL  flip axis_sign[{a}] in config.toml [rail]")
                ok = False
            drive(home)
        r.goto(home + np.array([0.030, 0, 0]))
        time.sleep(0.8)
        r.stop()
        time.sleep(1.5)
        drive(home)
        print(f"  measured speed {1e3 * r.speed_mps:.1f} mm/s; stops: {r.stops}; stopping distance used {1e3 * r.stop_m:.1f} mm")
    r.close()
    return ok and not r.error


def check_radar(cfg: dict, seconds: float = 5.0) -> bool:
    from backend.radar_stream import RadarStream
    from methods.waveforms import Library
    print("\n== radar ==")
    lib = Library(cfg)
    for line in lib.describe():
        print("  " + line)
    rs = RadarStream(cfg["radar"]["cli_port"], lib)
    try:
        rs.arm(cfg["radar"]["mode"])
        n0, t0 = rs.rx.frames, time.perf_counter()
        counts, last = {}, -1
        while time.perf_counter() - t0 < seconds:
            for idx, t, buf, name in rs.take(last, 0.5):
                counts[name] = counts.get(name, 0) + 1
                last = max(last, idx)
        dt = time.perf_counter() - t0
        print(f"  mode {rs.mode}" + (f" (fallback: {rs.fallback_reason})" if rs.fallback_reason else ""))
        print(f"  {rs.rx.frames - n0} frames in {dt:.1f} s: " + ", ".join(f"{k} {v / dt:.1f}/s" for k, v in counts.items())
              + f"; lost packets {rs.rx.lost}, incomplete frames {rs.rx.asm.incomplete}")
        good = rs.rx.lost == 0 and sum(counts.values()) > 0
        print(f"  {'PASS' if good else 'FAIL'}  (lost packets: the DCA1000 adapter on a USB-2 port? -> USB-3)")
        return good
    except Exception as e:                                        # noqa: BLE001
        print(f"  FAIL  {type(e).__name__}: {e}")
        return False
    finally:
        rs.disarm()


def check_camera(cfg: dict, seconds: float = 3.0) -> bool:
    from backend.camera import Camera
    print("\n== camera ==")
    try:
        cam = Camera(cfg["camera"]).start()
    except Exception as e:                                        # noqa: BLE001
        print(f"  FAIL  {type(e).__name__}: {e}")
        return False
    exp = cfg["camera"]["exposure"]
    cam.set_exposure(None if str(exp) == "auto" else int(exp))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        cam.read()
    st = cam.stats()
    cam.stop()
    good = st["fps_measured"] >= 20
    print(f"  {'PASS' if good else 'FAIL'}  {cam.describe()}: {st}")
    if str(exp) == "auto":
        print("  WARNING: automatic exposure - the light detection needs a FIXED exposure ([camera] exposure)")
    return good


def calibrate(cfg: dict, target_xyz, frames: int = 20) -> bool:
    from backend.radar_stream import RadarStream
    from methods.array_model import all_channel_steering
    from methods.radar_config import frame_from_words, range_fft
    from methods.waveforms import Library
    lib = Library(cfg)
    wf = lib.first
    x, y, z = (float(v) for v in target_xyz)
    R = float(np.sqrt(x * x + y * y + z * z))
    theta = float(np.degrees(np.arcsin(y / R)))
    phi = float(np.degrees(np.arcsin(z / R)))
    print(f"reflector at R {R:.3f} m, azimuth {theta:+.2f} deg, elevation {phi:+.2f} deg; recording {frames} frames")
    rs = RadarStream(cfg["radar"]["cli_port"], lib)
    acc, n, last = None, 0, -1
    try:
        rs.arm("single")
        t0 = time.perf_counter()
        while n < frames and time.perf_counter() - t0 < 20:
            for idx, t, buf, name in rs.take(last, 0.5):
                last = max(last, idx)
                X = range_fft(frame_from_words(np.frombuffer(buf, "<i2"), wf.cfg, True), wf.cfg).mean(axis=1)
                acc = X if acc is None else acc + X
                n += 1
    finally:
        rs.disarm()
    if n == 0:
        print("FAIL: no frames")
        return False
    X = acc / n
    res = wf.cfg.range_res_m
    ks = np.arange(int((R - 0.15) / res), int((R + 0.15) / res) + 1)
    az = list(wf.cfg.azimuth_channels)
    k = int(ks[np.argmax(np.sum(np.abs(X[ks][:, az]) ** 2, axis=1))])
    c = X[k] / all_channel_steering(wf.cfg, theta, phi)
    cal = (c / c[0]).astype(np.complex64)
    out = paths.reference_dir(cfg["session"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "channel_cal.npy", cal)
    print(f"echo at {k * res:.3f} m ({n} frames); |cal| {np.round(np.abs(cal), 2)}; -> {out / 'channel_cal.npy'}")
    return True


def reflector_range(X: np.ndarray, rcfg, r_guess: float, half_window_m: float = 0.15) -> float:
    res = rcfg.range_res_m
    az = list(rcfg.azimuth_channels)
    P = np.sum(np.abs(X[:, az]) ** 2, axis=1)
    ks = np.arange(max(int((r_guess - half_window_m) / res), 1), min(int((r_guess + half_window_m) / res) + 1, len(P) - 1))
    k = int(ks[np.argmax(P[ks])])
    m = np.sqrt(P[k - 1:k + 2])
    side = 1 if m[2] >= m[0] else -1
    alpha = m[1 + side] / max(m[1], 1e-30)
    return float((k + side * (2 * alpha - 1) / (alpha + 1)) * res)


def rail_scale(cfg: dict, reflector_m: float, frames: int = 10) -> bool:
    import json
    from backend.radar_stream import RadarStream
    from backend.rail import Rail
    from methods.radar_config import frame_from_words, range_fft
    from methods.waveforms import Library
    lib = Library(cfg)
    wf = lib.first
    end = min(float(cfg["rail"]["x_mm"][1]) / 1e3, reflector_m - cfg["radar"]["r_min_m"] - 0.2)
    xs = np.linspace(0.0, end, 6)
    print(f"corner reflector straight ahead at {reflector_m:.3f} m; the rail stops at {np.round(1e3 * xs).tolist()} mm "
          "and back home")
    if not ask("Nothing else in front of the radar, reflector placed? The rail will drive"):
        return False
    rail = Rail(cfg).start()
    rs = RadarStream(cfg["radar"]["cli_port"], lib)
    last, rows = -1, []

    def wait(seconds):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            rail.feed()
            time.sleep(0.05)

    def measure(x_cmd):
        nonlocal last
        rail.goto([x_cmd, 0.0, 0.0])
        t0 = time.perf_counter()
        while (rail.moving or abs(rail.pose[0] - x_cmd) > 0.001) and time.perf_counter() - t0 < 120:
            rail.feed()
            time.sleep(0.05)
        wait(0.5)
        for idx, _, _, _ in rs.take(last, 0.2):
            last = max(last, idx)
        acc, n = None, 0
        t0 = time.perf_counter()
        while n < frames and time.perf_counter() - t0 < 10:
            rail.feed()
            for idx, t, buf, name in rs.take(last, 0.5):
                last = max(last, idx)
                X = range_fft(frame_from_words(np.frombuffer(buf, "<i2"), wf.cfg, True), wf.cfg).mean(axis=1)
                acc, n = (X if acc is None else acc + X), n + 1
        if n == 0:
            raise SystemExit("no radar frames")
        r = reflector_range(acc / n, wf.cfg, reflector_m - x_cmd)
        rows.append({"x_cmd_m": round(float(x_cmd), 5), "x_counter_m": round(float(rail.pose[0]), 5), "range_m": round(r, 5)})
        print(f"  x {1e3 * x_cmd:7.1f} mm (counter {1e3 * rail.pose[0]:7.1f}): reflector at {1e3 * r:8.1f} mm")
    try:
        rs.arm("single")
        for x in [*xs, 0.0]:
            measure(float(x))
    finally:
        rs.disarm()
        rail.close()
    x = np.array([r["x_cmd_m"] for r in rows[:-1]])
    d = rows[0]["range_m"] - np.array([r["range_m"] for r in rows[:-1]])
    slope, icpt = np.polyfit(x, d, 1)
    resid = d - (slope * x + icpt)
    k0 = float(cfg["rail"]["steps_per_mm"][0])
    out = {"rows": rows, "scale": float(slope), "residual_rms_mm": float(1e3 * np.sqrt(np.mean(resid ** 2))),
           "residual_max_mm": float(1e3 * np.max(np.abs(resid))),
           "home_repeat_mm": float(1e3 * (rows[-1]["range_m"] - rows[0]["range_m"])),
           "steps_per_mm_now": k0, "steps_per_mm_measured": k0 / float(slope)}
    print(f"scale {slope:.4f} (1 = steps_per_mm right): over {1e3 * x[-1]:.0f} mm commanded the radar saw "
          f"{1e3 * slope * x[-1]:.1f} mm; residual rms {out['residual_rms_mm']:.2f} mm (max {out['residual_max_mm']:.2f}); "
          f"back home {out['home_repeat_mm']:+.2f} mm; steps_per_mm[0] measured {out['steps_per_mm_measured']:.2f} (config {k0})")
    dst = paths.reference_dir(cfg["session"])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "rail_scale.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {dst / 'rail_scale.json'}  (enter the measured steps_per_mm in config.toml [rail] if the scale is off by "
          "more than the residual allows)")
    return True


def reference(cfg: dict) -> bool:
    from backend.radar_process import RadarProcess
    from backend.rail import Rail
    out = paths.reference_dir(cfg["session"])
    out.mkdir(parents=True, exist_ok=True)
    print("EMPTY PATH: nothing on or beside the rail's path that will not be there in the runs, nobody in front.")
    if not ask("Ready? The rail will drive the whole forward travel"):
        return False
    rail = Rail(cfg).start()
    radar = RadarProcess(cfg, out, mode="reference").start()
    ok = False
    try:
        radar.wait_armed()
        rail.listeners.append(radar.send_pose)
        if np.linalg.norm(rail.pose) > 0.002:
            print(f"rail not at home ({np.round(1e3 * rail.pose, 1)} mm): driving home first")
            _drive(rail, radar, np.zeros(3))
        time.sleep(2.5)
        end = np.array([rail.limits[0, 1], 0.0, 0.0])
        _drive(rail, radar, end, verbose=True)
        radar.save_reference()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 10 and not any(m["type"] == "reference_saved" for m in radar.events):
            radar.poll()
            rail.feed()
            time.sleep(0.1)
        ok = any(m["type"] == "reference_saved" for m in radar.events)
        _drive(rail, radar, np.zeros(3))
    finally:
        radar.stop()
        rail.close()
    print("reference saved" if ok else "reference NOT saved")
    return ok


def _drive(rail, radar, target, verbose: bool = False):
    rail.goto(target)
    t0, t_print = time.perf_counter(), 0.0
    while time.perf_counter() - t0 < 600:
        rail.feed()
        radar.poll()
        if radar.errors:
            rail.stop()
            raise SystemExit("radar error: " + radar.errors[-1])
        if verbose and time.perf_counter() - t_print > 2.0:
            t_print = time.perf_counter()
            print(f"  x {1e3 * rail.pose[0]:7.1f} mm  ({1e3 * rail.speed_mps:.1f} mm/s)")
        if not rail.moving and np.linalg.norm(rail.pose - target) < 0.001:
            return
        time.sleep(0.05)
    rail.stop()


def selftest() -> bool:
    from methods.radar_config import range_fft
    from methods.waveforms import Library
    c = Library(paths.load_config()).first.cfg
    rng = np.random.default_rng(1)
    n = np.arange(c.n_samples)
    errs = []
    for R in np.linspace(0.30, 1.60, 14) + 0.0123:
        f = R / c.range_res_m / c.n_samples
        ph = rng.uniform(0, 2 * np.pi, c.n_virt)
        fr = 2000 * np.cos(2 * np.pi * f * n[:, None, None] + ph[None, None, :]) \
            + rng.normal(0, 20, (c.n_samples, c.n_loops, c.n_virt))
        X = range_fft(fr.astype(np.float32), c).mean(axis=1)
        errs.append(reflector_range(X, c, R) - R)
    e = 1e3 * float(np.max(np.abs(errs)))
    ok = e < 1.0
    print(f"  {'PASS' if ok else 'FAIL'}  reflector range on synthetic echoes 0.3-1.6 m: largest error {e:.2f} mm")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
