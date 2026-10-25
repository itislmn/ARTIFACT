"""
alignment.py - the radar-to-camera model: pinhole with one radial term, camera centre from a ruler, rotation and focal
length fitted by Levenberg-Marquardt, model chosen by leave-one-out error.

    from backend.alignment import Projector
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from backend.settings import CALIB, ROOT, load_config, radar_dir, take_dir

R0 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
NAMES = ("rot_x", "rot_y", "rot_z", "c_x", "c_y_left", "c_z", "f", "k1")


def right_handed(xyz) -> np.ndarray:
    return np.asarray(xyz, float).reshape(-1, 3) * [1.0, -1.0, 1.0]


def rotation(w) -> np.ndarray:
    a = float(np.linalg.norm(w))
    if a < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def project(p: np.ndarray, xyz_rh: np.ndarray, centre) -> tuple[np.ndarray, np.ndarray]:
    Xc = (xyz_rh - p[3:6]) @ (rotation(p[:3]) @ R0).T
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    d = 1 + p[7] * (x ** 2 + y ** 2)
    return np.c_[centre[0] + p[6] * x * d, centre[1] + p[6] * y * d], Xc[:, 2]


def levenberg_marquardt(res, p, free, iters=200):
    r = res(p)
    cost, lam = r @ r, 1e-3
    for _ in range(iters):
        J = np.empty((r.size, free.sum()))
        for j, k in enumerate(np.flatnonzero(free)):
            h = 1e-6 * max(1.0, abs(p[k]))
            q = p.copy()
            q[k] += h
            J[:, j] = (res(q) - r) / h
        A, g = J.T @ J, J.T @ r
        while True:
            q = p.copy()
            q[free] += np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            rq = res(q)
            if rq @ rq < cost:
                break
            lam *= 10
            if lam > 1e12:
                return p
        done = cost - rq @ rq < 1e-12 * cost
        p, r, cost, lam = q, rq, rq @ rq, lam * 0.3
        if done:
            break
    return p


def start_values(size_wh, a: dict) -> np.ndarray:
    f = size_wh[0] / 2 / np.tan(np.radians(a["hfov_deg"]) / 2)
    return np.r_[0.0, 0.0, 0.0, right_handed(a["camera_centre_m"])[0], f, 0.0]


def fit(xyz, uv, size_wh, a, with_k1: bool, p0=None, free=None) -> np.ndarray:
    xyz_rh, uv, c = right_handed(xyz), np.asarray(uv, float), (size_wh[0] / 2, size_wh[1] / 2)
    if free is None:
        free = np.array([True, True, True, False, False, False, True, with_k1])
    p0 = start_values(size_wh, a) if p0 is None else np.asarray(p0, float)
    return levenberg_marquardt(lambda p: (project(p, xyz_rh, c)[0] - uv).ravel(), p0, free)


def save_model(res: dict):
    CALIB.mkdir(exist_ok=True)
    path = CALIB / "alignment.json"
    if path.exists():
        path.rename(CALIB / f"alignment_{time.strftime('%Y%m%d_%H%M%S')}.json")
    path.write_text(json.dumps(res, indent=2))


def nominal(size_wh, a: dict) -> dict:
    p = start_values(size_wh, a)
    return {"model": "nominal (not calibrated)", "params": dict(zip(NAMES, p.round(6).tolist())),
            "hfov_deg": a["hfov_deg"], "camera_centre_in_radar_frame_m": a["camera_centre_m"],
            "rms_px": None, "loo_rms_px": round(float(p[6] * np.tan(np.radians(3.0))), 1),
            "image_size_wh": list(size_wh), "points": [], "date": time.strftime("%Y-%m-%d %H:%M")}


def errors(points: list[dict], base: dict) -> np.ndarray:
    p = np.array([base["params"][n] for n in NAMES])
    size = base["image_size_wh"]
    xyz = right_handed([q["xyz_m"] for q in points])
    uv = np.array([q["uv_px"] for q in points], float)
    return np.linalg.norm(project(p, xyz, (size[0] / 2, size[1] / 2))[0] - uv, axis=1)


def quick(points: list[dict], base: dict) -> dict:
    pts = [q for q in points if q.get("uv_px")]
    if len(pts) < 2:
        raise SystemExit(f"{len(pts)} clicked points; the quick re-alignment needs at least 2")
    size = base["image_size_wh"]
    p0 = np.array([base["params"][n] for n in NAMES])
    free = np.array([True, True, True, False, False, False, False, False])
    xyz = np.array([q["xyz_m"] for q in pts], float)
    uv = np.array([q["uv_px"] for q in pts], float)
    p = fit(xyz, uv, size, None, False, p0, free)
    c = (size[0] / 2, size[1] / 2)
    err = np.linalg.norm(project(p, right_handed(xyz), c)[0] - uv, axis=1)
    loo = None
    if len(pts) >= 3:
        loo = np.array([np.linalg.norm(project(fit(np.delete(xyz, i, 0), np.delete(uv, i, 0), size, None, False, p0, free),
                                               right_handed(xyz[i]), c)[0][0] - uv[i]) for i in range(len(pts))])
    old = np.degrees(p0[:3])
    return {**base, "model": base["model"].split(" +")[0] + " + quick rotation", "params": dict(zip(NAMES, p.round(6).tolist())),
            "rotation_change_deg": np.round(np.degrees(p[:3]) - old, 3).tolist(),
            "quick_points": pts, "quick_error_px": err.round(2).tolist(),
            "loo_rms_px": round(float(np.sqrt(np.mean(loo ** 2))), 2) if loo is not None else base["loo_rms_px"],
            "date": time.strftime("%Y-%m-%d %H:%M")}


def fit_heights(xyz, uv, labels, size_wh, a: dict, iters: int = 6):
    c = (size_wh[0] / 2, size_wh[1] / 2)
    dz = {lab: 0.0 for lab in set(labels)}
    for _ in range(iters):
        xc = xyz.copy()
        xc[:, 2] += [dz[lab] for lab in labels]
        p = fit(xc, uv, size_wh, a, False)
        r = project(p, right_handed(xc), c)[0] - uv
        for lab in dz:
            m = labels == lab
            dz[lab] += float(np.mean(r[m, 1] * xyz[m, 0] / p[6]))
    return p, dz


def solve(points: list[dict], size_wh, a: dict) -> dict:
    pts = [p for p in points if p.get("uv_px")]
    xyz = np.array([p["xyz_m"] for p in pts], float)
    uv = np.array([p["uv_px"] for p in pts], float)
    c = (size_wh[0] / 2, size_wh[1] / 2)
    if len(pts) < 4:
        raise SystemExit(f"{len(pts)} clicked points; the model needs at least 4 (better 8-12)")
    models = {}
    for with_k1 in (False, True):
        n_par = 4 + with_k1
        if 2 * (len(pts) - 1) <= n_par:
            continue
        p = fit(xyz, uv, size_wh, a, with_k1)
        err = np.linalg.norm(project(p, right_handed(xyz), c)[0] - uv, axis=1)
        loo = np.array([np.linalg.norm(project(fit(np.delete(xyz, i, 0), np.delete(uv, i, 0), size_wh, a, with_k1),
                                               right_handed(xyz[i]), c)[0][0] - uv[i]) for i in range(len(pts))])
        models["f+k1" if with_k1 else "f"] = {
            "params": dict(zip(NAMES, p.round(6).tolist())),
            "hfov_deg": round(float(np.degrees(2 * np.arctan(size_wh[0] / 2 / p[6]))), 2),
            "camera_centre_in_radar_frame_m": a["camera_centre_m"],
            "rms_px": round(float(np.sqrt(np.mean(err ** 2))), 2),
            "loo_rms_px": round(float(np.sqrt(np.mean(loo ** 2))), 2),
            "error_px": err.round(2).tolist(), "loo_error_px": loo.round(2).tolist()}
    labels = np.array([p["label"] for p in pts])
    if len(pts) >= 8:
        p, dz = fit_heights(xyz, uv, labels, size_wh, a)
        xc = xyz.copy()
        xc[:, 2] += [dz[lab] for lab in labels]
        err = np.linalg.norm(project(p, right_handed(xc), c)[0] - uv, axis=1)
        loo = []
        for i in range(len(pts)):
            k = np.arange(len(pts)) != i
            pi, dzi = fit_heights(xyz[k], uv[k], labels[k], size_wh, a)
            xi = xyz[i].copy()
            xi[2] += dzi.get(labels[i], 0.0)
            loo.append(np.linalg.norm(project(pi, right_handed(xi[None]), c)[0][0] - uv[i]))
        loo = np.array(loo)
        models["f+heights"] = {
            "params": dict(zip(NAMES, p.round(6).tolist())),
            "hfov_deg": round(float(np.degrees(2 * np.arctan(size_wh[0] / 2 / p[6]))), 2),
            "camera_centre_in_radar_frame_m": a["camera_centre_m"],
            "height_correction_mm": {lab: round(1000 * v, 1) for lab, v in dz.items()},
            "rms_px": round(float(np.sqrt(np.mean(err ** 2))), 2),
            "loo_rms_px": round(float(np.sqrt(np.mean(loo ** 2))), 2),
            "error_px": err.round(2).tolist(), "loo_error_px": loo.round(2).tolist()}
    best = min(models, key=lambda m: models[m]["loo_rms_px"])
    return {"model": best, **models[best], "models": {m: {k: v[k] for k in ("rms_px", "loo_rms_px")}
                                                      for m, v in models.items()},
            "image_size_wh": list(size_wh), "points": pts, "date": time.strftime("%Y-%m-%d %H:%M")}


class Projector:

    def __init__(self, calib: Path = CALIB):
        path = calib / "alignment.json"
        if not path.exists():
            raise SystemExit("calibration/alignment.json missing: python align.py nominal (rough) or points + solve")
        a = json.loads(path.read_text())
        self.p = np.array([a["params"][n] for n in NAMES])
        self.size_wh = tuple(a["image_size_wh"])
        self.centre = (self.size_wh[0] / 2, self.size_wh[1] / 2)
        self.f = self.p[6]
        self.loo_rms_px = a["loo_rms_px"]

    def point(self, range_m: float, azimuth_deg: float, z: float) -> np.ndarray | None:
        y = range_m * np.sin(np.radians(azimuth_deg))
        xyz = [np.sqrt(max(range_m ** 2 - y ** 2 - z ** 2, 0.0)), y, z]
        uv, depth = project(self.p, right_handed(xyz), self.centre)
        return uv[0] if depth[0] > 0 else None

    def bar(self, range_m, azimuth_deg, z_pair, n: int = 17):
        pts = [self.point(range_m, azimuth_deg, z) for z in np.linspace(z_pair[0], z_pair[1], n)]
        return None if any(e is None for e in pts) else np.array(pts)

    def u_at_row(self, range_m, azimuth_deg, v, z_pair) -> float:
        lo, hi = z_pair
        for _ in range(40):
            z = (lo + hi) / 2
            lo, hi = (lo, z) if self.point(range_m, azimuth_deg, z)[1] < v else (z, hi)
        return float(self.point(range_m, azimuth_deg, z)[0])


def selftest() -> bool:
    rng = np.random.default_rng(3)
    size = (1280, 720)
    a = {"hfov_deg": 70.0, "camera_centre_m": [0.0, 0.0, 0.06]}
    true = np.r_[np.radians([1.5, -2.0, 0.8]), right_handed([0.01, -0.02, 0.06])[0], 950.0, -0.08]
    n = 12
    R, th, z = rng.uniform(1.2, 2.0, n), rng.uniform(-35, 25, n), rng.choice([-0.11, -0.01, 0.04], n)
    y = R * np.sin(np.radians(th))
    uv = project(true, right_handed(np.c_[np.sqrt(R**2 - y**2 - z**2), y, z]), (640, 360))[0] + rng.normal(0, 2, (n, 2))
    Rm, thm = R + rng.normal(0, 0.02, n), th + rng.normal(0, 0.7, n)
    ym = Rm * np.sin(np.radians(thm))
    pts = [{"take": "syn", "label": str(i), "xyz_m": [float(np.sqrt(r**2 - b**2 - c**2)), float(b), float(c)],
            "uv_px": q.tolist()} for i, (r, b, c, q) in enumerate(zip(Rm, ym, z, uv))]
    res = solve(pts, size, a)
    p = np.array([res["params"][k] for k in NAMES])
    grid = right_handed([[1.6 * np.cos(np.radians(t)), 1.6 * np.sin(np.radians(t)), 0.0] for t in range(-35, 26, 5)])
    new_err = np.linalg.norm(project(p, grid, (640, 360))[0] - project(true, grid, (640, 360))[0], axis=1)
    checks = [
        (f"model chosen by leave-one-out: {res['model']} (rms / loo: {res['models']})", True),
        (f"focal length {res['params']['f']:.0f} px (true 950)", abs(res["params"]["f"] - 950) < 60),
        (f"error on new points: median {np.median(new_err):.1f} px, max {new_err.max():.1f} px", np.median(new_err) < 15),
    ]
    for text, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {text}")
    return all(ok for _, ok in checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    s = sub.add_parser("points")
    s.add_argument("takes", nargs="+")
    s.add_argument("--session", default=None)
    s.add_argument("--r-max-m", type=float, default=None, help="radar range limit for these takes (config: [radar] r_max_m)")
    sub.add_parser("solve")
    sub.add_parser("selftest")
    sub.add_parser("nominal")
    for name in ("check", "quick"):
        q = sub.add_parser(name)
        q.add_argument("takes", nargs="+")
        q.add_argument("--save-to", default=None, help="quick: write the new model to this folder (alignment.json), "
                                                         "leaving calibration/alignment.json unchanged")
    args = ap.parse_args()
    cfg = load_config()
    if args.step == "selftest":
        raise SystemExit(0 if selftest() else 1)
    if args.step == "nominal":
        res = nominal((cfg["camera"]["width"], cfg["camera"]["height"]), cfg["alignment"])
        save_model(res)
        print(f"nominal model (not calibrated): hfov {res['hfov_deg']} deg, uncertainty {res['loo_rms_px']} px -> calibration/alignment.json")
        return
    if args.step in ("check", "quick"):
        base = json.loads((CALIB / "alignment.json").read_text())
        pts = [q for q in json.loads((CALIB / "points.json").read_text()) if q["take"] in args.takes and q.get("uv_px")]
        if not pts:
            raise SystemExit(f"no clicked points of {args.takes} in calibration/points.json (python align.py points ...)")
        e = errors(pts, base)
        limit = 2 * base["loo_rms_px"]
        for q, x in zip(pts, e):
            print(f"  {q['take']:>10} / {q['label']:<12} error {x:6.1f} px with the current model ({base['model']})")
        print(f"median {np.median(e):.1f} px vs. 2 x leave-one-out error of the model = {limit:.1f} px -> "
              + ("still aligned" if np.median(e) <= limit else "MOVED: run  python align.py quick " + " ".join(args.takes)))
        if args.step == "quick":
            res = quick(pts, base)
            if args.save_to:
                out = Path(args.save_to).resolve()
                out.mkdir(parents=True, exist_ok=True)
                (out / "alignment.json").write_text(json.dumps(res, indent=2))
            else:
                save_model(res)
                out = CALIB
            print(f"rotation changed by {res['rotation_change_deg']} deg; errors now {res['quick_error_px']} px "
                  f"-> {out / 'alignment.json'}")
        return
    path = CALIB / "points.json"
    if args.step == "points":
        from backend.display import Window
        from backend.imgio import read_png
        from backend.radar_link import RadarTakes
        session = args.session or cfg["session"]
        rc = {**cfg["radar"], **({"r_max_m": args.r_max_m} if args.r_max_m else {})}
        radar, new = RadarTakes(radar_dir(session), rc), []
        for take in args.takes:
            pts = radar.points(take)
            t_radar = (radar_dir(session) / f"{take}.json").stat().st_mtime
            shots = [q for q in sorted(take_dir(session, take).glob("snapshot_*_raw.png"))
                     if abs(q.stat().st_mtime - t_radar) < 120]
            if not shots:
                print(f"  {take}: NO camera snapshot of this radar take - skipped; record it again: "
                      f"python align.py take {take} ... --overwrite")
                continue
            img = read_png(shots[-1])
            win = Window("alignment")
            for p in pts:
                win.title(f"{take}: click the apex of {p['label']} (radar {p['range_m']:.2f} m, "
                          f"{p['azimuth_deg']:+.1f} deg)   x = skip, Esc = stop")
                uv = None
                while win.alive and uv is None:
                    win.show(img)
                    keys = win.keys()
                    clicks = win.clicks()
                    if clicks:
                        uv = [round(clicks[0][0], 1), round(clicks[0][1], 1)]
                    elif "x" in keys or "escape" in keys:
                        break
                p.update(uv_px=uv, image=str(shots[-1].relative_to(ROOT)))
                if "escape" in keys:
                    break
            win.close()
            new += pts
        old = json.loads(path.read_text()) if path.exists() else []
        kept = [p for p in old if p["take"] not in args.takes]
        CALIB.mkdir(exist_ok=True)
        path.write_text(json.dumps(kept + new, indent=2))
        print(f"{sum(1 for p in kept + new if p.get('uv_px'))} clicked points in calibration/points.json")
    else:
        if not path.exists():
            raise SystemExit("no clicked points yet: first  python align.py points align_1 align_2 ...  (click every apex)")
        pts = json.loads(path.read_text())
        size = (cfg["camera"]["width"], cfg["camera"]["height"])
        res = solve(pts, size, cfg["alignment"])
        save_model(res)
        if "height_correction_mm" in res:
            print(f"reflector height corrections (tape heights of the plan): {res['height_correction_mm']} mm")
        print(f"model {res['model']} (all: {res['models']}), f {res['params']['f']:.0f} px = {res['hfov_deg']} deg "
              f"field of view (camera centre from the ruler: {res['camera_centre_in_radar_frame_m']} m)")
        for p, e, l in zip(res["points"], res["error_px"], res["loo_error_px"]):
            print(f"  {p['take']:>10} / {p['label']:<12} error {e:6.1f} px   leave-one-out {l:6.1f} px")
        print("-> calibration/alignment.json   (figure: python align.py figure)")


if __name__ == "__main__":
    main()
