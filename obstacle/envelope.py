"""
envelope.py - the spatial envelope of a run (Oliva and Torralba 2001): the camera's Gabor energy per orientation and
scale on a 4 x 4 grid, every radar point in the colour of the camera pixel it falls on, and both envelopes along the run.

    from envelope import run_envelope
"""
from __future__ import annotations

import numpy as np

import dataset

N = 256
SCALES = (0.3, 0.15, 0.075, 0.0375)
ORIENT = 8
GRID = 4
SECTORS = np.linspace(-60.0, 60.0, 13)
GIST_EVERY_S = 1.0


def _bank():
    fy, fx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij")
    f = np.maximum(np.hypot(fx, fy), 1e-9)
    th = np.arctan2(fy, fx)
    ang = lambda a: np.exp(-np.angle(np.exp(1j * (th - a))) ** 2 / (2 * (np.pi / ORIENT) ** 2))
    return np.array([[np.exp(-np.log(f / f0) ** 2 / (2 * np.log(0.55) ** 2)) * (ang(a) + ang(a + np.pi))
                      for a in np.arange(ORIENT) * np.pi / ORIENT] for f0 in SCALES])


def gist(gray, bank):
    import cv2
    g = cv2.resize(gray, (N, N), interpolation=cv2.INTER_AREA).astype(float)
    g = (g - g.mean()) / max(g.std(), 1e-9)
    E = np.abs(np.fft.ifft2(np.fft.fft2(g)[None, None] * bank))
    return E.reshape(len(SCALES), ORIENT, GRID, N // GRID, GRID, N // GRID).mean(axis=(3, 5))


def measures(E):
    o = E.sum(axis=(0, 2, 3))
    horizontal, vertical = o[[3, 4, 5]].sum(), o[[7, 0, 1]].sum()
    return 10 * np.log10(horizontal / vertical), float(E[:2].sum() / E.sum())


def run_envelope(run: str) -> dict | None:
    import cv2
    from backend.alignment import Projector
    tk = dataset.load(run)
    folder = dataset.recorded(tk.folder) if tk is not None else None
    if folder is None or not (folder / "camera.mp4").exists() or not (folder / "camera.csv").exists():
        return None
    video = folder / "camera.mp4"
    proj, bank = Projector(), _bank()
    log = np.genfromtxt(folder / "camera.csv", delimiter=",", names=True, usecols=("camera_frame", "t"))
    cam_f, cam_t = log["camera_frame"].astype(int), log["t"].astype(float)
    frame_of = lambda t: int(round(np.interp(t, cam_t, cam_f))) - cam_f[0]
    r_max = tk.cfg["radar"]["r_max_m"]
    pts, want = [], {}
    radar = []
    for lk in tk.looks:
        ds = [d for d in lk["dets"] if d.get("world") is not None]
        th = np.array([d["theta_deg"] for d in ds if d["range_m"] <= r_max])
        busy = np.histogram(th, SECTORS)[0] > 0 if th.size else np.zeros(len(SECTORS) - 1, bool)
        radar.append((lk["t"] - tk.t0, 1 - busy.mean(), np.median([d["range_m"] for d in ds]) if ds else np.nan))
        i = frame_of(lk["t"])
        for d in ds:
            uv = proj.point(d["range_m"], d["theta_deg"], d["world"][2])
            pts.append([*d["world"], lk["t"] - tk.t0, np.nan, np.nan, np.nan])
            if uv is not None and 0 <= uv[0] < proj.size_wh[0] and 0 <= uv[1] < proj.size_wh[1]:
                want.setdefault(i, []).append((len(pts) - 1, int(uv[0]), int(uv[1])))
    pts = np.array(pts, float).reshape(-1, 7)
    best = max(tk.looks, key=lambda lk: len(lk["dets"]))
    t_show = best["t"]
    show = frame_of(t_show)
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(GIST_EVERY_S * fps)))
    cam, image, E_show = [], None, None
    for i in range(n):
        if not cap.grab():
            break
        if i not in want and i % step and i != show:
            continue
        ok, img = cap.retrieve()
        if not ok:
            continue
        for k, u, v in want.get(i, []):
            pts[k, 4:] = img[v, u, ::-1] / 255.0
        if i % step == 0 or i == show:
            E = gist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), bank)
            if i % step == 0:
                cam.append((np.interp(i + cam_f[0], cam_f, cam_t) - tk.t0, *measures(E)))
            if i == show:
                image, E_show = img[:, :, ::-1].copy(), E
    cap.release()
    near = [lk for lk in tk.looks if abs(lk["t"] - t_show) <= 0.5]
    uv = [proj.point(d["range_m"], d["theta_deg"], d["world"][2]) for lk in near for d in lk["dets"] if d.get("world") is not None]
    return {"run": run, "image": image, "gist": E_show, "t_show": t_show - tk.t0, "uv": np.array([p for p in uv if p is not None]),
            "points": pts, "camera": np.array(cam, float).reshape(-1, 3), "radar": np.array(radar, float).reshape(-1, 3),
            "pose": np.array([lk["pose"] for lk in tk.looks], float), "truth": tk.truth, "size_wh": proj.size_wh}
