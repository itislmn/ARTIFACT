"""
world.py - the scene in world coordinates: static objects, known objects of the reference and movers with their
measured reach, as boxes the platform must not touch.

    python -m methods.world      (self-test)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

EL_MAX_DEG = 38.0
EL_FLOOR_DEG = 2.0
MIN_Z_POINTS = 5


def det_world(d: dict, pose: np.ndarray) -> tuple[np.ndarray, bool]:
    th = np.radians(d["theta_deg"])
    el = d.get("elev_deg", np.nan)
    fit = d.get("elev_fit", np.nan)
    ok = bool(np.isfinite(el) and abs(el) < EL_MAX_DEG and np.isfinite(fit) and 0.5 <= fit <= 2.0)
    ph = np.radians(el) if ok else 0.0
    R = d["range_m"]
    p = np.asarray(pose, float) + R * np.array([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)])
    return p, ok


def det_sigma(d: dict, wf) -> np.ndarray:
    snr = 10 ** (d["snr_db"] / 10)
    th = np.radians(d["theta_deg"])
    sr, st, _ = wf.sigmas(snr, th)
    R = d["range_m"]
    s_psi = 1.0 / np.sqrt(2 * max(snr, 1.0) * 4)
    s_el = np.hypot(s_psi / (2 * np.pi * 0.8), np.sin(np.radians(EL_FLOOR_DEG)))
    return np.array([sr, R * st, R * s_el])


@dataclass
class StaticObject:
    id: int
    points: deque = field(default_factory=lambda: deque(maxlen=1500))
    last_seen: float = 0.0
    absent_s: float = 0.0
    classes: dict = field(default_factory=dict)
    guess: dict | None = None
    lo: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hi: np.ndarray = field(default_factory=lambda: np.zeros(3))
    z_known: bool = False
    r_seen: float = np.inf

    def refit(self):
        P = np.array(self.points)
        lo, hi = np.zeros(3), np.zeros(3)
        for a in range(3):
            ok = P[:, 6] > 0 if a == 2 else np.ones(len(P), bool)
            if a == 2 and ok.sum() < MIN_Z_POINTS:
                lo[a], hi[a] = -np.inf, np.inf
                continue
            v, s = P[ok, a], P[ok, 3 + a]
            if len(v) >= 20:
                lo[a], hi[a] = np.quantile(v - 2 * s, 0.02), np.quantile(v + 2 * s, 0.98)
            else:
                lo[a], hi[a] = np.min(v - 2 * s), np.max(v + 2 * s)
        self.lo, self.hi = lo, hi
        self.z_known = bool(np.isfinite(lo[2]))


class World:
    def __init__(self, cfg: dict):
        tc, pc = cfg["tracking"], cfg["planner"]
        self.extent = tc["object_extent_m"]
        self.v_max = tc["max_speed_mps"]
        self.alpha_plan = pc["alpha"]
        self.footprint_s = pc["footprint_s"]
        self.absent_s = pc["absent_s"]
        self.r_lim = (cfg["radar"]["r_min_m"], cfg["radar"]["r_max_m"])
        self.floor_z = -float(cfg["platform"]["antenna_height_m"])
        self.person_top = self.floor_z + 1.95
        self.static: dict[int, StaticObject] = {}
        self.history: dict[int, deque] = {}
        self.boxes: dict[int, deque] = {}
        self.last_box: dict[int, tuple] = {}
        self.objects: list[dict] = []
        self.known: list[StaticObject] = []
        self.next_id = 1_000_000

    def update(self, tracker, dets: list[dict], pose, v_rail, t: float, wf, look_s: float) -> list[dict]:
        pose = np.asarray(pose, float)
        v_e = np.asarray(v_rail[:2], float)
        alive = {tr.id: tr for tr in tracker.tracks}
        by_track: dict[int, list] = {}
        for d in dets:
            if d.get("track", -1) in alive and d.get("label") in ("core", "part", "birth"):
                by_track.setdefault(d["track"], []).append(d)
        out = []
        for tid, tr in alive.items():
            ds = by_track.get(tid, [])
            h = self.history.setdefault(tid, deque(maxlen=600))
            for d in ds:
                if d["label"] in ("core", "birth"):
                    p, _ = det_world(d, pose)
                    h.append((t, p[0], p[1], not self._moving(d, v_e, wf, tracker.doppler_sign)))
            if not tr.confirmed:
                continue
            if self._is_static(h, t, wf.range_res_m):
                so = self.static.setdefault(tid, StaticObject(tid))
                for d in ds:
                    if not d.get("still", True):
                        continue
                    p, z_ok = det_world(d, pose)
                    s = det_sigma(d, wf)
                    so.points.append((*p, *s, float(z_ok)))
                    so.r_seen = min(so.r_seen, float(d["range_m"]))
                if so.points:
                    so.refit()
                    so.last_seen, so.absent_s = (t, 0.0) if ds else (so.last_seen, so.absent_s)
                    so.classes = tr.classes or so.classes
                    so.guess = getattr(tr, "guess", None) or so.guess
                self.boxes.pop(tid, None)
            else:
                so = self.static.pop(tid, None)
                if so is not None and so.points:
                    so.id, self.next_id = self.next_id, self.next_id + 1
                    self.static[so.id] = so
                obj = self._moving_object(tr, ds, pose, v_e, t, wf, tracker)
                if obj is not None:
                    out.append(obj)
        self._absence(dets, alive, pose, t, look_s)
        self._merge()
        for so in self.static.values():
            if so.points:
                out.append(self._static_dict(so, so.id in alive, t))
        for so in self.known:
            out.append({**self._static_dict(so, True, t), "known": True, "motion": "static"})
        for tid in [k for k in self.history if k not in alive]:
            self.history.pop(tid)
            self.boxes.pop(tid, None)
            self.last_box.pop(tid, None)
        self.objects = out
        return out

    @staticmethod
    def _moving(d, v_e, wf, sign) -> bool:
        u = np.array([np.cos(np.radians(d["theta_deg"])), np.sin(np.radians(d["theta_deg"]))])
        return abs(d["v_mps"] * (sign or 1) + float(v_e @ u)) > 2 * wf.dv

    @staticmethod
    def _is_static(h, t, cell) -> bool:
        recent = [(x, y, s) for ti, x, y, s in h if t - ti <= 2.0]
        if len(recent) < 5 or t - h[0][0] < 1.0:
            return False
        P = np.array(recent)
        return bool(P[:, 2].mean() >= 0.8 and np.ptp(P[:, 0]) <= cell and np.ptp(P[:, 1]) <= cell)

    @staticmethod
    def _person(classes: dict) -> bool:
        return label(classes, "material") == "waterlike" and label(classes, "motion") == "articulated"

    def _moving_object(self, tr, ds, pose, v_e, t, wf, tracker) -> dict | None:
        c = np.r_[pose[:2] + tr.kf.s[:2], pose[2]]
        person = self._person(tr.classes)
        pts = [q for q in (det_world(d, pose) for d in ds) if np.hypot(*(q[0][:2] - c[:2])) <= self.extent]
        spread = np.ptp([p for p, _ in pts], axis=0) / 2 if len(pts) > 1 else np.zeros(3)
        half = np.maximum(spread, 2 * np.sqrt(np.clip(np.diag(tr.kf.P)[:2], 0, None)).max())
        reach = self.extent / 2 if person else 0.0
        zs = [p[2] for p, ok in pts if ok]
        if person:
            z_lo, z_hi = self.floor_z, self.person_top
        elif zs:
            z_lo, z_hi = min(zs) - 0.1, max(zs) + 0.1
        else:
            z_lo, z_hi = -np.inf, np.inf
        lo = np.r_[c[:2] - half[:2] - reach, z_lo]
        hi = np.r_[c[:2] + half[:2] + reach, z_hi]
        last = self.last_box.get(tr.id)
        if tr.last_update >= t - 1e-9 or last is None:
            self.last_box[tr.id] = (t, lo.copy(), hi.copy())
        else:
            run = self.v_max * (t - last[0])
            lo[:2] = np.maximum(lo[:2], last[1][:2] - run)
            hi[:2] = np.minimum(hi[:2], last[2][:2] + run)
            hi[:2] = np.maximum(hi[:2], lo[:2])
        regions, calibrated, reach_m = {}, True, {}
        for hz in tracker.horizons:
            d, ok = self._reach(self.history.get(tr.id, ()), hz, wf.frame_period_s)
            calibrated &= ok
            reach_m[str(hz)] = round(d, 4)
            regions[str(hz)] = [[*(lo[:2] - d), z_lo], [*(hi[:2] + d), z_hi]]
        q = self.boxes.setdefault(tr.id, deque(maxlen=400))
        q.append((t, lo, hi))
        while q and t - q[0][0] > self.footprint_s:
            q.popleft()
        flo = np.min([b[1] for b in q], axis=0)
        fhi = np.max([b[2] for b in q], axis=0)
        h = self.history.get(tr.id, ())
        still = len(h) > 0 and all(s for *_, s in h)
        return {"id": tr.id, "kind": "moving", "still": bool(still), "lo": lo.tolist(), "hi": hi.tolist(),
                "z_known": bool(zs) or person,
                "person": person, "material": label(tr.classes, "material"), "motion": label(tr.classes, "motion"),
                "guess": getattr(tr, "guess", None),
                "speed": float(np.hypot(tr.kf.s[2], tr.kf.s[3])), "v": [float(tr.kf.s[2]), float(tr.kf.s[3])],
                "regions": regions, "calibrated": bool(calibrated), "reach_m": reach_m,
                "footprint": [flo.tolist(), fhi.tolist()], "live": True, "n_points": len(ds),
                "centre": c.tolist(), "range_m": float(np.hypot(*tr.kf.s[:2])),
                "theta_deg": float(np.degrees(np.arctan2(tr.kf.s[1], tr.kf.s[0])))}

    def _reach(self, h, hz: float, frame_s: float) -> tuple[float, bool]:
        bound = self.v_max * hz
        if len(h) < 2:
            return bound, False
        P = np.array([(t, x, y) for t, x, y, _ in h])
        ts = P[:, 0]
        j = np.clip(np.searchsorted(ts, ts + hz), 1, len(ts) - 1)
        j = np.where(np.abs(ts[j - 1] - ts - hz) < np.abs(ts[j] - ts - hz), j - 1, j)
        ok = (np.abs(ts[j] - ts - hz) <= frame_s / 2) & (j > np.arange(len(ts)))
        sc = np.max(np.abs(P[j[ok], 1:] - P[ok, 1:]), axis=1)
        k = int(np.ceil((1 - self.alpha_plan) * (len(sc) + 1)))
        if k > len(sc):
            return bound, False
        return float(min(np.sort(sc)[k - 1], bound)), True

    def _static_dict(self, so: StaticObject, live: bool, t: float) -> dict:
        fin = np.isfinite(so.lo) & np.isfinite(so.hi)
        c = np.where(fin, (np.where(fin, so.lo, 0) + np.where(fin, so.hi, 0)) / 2, 0.0)
        return {"id": so.id, "kind": "static", "lo": so.lo.tolist(), "hi": so.hi.tolist(), "z_known": so.z_known,
                "person": self._person(so.classes), "material": label(so.classes, "material"),
                "motion": label(so.classes, "motion"), "guess": so.guess, "speed": 0.0, "live": live, "n_points": len(so.points),
                "last_seen": so.last_seen, "centre": c.tolist()}

    def set_known(self, reference) -> int:
        P = getattr(reference, "static_pts", None)
        self.known = []
        if P is None or len(P) == 0:
            return 0
        lo, hi = P[:, :2] - 2 * P[:, 3:5], P[:, :2] + 2 * P[:, 3:5]
        parent = np.arange(len(P))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(P)):
            touch = np.flatnonzero(np.all(lo[i] <= hi[i + 1:], axis=1) & np.all(lo[i + 1:] <= hi[i], axis=1)) + i + 1
            for j in touch:
                parent[root(j)] = root(i)
        roots = np.array([root(i) for i in range(len(P))])
        poses = np.c_[reference.x, reference.y, reference.z]
        for g in np.unique(roots):
            pts = P[roots == g]
            c = pts[:, :2].mean(axis=0)
            rel = c[None, :] - poses[:, :2]
            R, th = np.hypot(rel[:, 0], rel[:, 1]), np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
            in_view = (R > self.r_lim[0]) & (R < self.r_lim[1]) & (np.abs(th) < 50.0)
            if not in_view.any() or len(np.unique(pts[:, 7])) < 0.5 * in_view.sum():
                continue
            so = StaticObject(-(len(self.known) + 1))
            so.points.extend(tuple(p) for p in pts[:, :7])
            so.refit()
            so.classes = {"motion": {"label": "static"}}
            self.known.append(so)
        return len(self.known)

    def _absence(self, dets, alive, pose, t, look_s):
        pts = [det_world(d, pose)[0] for d in dets]
        for so in list(self.static.values()):
            if so.id in alive and alive[so.id].last_update == t:
                continue
            rel = (so.lo[:2] + so.hi[:2]) / 2 - pose[:2]
            R, th = float(np.hypot(*rel)), float(np.degrees(np.arctan2(rel[1], rel[0])))
            in_view = self.r_lim[0] + 0.1 < R < self.r_lim[1] - 0.1 and abs(th) < 50.0 and R >= so.r_seen
            if not in_view:
                continue
            near = any(np.all(p[:2] >= so.lo[:2] - 0.084) and np.all(p[:2] <= so.hi[:2] + 0.084) for p in pts)
            so.absent_s = 0.0 if near else so.absent_s + look_s
            if so.absent_s >= self.absent_s:
                self.static.pop(so.id)

    def _merge(self):
        ids = sorted(self.static)
        for i in ids:
            for j in ids:
                a, b = self.static.get(i), self.static.get(j)
                if a is None or b is None or i >= j or not a.points or not b.points:
                    continue
                size = np.maximum(a.hi[:2], b.hi[:2]) - np.minimum(a.lo[:2], b.lo[:2])
                ma, mb = label(a.classes, "material"), label(b.classes, "material")
                if ma and mb and ma != mb:
                    continue
                if np.all(a.lo[:2] <= b.hi[:2]) and np.all(b.lo[:2] <= a.hi[:2]) and np.all(size <= self.extent):
                    a.points.extend(b.points)
                    a.classes = a.classes or b.classes
                    a.last_seen = max(a.last_seen, b.last_seen)
                    a.r_seen = min(a.r_seen, b.r_seen)
                    a.refit()
                    self.static.pop(j)


def label(classes: dict, head: str):
    return ((classes or {}).get(head) or {}).get("label")


def polar_box_xy(r_lo, r_hi, th_lo, th_hi) -> tuple[np.ndarray, np.ndarray]:
    th = np.linspace(th_lo, th_hi, 9)
    if th_lo < 0 < th_hi:
        th = np.r_[th, 0.0]
    r = np.array([max(r_lo, 0.0), max(r_hi, 0.0)])
    x = (r[:, None] * np.cos(th)[None]).ravel()
    y = (r[:, None] * np.sin(th)[None]).ravel()
    return np.array([x.min(), y.min()]), np.array([x.max(), y.max()])


def selftest() -> bool:
    import paths
    from types import SimpleNamespace
    cfg = paths.load_config()
    w = World(cfg)
    rng = np.random.default_rng(0)
    pts = [(1.65 + rng.normal(0, .005), 0.48 + rng.normal(0, .005), 0.0, .01, .01, .05, 0.0, f) for f in range(20) if f % 10]
    pts.append((1.0, -0.6, 0.0, .01, .01, .05, 0.0, 7))
    ref = SimpleNamespace(static_pts=np.array(pts), x=np.zeros(20), y=np.zeros(20), z=np.zeros(20))
    n = w.set_known(ref)
    ok1 = n == 1 and abs(w.known[0].lo[0] - 1.62) < 0.05 and w.known[0].hi[1] > 0.48
    t = np.arange(0, 30, 0.1)
    ball = [(ti, 1.55, 0.14 * np.sin(np.pi * ti), True) for ti in t]
    walk = [(ti, 2.0, -2.0 + 1.2 * abs(ti % 6.0 - 3.0), False) for ti in t]
    rb, okb = w._reach(ball, 1.0, 0.1)
    rw, okw = w._reach(walk, 1.0, 0.1)
    rs, oks = w._reach(ball[:10], 1.0, 0.1)
    ok2 = okb and okw and rb <= 0.29 and 1.0 < rw <= 1.5 and not oks and rs == w.v_max * 1.0
    print(f"  {'PASS' if ok1 and ok2 else 'FAIL'}  known objects: {n} (the stand; one false detection left out); reach in 1 s: "
          f"swinging ball {rb:.2f} m, walker {rw:.2f} m (bound {w.v_max:.1f} m), 1 s of looks: {rs:.1f} m, calibrated {oks}")
    return ok1 and ok2


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
