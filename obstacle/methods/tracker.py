"""
tracker.py - multi-object tracking with KalmanCQR on the FDR-controlled detections: roles of the echoes (core, part,
static, ghost, birth), M-of-N confirmation, deletion after unseen time (shared with obstacle/).

    from methods.tracker import Tracker
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from methods.kalman_cqr import EKF, ConformalRegion, wrap_v

HANN_SIDELOBE_DB = 31.5


def entropy_bits(P2: np.ndarray) -> float:
    return 0.5 * np.log2((2 * np.pi * np.e) ** 2 * max(np.linalg.det(P2), 1e-30))


@dataclass
class Track:
    id: int
    kf: EKF
    t: float
    born: float
    status: str = "tentative"
    hits: list = field(default_factory=list)
    last_update: float = 0.0
    n_updates: int = 0
    snr_db: float = 0.0
    snr_mean: float = 0.0
    snr_var: float = 9.0
    info_bits: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=64))
    pred: dict = field(default_factory=dict)
    parts: list = field(default_factory=list)
    features: deque = field(default_factory=lambda: deque(maxlen=200))
    region: ConformalRegion | None = None
    n_scat: float = 1.0
    last_moving: float = 0.0
    last_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    meas: deque = field(default_factory=lambda: deque(maxlen=200))
    r_extra: np.ndarray = field(default_factory=lambda: np.zeros(3))
    unseen: float = 0.0
    vr_meas: deque = field(default_factory=lambda: deque(maxlen=20))
    classes: dict = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"


class Tracker:
    def __init__(self, cfg: dict, doppler_sign="auto", birth_static: bool = False):
        self.r_lim = (cfg["radar"]["r_min_m"], cfg["radar"]["r_max_m"])
        tc = cfg["tracking"]
        self.alpha = tc["alpha"]
        self.sigma_a = tc["accel_sigma_mps2"]
        self.v_init = tc.get("max_speed_mps", 2.0)
        self.confirm_hits, self.confirm_window = tc["confirm_hits"], tc["confirm_window"]
        self.lost_after = tc["lost_after_s"]
        self.horizons = list(tc["horizons_s"])
        self.ghost_db = tc.get("ghost_db", 17.8)
        self.extent = tc["object_extent_m"]
        self.reacquire = tc.get("reacquire", True)
        self.rayleigh_u = 1.0 / 6.0
        self.births_log = deque(maxlen=400)
        mk = lambda: ConformalRegion(tc["alpha"], tc["score_window"], tc["aci_gamma"], 3)
        self.regions = {"step": mk(), **{h: mk() for h in self.horizons}}
        self.tracks: list[Track] = []
        self.next_id = 1
        self.doppler_sign = 0 if doppler_sign == "auto" else int(doppler_sign)
        self._sign_votes = []
        self.sign_check = deque(maxlen=500)
        self._near = {}
        self.recent = deque(maxlen=400)
        self.birth_static = birth_static
        self.t_prev = None
        self.t_first = None

    def _measurement_box(self, tr: Track, s, P, region: ConformalRegion, wf, v_ego):
        z, H = EKF.h(s, v_ego)
        Rm = np.diag(np.square(tr.pred.get("sig_meas", self._sig(wf, tr.snr_db, z[1]))) + tr.r_extra)
        S = H @ P @ H.T + Rm
        sig = np.sqrt(np.diag(S))
        lo, hi, ok = region.box(z, sig)
        return z, H, S, sig, lo, hi, ok

    def _sig(self, wf, snr_db, theta):
        return np.array(wf.sigmas(10 ** (snr_db / 10), theta))

    def predict(self, t: float, wf, ego: dict | None = None) -> list[dict]:
        ego = ego or {}
        v_e, dpsi = ego.get("v", (0.0, 0.0)), ego.get("dpsi", 0.0)
        out = []
        for tr in self.tracks:
            dt = max(t - tr.t, 0.0)
            s, P = tr.kf.predicted(dt, self.sigma_a, v_e, dpsi)
            z, H, S, sig, lo, hi, ok = self._measurement_box(tr, s, P, self._region_for(dt, wf, tr), wf, v_e)
            if self.doppler_sign == 0:
                lo[2], hi[2] = -np.inf, np.inf
            tr.pred = {"t": t, "s": s, "P": P, "z": z, "H": H, "S": S, "sig": sig, "lo": lo, "hi": hi,
                       "calibrated": ok, "v_e": v_e, "sig_meas": tr.pred.get("sig_meas", self._sig(wf, tr.snr_db, z[1]))}
            if tr.confirmed:
                out.append({"track": tr.id, "r_lo": lo[0], "r_hi": hi[0],
                            "u_lo": np.sin(np.clip(lo[1], -1.5, 1.5)), "u_hi": np.sin(np.clip(hi[1], -1.5, 1.5)),
                            "v_lo": lo[2], "v_hi": hi[2], "v_span": 2 * wf.v_max, "u": float(np.sin(z[1]))})
        return out

    def _region_for(self, dt: float, wf, tr: Track | None = None) -> ConformalRegion:
        if dt <= 1.5 * wf.frame_period_s or not self.horizons:
            if tr is not None and tr.region is not None and tr.region.fit()[3]:
                return tr.region
            return self.regions["step"]
        h = min(self.horizons, key=lambda x: abs(x - dt))
        return self.regions[h] if self.regions[h].fit()[3] else self.regions["step"]

    def ahead(self, tr: Track, h: float, wf, ego: dict | None = None) -> dict:
        ego = ego or {}
        s, P = tr.kf.predicted(h, self.sigma_a, ego.get("v", (0.0, 0.0)), 0.0)
        reg = self.regions.get(h) or self.regions["step"]
        z, H, S, sig, lo, hi, ok = self._measurement_box(tr, s, P, reg, wf, ego.get("v", (0.0, 0.0)))
        return {"s": s, "P": P, "z": z, "lo": lo, "hi": hi, "calibrated": ok}

    def lam(self, t: float, memory_s: float = 3.0) -> float:
        rec = [1 for (ti, inside) in self.recent if t - ti <= memory_s]
        alive = {tr.id: tr for tr in self.tracks}
        new = sum(1 for (ti, k) in self.births_log if t - ti <= memory_s and k in alive and alive[k].confirmed)
        return (new + 1) / (len(rec) + 2)

    def update(self, dets: list[dict], t: float, wf, ego: dict | None = None) -> dict:
        ego = ego or {}
        v_e = np.asarray(ego.get("v", (0.0, 0.0)), float)
        self.t_first = t if self.t_first is None else self.t_first
        self._range_res = wf.range_res_m
        if any(tr.pred.get("t") != t for tr in self.tracks):
            self.predict(t, wf, ego)
        n_det = len(dets)
        for d in dets:
            d["label"], d["track"] = "unassigned", -1
        self._flag_ghosts(dets, wf)
        room = {i for i, d in enumerate(dets) if self._residual(d, v_e, wf)}

        pairs = []
        for tr in self.tracks:
            c, lo, hi = tr.pred["z"], tr.pred["lo"], tr.pred["hi"]
            gate = 2.0 if tr.confirmed else 1.0
            reach = self.v_init * max(t - tr.last_update, wf.frame_period_s) + self.extent / 2
            for i, d in enumerate(dets):
                if d["label"] == "ghost":
                    continue
                z = self._z(d, tr, wf)
                if self._box_dist(z, c, lo, hi) <= gate and np.linalg.norm(self._xy(d) - tr.last_xy) <= reach:
                    pairs.append((self._cost(z, d, tr), -d["snr_db"], i, tr))
        for tr in self.tracks:
            tr.parts = []
        taken_d, taken_t = set(), set()
        for dist, _, i, tr in sorted(pairs, key=lambda q: (q[0], q[1])):
            if i in taken_d or tr.id in taken_t:
                continue
            taken_d.add(i)
            taken_t.add(tr.id)
            dets[i]["label"], dets[i]["track"] = "core", tr.id
            tr.parts.append((i, "core", dist))

        for tr in self.tracks:
            if not self.reacquire:
                break
            if not tr.confirmed or tr.id in taken_t or tr.n_updates < 3 or t - tr.last_update > 2.5 * wf.frame_period_s:
                continue
            reach = self.v_init * (t - tr.last_update) + self.extent / 2
            s_snr = max(np.sqrt(tr.snr_var), 3.0)
            cand = []
            for i, d in enumerate(dets):
                if i in taken_d or i in room or d["label"] == "ghost" or abs(d["snr_db"] - tr.snr_mean) > 3 * s_snr:
                    continue
                dist = float(np.linalg.norm(self._xy(d) - tr.last_xy))
                if dist <= reach:
                    cand.append((dist, i))
            if cand:
                dist, i = min(cand)
                taken_d.add(i)
                taken_t.add(tr.id)
                dets[i]["label"], dets[i]["track"] = "core", tr.id
                tr.parts.append((i, "core", np.inf))

        for i, d in enumerate(dets):
            if d["label"] != "unassigned" or i in room:
                continue
            moving = self._moving(d, v_e, wf)
            xy = self._xy(d)
            best_tr, best_dist = None, np.inf
            for tr in self.tracks:
                c, lo, hi = tr.pred["z"], tr.pred["lo"], tr.pred["hi"]
                dist = float(np.linalg.norm(xy - tr.pred["s"][:2]))
                in_gate = self._box_dist(self._z(d, tr, wf)[:2], c[:2], lo[:2], hi[:2]) <= 2.0
                reach = self.extent if moving else self.extent / 2
                limbs = tr.n_scat > 1.5 and self._may_have_limbs(tr)
                if (in_gate or (limbs and dist <= reach)) and dist < best_dist:
                    best_tr, best_dist = tr, dist
            if best_tr is not None:
                d["label"], d["track"] = "part", best_tr.id
                best_tr.parts.append((i, "part" if moving else "near", best_dist))

        info_frame, scored = 0.0, 0
        for tr in self.tracks:
            core = [i for i, kind, _ in tr.parts if kind == "core"]
            if core:
                n_obj = sum(1 for _, kind, _ in tr.parts if kind != "near")
                tr.n_scat += (n_obj - tr.n_scat) / min(tr.n_updates + 1, 20)
                if any(self._moving(dets[i], v_e, wf) for i in core):
                    tr.last_moving = t
            evidence = any(i not in room for i in core)
            if tr.status == "tentative":
                tr.hits.append(1 if evidence else 0)
            if not core:
                tr.kf.s, tr.kf.P, tr.t = tr.pred["s"], tr.pred["P"], t
                tr.unseen += wf.frame_period_s
                continue
            tr.unseen = 0.0 if evidence else tr.unseen + wf.frame_period_s
            info, missed = self._kalman_update(tr, [dets[i] for i in core], t, wf, v_e)
            info_frame += info
            scored += missed is not None
            for i in core:
                dets[i]["inside_region"] = not missed

        births = self._births(dets, t, wf, v_e)

        for tr in self.tracks:
            if tr.status == "tentative" and len(tr.hits) >= self.confirm_window:
                tr.status = "confirmed" if sum(tr.hits[:self.confirm_window]) >= self.confirm_hits else "dead"
            elif tr.status == "tentative" and len(tr.hits) - sum(tr.hits) > self.confirm_window - self.confirm_hits:
                tr.status = "dead"
            if tr.confirmed and tr.unseen > self.lost_after:
                tr.status = "dead"
            if tr.region is not None and len(tr.region.misses) >= 30:
                recent = [m for m, _ in tr.region.misses[-30:]]
                if np.mean(recent) >= min(5 * self.alpha, 0.5):
                    tr.status = "dead"
            r = float(np.hypot(tr.kf.s[0], tr.kf.s[1]))
            if r > self.r_lim[1] + self.extent or r < max(self.r_lim[0] - self.extent, 0.0):
                tr.status = "dead"
            if tr.kf.s[0] <= 0 or abs(np.degrees(np.arctan2(tr.kf.s[1], tr.kf.s[0]))) > 65.0:
                tr.status = "dead"
        self._merge(t)
        dead = [tr.id for tr in self.tracks if tr.status == "dead"]
        self.tracks = [tr for tr in self.tracks if tr.status != "dead"]
        for d in dets:
            if d["label"] in ("core", "part", "birth", "unassigned"):
                inside = d["label"] in ("core", "part") and any(tr.id == d["track"] and tr.confirmed for tr in self.tracks)
                self.recent.append((t, inside))
        self.t_prev = t
        return {"info_bits": info_frame, "births": births, "deleted": dead, "n_det": n_det,
                "doppler_sign": self.doppler_sign,
                "doppler_agreement": float(np.mean(self.sign_check)) if self.sign_check else None}

    def _z(self, d: dict, tr: Track, wf) -> np.ndarray:
        v = d["v_mps"] * (self.doppler_sign or 1)
        vp = tr.pred["z"][2]
        return np.array([d["range_m"], np.radians(d["theta_deg"]), vp + wrap_v(v - vp, wf.v_max)])

    @staticmethod
    def _xy(d) -> np.ndarray:
        return d["range_m"] * np.array([np.cos(np.radians(d["theta_deg"])), np.sin(np.radians(d["theta_deg"]))])

    def _moving(self, d, v_e, wf) -> bool:
        u = np.array([np.cos(np.radians(d["theta_deg"])), np.sin(np.radians(d["theta_deg"]))])
        return abs(d["v_mps"] * (self.doppler_sign or 1) + float(np.asarray(v_e) @ u)) > 2 * wf.dv

    def _cost(self, z, d, tr: Track) -> float:
        c, lo, hi = tr.pred["z"], tr.pred["lo"], tr.pred["hi"]
        hw = np.where(z >= c, hi - c, c - lo)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(np.isfinite(hw), (z - c) / np.maximum(hw, 1e-9), 0.0)
        cost = float(np.sum(r ** 2))
        fin = np.isfinite(hw) & (hw > 0)
        cost += float(np.sum(2 * np.log(hw[fin])))
        if tr.n_updates >= 3:
            cost += ((d["snr_db"] - tr.snr_mean) / max(np.sqrt(tr.snr_var), 3.0)) ** 2
        return cost

    @staticmethod
    def _box_dist(z, c, lo, hi) -> float:
        hw = np.where(z >= c, hi - c, c - lo)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(z - c) / np.maximum(hw, 1e-9)
        r = np.where(np.isfinite(hw), r, 0.0)
        return float(np.max(r))

    def _flag_ghosts(self, dets, wf):
        n = len(dets)
        for i in range(n):
            a = dets[i]
            for j in range(n):
                if i == j:
                    continue
                b = dets[j]
                diff = b["snr_db"] - a["snr_db"]
                if diff <= 0:
                    continue
                if tuple(a["cell"]) == tuple(b["cell"]) and diff > self.ghost_db:
                    a["label"] = "ghost"
                    break
                same_angle = abs(a["u"] - b["u"]) <= self.rayleigh_u / 2
                same_range = abs(a["cell"][0] - b["cell"][0]) <= 1
                same_dopp = abs(a["cell"][1] - b["cell"][1]) <= 1
                if same_angle and (same_range or same_dopp) and diff > HANN_SIDELOBE_DB:
                    a["label"] = "ghost"
                    break

    def _measurement(self, ds: list[dict], tr: Track, wf):
        w = np.array([10 ** (d["snr_db"] / 10) for d in ds])
        w = w / w.sum()
        zs = np.array([self._z(d, tr, wf) for d in ds])
        xy = np.c_[zs[:, 0] * np.cos(zs[:, 1]), zs[:, 0] * np.sin(zs[:, 1])]
        c = w @ xy
        r, th = float(np.hypot(*c)), float(np.arctan2(c[1], c[0]))
        v = float(w @ zs[:, 2])
        snr = max(d["snr_db"] for d in ds)
        sig = self._sig(wf, snr, th)
        if len(ds) > 1:
            spread = np.sqrt(np.c_[(zs[:, 0] - r) ** 2, (zs[:, 1] - th) ** 2, (zs[:, 2] - v) ** 2].T @ w)
            sig = np.sqrt(sig ** 2 + spread ** 2 / len(ds))
        return np.array([r, th, v]), sig, snr

    def _kalman_update(self, tr: Track, ds, t, wf, v_e):
        p = tr.pred
        z, sig_m, snr = self._measurement(ds, tr, wf)
        use = [0, 1, 2] if self.doppler_sign != 0 else [0, 1]
        Rm = np.diag(sig_m ** 2 + tr.r_extra)
        H, zp = p["H"], p["z"]
        S = H @ p["P"] @ H.T + Rm
        nu = z - zp
        nu[2] = wrap_v(nu[2], wf.v_max)
        sig = np.sqrt(np.diag(S))
        missed = None
        if True:
            inside = bool(np.all((z[use] >= p["lo"][use]) & (z[use] <= p["hi"][use])))
            missed = not inside
            u = nu / sig
            if self.doppler_sign == 0:
                u[2] = 0.0
            self.regions["step"].add(u, sig, missed)
            if tr.region is None:
                tr.region = ConformalRegion(self.alpha, self.regions["step"].window, self.regions["step"].gamma, 3)
            tr.region.add(u, sig, missed)
        a = 1.0 / min(tr.n_updates + 1, 20)
        excess = nu ** 2 - np.diag(H @ p["P"] @ H.T) - sig_m ** 2
        tr.r_extra = np.maximum((1 - a) * tr.r_extra + a * excess, 0.0)
        Hu, Su, nuu = H[use], S[np.ix_(use, use)], nu[use]
        K = p["P"] @ Hu.T @ np.linalg.inv(Su)
        s_new = p["s"] + K @ nuu
        P_new = (np.eye(4) - K @ Hu) @ p["P"]
        P_new = 0.5 * (P_new + P_new.T)
        info = 0.5 * np.log2(max(np.linalg.det(p["P"][:2, :2]), 1e-30) / max(np.linalg.det(P_new[:2, :2]), 1e-30))
        self._score_horizons(tr, z, t, wf, v_e)
        self._vote_sign(tr, z)
        tr.kf.s, tr.kf.P, tr.t = s_new, P_new, t
        tr.last_xy = s_new[:2].copy()
        tr.meas.append((t, np.array([z[0] * np.cos(z[1]), z[0] * np.sin(z[1])])))
        tr.vr_meas.append(abs(float(z[2])))
        tr.last_update, tr.n_updates, tr.snr_db = t, tr.n_updates + 1, snr
        a = 1.0 / min(tr.n_updates, 20)
        d_snr = snr - tr.snr_mean
        tr.snr_mean += a * d_snr
        tr.snr_var = (1 - a) * (tr.snr_var + a * d_snr ** 2)
        tr.info_bits += info
        tr.pred["sig_meas"] = sig_m
        tr.history.append((t, s_new.copy(), P_new.copy()))
        return float(info), missed

    def _score_horizons(self, tr, z, t, wf, v_e):
        for h in self.horizons:
            cand = [(abs(t - ti - h), ti, s, P) for (ti, s, P) in tr.history]
            if not cand:
                continue
            gap, ti, s, P = min(cand, key=lambda c: c[0])
            if gap > 0.5 * wf.frame_period_s + 1e-9:
                continue
            sp, Pp = EKF(s, P).predicted(t - ti, self.sigma_a, v_e, 0.0)
            zp, H = EKF.h(sp, v_e)
            S = H @ Pp @ H.T + np.diag(tr.pred["sig_meas"] ** 2)
            sig = np.sqrt(np.diag(S))
            nu = z - zp
            nu[2] = wrap_v(nu[2], wf.v_max)
            lo, hi, ok = self.regions[h].box(zp, sig)
            use = [0, 1, 2] if self.doppler_sign != 0 else [0, 1]
            u = nu / sig
            if self.doppler_sign == 0:
                u[2] = 0.0
            self.regions[h].add(u, sig, not bool(np.all((z[use] >= lo[use]) & (z[use] <= hi[use]))))

    def _vote_sign(self, tr, z):
        if not tr.confirmed or tr.n_updates < 5:
            return
        if self.doppler_sign != 0:
            if len(tr.history) >= 6:
                (t0, s0, _), (t1, s1, _) = tr.history[-6], tr.history[-1]
                rdot_pos = (np.hypot(*s1[:2]) - np.hypot(*s0[:2])) / max(t1 - t0, 1e-6)
                if abs(rdot_pos) > 0.15:
                    self.sign_check.append(float(np.sign(rdot_pos) * np.sign(z[2])))
            return
        s, P = tr.pred["s"], tr.pred["P"]
        r = np.hypot(s[0], s[1])
        rdot = (s[0] * s[2] + s[1] * s[3]) / r
        e = s[:2] / r
        sd = np.sqrt(e @ P[2:, 2:] @ e)
        if abs(rdot) > 3 * sd and abs(rdot) > 0.15:
            self._sign_votes.append(np.sign(rdot) * np.sign(z[2]))
        if len(self._sign_votes) >= 15:
            m = float(np.mean(self._sign_votes))
            if abs(m) >= 0.6:
                self.doppler_sign = 1 if m > 0 else -1
                print(f"  [tracker] Doppler sign from data: {self.doppler_sign:+d} (agreement {m:+.2f} over "
                      f"{len(self._sign_votes)} updates); the radial speed is used from now on", flush=True)
            self._sign_votes = self._sign_votes[-30:]

    def _residual(self, d, v_e, wf) -> bool:
        return not self._moving(d, v_e, wf) and d.get("change_db", np.inf) < 0.0

    def _births(self, dets, t, wf, v_e) -> list[int]:
        seeds = []
        for i, d in enumerate(dets):
            if d["label"] != "unassigned":
                continue
            if self._residual(d, v_e, wf):
                d["label"] = "static"
                continue
            seeds.append(i)
        pool = seeds
        groups, used = [], set()
        for i in sorted(seeds, key=lambda i: -dets[i]["snr_db"]):
            if i in used:
                continue
            g = [j for j in pool if j not in used and self._dist(dets[i], dets[j]) <= self.extent]
            used.update(g)
            groups.append(sorted(g, key=lambda j: -dets[j]["snr_db"]))
        new = []
        for g in groups:
            d = dets[g[0]]
            th = np.radians(d["theta_deg"])
            sr, st, sv = wf.sigmas(10 ** (d["snr_db"] / 10), th)
            vr = d["v_mps"] * (self.doppler_sign or 1)
            kf = EKF.from_measurement(d["range_m"], th, vr if self.doppler_sign else 0.0, sr, st,
                                      sv if self.doppler_sign else self.v_init, self.v_init)
            tr = Track(self.next_id, kf, t, t, last_update=t, snr_db=d["snr_db"], snr_mean=d["snr_db"], hits=[1], n_updates=1,
                       n_scat=float(len(g)), last_moving=t)
            tr.meas.append((t, self._xy(d)))
            tr.pred = {"sig_meas": np.array([sr, st, sv])}
            tr.history.append((t, kf.s.copy(), kf.P.copy()))
            tr.last_xy = kf.s[:2].copy()
            self.tracks.append(tr)
            for i in g:
                dets[i]["label"], dets[i]["track"] = "birth", tr.id
            new.append(tr.id)
            self.births_log.append((t, tr.id))
            self.next_id += 1
        return new

    @staticmethod
    def _dist(a, b) -> float:
        pa = a["range_m"] * np.array([np.cos(np.radians(a["theta_deg"])), np.sin(np.radians(a["theta_deg"]))])
        pb = b["range_m"] * np.array([np.cos(np.radians(b["theta_deg"])), np.sin(np.radians(b["theta_deg"]))])
        return float(np.linalg.norm(pa - pb))

    def _merge(self, t: float, window_s: float = 1.0):
        live = [tr for tr in self.tracks if tr.status != "dead" and len(tr.history) >= 3]
        for a in live:
            for b in live:
                if a.id >= b.id or a.status == "dead" or b.status == "dead":
                    continue
                ha = [(ti, s) for ti, s, _ in a.history if t - ti <= window_s]
                hb = [(ti, s) for ti, s, _ in b.history if t - ti <= window_s]
                if len(ha) < 3 or len(hb) < 3 or min(ha[-1][0], hb[-1][0]) - max(ha[0][0], hb[0][0]) < 0.6 * window_s:
                    continue
                if np.linalg.norm(a.kf.s[:2] - b.kf.s[:2]) > self.extent:
                    continue
                times = np.linspace(max(ha[0][0], hb[0][0]), min(ha[-1][0], hb[-1][0]), 5)
                pa = np.array([np.interp(times, [x[0] for x in ha], [x[1][k] for x in ha]) for k in (0, 1)]).T
                pb = np.array([np.interp(times, [x[0] for x in hb], [x[1][k] for x in hb]) for k in (0, 1)]).T
                close = np.all(np.linalg.norm(pa - pb, axis=1) <= self.extent)
                alike = np.linalg.norm((pa[-1] - pa[0]) - (pb[-1] - pb[0])) <= self.extent / 2
                still_a = np.mean([np.hypot(s[2], s[3]) for ti, s, _ in a.history if t - ti <= window_s]) <= self._range_res / window_s
                still_b = np.mean([np.hypot(s[2], s[3]) for ti, s, _ in b.history if t - ti <= window_s]) <= self._range_res / window_s
                if close and alike and still_a == still_b:
                    (b if b.n_updates <= a.n_updates else a).status = "dead"

    @staticmethod
    def _may_have_limbs(tr: Track) -> bool:
        label = ((tr.classes or {}).get("material") or {}).get("label")
        return label in (None, "waterlike")

    def snapshot(self) -> list[dict]:
        out = []
        for tr in self.tracks:
            P2 = tr.kf.P[:2, :2]
            p = tr.pred
            out.append({"id": tr.id, "status": tr.status, "x": float(tr.kf.s[0]), "y": float(tr.kf.s[1]),
                        "vx": float(tr.kf.s[2]), "vy": float(tr.kf.s[3]),
                        "range_m": float(np.hypot(tr.kf.s[0], tr.kf.s[1])),
                        "theta_deg": float(np.degrees(np.arctan2(tr.kf.s[1], tr.kf.s[0]))),
                        "P_pos": P2.round(8).tolist(), "entropy_bits": entropy_bits(P2), "info_bits": tr.info_bits,
                        "snr_db": tr.snr_db, "n_updates": tr.n_updates, "updated": bool(tr.last_update == tr.t),
                        "region": {"lo": np.asarray(p.get("lo", [np.nan] * 3)).tolist(),
                                   "hi": np.asarray(p.get("hi", [np.nan] * 3)).tolist(),
                                   "calibrated": bool(p.get("calibrated", False))},
                        "n_parts": sum(1 for _, k, _ in tr.parts if k == "part"), "classes": tr.classes,
                        "window": getattr(tr, "window", None), "extended": bool(tr.n_scat > 1.5),
                        "n_scat": round(float(tr.n_scat), 2)})
        return out
