"""
chirp_choice.py - the cognitive chirp choice: of the preloaded sub-frames, the plan whose newest result has the
lowest predicted position entropy over the next horizon, processing time and latency counted; an extra look only if it
gains at least one bit (halves the predicted position area) (shared with obstacle/).

    from methods.chirp_choice import ChirpChoice      (python -m methods.chirp_choice: self-tests)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from methods.kalman_cqr import EKF

POLICIES = ("cognitive", "search", "all")
LOG2_2PIE = math.log2(2 * math.pi * math.e)
MIN_GAIN_BITS = 1.0
GRID_S = 0.005


@dataclass(frozen=True)
class SubFrame:
    name: str
    wf: object
    start_s: float
    active_s: float


def pick_looks(items, main: str, wanted, t_of, name_of) -> list:
    mains = [x for x in items if name_of(x) == main]
    pick = [max(mains, key=t_of)] if mains else []
    t0 = t_of(pick[0]) if pick else -math.inf
    for n in wanted:
        if n == main:
            continue
        later = [x for x in items if name_of(x) == n and t_of(x) > t0]
        if later:
            pick.append(max(later, key=t_of))
    return sorted(pick, key=t_of)


class ChirpChoice:
    def __init__(self, subframes: list[SubFrame], frame_s: float, policy: str, sigma_a: float, horizon_s: float):
        if policy not in POLICIES:
            raise ValueError(f"look_policy {policy!r}: one of {POLICIES}")
        self.sf, self.frame_s, self.policy = list(subframes), float(frame_s), policy
        self.sigma_a, self.H = float(sigma_a), float(horizon_s)
        self.names = [s.name for s in self.sf]
        self.main = self.names[0]
        extra = self.names[1:]
        self.plans = [(self.main,) + c for k in range(len(extra) + 1) for c in combinations(extra, k)]
        self.proc = {n: deque(maxlen=30) for n in self.names}
        self._cost, self._FQ, self._last = {}, {}, None

    def observe(self, name: str, proc_s: float):
        if name in self.proc:
            self.proc[name].append(proc_s)

    def proc_s(self, name: str) -> float:
        if self.proc[name]:
            return float(np.median(self.proc[name]))
        others = [v for n in self.names[1:] for v in self.proc[n]] if name != self.main else []
        for vals in (others, self.proc[self.main]):
            if vals:
                return float(np.median(vals))
        return self.frame_s / len(self.names)

    def decide(self, tracks, doppler_sign: int = 1, after: str | None = None) -> tuple[list[str], str, dict]:
        if after not in (None, self.main) and self._last is not None:
            return self._last
        self._last = self._decide(tracks, doppler_sign)
        return self._last

    def _decide(self, tracks, doppler_sign: int) -> tuple[list[str], str, dict]:
        if len(self.names) == 1:
            return [self.main], "single chirp armed: nothing to choose", {}
        if self.policy == "search":
            return [self.main], "policy: search chirp only", {}
        if self.policy == "all":
            return list(self.names), "policy: every sub-frame", {}
        tracks = [tr for tr in tracks if np.all(np.isfinite(tr.kf.P))]
        if not tracks:
            return [self.main], "no object: the search chirp alone", {}
        self._cost = {n: self.proc_s(n) for n in self.names}
        res = {plan: self.uncertainty(plan, tracks, doppler_sign) for plan in self.plans}
        base = res[(self.main,)][0]
        u_min = min(u for u, _, _ in res.values())
        best = min((p for p in self.plans if res[p][0] <= u_min + MIN_GAIN_BITS), key=lambda p: (len(p), res[p][0]))
        table = {"+".join(p): {"gain_bits": round(base - u, 3), "looks_per_s": round(n, 1), "latency_ms": round(lat, 1)}
                 for p, (u, n, lat) in res.items()}
        why = (f"{'+'.join(best)}: {base - res[best][0]:+.2f} bit vs the search chirp alone "
               f"({res[best][1]:.0f} looks/s, latency {res[best][2]:.0f} ms)")
        return list(best), why, table

    def schedule(self, plan) -> list[tuple[float, float, str]]:
        n_frames = int(math.ceil(self.H / self.frame_s)) + 1
        looks = sorted((k * self.frame_s + s.start_s + s.active_s, s.name, k * self.frame_s + s.start_s + s.active_s / 2)
                       for k in range(n_frames) for s in self.sf if s.name in plan)
        done, i, free = [], 0, 0.0
        while i < len(looks):
            now = max(free, looks[i][0])
            j = i
            while j < len(looks) and looks[j][0] <= now:
                j += 1
            batch, i = looks[i:j], j
            for t_ready, name, t_mid in pick_looks(batch, self.main, plan, lambda x: x[0], lambda x: x[1]):
                now = max(now, t_ready) + (self._cost.get(name) or self.proc_s(name))
                done.append((t_mid, now, name))
            free = now
        return done

    def uncertainty(self, plan, tracks, doppler_sign: int) -> tuple[float, float, float]:
        wf = {s.name: s.wf for s in self.sf}
        P0 = np.stack([tr.kf.P for tr in tracks])
        use = [0, 1, 2] if doppler_sign != 0 else [0, 1]
        zH = [EKF.h(tr.kf.s) for tr in tracks]
        Hs = np.stack([H[use] for _, H in zH])
        Rs, ok = {}, {}
        for n in plan:
            R, v = [], []
            for tr, (z, _) in zip(tracks, zH):
                sig = np.array(wf[n].sigmas(10 ** (tr.snr_db / 10), z[1]))
                R.append(np.diag(sig ** 2 + tr.r_extra)[np.ix_(use, use)])
                reach = z[0] + 3 * math.sqrt(max(np.trace(tr.kf.P[:2, :2]) / 2, 0.0))
                vr = max(tr.vr_meas) if tr.vr_meas else 0.0
                v.append(reach < wf[n].max_range_m and vr + wf[n].dv < wf[n].v_max)
            Rs[n], ok[n] = np.stack(R), np.array(v)
        done = self.schedule(plan)
        posts, P, t_prev = [], P0, 0.0
        Ht = np.swapaxes(Hs, 1, 2)
        for t_mid, t_fin, n in done:
            P = self._predict(P, t_mid - t_prev)
            t_prev = t_mid
            S = Hs @ P @ Ht + Rs[n]
            K = P @ Ht @ np.linalg.inv(S)
            Pu = P - K @ Hs @ P
            P = np.where(ok[n][:, None, None], Pu, P)
            posts.append((t_mid, t_fin, P))
        grid = np.arange(GRID_S / 2, self.H, GRID_S)
        fin = np.array([f for _, f, _ in posts]) if posts else np.zeros(0)
        k = np.searchsorted(fin, grid, side="right") - 1
        Pk = np.stack([P0] + [p for _, _, p in posts])[k + 1]
        t0 = np.r_[0.0, [m for m, _, _ in posts]][k + 1]
        dt = (grid - t0)[:, None]
        pos = (Pk[..., :2, :2] + dt[..., None, None] * (Pk[..., :2, 2:] + Pk[..., 2:, :2])
               + (dt ** 2)[..., None, None] * Pk[..., 2:, 2:]
               + (self.sigma_a ** 2 * dt ** 4 / 4)[..., None, None] * np.eye(2))
        det = np.maximum(pos[..., 0, 0] * pos[..., 1, 1] - pos[..., 0, 1] * pos[..., 1, 0], 1e-30)
        U = float(np.sum(np.mean(0.5 * (2 * LOG2_2PIE + np.log2(det)), axis=0)))
        lat = [1e3 * (f - m) for m, f, _ in done]
        n_in = sum(1 for _, f, _ in done if f <= self.H)
        return U, n_in / self.H, float(np.median(lat)) if lat else 0.0

    def _predict(self, P, dt: float):
        if dt <= 0:
            return P
        key = round(dt, 6)
        if key not in self._FQ:
            self._FQ[key] = EKF.F_Q(dt, self.sigma_a)
        F, Q = self._FQ[key]
        return F @ P @ F.T + Q


def selftest() -> bool:
    from types import SimpleNamespace as NS
    lam, n_loops, n_az = 3.0e8 / 78.39e9, 128, 12

    def chirp(idle_us):
        loop = 4 * (idle_us + 36.0) * 1e-6
        dv, v_max = lam / (2 * n_loops * loop), lam / (4 * loop)

        def sigmas(snr, th, dv=dv):
            sb = math.sqrt(1.5 * 6) / (2 * math.pi) / math.sqrt(max(snr, 1.0))
            su = math.sqrt(6.0 / ((2 * math.pi * 0.5) ** 2 * max(snr, 1.0) * (n_az ** 2 - 1)))
            return 0.084 * max(sb, 0.05), su / max(math.cos(th), 0.2), dv * max(sb, 0.05)
        return NS(dv=dv, v_max=v_max, max_range_m=9.6, sigmas=sigmas), n_loops * loop

    (s_wf, s_act), (f_wf, f_act) = chirp(26.5), chirp(10.0)
    frame = 0.105
    slack = (frame - 2 * s_act - f_act) / 3
    sf = [SubFrame("search", s_wf, 0.0, s_act), SubFrame("fast", f_wf, s_act + slack, f_act),
          SubFrame("track", s_wf, s_act + f_act + 2 * slack, s_act)]

    def walker(vr):
        P = np.diag([0.03 ** 2, 0.03 ** 2, 0.3 ** 2, 0.3 ** 2])
        return NS(kf=EKF(np.array([1.5, 0.0, vr, 0.0]), P), snr_db=25.0, r_extra=np.zeros(3), vr_meas=deque([abs(vr)]))

    def run(proc_ms, tracks):
        cc = ChirpChoice(sf, frame, "cognitive", 4.9, 1.0)
        cost = proc_ms if isinstance(proc_ms, dict) else {n: proc_ms for n in cc.names}
        for n, v in cost.items():
            cc.observe(n, v / 1e3)
        return cc.decide(tracks, 1)

    cases = [("no object", run(30, []), lambda p: p == ["search"]),
             ("walker, 30 ms per look (3 fit; one extra look gains > 1 bit, the third < 1 bit more)",
              run(30, [walker(1.0)]), lambda p: len(p) == 2),
             ("walker, 45 ms per look (2 fit, but the second gains < 1 bit)", run(45, [walker(1.0)]),
              lambda p: p == ["search"]),
             ("object at 4.5 m/s (> 3.8 m/s of search), 45 ms", run(45, [walker(4.5)]), lambda p: "fast" in p),
             ("walker, 120 ms per look (not even 1 fits)", run(120, [walker(1.0)]), lambda p: p == ["search"]),
             ("search 52 ms, track 26 ms, fast never processed (costs what track costs; the third look < 1 bit more)",
              run({"search": 52, "track": 26}, [walker(1.0)]), lambda p: len(p) == 2)]
    ok = True
    for name, (plan, why, table), check in cases:
        good = check(plan)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: {'+'.join(plan)}  ({why})")
        for p, row in table.items():
            print(f"          {p:20s} {row['gain_bits']:+7.2f} bit  {row['looks_per_s']:5.1f} looks/s  latency {row['latency_ms']:5.0f} ms")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
