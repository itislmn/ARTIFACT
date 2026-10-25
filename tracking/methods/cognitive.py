"""
cognitive.py - the chirp controller of the reconfigure mode (stop, new chirp over UART, restart): a change only when
the predicted entropy gain outweighs the measured blind gap.

    from methods.cognitive import Controller
"""
from __future__ import annotations

from collections import deque

import numpy as np

from methods.kalman_cqr import EKF

LOG2_2PIE = np.log2(2 * np.pi * np.e)


def _h_pos(P) -> float:
    return 0.5 * (2 * LOG2_2PIE + np.log2(max(np.linalg.det(P[:2, :2]), 1e-30)))


class Controller:
    def __init__(self, library: dict, cfg: dict, start: str):
        self.lib = library
        self.H = float(cfg["cognitive"]["horizon_s"])
        self.gaps = deque([float(cfg["cognitive"]["gap_s"])], maxlen=20)
        self.proc_ms = deque(maxlen=50)
        self.current = start
        self.t_switch = -np.inf
        self.history = []
        self._table, self._t_table = None, -np.inf
        self._bad = 0

    @property
    def gap_s(self) -> float:
        return float(np.median(self.gaps))

    def observe(self, proc_ms: float | None = None, gap_s: float | None = None):
        if proc_ms is not None:
            self.proc_ms.append(proc_ms)
        if gap_s is not None:
            self.gaps.append(gap_s)

    def mean_entropy(self, tracker, tr, w, blind: float) -> float:
        proc = float(np.median(self.proc_ms)) / 1000 if self.proc_ms else 0.0
        T = max(w.frame_period_s, proc)
        P = tr.kf.P.copy()
        z, H = EKF.h(tr.kf.s)
        sig = np.array(w.sigmas(10 ** (tr.snr_db / 10), z[1]))
        Rm = np.diag(sig ** 2 + tr.r_extra)
        use = [0, 1, 2] if tracker.doppler_sign != 0 else [0, 1]
        Hu, Ru = H[use], Rm[np.ix_(use, use)]
        acc, t = 0.0, 0.0
        if blind > 0:
            F, Q = EKF.F_Q(blind, tracker.sigma_a)
            P1 = F @ P @ F.T + Q
            acc += 0.5 * (_h_pos(P) + _h_pos(P1)) * blind
            P, t = P1, blind
        F, Q = EKF.F_Q(T, tracker.sigma_a)
        n, cycle = 0, 0.0
        while t < self.H - 1e-9 and n < 30:
            K = P @ Hu.T @ np.linalg.inv(Hu @ P @ Hu.T + Ru)
            Pp = (np.eye(4) - K @ Hu) @ P
            Pm = F @ Pp @ F.T + Q
            dt = min(T, self.H - t)
            cycle = 0.5 * (_h_pos(Pp) + _h_pos(Pm))
            acc += cycle * dt
            P, t, n = Pm, t + dt, n + 1
        if t < self.H:
            acc += cycle * (self.H - t)
        return acc / self.H

    def predictable(self, tracker) -> bool:
        hs = sorted(h for h in tracker.horizons if h >= self.gap_s) or sorted(tracker.horizons)[-1:]
        if not hs:
            return True
        reg = tracker.regions[hs[0]]
        return bool(reg.fit()[3] and len(reg.misses) >= 20 and reg.coverage(50) >= 1 - tracker.alpha)

    def evaluate(self, tracker) -> dict:
        conf = [tr for tr in tracker.tracks if tr.confirmed and tr.n_updates >= 10]
        proc = float(np.median(self.proc_ms)) / 1000 if self.proc_ms else 0.0
        U, table = {}, {}
        for name, w in self.lib.items():
            ok, why = True, ""
            for tr in conf:
                s, P = tr.kf.s, tr.kf.P
                r = float(np.hypot(s[0], s[1]))
                if r + 3 * np.sqrt(np.trace(P[:2, :2]) / 2) > w.max_range_m:
                    ok, why = False, f"track {tr.id} beyond R_max {w.max_range_m:.1f} m"
                vr_max = max(tr.vr_meas) if tr.vr_meas else 0.0
                if vr_max + w.dv > w.v_max:
                    ok, why = False, f"track {tr.id}: radial speed near v_max {w.v_max:.1f} m/s"
            blind = 0.0 if name == self.current else self.gap_s
            U[name] = sum(self.mean_entropy(tracker, tr, w, blind) for tr in conf)
            table[name] = {"valid": ok, "why": why, "revisit_s": max(w.frame_period_s, proc), "mb_per_s": w.mb_per_s,
                           "mean_entropy_bits": U[name]}
        for name in table:
            table[name]["gain_bits"] = (U[self.current] - U[name]) if conf else 0.0
        return table

    def decide(self, tracker, t: float):
        if self._table is None or t - self._t_table >= 0.5 or t < self._t_table:
            self._table, self._t_table = self.evaluate(tracker), t
        table = self._table
        if len(self.proc_ms) < 10:
            return None, table
        cur = table[self.current]
        valid = {n: v for n, v in table.items() if v["valid"]}
        self._bad = self._bad + 1 if not cur["valid"] else 0
        choice = None
        if valid:
            if self._bad >= 3:
                choice = max(valid, key=lambda n: (valid[n]["gain_bits"], -valid[n]["mb_per_s"]))
            elif t - self.t_switch >= self.H and self.predictable(tracker):
                others = {n: v for n, v in valid.items() if n != self.current}
                if others:
                    best = max(others, key=lambda n: (others[n]["gain_bits"], -others[n]["mb_per_s"]))
                    idle = all(abs(v["gain_bits"]) < 1e-12 for v in table.values())
                    if others[best]["gain_bits"] > 0 or (idle and others[best]["mb_per_s"] < cur["mb_per_s"]):
                        choice = best
        if choice is not None and choice != self.current:
            self.history.append({"t": t, "from": self.current, "to": choice,
                                 "gain_bits": {n: round(v["gain_bits"], 2) for n, v in table.items()}})
            return choice, table
        return None, table

    def switched(self, name: str, t: float, gap_s: float | None = None):
        self.current, self.t_switch = name, t
        self._table = None
        if gap_s is not None:
            self.gaps.append(gap_s)
