"""
cognitive_tracking.py - the method: one radar frame in (prediction, range-Doppler, Infini-Rex with weighted e-BH,
ego-motion, tracking, M2-SRC, chirp choice), tracked objects with prediction regions and classes out.

    ct = CognitiveTracking(cfg, library, references); rec = ct.step(frame, t)
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from methods.chirp_choice import ChirpChoice, SubFrame
from methods.cognitive import Controller
from methods.ego_motion import EgoMotion
from methods.infinirex_frame import FrameSelector, prior_weights
from methods.range_doppler import FrontEnd
from methods.tracker import Tracker


class CognitiveTracking:
    def __init__(self, cfg: dict, library: dict, references: dict, start: str | None = None,
                 doppler_sign="auto", material=None, compare: bool = False, birth_static: bool = False,
                 subframes: bool = False):
        self.cfg, self.lib, self.refs = cfg, library, references
        rc, dc = cfg["radar"], cfg["detection"]
        self.r_min, self.r_max = rc["r_min_m"], rc["r_max_m"]
        self.y_lim = (rc.get("y_min_m", -np.inf), rc.get("y_max_m", np.inf))
        self.alpha, self.pfa, self.cap = dc["alpha"], dc["gate_pfa"], dc["gates_per_frame"]
        self.tracker = Tracker(cfg, doppler_sign, birth_static)
        self.controller = Controller(library, cfg, start or rc.get("start_waveform", next(iter(library))))
        self.policy = chirp_choice(cfg, library) if subframes else None
        self.material = material
        self.compare = compare
        self._fe, self._sel = {}, {}
        self.set_waveform(self.controller.current)
        self.ego_est = EgoMotion(self.wf, cfg.get("ego", {}).get("enabled", True), cfg["tracking"]["max_speed_mps"])
        self.ego = {"v": (0.0, 0.0), "dpsi": 0.0, "moving": False}
        self.t_last = None

    def set_waveform(self, name: str, t: float | None = None, gap_s: float | None = None, revisit_s: float | None = None):
        self.wf = self.lib[name] if revisit_s is None else replace(self.lib[name], frame_ms=1e3 * revisit_s)
        if name not in self._fe:
            ref = self.refs.get(name) or self.refs.get("*") or {}
            self._fe[name] = FrontEnd(self.lib[name], self.r_min, self.r_max, ref.get("cal"), ref.get("bg"))
            self._sel[name] = FrameSelector(self._fe[name].array, self.alpha)
        self.fe = self._fe[name]
        self.selector = self._sel[name]
        if t is not None:
            self.controller.switched(name, t, gap_s)
        if hasattr(self, "ego_est"):
            self.ego_est.wf = self.wf

    @property
    def waveform(self) -> str:
        return self.wf.name

    def step(self, frame: np.ndarray, t: float, raw_for_material: bool = True) -> dict:
        t0 = time.perf_counter()
        wf, fe, sel, tr = self.wf, self.fe, self.selector, self.tracker
        regions = tr.predict(t, wf, self.ego)
        t1 = time.perf_counter()
        rd = fe.process(frame)
        gates = fe.search_gates(rd, self.pfa)
        beams = fe.beam_gates(rd, regions, self.pfa, gates)
        gates = self._prioritise(gates + beams, regions)
        t2 = time.perf_counter()
        ev = sel.evalues(rd, gates)
        lam = tr.lam(t)
        merge_u = 0.5 * fe.array.rayleigh_u
        W = prior_weights(gates, sel.u_grid, regions, lam, merge_u)
        picked = sel.weighted_ebh(ev["E"], W)
        dets = sel.detections(rd, gates, picked, fe, ev, W, wf.cfg)
        dets = [d for d in dets if self.y_lim[0] <= d["range_m"] * np.sin(np.radians(d["theta_deg"])) <= self.y_lim[1]]
        for d in dets:
            d["change_db"] = fe.change_db(d["cell"][0], d["u"], 10 ** (d["power_db"] / 10))
        t3 = time.perf_counter()
        inside = {i for i, d in enumerate(dets) if self._in_any(d, regions)}
        dt = t - self.t_last if self.t_last is not None else 0.0
        self.ego = self.ego_est.estimate(dets, inside, fe.array.rayleigh_u, tr.doppler_sign, dt)
        rep = tr.update(dets, t, wf, self.ego)
        t4 = time.perf_counter()
        if self.material is not None:
            self.material.update(tr, rd, fe, frame if raw_for_material else None, t, dets)
        t5 = time.perf_counter()
        looks, why, look_table = None, None, {}
        if self.policy is None:
            switch, table = self.controller.decide(tr, t)
        else:
            switch, table = None, {}
            looks, why, look_table = self.policy.decide(tr.tracks, tr.doppler_sign, after=wf.name)
        t6 = time.perf_counter()
        ms = {"predict": 1e3 * (t1 - t0), "range_doppler": 1e3 * (t2 - t1), "infinirex_fdr": 1e3 * (t3 - t2),
              "track": 1e3 * (t4 - t3), "material": 1e3 * (t5 - t4), "cognitive": 1e3 * (t6 - t5),
              "total": 1e3 * (t6 - t0)}
        self.controller.observe(proc_ms=ms["total"])
        if self.policy is not None:
            self.policy.observe(wf.name, ms["total"] / 1e3)
        self.t_last = t
        rec = {"t": t, "waveform": wf.name, "gates": len(gates), "beam_gates": len(beams),
               "gates_search": len(gates) - sum(g.source == "beam" for g in gates),
               "lambda": lam, "alpha": self.alpha, "dets": dets, "tracks": tr.snapshot(),
               "info_bits": rep["info_bits"], "births": rep["births"], "deleted": rep["deleted"],
               "doppler_sign": tr.doppler_sign, "doppler_agreement": rep.get("doppler_agreement"),
               "ego": {k: v for k, v in self.ego.items()},
               "coverage": {str(h): r.coverage() for h, r in tr.regions.items()},
               "alpha_t": {str(h): r.alpha_t for h, r in tr.regions.items()},
               "attention_bits": attention_entropy(W), "waveform_table": table,
               "switch_to": switch, "ms": ms, "spectrum_entropy_bits": spectrum_entropy(rd, fe), "noise_db": float(10 * np.log10(np.mean(rd.noise_ch[fe.az]))),
               "static_floor_db": static_floor_db(rd), "bg_used": bool(rd.bg_used), "looks": looks, "look_reason": why, "look_table": look_table}
        if self.compare:
            uni = sel.weighted_ebh(ev["E"], np.ones_like(ev["E"]))
            per_gate = np.zeros_like(picked)
            for i in range(len(gates)):
                per_gate[i] = sel.weighted_ebh(ev["E"][i:i + 1], np.ones((1, sel.p)))[0]
            rec["compare"] = {"uniform": sel.detections(rd, gates, uni, fe, ev, np.ones_like(W), wf.cfg),
                              "per_gate": sel.detections(rd, gates, per_gate, fe, ev, np.ones_like(W), wf.cfg)}
        return rec

    def _prioritise(self, gates, regions):
        def in_region(g):
            return any(r["r_lo"] <= g.range_m <= r["r_hi"] for r in regions)
        gates = sorted(gates, key=lambda g: (not in_region(g), -g.excess))
        return gates[:self.cap]

    @staticmethod
    def _in_any(d, regions) -> bool:
        u = np.sin(np.radians(d["theta_deg"]))
        return any(r["r_lo"] <= d["range_m"] <= r["r_hi"] and r["u_lo"] - 0.1 <= u <= r["u_hi"] + 0.1 for r in regions)


def chirp_choice(cfg: dict, library: dict) -> ChirpChoice:
    wfs = sorted(library.values(), key=lambda w: w.subframe)
    sf = [SubFrame(w.name, w, w.start_ms / 1e3, w.active_s) for w in wfs]
    return ChirpChoice(sf, wfs[0].frame_period_s, cfg["radar"].get("look_policy", "cognitive"),
                       cfg["tracking"]["accel_sigma_mps2"], max(cfg["tracking"]["horizons_s"]))


def attention_entropy(W: np.ndarray) -> float:
    if W.size == 0:
        return 0.0
    p = W.ravel() / W.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def static_floor_db(rd) -> float:
    P = rd.P
    return float(10 * np.log10(np.median(P[:, P.shape[1] // 2]) / max(np.median(P), 1e-30)))


def spectrum_entropy(rd, fe) -> float:
    A = fe.array.steering(fe.array.u_grid)
    keep = rd.P > 10 * np.median(rd.P)
    if not keep.any():
        return float(np.log2(A.shape[1]))
    B = (np.abs(rd.D[keep][:, fe.az] @ A.conj()) ** 2).sum(axis=0)
    p = B / B.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
