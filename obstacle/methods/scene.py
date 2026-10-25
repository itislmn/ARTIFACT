"""
scene.py - the method, one radar look: frame and rail position in, detections, tracks, labels and world objects out;
the same live and on recordings.

    python -m methods.scene      (self-test)
"""
from __future__ import annotations

import time

import numpy as np

from methods.beam_search import BeamSearch
from methods.infinirex import FrameSelector, prior_weights
from methods.range_doppler import FrontEnd
from methods.tracker import Tracker
from methods.world import World, det_world


class ObstacleScene:
    def __init__(self, cfg: dict, lib, channel_cal=None, reference=None, material=None, detector: str = "infinirex"):
        self.cfg, self.lib, self.detector = cfg, lib, detector
        rc, dc = cfg["radar"], cfg["detection"]
        self.r_min, self.r_max = rc["r_min_m"], rc["r_max_m"]
        self.x_max, self.y_lim = rc["x_max_m"], (rc["y_min_m"], rc["y_max_m"])
        self.alpha, self.pfa, self.cap = dc["alpha"], dc["gate_pfa"], dc["gates_per_frame"]
        self.tracker = Tracker(cfg, cfg["tracking"].get("doppler_sign", 1))
        self.world = World(cfg)
        self.n_known = self.world.set_known(reference) if reference is not None else 0
        self.fe = {n: FrontEnd(lib[n], self.r_min, self.r_max, channel_cal, None) for n in lib.names}
        self.sel = {n: FrameSelector(self.fe[n].array, self.alpha) for n in lib.names}
        self.beam = {n: BeamSearch(self.fe[n].array, self.pfa) for n in lib.names}
        self.fe_still = {}
        if reference is not None and reference.bg_rfft is not None:
            self.fe_still = {n: FrontEnd(lib[n], self.r_min, self.r_max, channel_cal, {"rfft": reference.bg_rfft})
                             for n in lib.names}
        self.reference, self.material = reference, material
        self.t_last = None
        self._n_miss = {}
        self._A = {}
        if detector != "infinirex":
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from estimation.core.interfaces import Settings
            from estimation.methods import doa_methods
            self._cell_fn = doa_methods([{"ctrex": "ctrex_tomp", "cfar": "os_cfar"}[detector]])[0].fn
            self._settings = Settings(alpha=self.alpha, pfa=self.pfa, gate_pfa=self.pfa, r_min_m=self.r_min)
            self._rng = np.random.default_rng(0)

    def step(self, frame: np.ndarray, t: float, name: str, pose, v_rail, keep_rd: bool = False) -> dict:
        t0 = time.perf_counter()
        pose, v_rail = np.asarray(pose, float), np.asarray(v_rail, float)
        revisit = float(np.clip(t - self.t_last, self.lib.min_period_ms / 1e3, 3 * self.lib.frame_ms / 1e3)) \
            if self.t_last is not None else self.lib.frame_ms / 1e3
        wf = self.lib.look(name, round(revisit, 2))
        sel, tr = self.sel[name], self.tracker
        still = self.reference is not None and self.fe_still and self.reference.still_background(pose, v_rail) is not None
        fe = self.fe_still[name] if still else self.fe[name]
        ego = {"v": (float(v_rail[0]), float(v_rail[1])), "dpsi": 0.0, "moving": bool(np.hypot(*v_rail[:2]) > 1e-3)}
        regions = tr.predict(t, wf, ego)
        t1 = time.perf_counter()
        rd = fe.process(frame)
        still = bool(still and rd.bg_used)
        gates = fe.search_gates(rd, self.pfa, static_line=True)
        gates += self.beam[name](fe, rd, gates)
        n_search = len(gates)
        explained = set() if still else self._explained(gates, rd, fe, regions, pose, t, revisit)
        beams = fe.beam_gates(rd, regions, self.pfa, gates)
        gates = self._prioritise(gates + beams, regions, explained)
        left_out = sum(1 for c in explained if c not in {g.cell for g in gates})
        t2 = time.perf_counter()
        ev = sel.evalues(rd, gates)
        lam = tr.lam(t)
        W = prior_weights(gates, sel.u_grid, regions, lam, 0.5 * fe.array.rayleigh_u)
        picked = sel.weighted_ebh(ev["E"], W) if self.detector == "infinirex" else self._per_cell(rd, fe, gates, sel.u_grid)
        dets = sel.detections(rd, gates, picked, fe, ev, W, wf.cfg)
        t3 = time.perf_counter()
        keep = []
        sign = tr.doppler_sign or 1
        for d in dets:
            p, z_ok = det_world(d, pose)
            if p[0] > self.x_max or not (self.y_lim[0] <= p[1] <= self.y_lim[1]) or self._shadowed(p, pose):
                continue
            th = np.radians(d["theta_deg"])
            static = abs(d["v_mps"] * sign + v_rail[0] * np.cos(th) + v_rail[1] * np.sin(th)) <= 2 * wf.dv
            d["world"], d["z_ok"], d["still"] = [round(float(c), 4) for c in p], z_ok, bool(static)
            if still:
                d["change_db"] = fe.change_db(d["cell"][0], d["u"], 10 ** (d["power_db"] / 10))
            else:
                d["change_db"] = (self.reference.change_db(d["range_m"], 10 ** (d["power_db"] / 10), p, pose)
                                  if (static and self.reference is not None) else float("inf"))
            keep.append(d)
        dets = keep
        t4 = time.perf_counter()
        rep = tr.update(dets, t, wf, ego)
        t5 = time.perf_counter()
        products = None
        if self.material is not None and name == self.lib.names[0]:
            products = self.material.update(tr, frame, t, pose)
        t6 = time.perf_counter()
        objects = self.world.update(tr, dets, pose, v_rail, t, wf, revisit)
        t7 = time.perf_counter()
        self.t_last = t
        ms = {"predict": 1e3 * (t1 - t0), "range_doppler": 1e3 * (t2 - t1), "infinirex_fdr": 1e3 * (t3 - t2),
              "place": 1e3 * (t4 - t3), "track": 1e3 * (t5 - t4), "material": 1e3 * (t6 - t5),
              "world": 1e3 * (t7 - t6), "total": 1e3 * (t7 - t0)}
        snap = tr.snapshot()
        for s in snap:
            s["wx"], s["wy"] = float(pose[0] + s["x"]), float(pose[1] + s["y"])
        misses = {}
        for h, r in tr.regions.items():
            n0 = self._n_miss.get(h, 0)
            misses[str(h)] = [int(m) for m, _ in r.misses[n0:]]
            self._n_miss[h] = len(r.misses)
        rec = {"t": t, "chirp": name, "revisit_s": round(revisit, 4), "pose": [round(float(c), 5) for c in pose],
               "v_rail": [round(float(c), 5) for c in v_rail], "gates": len(gates), "beam_gates": len(beams),
               "search_gates": n_search, "gates_explained": len(explained), "explained_left_out": left_out,
               "subtracted": bool(still),
               "lambda": lam, "alpha": self.alpha, "dets": [_compact(d) for d in dets], "tracks": snap,
               "objects": objects, "info_bits": rep["info_bits"], "births": rep["births"], "deleted": rep["deleted"],
               "doppler_sign": tr.doppler_sign,
               "coverage": {str(h): r.coverage() for h, r in tr.regions.items()}, "misses": misses,
               "alpha_t": {str(h): r.alpha_t for h, r in tr.regions.items()},
               "material_paused": getattr(self.material, "paused", "no material model"),
               "noise_db": float(10 * np.log10(np.mean(rd.noise_ch[fe.az]))), "ms": ms}
        if keep_rd:
            rec["_rd"], rec["_fe"], rec["_products"] = rd, fe, products
        return rec

    def _per_cell(self, rd, fe, gates, u_grid) -> np.ndarray:
        from estimation.core.interfaces import Gate as CellGate
        picked = np.zeros((len(gates), len(u_grid)), bool)
        for i, g in enumerate(gates):
            y = rd.D[g.k, g.d, fe.az].astype(np.complex128)
            cell = CellGate(int(g.k), g.range_m, y[:, None], float(np.mean(rd.noise_ch[fe.az])),
                            channel_noise=rd.noise_ch[fe.az].astype(float))
            for u in np.sin(np.radians(np.asarray(self._cell_fn(cell, fe.array, self._settings, self._rng).angles_deg, float))):
                picked[i, int(np.argmin(np.abs(u_grid - u)))] = True
        return picked

    def _shadowed(self, p, pose) -> bool:
        q = np.asarray(p[:2]) - pose[:2]
        for o in self.world.objects:
            if not o.get("n_points") or not o.get("person"):
                continue
            c = [np.array([x, y]) - pose[:2] for x in (o["lo"][0], o["hi"][0]) for y in (o["lo"][1], o["hi"][1])]
            if min(v[0] for v in c) <= 0:
                continue
            a = [np.arctan2(v[1], v[0]) for v in c]
            if min(a) <= np.arctan2(q[1], q[0]) <= max(a) and np.hypot(*q) > max(np.hypot(*v) for v in c):
                return True
        return False

    def _explained(self, gates, rd, fe, regions, pose, t_now: float, look_s: float) -> set:
        if self.reference is None:
            return set()
        L = self.reference.level_map(pose)
        if L is None:
            return set()
        if id(fe) not in self._A:
            self._A[id(fe)] = fe.array.steering(fe.array.u_grid).conj()
        A = self._A[id(fe)]
        tr = self.tracker
        near = [(float(np.hypot(*t.pred["s"][:2])), tr.v_init * max(t_now - t.last_update, look_s) + tr.extent / 2)
                for t in tr.tracks if "s" in t.pred]
        out = set()
        for g in gates:
            k = self.reference.row(g.range_m)
            if abs(g.d - fe.K // 2) > 1 or k < 0 or self._in_region(g, regions) or \
                    any(abs(g.range_m - r) <= reach + fe.cfg.range_res_m for r, reach in near):
                continue
            if not np.any(np.abs(rd.D[g.k, g.d, fe.az] @ A) ** 2 / fe.array.n > L[k]):
                out.add(g.cell)
        return out

    @staticmethod
    def _in_region(g, regions) -> bool:
        return any(r["r_lo"] <= g.range_m <= r["r_hi"] for r in regions)

    def _prioritise(self, gates, regions, explained: set = frozenset()):
        gates = sorted(gates, key=lambda g: (not self._in_region(g, regions), g.cell in explained, -g.excess))
        return gates[:self.cap]


class StaticDetector:

    def __init__(self, cfg: dict, fe, wf):
        self.fe, self.wf = fe, wf
        self.sel = FrameSelector(fe.array, cfg["detection"]["alpha"])
        self.pfa = cfg["detection"]["gate_pfa"]
        self.y_lim = (cfg["radar"]["y_min_m"], cfg["radar"]["y_max_m"])

    def __call__(self, rd, pose) -> list[tuple]:
        from methods.world import det_sigma
        fe, sel = self.fe, self.sel
        gates = [g for g in fe.search_gates(rd, self.pfa, static_line=True) if abs(g.d - fe.K // 2) <= 1]
        if not gates:
            return []
        ev = sel.evalues(rd, gates)
        W = np.ones_like(ev["E"])
        picked = sel.weighted_ebh(ev["E"], W)
        out = []
        for d in sel.detections(rd, gates, picked, fe, ev, W, self.wf.cfg):
            p, z_ok = det_world(d, pose)
            if self.y_lim[0] <= p[1] <= self.y_lim[1]:
                out.append((*p, *det_sigma(d, self.wf), float(z_ok)))
        return out


def selftest() -> bool:
    import paths
    from methods.range_doppler import Gate
    from methods.waveforms import Library
    cfg = paths.load_config()
    sc = ObstacleScene(cfg, Library(cfg), None, None, None)
    fe = sc.fe[sc.lib.names[0]]
    n = len(fe.range_m)
    strong = [Gate(k % n, fe.K // 2 + k // n, float(fe.range_m[k % n]), 0.0, 1e6, 50.0 - k, "search") for k in range(sc.cap)]
    weak = Gate(n // 2, fe.K // 2 + 5, float(fe.range_m[n // 2]), 0.3, 10.0, 1.1, "search")
    explained = {g.cell for g in strong}
    with_c = weak in sc._prioritise(strong + [weak], [], explained)
    without = weak in sc._prioritise(strong + [weak], [], set())
    few = sc._prioritise(strong[:5] + [weak], [], explained)
    ok = with_c and not without and len(few) == 6
    print(f"  {'PASS' if ok else 'FAIL'}  weak new cell behind {sc.cap} empty-path cells: decided with the competition "
          f"{with_c}, without {without}; budget not full: all {len(few)} cells kept")
    return ok


def _compact(d: dict) -> dict:
    keys = ("range_m", "theta_deg", "elev_deg", "elev_fit", "v_mps", "snr_db", "power_db", "e", "w", "label", "track",
            "world", "z_ok", "still", "change_db", "source", "cell")
    out = {}
    for k in keys:
        v = d.get(k)
        if isinstance(v, float):
            v = round(v, 5) if np.isfinite(v) else (None if np.isnan(v) else (1e9 if v > 0 else -1e9))
        out[k] = v
    return out


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
