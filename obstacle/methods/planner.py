"""
planner.py - what the rail does next: drive, stop, wait for a mover that could reach the platform within the
measured response time, or step aside and look again.

    python -m methods.planner      (self-test)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 0.002


def response_time(frame_s: float, proc_s: float, loop_s: float, stop_s: float) -> float:
    return float(frame_s + proc_s + loop_s + stop_s)


def boxes_touch(lo_a, hi_a, lo_b, hi_b, clearance: float = 0.0) -> bool:
    lo_a, hi_a, lo_b, hi_b = (np.asarray(v, float) for v in (lo_a, hi_a, lo_b, hi_b))
    return bool(np.all(lo_a - clearance < hi_b) and np.all(lo_b < hi_a + clearance))


@dataclass
class Platform:
    head: np.ndarray
    carriage: np.ndarray
    limits: np.ndarray

    @classmethod
    def from_config(cls, cfg: dict) -> "Platform":
        p, r = cfg["platform"], cfg["rail"]
        head = np.array([p["head_x_m"], p["head_y_m"], p["head_z_m"]], float)
        car = np.array([p["carriage_x_m"], p["carriage_y_m"], p["carriage_z_m"]], float)
        lim = np.array([r["x_mm"], r["y_mm"], r["z_mm"]], float) / 1000.0
        return cls(head, car, lim)

    def head_box(self, p):
        p = np.asarray(p, float)
        return p + self.head[:, 0], p + self.head[:, 1]

    def head_sweep(self, p0, p1):
        a0, b0 = self.head_box(p0)
        a1, b1 = self.head_box(p1)
        return np.minimum(a0, a1), np.maximum(b0, b1)

    def carriage_sweep(self, x0: float, x1: float):
        lo = self.carriage[:, 0] + np.array([min(x0, x1), 0, 0])
        hi = self.carriage[:, 1] + np.array([max(x0, x1), 0, 0])
        return lo, hi

    def danger_zone(self, pose, v_fwd: float, horizon: float, stop_m: float, target=None):
        pose = np.asarray(pose, float)
        ahead = pose.copy()
        if target is None:
            ahead[0] = pose[0] + max(v_fwd, 0.0) * horizon + stop_m
        else:
            d = np.asarray(target, float) - pose
            n = float(np.linalg.norm(d))
            step = min(n, max(v_fwd, 0.0) * horizon + stop_m)
            ahead = pose + (d / n * step if n > 1e-9 else 0.0)
        return [self.head_sweep(pose, ahead), self.carriage_sweep(pose[0], ahead[0])]


@dataclass
class Command:
    kind: str
    target: np.ndarray | None
    state: str
    reason: str
    info: dict = field(default_factory=dict)


class Planner:
    def __init__(self, cfg: dict):
        self.pf = Platform.from_config(cfg)
        pc = cfg["planner"]
        self.clear = float(pc["clearance_m"])
        self.horizon = float(pc["horizon_s"])
        self.verify_s = float(pc["verify_s"])
        self.step_m = float(pc["step_m"])
        self.return_home = bool(pc["return_to_centre"])
        self.reach = float(cfg["tracking"]["object_extent_m"]) / 2
        self.v_max = float(cfg["tracking"]["max_speed_mps"])
        self.cell = 0.0837
        self.r_obs = float(cfg["radar"]["r_min_m"]) + self.cell
        self.target_x = None
        self.goal = float(self.pf.limits[0, 1])
        self.home_offset = np.zeros(2)
        self.state, self.reason = "idle", "not started"
        self.t_state = 0.0
        self.t_clear = None
        self.avoid_to = None
        self.events: list[dict] = []
        self.started = False

    def blockers(self, objects: list[dict]) -> list[tuple]:
        out = []
        for o in objects:
            if o.get("known"):
                continue
            if o["kind"] == "static":
                out.append((np.array(o["lo"]), np.array(o["hi"]), o))
            elif not o.get("person"):
                lo, hi = o.get("footprint", [o["lo"], o["hi"]])
                out.append((np.array(lo), np.array(hi), o))
        return out

    def _cross(self, part: np.ndarray, yz, lo, hi) -> bool:
        y0, y1 = yz[0] + part[1, 0], yz[0] + part[1, 1]
        z0, z1 = yz[1] + part[2, 0], yz[1] + part[2, 1]
        return bool(y0 - self.clear < hi[1] and lo[1] < y1 + self.clear and z0 - self.clear < hi[2] and lo[2] < z1 + self.clear)

    def x_head(self, x0: float, yz, blk) -> tuple[float, dict | None]:
        best, who = self.goal, None
        for lo, hi, o in blk:
            if hi[0] < x0 + self.pf.head[0, 0] - self.clear or not self._cross(self.pf.head, yz, lo, hi):
                continue
            lim = min(lo[0] - self.clear - self.pf.head[0, 1], lo[0] - self.r_obs)
            if lim < best:
                best, who = lim, o
        return float(best), who

    def x_carriage(self, x0: float, blk) -> tuple[float, dict | None]:
        best, who = self.goal, None
        for lo, hi, o in blk:
            if hi[0] < x0 + self.pf.carriage[0, 0] - self.clear or not self._cross(self.pf.carriage, (0.0, 0.0), lo, hi):
                continue
            lim = min(lo[0] - self.clear - self.pf.carriage[0, 1], lo[0] - self.r_obs)
            if lim < best:
                best, who = lim, o
        return float(best), who

    def x_free(self, x0: float, yz, blk) -> tuple[float, dict | None, str]:
        xh, oh = self.x_head(x0, yz, blk)
        xc, oc = self.x_carriage(x0, blk)
        if oc is not None and xc <= xh:
            return xc, oc, "carriage"
        return xh, oh, ("head" if oh is not None else "")

    def move_free(self, p0, p1, blk) -> bool:
        lo, hi = self.pf.head_sweep(p0, p1)
        return not any(boxes_touch(lo, hi, a, b, self.clear) for a, b, _ in blk)

    def offsets(self):
        ys = np.arange(self.pf.limits[1, 0], self.pf.limits[1, 1] + 1e-9, self.step_m)
        zs = np.arange(self.pf.limits[2, 0], self.pf.limits[2, 1] + 1e-9, self.step_m)
        return [(y, z) for y in ys for z in zs]

    def best_offset(self, pose, blk, x_need: float):
        cands = []
        for y, z in self.offsets():
            p1 = np.array([pose[0], y, z])
            xh, _ = self.x_head(pose[0], (y, z), blk)
            if xh >= x_need - EPS and self.move_free(pose, p1, blk):
                cands.append((abs(y - pose[1]) + abs(z - pose[2]) + 1e-6 * abs(z - pose[2]), y, z, xh))
        return min(cands) if cands else None

    def step(self, objects: list[dict], pose, t: float, v_fwd: float, stop_m: float, moving: bool,
             t_resp: float) -> Command:
        pose = np.asarray(pose, float)
        if not self.started:
            return Command("hold", None, "idle", "waiting for start")
        blk = self.blockers(objects)
        x_lim, who, part = self.x_free(pose[0], pose[1:], blk)
        info = {"x_free": round(x_lim, 4), "limit": part, "limit_obj": who["id"] if who else None,
                "blockers": [o["id"] for _, _, o in blk], "t_resp": round(t_resp, 3)}

        hit = self.safety_hit(objects, pose, v_fwd, stop_m, t_resp)
        standing = who is not None and who.get("person") and x_lim - stop_m <= pose[0] + EPS
        if hit is not None or standing:
            self.t_clear = None
            if hit is not None:
                o, hz, kind = hit
                why = f"{kind} {o['id']} could reach the platform before it stops ({t_resp:.2f} s)"
            else:
                o, why = who, f"person {who['id']} standing in the path"
            return self._set("wait", Command("stop", None, "wait", why, {**info, "cause": o["id"]}), t)
        if self.state == "wait":
            self.t_clear = t if self.t_clear is None else self.t_clear
            if t - self.t_clear < self.horizon:
                return Command("hold", None, "wait", f"path clear for {t - self.t_clear:.1f} s of {self.horizon:.1f} s", info)
            self._set("cruise", None, t, "path clear again")

        if pose[0] >= self.goal - EPS:
            return self._set("done", Command("stop", None, "done", "end of the forward travel", info), t)

        if self.state == "avoid":
            if np.linalg.norm(pose[1:] - self.avoid_to) > EPS:
                return Command("goto", np.r_[pose[0], self.avoid_to], "avoid", self.reason, info)
            self._set("verify", None, t, f"at offset y {1e3 * pose[1]:+.0f} mm, z {1e3 * pose[2]:+.0f} mm: looking again")
        if self.state == "verify":
            if t - self.t_state < self.verify_s:
                return Command("hold", None, "verify", self.reason, info)
            if x_lim > pose[0] + stop_m + EPS:
                self._set("cruise", None, t, "the new view confirms the path")
            else:
                self._set("decide", None, t, "the new view does not confirm the path")

        if self.return_home and np.linalg.norm(pose[1:] - self.home_offset) > EPS and self.state in ("cruise", "approach"):
            home = np.r_[pose[0], self.home_offset]
            x_home = self.x_free(pose[0], self.home_offset, blk)[0]
            if self.move_free(pose, home, blk) and x_home >= x_lim - EPS and x_home > pose[0] + stop_m + EPS:
                return self._set("return", Command("goto", home, "return", "back to the home offset", info), t)
        if self.state == "return":
            if np.linalg.norm(pose[1:] - self.home_offset) > EPS:
                return Command("goto", np.r_[pose[0], self.home_offset], "return", self.reason, info)
            self._set("cruise", None, t, "home offset reached")

        target_x = x_lim - stop_m
        if self.target_x is not None and self.state == "approach" and self.target_x < target_x < self.target_x + self.cell:
            target_x = self.target_x
        self.target_x = target_x
        if target_x > pose[0] + EPS:
            st = "cruise" if who is None else "approach"
            why = "path free to the end" if who is None else f"object {who['id']} limits the {part} at x {x_lim:.3f} m"
            if self.state != st:
                self._set(st, None, t, why)
            return Command("goto", np.r_[min(target_x, self.goal), pose[1:]], st, why, info)

        x_car, o_car = self.x_carriage(pose[0], blk)
        if part == "carriage" or x_car - stop_m <= pose[0] + EPS:
            o = o_car or who
            why = f"object {o['id']} is on the rail's path: no sidestep can help"
            if not o.get("z_known", True):
                why += " (its height is not measured: it may reach down to the rail)"
            return self._set("blocked", Command("stop", None, "blocked", why, info), t)
        x_need = min(self.goal, x_car)
        best = self.best_offset(pose, blk, x_need)
        if best is None:
            return self._set("blocked", Command("stop", None, "blocked",
                                                f"object {who['id'] if who else '?'}: no side or height offset within the "
                                                "limits clears it", info), t)
        cost, y, z, xf = best
        self.avoid_to = np.array([y, z])
        way = "sideways" if abs(y - pose[1]) >= abs(z - pose[2]) else ("up" if z > pose[2] else "down")
        return self._set("avoid", Command("goto", np.r_[pose[0], y, z], "avoid",
                                          f"object {who['id'] if who else '?'}: step {way} to y {1e3 * y:+.0f} mm, "
                                          f"z {1e3 * z:+.0f} mm (free to x {xf:.3f} m)", {**info, "offset": [y, z]}), t)

    def safety_hit(self, objects, pose, v_fwd, stop_m, t_resp: float, extra_s: float = 0.0):
        need = t_resp + extra_s
        zones = self.pf.danger_zone(pose, v_fwd, need, stop_m)

        def touches(lo, hi):
            return any(boxes_touch(zl, zh, lo, hi, self.clear) for zl, zh in zones)
        for o in objects:
            if o["kind"] != "moving" or (o.get("still") and not o.get("person")):
                continue
            unclassified = o.get("material") is None
            reach = self.reach if (unclassified and not o.get("person")) else 0.0
            pad = np.array([reach, reach, 0.0])
            lo, hi = np.asarray(o["lo"], float) - pad, np.asarray(o["hi"], float) + pad
            run = np.array([self.v_max * need, self.v_max * need, 0.0])
            if not touches(lo - run, hi + run):
                continue
            regs = sorted((float(h), lo_r, hi_r) for h, (lo_r, hi_r) in o["regions"].items())
            pick = next((r for r in regs if r[0] >= need - 1e-9), regs[-1] if regs else None)
            hz = need
            if pick is not None and o.get("calibrated"):
                hz = pick[0]
                if not touches(np.minimum(lo, np.asarray(pick[1], float) - pad), np.maximum(hi, np.asarray(pick[2], float) + pad)):
                    continue
            kind = "person" if o.get("person") else ("unclassified mover" if unclassified else "moving object")
            return o, hz, kind
        return None

    def _set(self, state: str, cmd: Command | None, t: float, reason: str | None = None):
        reason = reason if reason is not None else (cmd.reason if cmd else "")
        if state != self.state or (cmd is not None and cmd.kind == "stop" and reason != self.reason):
            self.events.append({"t": t, "from": self.state, "to": state, "reason": reason,
                                **({"info": cmd.info} if cmd is not None else {})})
            self.t_state = t
        self.state, self.reason = state, reason
        return cmd


def selftest() -> bool:
    cfg = {"platform": {"head_x_m": [-0.10, 0.04], "head_y_m": [-0.08, 0.08], "head_z_m": [-0.08, 0.12],
                        "carriage_x_m": [-0.30, 0.05], "carriage_y_m": [-0.10, 0.10], "carriage_z_m": [-0.45, -0.20],
                        "antenna_height_m": 0.6},
           "rail": {"x_mm": [0, 1250], "y_mm": [-150, 150], "z_mm": [0, 200]},
           "planner": {"clearance_m": 0.03, "horizon_s": 1.0, "verify_s": 1.0, "step_m": 0.01, "return_to_centre": True},
           "radar": {"r_min_m": 0.25}, "tracking": {"object_extent_m": 0.6, "max_speed_mps": 1.5}}
    at_decision = 0.9 - (0.25 + 0.0837) - 0.005 + 0.0001

    def box(i, lo, hi, kind="static", **kw):
        return {"id": i, "kind": kind, "lo": lo, "hi": hi, "z_known": bool(np.isfinite(lo[2])), "material": "metal", **kw}

    def run(objs, x=at_decision):
        pl = Planner(cfg)
        pl.started = True
        return pl.step(objs, np.array([x, 0.0, 0.0]), 1.0, 0.025, 0.005, True, t_resp=0.45)

    def mover(i, lo, hi, reg, **kw):
        kw.setdefault("material", "waterlike")
        kw.setdefault("calibrated", True)
        return {"id": i, "kind": "moving", "lo": lo, "hi": hi, "z_known": True, "regions": {"0.5": reg},
                "footprint": [lo, hi], **kw}

    cases = [
        ("object on the rail (below the head)", [box(1, [0.9, -0.05, -0.40], [1.0, 0.05, -0.25])], at_decision,
         lambda c: c.state == "blocked" and "rail" in c.reason),
        ("box over the rail edge into the head's way", [box(2, [0.9, 0.02, -0.15], [1.0, 0.40, 0.5])], at_decision,
         lambda c: c.state == "avoid" and c.target[1] < -0.08),
        ("thin plate at the head's lower edge", [box(3, [0.9, -0.5, -0.15], [1.0, 0.5, -0.05])], at_decision,
         lambda c: c.state == "avoid" and c.target[2] > 0.05),
        ("the same plate, height not measured", [box(4, [0.9, -0.5, -np.inf], [1.0, 0.5, np.inf])], at_decision,
         lambda c: c.state == "blocked"),
        ("obstacle far ahead: approach first", [box(6, [0.9, -0.05, -0.40], [1.0, 0.05, -0.25])], 0.3,
         lambda c: c.state == "approach" and abs(c.target[0] - (0.9 - 0.3337 - 0.005)) < 1e-3),
        ("person crossing 1 m ahead: not close enough", [mover(5, [1.3, 0.3, -0.6], [1.5, 0.6, 1.3],
                                                               [[1.2, -0.2, -0.6], [1.6, 0.7, 1.3]], person=True)], 0.4,
         lambda c: c.state == "cruise"),
        ("person stepping in right in front", [mover(7, [0.5, 0.1, -0.6], [0.7, 0.5, 1.3],
                                                     [[0.45, -0.2, -0.6], [0.75, 0.5, 1.3]], person=True)], 0.4,
         lambda c: c.state == "wait" and "person 7" in c.reason),
        ("unclassified mover close beside the head", [mover(8, [0.42, 0.2, -0.1], [0.52, 0.3, 0.1],
                                                            [[0.40, 0.15, -0.1], [0.55, 0.35, 0.1]], material=None)], 0.4,
         lambda c: c.state == "wait" and "unclassified" in c.reason),
        ("new track, every echo still (a reflector)", [mover(12, [0.42, 0.2, -0.1], [0.52, 0.3, 0.1],
                                                             [[0.40, 0.15, -0.1], [0.55, 0.35, 0.1]], material=None,
                                                             still=True, calibrated=False)], 0.4,
         lambda c: c.state == "cruise"),
        ("the same mover classified as metal",[mover(9, [0.42, 0.2, -0.1], [0.52, 0.3, 0.1],
                                                      [[0.40, 0.15, -0.1], [0.55, 0.35, 0.1]], material="metal")], 0.4,
         lambda c: c.state == "cruise"),
        ("mover 1.2 m away, region not calibrated yet", [mover(11, [1.6, 0.3, -0.1], [1.7, 0.4, 0.1],
                                                              [[-0.5, -1.2, -0.1], [2.6, 1.9, 0.1]], material="metal",
                                                              calibrated=False)], 0.4,
         lambda c: c.state == "cruise"),
        ("person standing where the platform decides", [box(10, [0.9, 0.0, -0.6], [1.1, 0.05, 1.4], person=True)],
         at_decision, lambda c: c.state == "wait" and "standing" in c.reason),
        ("nothing in the way", [], 0.4, lambda c: c.state == "cruise" and abs(c.target[0] - 1.245) < 1e-6),
    ]
    ok = True
    for name, objs, x, check in cases:
        c = run(objs, x)
        good = bool(check(c))
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} {name:44s} -> {c.state}: {c.reason}")
    print(f"  {'PASS' if ok else 'FAIL'}  planner decisions")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
