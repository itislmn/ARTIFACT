"""
subframe_replay.py - a sub-frame stream played through the method as live: one processor, the newest look of each
chosen chirp, looks that arrive while it is busy are skipped; or exactly the looks the live program chose.

    recs, counts = play(ct, looks, main)          (play_live(ct, looks, live): the live chirp choices)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from methods.chirp_choice import pick_looks


@dataclass
class Look:
    t: float
    name: str
    load: Callable[[], Any]
    truth: Any = None


def play(ct, looks: list[Look], main: str, proc_scale: float = 1.0, on_record=None,
         cost_s: dict | None = None) -> tuple[list, dict]:
    recs, n = [], len(looks)
    idx, free, t_prev, wanted = 0, -np.inf, None, [main]
    counts = {"arrived": n, "processed": 0, "not_looked": 0, "skipped": 0}
    while idx < n:
        now = max(free, looks[idx].t)
        j = idx
        while j < n and looks[j].t <= now:
            j += 1
        batch, idx = looks[idx:j], j
        pick = pick_looks(batch, main, wanted, lambda f: f.t, lambda f: f.name)
        not_looked = sum(1 for f in batch if f.name != main and f.name not in wanted)
        counts["not_looked"] += not_looked
        counts["skipped"] += len(batch) - len(pick) - not_looked
        for f in pick:
            ct.set_waveform(f.name, revisit_s=(f.t - t_prev) if t_prev is not None else ct.lib[f.name].frame_period_s)
            t_prev = f.t
            rec = ct.step(f.load(), f.t)
            proc = proc_scale * (cost_s[f.name] if cost_s else rec["ms"]["total"] / 1e3)
            if cost_s and getattr(ct, "policy", None) is not None:
                ct.policy.proc[f.name][-1] = proc
                if f.name == main:
                    rec["looks"], rec["look_reason"], rec["look_table"] = ct.policy.decide(ct.tracker.tracks,
                                                                                             ct.tracker.doppler_sign)
            now = max(now, f.t) + proc
            rec["t_ready"] = now
            rec["truth"] = f.truth
            wanted = rec.get("looks") or [main]
            recs.append(rec)
            counts["processed"] += 1
            if on_record is not None:
                on_record(rec)
        free = now
    return recs, counts


def play_live(ct, looks: list[Look], live: list[dict], on_record=None, cost_s: dict | None = None) -> tuple[list, dict]:
    recs, t_prev = [], None
    for lv in live:
        f = looks[lv["frame"]]
        ct.set_waveform(f.name, revisit_s=(f.t - t_prev) if t_prev is not None else ct.lib[f.name].frame_period_s)
        t_prev = f.t
        rec = ct.step(f.load(), f.t)
        rec["t_ready"] = f.t + (cost_s[f.name] if cost_s else lv["ms"]["total"] / 1e3)
        rec["truth"] = f.truth
        rec.update({k: lv[k] for k in ("looks", "look_reason", "look_table") if k in lv})
        recs.append(rec)
        if on_record is not None:
            on_record(rec)
    counts = {"arrived": len(looks), "processed": len(recs), "not_looked": live[-1].get("not_looked", 0),
              "skipped": live[-1].get("skipped", 0)}
    return recs, counts
