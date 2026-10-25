"""
replay.py - the method on recorded radar data exactly as live, frame by frame; a sub-frame take is played as live
with a look policy.

    python replay.py recordings/TR1/final_fixed_1 --out final_fixed_1     (-> output/cache/replay/)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import paths
from estimation.core.frame import background_from_recording
from estimation.core.radar import RadarConfig, open_recording, with_layout
from methods.cognitive_tracking import CognitiveTracking
from methods.material import MaterialHead
from methods.material_src import for_session
from methods.radar_model import Waveform

MATERIAL_CFG = paths.MATERIAL / "config" / "awr2944_material_spec.cfg"
MATERIAL_CAL = paths.MATERIAL / "config" / "channel_cal.npy"


def material_waveform() -> Waveform:
    rc = with_layout(RadarConfig.from_cfg(MATERIAL_CFG), paths.LAYOUT)
    return Waveform("material", rc.slope_hz_per_s / 1e12, rc.idle_s * 1e6, rc.frame_period_s * 1e3, rc, MATERIAL_CFG)


def material_references(take: Path, wf: Waveform) -> dict:
    session = take.parent.parent
    parts = []
    for p in sorted((session / "background").glob("*.npy")):
        n, get = open_recording(p, wf.cfg)
        parts.append(background_from_recording(min(n, 100), get, wf.cfg))
    bg = {k: np.mean([q[k] for q in parts], axis=0) for k in ("rfft", "adc")} if parts else None
    cal = np.load(MATERIAL_CAL) if MATERIAL_CAL.exists() else None
    if bg is not None and cal is not None:
        bg = {"rfft": bg["rfft"], "adc": bg["adc"]}
    return {"bg": bg, "cal": cal}


def frame_times(take: Path, n: int, period: float) -> np.ndarray:
    meta = take.with_name(take.stem + ".meta.json")
    if meta.exists():
        t = np.asarray(json.loads(meta.read_text()).get("frame_t") or [], float)
        if t.size >= n:
            t = t[-n:]
            return t - t[0]
    return np.arange(n) * period


def open_take(path: Path, wf: Waveform):
    n, get = open_recording(path, wf.cfg)
    return n, get, frame_times(path, n, wf.frame_period_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", type=Path)
    ap.add_argument("--superpose", type=Path, default=None)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--doppler-sign", default=None, help="1, -1 or auto (default [tracking] doppler_sign)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--policy", default=None, help="sub-frame takes: the look policy (cognitive / search / all, or live: "
                                                   "the live program's own choices; default the take's [radar] "
                                                   "look_policy); played as live")
    ap.add_argument("--costs", type=Path, default=None, help="sub-frame takes: JSON {chirp: seconds}, a fixed processing "
                                                             "time per look instead of this machine's measured one")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a setting of config.toml for this replay, e.g. --set radar.y_max_m=1.0")
    args = ap.parse_args()
    cfg = paths.load_config()
    for item in args.set:
        key, value = item.split("=", 1)
        sec, k = key.split(".", 1)
        cfg[sec][k] = type(cfg[sec].get(k, 0.0))(value) if not isinstance(cfg[sec].get(k), bool) else value == "true"
        print(f"setting {sec}.{k} = {cfg[sec][k]}")
    take = args.take
    if take.is_dir():
        from hardware.recording import open_live_take
        lib, refs, n, get, times, wf_of = open_live_take(take, cfg)
    else:
        wf = material_waveform()
        lib = {"material": wf}
        refs = {"material": material_references(take, wf)}
        n, get, times = open_take(take, wf)
        wf_of = lambda i: "material"
        if args.superpose is not None:
            n2, get2, _ = open_take(args.superpose, wf)
            ref2 = material_references(args.superpose, wf)
            bg2 = ref2["bg"]
            n = min(n, n2)
            g1 = get
            adc2 = bg2["adc"][:, None, :].astype(np.float32) if bg2 is not None else 0.0
            get = lambda i: g1(i) + get2(i) - adc2
    n = min(n, args.frames) if args.frames else n
    name = args.out or (take.stem if not take.is_dir() else take.name) + (f"+{args.superpose.stem}" if args.superpose else "")
    out = paths.RESULTS / "replay" / name
    out.mkdir(parents=True, exist_ok=True)
    first = wf_of(0)
    ds = args.doppler_sign or cfg["tracking"].get("doppler_sign", 1)
    ds = ds if ds == "auto" else int(ds)
    sub = take.is_dir() and any(getattr(w, "subframe", -1) >= 0 for w in lib.values())
    if sub:
        import copy
        from hardware.subframe_replay import Look, play, play_live
        main = sorted(lib, key=lambda k: lib[k].subframe)[0]
        cfg = copy.deepcopy(cfg)
        policy = args.policy or cfg["radar"].get("look_policy", "cognitive")
        cfg["radar"]["look_policy"] = "cognitive" if policy == "live" else policy
        ct = CognitiveTracking(cfg, lib, refs, start=main, doppler_sign=ds, material=for_session(cfg, lib, only=main),
                               subframes=True)
        log = open(out / "frames.jsonl", "w", encoding="utf-8")
        t_wall = time.perf_counter()
        looks = [Look(float(times[i]), wf_of(i), (lambda i=i: get(i))) for i in range(n)]
        costs = json.loads(args.costs.read_text()) if args.costs else None
        write = lambda r: log.write(json.dumps(r, default=_js) + "\n")
        if policy == "live":
            live = [json.loads(line) for line in open(take / "radar_frames.jsonl", encoding="utf-8")]
            recs, counts = play_live(ct, looks, [lv for lv in live if lv["frame"] < n], on_record=write, cost_s=costs)
        else:
            recs, counts = play(ct, looks, main, on_record=write, cost_s=costs)
        log.close()
        cfg["radar"]["look_policy"] = policy
        (out / "meta.json").write_text(json.dumps({"take": str(take), "frames": n, "policy": policy,
                                                   "counts": counts, "costs_s": costs, "wall_s": round(time.perf_counter() - t_wall, 1),
                                                   "waveforms": {k: w.describe() for k, w in lib.items()}}, indent=2, default=_js))
        print(f"{counts['processed']} of {n} looks processed (policy {cfg['radar']['look_policy']}; {counts['not_looked']} not "
              f"looked, {counts['skipped']} skipped) -> {out}")
        return
    mat = for_session(cfg, lib) if take.is_dir() else MaterialHead(lib[first])
    ct = CognitiveTracking(cfg, lib, refs, start=first, doppler_sign=ds, material=mat)
    log = open(out / "frames.jsonl", "w", encoding="utf-8")
    t_wall = time.perf_counter()
    for i in range(n):
        w = wf_of(i)
        if w != ct.waveform:
            ct.set_waveform(w, float(times[i]))
        rec = ct.step(get(i), float(times[i]))
        rec["frame"] = i
        log.write(json.dumps(rec, default=_js) + "\n")
        if not args.quiet and i % 25 == 0:
            conf = [tr for tr in rec["tracks"] if tr["status"] == "confirmed"]
            print(f"frame {i:4d} t {times[i]:6.2f} s  dets {len(rec['dets']):2d}  tracks " + "; ".join(
                f"{tr['id']}: R {tr['range_m']:.2f} th {tr['theta_deg']:+.0f} v ({tr['vx']:+.2f},{tr['vy']:+.2f})"
                + (f" {tr['classes'].get('material', {}).get('label', '')}/{tr['classes'].get('motion', {}).get('label', '')}" if tr['classes'] else "")
                for tr in conf) + f"   {rec['ms']['total']:.0f} ms  doppler sign {rec['doppler_sign']:+d}", flush=True)
    log.close()
    (out / "meta.json").write_text(json.dumps({"take": str(take), "superpose": str(args.superpose) if args.superpose else None,
                                               "frames": n, "wall_s": round(time.perf_counter() - t_wall, 1),
                                               "waveforms": {k: w.describe() for k, w in lib.items()},
                                               "decisions": ct.controller.history}, indent=2, default=_js))
    print(f"{n} frames -> {out}")


def _js(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    main()
