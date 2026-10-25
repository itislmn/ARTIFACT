"""
radar_setup.py - the radar references of a session: the empty-room background, the channel calibration with one
reflector, and the surveyed light-source position.

    python radar_setup.py background                (calibration --target-xy ..., show, source --side left)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import argparse
import subprocess
import sys

from backend.settings import ROOT, load_config, radar_dir


def capture(session, name, scenario, port, targets, overwrite):
    cmd = [sys.executable, "-m", "backend.radar_capture", "--session", session, "--name", name,
           "--scenario", scenario, "--cli", port] + sum((["--target-xy", t] for t in targets), [])
    if overwrite:
        cmd.append("--overwrite")
    print(" ".join(cmd[1:]))
    return subprocess.run(cmd, cwd=ROOT).returncode


def survey_source(session: str, cfg: dict, side: str):
    import json
    import numpy as np
    from backend.radar_link import cluster
    from backend.settings import CALIB, RECORDINGS
    checks = sorted((RECORDINGS / session).glob("check_*/radar_live.jsonl"))
    if not checks:
        raise SystemExit("no radar log: run  python check_system.py radar  with the light source in place first")
    rows = [json.loads(line) for line in open(checks[-1], encoding="utf-8")]
    dets = [np.array(r["dets"], float).reshape(-1, 2) for r in rows if r.get("type") == "frame"]
    objs = cluster(dets, 2 * 0.0836, 4.78, 0.2)
    print(f"{checks[-1].parent.name}: {len(dets)} radar frames, objects: "
          + "; ".join(f"{o['range_m']:.2f} m {o['azimuth_deg']:+.1f} deg ({100 * o['rate']:.0f} %)" for o in objs))
    if len(objs) < 2:
        raise SystemExit("the radar sees fewer than 2 objects (text + light source): turn the source and check again")
    text = min(objs, key=lambda o: abs(o["azimuth_deg"]))
    cand = [o for o in objs if o is not text and (o["azimuth_deg"] < text["azimuth_deg"]) == (side == "left")]
    if not cand:
        raise SystemExit(f"no object on the {side} of the text object ({text['range_m']:.2f} m {text['azimuth_deg']:+.1f} deg)")
    src = max(cand, key=lambda o: o["rate"])
    CALIB.mkdir(exist_ok=True)
    (CALIB / "light_source.json").write_text(json.dumps({**src, "surveyed_from": checks[-1].parent.name}, indent=2))
    print(f"light source saved: {src['range_m']:.3f} m, {src['azimuth_deg']:+.2f} deg ({100 * src['rate']:.0f} % of frames) "
          f"-> calibration/light_source.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["background", "calibration", "show", "source"])
    ap.add_argument("--target-xy", action="append", default=[], help='calibration: "x:0.00:big CR" (tape, metres)')
    ap.add_argument("--session", default=None)
    ap.add_argument("--cli", default=None)
    ap.add_argument("--overwrite", action="store_true", help="record again (the old take is kept in _previous/)")
    ap.add_argument("--side", choices=["left", "right"], default="left", help="source: on which side of the text the light is")
    args = ap.parse_args()
    cfg = load_config()
    session, port = args.session or cfg["session"], args.cli or cfg["radar"]["cli_port"]
    if args.what == "source":
        survey_source(session, cfg, args.side)
        return
    if args.what == "background":
        input("Empty scene: no reflectors, no lamp, no board, nobody in front of the radar. Enter = record ...")
        rc = capture(session, "empty", "background", port, [], args.overwrite)
    elif args.what == "calibration":
        if len(args.target_xy) != 1:
            raise SystemExit('one reflector straight ahead: --target-xy "2.00:0.00:big CR"')
        input(f"Reflector at {args.target_xy[0]}, nothing else new in the scene. Enter = record ...")
        rc = capture(session, "cal_0deg", "calibration", port, args.target_xy, args.overwrite)
    else:
        rc = 0
    if rc:
        raise SystemExit(rc)
    from backend.radar_link import RadarTakes
    rt = RadarTakes(radar_dir(session), cfg["radar"])
    for name, (path, doc) in rt.takes.items():
        objs, ms = rt.objects(name)
        print(f"  {name:<12} [{doc.get('scenario')}]  {doc.get('frames')} frames, {doc.get('lost_packets')} packets lost; "
              f"Infini-Rex objects after background subtraction: "
              + ("; ".join(f"{o['range_m']:.2f} m {o['azimuth_deg']:+.1f} deg" for o in objs) or "none"))
    print("expected: 'empty' -> none (it is the background itself); 'cal_0deg' -> the reflector near 0 deg")


if __name__ == "__main__":
    main()
