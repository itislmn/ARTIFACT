"""
align.py - radar-camera alignment: records reflector takes, clicks their apexes and fits the camera model (rotation,
focal length) by leave-one-out choice.

    python align.py take align_1 --target-xy "0.50:-0.20:small CR 2"   (then: points, solve, check, quick, nominal)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import argparse
import sys
import time

import numpy as np

from backend import overlay as ov
from backend.session import Live
from backend.settings import load_config


def take(argv):
    ap = argparse.ArgumentParser(prog="align.py take")
    ap.add_argument("take")
    ap.add_argument("--target-xy", action="append", default=[], required=True,
                    help='"x:y:label" tape position of a reflector apex (x forward, y right, m); labels from the plan '
                         '(big CR, small CR 1, small CR 2) give the heights')
    ap.add_argument("--session", default=None)
    ap.add_argument("--cli", default=None)
    ap.add_argument("--exposure", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--overwrite", action="store_true", help="record this take again (old one kept in radar/_previous/)")
    args = ap.parse_args(argv)
    cfg = load_config()
    import json
    from backend.settings import ROOT
    plan = json.loads(((ROOT / cfg["radar"]["estimation_dir"]).resolve() / "config" / "session_plan.json").read_text())
    for t in args.target_xy:
        label = t.split(":", 2)[2].strip() if t.count(":") >= 2 else ""
        if label not in plan["reflectors"]:
            raise SystemExit(f'reflector label "{label}" is not in the session plan (heights): use one of '
                             f'{list(plan["reflectors"])} - heights {[(k, v["apex_height_m"]) for k, v in plan["reflectors"].items()]} m, '
                             f'antenna {plan["antenna_height_m"]} m (estimation/config/session_plan.json)')
    live = Live("ALIGNMENT TAKE", cfg, args.session or cfg["session"], args.take, args.exposure, args.camera)
    try:
        t0 = time.perf_counter()
        while live.alive and time.perf_counter() - t0 < 2:
            f = live.frame()
            view = np.array(f.image[::2, ::2])
            ov.text(view, f"ALIGNMENT TAKE {args.take}: RADAR STARTS", 6, 6, ov.BLACK, 2, keep=(args.take,))
            live.show(view)
        rec = live.radar_take(args.cli, "single" if len(args.target_xy) == 1 else "multi", args.target_xy,
                              overwrite=args.overwrite)
    finally:
        live.close({"kind": "alignment take", "targets": args.target_xy})
    print(f"\nnext spot:  python align.py take align_<n> --target-xy ...\nwhen all are recorded:  python align.py points "
          f"{args.take} ...  then  python align.py solve")
    return rec


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "take":
        take(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] in ("nominal", "points", "solve", "check", "quick", "selftest"):
        from backend import alignment
        sys.argv = ["align.py"] + sys.argv[1:]
        alignment.main()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
