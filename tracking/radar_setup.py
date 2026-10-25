"""
radar_setup.py - the radar references of a session: the empty-room background of every chirp profile and the channel
calibration on a corner reflector.

    python radar_setup.py background        (then calibration --target-xy "1.00:0.00:big CR", show)
"""
from __future__ import annotations

import argparse
import sys

import paths
from hardware.references import load_references, profile_key
from methods.radar_model import waveform_library


def finite_cfg(wf, n_frames: int = 20):
    lines = []
    for line in wf.path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("frameCfg"):
            v = line.split()
            v[4] = str(n_frames)
            line = " ".join(v)
        lines.append(line)
    out = wf.path.with_name(wf.path.stem + "_take.cfg")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def capture(args_list: list[str]):
    import estimation.capture as cap
    cap.RAW = paths.RECORDINGS
    sys.argv = ["core.capture"] + args_list
    cap.main()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["background", "calibration", "show"])
    ap.add_argument("--target-xy", default=None, help="calibration reflector 'x:y:label' (tape measure)")
    ap.add_argument("--cli", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = paths.load_config()
    lib = waveform_library(cfg)
    port = args.cli or cfg["radar"]["cli_port"]
    session = f"{cfg['session']}/radar"
    if args.step == "show":
        load_references(cfg, lib)
        return
    if args.step == "background":
        done = set()
        for wf in lib.values():
            key = profile_key(wf.cfg)
            if key in done:
                continue
            done.add(key)
            print(f"\n=== background of chirp profile {key} (waveform '{wf.name}') - nobody in front of the radar ===")
            capture(["--cli", port, "--session", session, "--name", f"background_{key}", "--scenario", "background",
                     "--cfg", str(finite_cfg(wf))] + (["--overwrite"] if args.overwrite else []))
    else:
        if not args.target_xy:
            raise SystemExit('--target-xy "x:y:label" of the reflector (e.g. "1.00:0.00:big CR")')
        wf = lib[cfg["radar"]["start_waveform"]]
        capture(["--cli", port, "--session", session, "--name", "calibration", "--scenario", "calibration",
                 "--target-xy", args.target_xy, "--cfg", str(finite_cfg(wf)),
                 "--plan", str(paths.ESTIMATION / "config" / "session_plan.json")] + (["--overwrite"] if args.overwrite else []))
    load_references(cfg, lib)


if __name__ == "__main__":
    main()
