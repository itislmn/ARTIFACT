r"""
record_session.py - guided recording of a session: shows what to place, counts down, records the take, checks it
against the empty room and logs it in data/raw/LOG.md (plans A, A2, B; existing takes are skipped).

    python record_session.py --cli COM4 --plan B          (--list shows the plan, --redo --only <name> records again)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from material_classification.dataset import label_of  # noqa: E402
from material_classification.utils.recording import last_stream_start, load_frame_times, load_take, meta_path  # noqa: E402

PLAN = [
    ("background", "bg_take1", 15, 8, "EMPTY scene: no object; stool/box, rain rig (dry), tub stay. Stand behind the radar."),
    ("background", "bg_take2", 15, 8, "Same as before, nothing moved."),
    ("none", "empty_take1", 15, 8, "EMPTY scene = EXACTLY the scene of the background takes: the mount/frame, stool, rain rig and everything else that was there stays. Remove only the test object, hands and funnel. You at your marked spot."),
    ("none", "empty_take2", 15, 8, "EMPTY scene = EXACTLY the scene of the background takes: the mount/frame, stool, rain rig and everything else that was there stays. Remove only the test object, hands and funnel. You at your marked spot."),
    ("none", "empty_take3", 15, 8, "EMPTY scene = EXACTLY the scene of the background takes: the mount/frame, stool, rain rig and everything else that was there stays. Remove only the test object, hands and funnel. You at your marked spot."),
    ("none", "empty_take4", 15, 8, "EMPTY scene = EXACTLY the scene of the background takes: the mount/frame, stool, rain rig and everything else that was there stays. Remove only the test object, hands and funnel. You at your marked spot."),

    ("metal-static", "gold8cm_aimed_take1", 12, 6, "Gold 8 cm corner reflector on its plastic stand at the spot, opening AIMED at the radar, centre at radar height."),
    ("metal-static", "gold8cm_rot15_take2", 12, 6, "Same, rotated about 15 degrees about the vertical axis."),
    ("metal-static", "alu2p8cm_aimed_take1", 12, 6, "Aluminium 2.8 cm corner reflector on a plastic stand, aimed at the radar, at radar height."),
    ("metal-static", "alu2p8cm_rot15_take2", 12, 6, "Same, rotated about 15 degrees."),
    ("metal-static", "steelball_box_take1", 12, 6, "Large steel ball on the plastic box (ball centre at radar height)."),
    ("metal-static", "steelball_box_take2", 12, 6, "Same, shifted about 10 cm sideways."),
    ("metal-rigid", "gold8cm_pendulum_take1", 12, 12, "Gold 8 cm corner reflector hanging on a thin string from the frame at the spot (frame in the background). Pull it about 10 cm toward you along the boresight, release, then WALK to your spot at once: the countdown (12 s) plus the radar setup (8 s) give it time. It must be swinging along the boresight during the take, and you must not be near it."),
    ("metal-rigid", "alu2p8cm_pendulum_take2", 12, 12, "Aluminium 2.8 cm corner reflector hanging on a thin string from the frame, same: release, then walk to your spot."),
    ("dielectric-rigid", "plasticbox_pendulum_take1", 12, 12, "Light plastic box (or your bag) hanging on a thin string from the frame; make sure it is big enough (about 20 cm) so that it is clearly detected. Release, walk to your spot."),

    ("dielectric-static", "stand_alone_take1", 12, 6, "CONTROL: the plastic stand of the corner reflector alone, same height."),
    ("dielectric-static", "box_alone_take1", 12, 6, "CONTROL: the plastic box alone, same height."),
    ("dielectric-static", "bag_take1", 12, 6, "Your bag, empty, no metal parts facing the radar, on the spot."),
    ("dielectric-static", "bag_take2", 12, 6, "Bag turned 90 degrees."),
    ("dielectric-static", "plasticboxes_take1", 12, 6, "Plastic boxes stacked at the spot (no metal clips)."),
    ("dielectric-static", "plasticboxes_take2", 12, 6, "Boxes rotated 90 degrees."),
    ("dielectric-static", "radome_take1", 12, 6, "Radome plate upright at the spot, face-on."),
    ("dielectric-static", "radome_take2", 12, 6, "Radome tilted about 15 degrees. (Skip this and the previous one with q if you have none.)"),

    ("waterlike-static", "bottle_still_take1", 12, 6, "Transparent plastic bottle, filled >= 80 %, at the spot. Let the water settle 30 s first."),
    ("waterlike-static", "bottle_rippled_take2", 12, 6, "Same bottle: stir once with a plastic stick, step away, press Enter."),
    ("waterlike-static", "bottle_turned_take3", 12, 6, "Same bottle turned 90 degrees, still water (wait 30 s after moving it)."),
    ("waterlike-static", "bottle_fill80_take4", 12, 6, "Same bottle, fill level about 80 %, still water (wait 30 s)."),

    ("waterlike-static", "human_still_take1", 12, 15, "YOU: stand at the spot facing the radar, arms down, normal breathing. Walk there DURING the countdown."),
    ("waterlike-static", "human_side_take2", 12, 15, "YOU: turned 90 degrees to the radar, standing still."),
    ("waterlike-articulated", "human_gestures_take1", 12, 15, "YOU: at the spot, move arms and hands slowly the whole time."),
    ("waterlike-articulated", "human_walk_take1", 12, 15, "YOU: walk slowly back and forth along the boresight between 1 m and 3 m from the radar."),

    ("waterlike-distributed", "rain_holed_bottle_take1", 10, 8, "RAIN: holed bottle on the rig with all holes open, tub below. Feed water from behind the radar through the funnel; start pouring ~2 s before the [RECORDING] line."),
    ("waterlike-distributed", "rain_holed_bottle_take2", 10, 8, "Same, a bit more water per second."),
    ("waterlike-distributed", "pour_stream_take1", 10, 8, "STEADY STREAM: tape over all holes except one, feed through the funnel, same timing."),
    ("waterlike-distributed", "pour_stream_take2", 10, 8, "Same with a second hole open."),

    ("none", "empty_end_take5", 15, 8, "END OF SESSION: the same EMPTY scene as the backgrounds (mount stays, objects removed). If this differs from the background, the room drifted."),
]

SWING = (" Hold it at the spot and swing it gently toward and away from the radar (about 10 cm) "
         "from when [RECORDING] appears until the end. Stand at the left of the radar.")
PLAN_A2 = [
    ("background", "bg_take1", 15, 8, "EMPTY scene as it stays for this session (table, no object). You at the left of the radar."),
    ("background", "bg_take2", 15, 8, "Same as before, nothing moved."),
    ("none", "empty_take1", 15, 8, "EMPTY scene again, you at the left of the radar."),

    ("dielectric-rigid", "plasticbox_swing_take2", 12, 6, "Plastic box." + SWING),
    ("dielectric-rigid", "bag_swing_take1", 12, 6, "Your bag (no metal parts facing the radar)." + SWING),
    ("metal-rigid", "gold8cm_swing_take2", 12, 6, "Gold 8 cm corner reflector, opening toward the radar." + SWING),

    ("waterlike-articulated", "human_gestures_take2", 12, 10, "YOU: behind the table, body still, move both arms the whole time."),
    ("waterlike-articulated", "human_march_take1", 12, 10, "YOU: behind the table, march on the spot (knees up, arms swinging)."),
    ("waterlike-articulated", "human_walk_take2", 12, 10, "YOU: behind the table, walk slowly left-right-left (short steps), stay within 1.6 m of the radar."),

    ("none", "empty_end_take2", 15, 8, "END: the same EMPTY scene as the backgrounds."),
]

PUT = " Put it on the table at the spot, then step to the left of the radar."
PLAN_B = [
    ("background", "bg_take1", 15, 8, "EMPTY scene (table, no object). You at the left of the radar."),
    ("background", "bg_take2", 15, 8, "Same as before, nothing moved."),
    ("none", "empty_take1", 15, 8, "EMPTY scene again."),

    ("metal-static", "gold8cm_take1", 12, 6, "Gold 8 cm corner reflector, aimed at the radar." + PUT),
    ("metal-static", "steelball_take1", 12, 6, "Steel ball." + PUT),
    ("metal-rigid", "gold8cm_swing_take1", 12, 6, "Gold 8 cm corner reflector." + SWING),
    ("dielectric-static", "bag_take1", 12, 6, "Your bag." + PUT),
    ("dielectric-static", "plasticboxes_take1", 12, 6, "Stacked plastic boxes." + PUT),
    ("dielectric-rigid", "plasticbox_swing_take1", 12, 6, "Plastic box." + SWING),
    ("waterlike-static", "bottle_take1", 12, 6, "Water bottle (>= 80 % full), still water." + PUT),
    ("waterlike-static", "human_still_take1", 12, 10, "YOU: stand behind the table facing the radar, still."),
    ("waterlike-articulated", "human_gestures_take1", 12, 10, "YOU: behind the table, body still, move both arms the whole time."),
    ("waterlike-articulated", "human_march_take1", 12, 10, "YOU: behind the table, march on the spot."),
    ("waterlike-distributed", "rain_take1", 10, 8, "RAIN: holed bottle on the rig, all holes open; pour from the side, start ~2 s before [RECORDING]."),
    ("waterlike-distributed", "stream_take1", 10, 8, "STREAM: one hole open, same timing."),

    ("none", "empty_end_take2", 15, 8, "END: the same EMPTY scene as the backgrounds."),
]
PLANS = {"A": PLAN, "A2": PLAN_A2, "B": PLAN_B}


class TakeChecker:

    MIN_DB = {"dielectric-static": 11, "dielectric-rigid": 11}

    def __init__(self, session, spot, max_range=1.6):
        from material_classification.features import FeatureExtractor
        from material_classification.run_realtime_sensing import load_background
        from material_classification.utils.micro_doppler import FramePreprocessor, RadarParams, load_channel_cal
        self.params = RadarParams.from_cfg(HERE / "config" / "awr2944_material_spec.cfg")
        self.pre = FramePreprocessor(self.params, max_range_m=max_range + 1.0, channel_cal=load_channel_cal())
        self.fx = FeatureExtractor(self.params, self.pre, max_range_m=max_range)
        self.spot = spot
        bg_dir = Path(session) / "background"
        self.bg = None
        if bg_dir.is_dir() and list(bg_dir.glob("*.npy")):
            self.bg = load_background(bg_dir, self.params, self.pre)
        else:
            print("[check] no background takes yet: only the signal level is checked")

    def _frames(self, stack, start, n=40):
        return [self.pre.process(c) for c in stack[start:start + n]]

    def check(self, path, folder, window=40):
        from material_classification.features import background_map
        stack = load_take(path)
        x = np.abs(np.asarray(stack[:3], dtype=np.float32))
        m = float(x.max())
        clipped = m > 0 and float((x == m).mean()) > 1e-4
        mid = max(0, (len(stack) - window) // 2)
        if folder == "background":
            others = [f for f in sorted(Path(path).parent.glob("*.npy")) if f != Path(path)]
            if not others or clipped:
                return (not clipped), f"max sample {m:.0f}" + ("  !! CLIPPED" if clipped else "")
            ref = background_map(f for o in others for f in self._frames(load_take(o), mid, 60))
            ex = self.fx.window(self._frames(stack, mid), ref).features["change_db"]
            return ex < 6, (f"differs from the other background take(s) by at most {ex:.1f} dB" +
                            ("" if ex < 6 else "  !! something in the room changed or moved: find it and retake"))
        if self.bg is None:
            return (not clipped), f"no background takes yet, max sample {m:.0f}"
        n = len(stack)
        wins = [self.fx.window(self._frames(stack, max(0, min(n - window, int(q * n) - window // 2))), self.bg)
                for q in (0.2, 0.5, 0.8)]
        ex = np.array([w.features["change_db"] for w in wins])
        rg = np.array([w.range_m for w in wins])
        info = ("change " + " / ".join(f"{e:.0f}" for e in ex) + " dB at " +
                " / ".join(f"{r:.2f}" for r in rg) + " m (20 / 50 / 80 % of the take)")
        if folder == "none":
            return ex.max() < 6 and not clipped, info + ("" if ex.max() < 6 else
                   "  !! the scene differs from the background: something moved or you stood elsewhere")
        good, notes = True, ""
        motion = folder.partition("-")[2]
        if motion == "distributed":
            if (ex >= 8).sum() < 2:
                good, notes = False, "  !! the water is hardly detected (needs 8 dB in two of three windows): pour earlier/more"
        elif ex.min() < self.MIN_DB.get(folder, 15):
            good, notes = False, f"  !! OBJECT NOT DETECTED in every window (weakest {ex.min():.0f} dB, needs {self.MIN_DB.get(folder, 15)})"
        elif motion != "articulated":
            if rg.max() - rg.min() > 0.3:
                good, notes = False, "  !! the strongest change jumps between ranges: you or another object is detected"
            elif self.spot and abs(np.median(rg) - self.spot) > 0.25:
                good, notes = False, f"  !! strongest change is not at the spot ({self.spot:.2f} m)"
        if rg.min() < 0.45:
            notes += "  (closer than 0.45 m: the radar leakage distorts the signal, move the object out)"
        if clipped:
            good, notes = False, notes + "  !! signal CLIPPED: move the object further away or use a smaller one"
        return good, info + notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", required=True, help="radar CLI port, e.g. COM4")
    ap.add_argument("--plan", default="A", choices=list(PLANS), help="A: main session (done), A2: short top-up of A (no water), B: replication test")
    ap.add_argument("--session", default=None, help="folder (default: data/raw/session<plan>)")
    ap.add_argument("--only", default=None, help="only takes whose path contains this text")
    ap.add_argument("--redo", action="store_true", help="record takes again even if the file exists")
    ap.add_argument("--list", action="store_true", help="show the plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="do not start the radar")
    ap.add_argument("--spot", type=float, default=None,
                    help="distance of the object spot in m: the check warns if the strongest change is elsewhere")
    ap.add_argument("--no-check", action="store_true", help="skip the automatic check after every take")
    ap.add_argument("--max-range", type=float, default=1.6, help="objects are searched up to this range (m)")
    args = ap.parse_args()

    session = Path(args.session or HERE / "data" / "raw" / f"session{args.plan}")
    todo = []
    for folder, name, secs, cd, what in PLANS[args.plan]:
        if folder != "background" and label_of(folder) is None:
            sys.exit(f"plan error: folder '{folder}' is not a known label")
        path = session / folder / f"{name}.npy"
        if args.only and args.only not in path.relative_to(session).as_posix():
            continue
        todo.append((path, secs, cd, what, path.exists()))
    if args.list:
        for i, (p, secs, cd, what, done) in enumerate(todo, 1):
            print(f"{i:2d}. [{'done' if done else '    '}] {p.relative_to(session).as_posix():48s} {secs:2d} s  {what}")
        return

    todo = [t for t in todo if args.redo or not t[4]]
    checker = None if (args.no_check or args.dry_run) else TakeChecker(session, args.spot, args.max_range)
    print(f"{len(todo)} take(s) to record into {session}. Enter = start, s = skip, q = quit.\n")
    quit_all = False
    for i, (path, secs, cd, what, _) in enumerate(todo, 1):
        if quit_all:
            break
        folder = path.parent.name
        print(f"--- take {i}/{len(todo)}: {path.relative_to(session).as_posix()}  ({secs} s)")
        print(f"    {what}")
        while True:
            ans = input("    ready? [Enter/s/q] ").strip().lower()
            if ans == "q":
                quit_all = True
                break
            if ans == "s":
                break
            path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(HERE / "run_realtime_sensing.py"), "--cli", args.cli, "--no-plot",
                   "--countdown", str(cd), "--duration", str(secs), "--record", str(path)]
            if args.dry_run:
                print("    would run:", " ".join(cmd))
                break
            try:
                subprocess.run(cmd, check=False)
            except KeyboardInterrupt:
                print("\n    interrupted")
                quit_all = True
                break
            ok, report = True, ""
            if not path.exists():
                print("    !! no file was written")
                ok = False
            else:
                frame_t = load_frame_times(path)
                n = len(frame_t) - last_stream_start(frame_t) if frame_t is not None else 0
                need = int(0.85 * secs * 25)
                short = n < need
                print(f"    {'!! TOO SHORT' if short else 'OK'}: {n} frames (expected about {secs * 25})")
                ok = not short
                report = f"{n} frames"
                if checker is not None and ok:
                    good, msg = checker.check(path, folder)
                    print(f"    {'OK' if good else '!!'} {msg}")
                    ok, report = good, f"{report}, {msg}"
            choice = input(f"    keep this take? [Enter = {'keep' if ok else 'REDO'}, k = keep, r = redo, q = quit] ").strip().lower()
            if choice == "q":
                quit_all = True
                break
            if choice == "r" or (choice == "" and not ok):
                path.unlink(missing_ok=True)
                meta_path(path).unlink(missing_ok=True)
                print("    take deleted - record it again")
                continue
            with open(session.parent / "LOG.md", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M')} | {path.relative_to(session.parent).as_posix()} | "
                        f"{report} | {what}\n")
            print()
            break
    print("done.")


if __name__ == "__main__":
    main()
