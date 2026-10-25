"""
train_frame_spacing.py - the same model trained on every 2nd and 3rd frame of each window (80/120 ms spacing, as in
the tracking and obstacle experiments); writes models/material_motion_model_100ms.json.

    python train_frame_spacing.py            (--add / --check: tracked windows.csv to learn from / to test on)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from material_classification.classifier import MaterialMotionModel  # noqa: E402
from material_classification.dataset import CFG, HOP, WINDOW, label_of, read_table  # noqa: E402
from material_classification.features import FEATURES, FeatureExtractor, background_map  # noqa: E402
from material_classification.evaluate import evaluate, subset, summary, take_table  # noqa: E402
from material_classification.utils.micro_doppler import FramePreprocessor, RadarParams, load_channel_cal  # noqa: E402
from material_classification.utils.recording import load_take  # noqa: E402

STEPS = (2, 3)
TEST_SESSION = "sessionB"


def extract(session: Path, steps=STEPS) -> list[Path]:
    params = RadarParams.from_cfg(CFG)
    pre = FramePreprocessor(params, max_range_m=2.6, channel_cal=load_channel_cal())
    fx = FeatureExtractor(params, pre)
    outs = {k: HERE / "output" / f"features_{session.name}_every{k}.csv" for k in steps}
    if all(p.exists() for p in outs.values()):
        print(f"[{session.name}] feature tables exist")
        return list(outs.values())
    background = background_map(pre.process(c) for f in sorted((session / "background").glob("*.npy"))
                                for c in load_take(f))
    rows = {k: [] for k in steps}
    for folder in sorted(p for p in session.iterdir() if p.is_dir()):
        label = label_of(folder.name)
        if label is None:
            continue
        for path in sorted(folder.glob("*.npy")):
            frames = [pre.process(c, i * params.frame_period_s) for i, c in enumerate(load_take(path))]
            for k in steps:
                for s in range(0, len(frames) - WINDOW + 1, HOP):
                    w = fx.window(frames[s:s + WINDOW:k], background)
                    rows[k].append({"session": session.name, "take": f"{folder.name}/{path.stem}",
                                    "material": label[0], "motion": label[1],
                                    "t_start_s": round(s * params.frame_period_s, 2), "detected": int(w.detected),
                                    "range_m": round(w.range_m, 3), "azimuth_deg": round(w.azimuth_deg, 1),
                                    **{n: round(v, 5) if np.isfinite(v) else "" for n, v in w.features.items()}})
            print(f"  {session.name} {folder.name}/{path.stem}: {len(frames)} frames", flush=True)
    for k, p in outs.items():
        with open(p, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[k][0]))
            wr.writeheader()
            wr.writerows(rows[k])
        print(f"[{session.name}] every {k}: {len(rows[k])} windows -> {p}")
    return list(outs.values())


def report(name, r, outdir: Path):
    lines, nums = summary(r)
    takes = take_table(r)
    lines.append(f"takes recognised by majority vote: {np.mean([t['vote_ok'] for t in takes]) * 100:.0f} % of {len(takes)}")
    det = r["detected_object"]
    for head in ("motion", "material"):
        per = []
        for c in sorted(set(r[head][det])):
            m = det & (r[head] == c)
            per.append(f"{c} {np.mean(r[head + '_pred'][m] == c) * 100:.0f} %")
        lines.append(f"per class {head}: " + ", ".join(per))
    lines.insert(0, name)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


TRUTH = {"ball": ("dielectric", "rigid"), "person": ("waterlike", "articulated")}


def tracked_table(path: Path) -> Path:
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["who"] not in TRUTH:
            continue
        mat, mot = TRUTH[r["who"]]
        rows.append({"session": f"tracked_{r['take']}", "take": f"{r['who']}/track{r['track']}", "material": mat,
                     "motion": mot, "t_start_s": r["t"], "detected": 1, "range_m": r["range_m"], "azimuth_deg": 0.0,
                     **{n: r[n] for n in FEATURES}})
    out = HERE / "output" / f"features_tracked_{Path(path).parent.name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{path}: {len(rows)} windows with a known object -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add", nargs="*", default=[], help="windows.csv of tracked takes to learn from")
    ap.add_argument("--check", nargs="*", default=[], help="windows.csv of OTHER tracked takes to test on")
    args = ap.parse_args()
    sessions = sorted(p for p in (HERE / "data" / "raw").glob("session*") if (p / "background").is_dir())
    tables = {s.name: extract(s) for s in sessions}
    train = [p for s, ps in tables.items() if s != TEST_SESSION for p in ps] + [tracked_table(Path(a)) for a in args.add]
    final = HERE / "output" / "final_100ms"
    tr = read_table(*train)
    report(f"LEAVE-ONE-TAKE-OUT on {', '.join(p.stem for p in train)} (frames every 80 / 120 ms)",
           evaluate(tr), final / "crossval")
    if TEST_SESSION in tables:
        report(f"TEST on {TEST_SESSION} every 2 / 3 frames (trained on the training sessions, same spacings)",
               evaluate(tr, read_table(*tables[TEST_SESSION])), final / "test_sessionB")
    for c in args.check:
        report(f"TEST on the tracked windows of {Path(c).parent.name} (not learned from)",
               evaluate(tr, read_table(tracked_table(Path(c)))), final / f"test_{Path(c).parent.name}")
    model = MaterialMotionModel().fit(subset(tr, tr["detected"] & (tr["motion"] != "none")))
    model.settings = {"trained_on": [p.stem for p in train], "window": WINDOW, "hop": HOP, "frame_steps": list(STEPS),
                      "frame_spacing_ms": [40 * k for k in STEPS]}
    model.save(HERE / "models" / "material_motion_model_100ms.json")
    print("model -> models/material_motion_model_100ms.json")


if __name__ == "__main__":
    main()
