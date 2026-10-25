"""
enrol.py - writer enrolment: letters of a written sheet are cut at their gaps and added to the training data of both
readers.

    python -m readers.enrol recordings/CV1/enrol_1 --text ABCDEFGHI
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.settings import MODELS, load_config
from readers.metricsight import ALPHABET
from readers.segment import characters, normalise_char

ENROLLED = MODELS / "enrolled.npz"


def cut(rgb, obj, proj, cfg, n_expected: int):
    from readers import metricsight
    from readers.segment import drop_frame_lines
    c = proj.point(obj["range_m"], obj["azimuth_deg"], cfg["text"]["centre_z_m"])
    rd = metricsight.read(rgb, c, obj["range_m"], proj.f, cfg, keep_debug=True)
    d = rd.debug
    if not d or d.get("line") is None:
        return None, "no text line at this radar object"
    a, b = d["line"]
    h = b - a
    li = drop_frame_lines(d["ink"], d["line"])
    soft = np.clip(-d["norm"][a:b] / 2, 0, 1) * li
    cols = characters(li, min_gap=max(1, int(0.04 * h)), min_width=0.1 * h, max_width=1.1 * h, typical=0.7 * h)
    if len(cols) != n_expected:
        return None, f"{len(cols)} pieces instead of {n_expected} (write with clear gaps; nothing else on the sheet)"
    return np.array([normalise_char(soft[:, c0:c1], li[:, c0:c1]) for c0, c1 in cols]), ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", type=Path, help="take folder with snapshots of ONE enrolment sheet")
    ap.add_argument("--text", required=True, help="the characters on the sheet, left to right, e.g. ABCDEFGHI")
    args = ap.parse_args()
    text = args.text.upper().replace(" ", "")
    if any(ch not in ALPHABET for ch in text):
        raise SystemExit(f"only {ALPHABET}")
    cfg = load_config()
    from backend.alignment import Projector
    from backend.imgio import read_png
    proj = Projector()
    X, y, src = [], [], []
    for js in sorted(args.take.glob("snapshot_*.json")):
        meta = json.loads(js.read_text())
        png = js.with_name(js.stem + "_raw.png")
        if not png.exists() or not meta.get("objects"):
            continue
        rgb = read_png(png)
        best = None
        for o in meta["objects"]:
            crops, why = cut(rgb, o, proj, cfg, len(text))
            if crops is not None:
                best = crops
                break
        if best is None:
            print(f"  {png.name}: skipped - {why}")
            continue
        X.append(best)
        y += [ALPHABET.index(ch) for ch in text]
        src += [f"{args.take.name}/{png.name}"] * len(text)
        print(f"  {png.name}: {len(text)} letters {text}")
    if not X:
        raise SystemExit("no usable snapshot: write the sheet with clear gaps, press s while the bar is on it")
    X = np.concatenate(X)
    if ENROLLED.exists():
        old = np.load(ENROLLED)
        keep = np.array([not s.startswith(args.take.name + "/") for s in old["src"]])
        X = np.concatenate([old["X"][keep], X])
        y = list(old["y"][keep]) + y
        src = list(old["src"][keep]) + src
    np.savez(ENROLLED, X=X.astype(np.float32), y=np.array(y), src=np.array(src))
    have = sorted({ALPHABET[i] for i in y})
    print(f"-> {ENROLLED}: {len(y)} letters, classes {''.join(have)}; then: python -m readers.train")


if __name__ == "__main__":
    main()
