"""
annotate.py - the ground truth of the events of a run, marked by hand on its camera video: light on, flashing, a
person in view, each obstacle of scene.json placed (keys toggle, times are the camera's) -> annotations.json.

    python annotate.py OB1/run_1          (SPACE play/pause, , . one frame, [ ] 2 s, - + speed, l f p 1-9 toggle, u undo, q save)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import paths

KEYS = {ord("l"): "light", ord("f"): "flashing", ord("p"): "person"}


def frame_times(folder: Path) -> tuple[np.ndarray, int]:
    rows = list(csv.DictReader(open(folder / "camera.csv", encoding="utf-8")))
    idx = np.array([int(r["camera_frame"]) for r in rows])
    t = np.array([float(r["t"]) for r in rows])
    return t[np.clip(np.searchsorted(idx, idx[0] + np.arange(idx[-1] - idx[0] + 1), side="right") - 1, 0, len(t) - 1)], idx[0]


def intervals(marks: list, end: float) -> dict:
    out = {}
    for t, name in sorted(marks):
        ivs = out.setdefault(name, [])
        if ivs and ivs[-1][1] is None:
            ivs[-1][1] = t
        else:
            ivs.append([t, None])
    return {k: [[a, b if b is not None else end] for a, b in v] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", help="<session>/<take> under output/takes")
    args = ap.parse_args()
    folder = paths.TAKES / args.take
    scene = json.loads((folder / "scene.json").read_text()) if (folder / "scene.json").exists() else {}
    labels = {f"obstacle:{i}": o.get("label", f"obstacle {i}") for i, o in enumerate(scene.get("obstacles", []), 1)}
    keys = {**KEYS, **{ord(str(i)): f"obstacle:{i}" for i in range(1, min(len(labels), 9) + 1)}}
    times, _ = frame_times(folder)
    cap = cv2.VideoCapture(str(folder / "camera.mp4"))
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(times))
    out = folder / "annotations.json"
    marks = [(t, k) for k, ivs in json.loads(out.read_text())["events"].items() for iv in ivs for t in iv] if out.exists() else []
    j, cur, play, speed, img = 0, -2, False, 1.0, None
    while True:
        if j != cur + 1 and j != cur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        if j != cur:
            ok, img = cap.read()
            cur = j
        if not ok:
            j, play = max(0, j - 1), False
            continue
        t = times[j]
        on = {k for k, ivs in intervals(marks, np.inf).items() for a, b in ivs if a <= t < b}
        view = img.copy()
        lines = [f"t = {t - times[0]:7.2f} s   frame {j}/{n - 1}   speed x{speed:g}   {'PLAY' if play else 'PAUSE'}",
                 "on: " + (", ".join(labels.get(k, k) for k in sorted(on)) or "-"),
                 "l light  f flashing  p person  " + "  ".join(f"{i[-1]} {labels[i]}" for i in labels) + "   u undo  q save"]
        for k, line in enumerate(lines):
            cv2.putText(view, line, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, line, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("annotate", view)
        k = cv2.waitKey(max(1, int(33 / speed)) if play else 0) & 0xFF
        if k in keys:
            marks.append((float(t), keys[k]))
        elif k == ord(" "):
            play = not play
        elif k == ord(","):
            j = max(0, j - 1)
        elif k == ord("."):
            j = min(n - 1, j + 1)
        elif k == ord("["):
            j = max(0, j - 60)
        elif k == ord("]"):
            j = min(n - 1, j + 60)
        elif k in (ord("+"), ord("=")):
            speed = min(8.0, speed * 2)
        elif k == ord("-"):
            speed = max(0.25, speed / 2)
        elif k == ord("u") and marks:
            marks.pop()
        elif k in (ord("q"), 27):
            break
        if play:
            j = min(n - 1, j + 1)
    cv2.destroyAllWindows()
    out.write_text(json.dumps({"video": "camera.mp4", "clock": "camera t (camera.csv), the clock of radar.jsonl.gz",
                               "labels": labels, "events": intervals(marks, float(times[-1]))}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
