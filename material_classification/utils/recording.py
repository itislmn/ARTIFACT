"""
recording.py - loads a take (.npy) and its frame time stamps (.meta.json), dropping leftover frames of an earlier run.

    from material_classification.utils.recording import load_take
"""
import json
from pathlib import Path

import numpy as np


def meta_path(recording_path):
    p = Path(recording_path)
    return p.with_name(p.stem + ".meta.json")


def save_meta(path, frame_t, extra=None):
    with open(path, "w") as f:
        json.dump({"frame_t": np.asarray(frame_t, float).round(6).tolist(), **(extra or {})}, f)


def load_frame_times(path):
    p = meta_path(path)
    if not p.exists():
        return None
    t = json.load(open(p)).get("frame_t")
    return np.asarray(t, float) if t else None


def last_stream_start(frame_t, gap_s=1.0):
    gaps = np.flatnonzero(np.diff(np.asarray(frame_t, float)) > gap_s)
    return int(gaps[-1]) + 1 if gaps.size else 0


def load_take(path):
    stack = np.load(path, mmap_mode="r")
    t = load_frame_times(path)
    if t is not None and len(t) == len(stack):
        stack = stack[last_stream_start(t):]
    return stack
