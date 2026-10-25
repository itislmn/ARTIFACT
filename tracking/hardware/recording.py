"""
recording.py - a live take's raw radar data back as frames with their chirp, time and references, for replay.py.

    from hardware.recording import open_live_take
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np

import paths  # noqa: F401  (import paths of ../estimation, ../computer_vision)
from estimation.core.radar import frame_from_words
from hardware.references import load_references
from methods.radar_model import library_for


def open_live_take(take: Path, cfg_now: dict):
    take = Path(take)
    raw = take / "radar_raw.bin"
    if not raw.exists():
        raise SystemExit(f"{raw} missing: record the take with  python track_live.py --take ... --raw")
    cfg_file = take / "config.toml"
    cfg = tomllib.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else cfg_now
    lib = library_for(cfg)
    refs, unusable = load_references(cfg, lib)
    index = [json.loads(line) for line in open(take / "radar_raw.jsonl", encoding="utf-8")]
    fb = next(iter(lib.values())).cfg.wire_bytes_per_frame(True)
    words = np.memmap(raw, dtype="<i2", mode="r")
    n = min(len(index), words.size * 2 // fb)
    per = fb // 2
    times = np.array([r["t"] for r in index[:n]], float)
    names = [r["waveform"] or cfg["radar"].get("start_waveform", next(iter(lib))) for r in index[:n]]

    def get(i):
        return frame_from_words(np.asarray(words[i * per:(i + 1) * per]), lib[names[i]].cfg, True)

    return lib, refs, n, get, times - times[0], lambda i: names[i]
