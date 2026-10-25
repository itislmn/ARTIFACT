"""
settings.py - where everything lives (recordings, calibration, models, output) and the one settings file config.toml.

    from backend.settings import OUTPUT, load_config
"""
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.toml"
RECORDINGS = ROOT / "recordings"
CALIB = ROOT / "calibration"
MODELS = ROOT / "readers" / "models"
OUTPUT = ROOT / "output"
FIGURES, VIDEOS, CACHE = OUTPUT / "figures", OUTPUT / "videos", OUTPUT / "cache"


def load_config(path=CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def radar_dir(session: str) -> Path:
    return RECORDINGS / session / "radar"


def take_dir(session: str, take: str) -> Path:
    return RECORDINGS / session / take
