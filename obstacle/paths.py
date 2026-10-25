"""
paths.py - where everything is (configuration, calibration, takes, videos, output) and the configuration loader;
every program imports it first.

    import paths
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.toml"
CALIB = ROOT / "backend" / "calibration"
WAVEFORMS = ROOT / "backend" / "waveforms"
LAYOUT = ROOT / "backend" / "awr2944pevm_layout.json"
OUTPUT = ROOT / "output"
TAKES = OUTPUT / "takes"
VIDEOS = OUTPUT / "videos"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
DATASET = OUTPUT / "dataset"


def load_config(path: Path = CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def session_dir(session: str) -> Path:
    return TAKES / session


def reference_dir(session: str) -> Path:
    return TAKES / session / "reference"
