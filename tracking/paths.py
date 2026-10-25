"""
paths.py - where everything is (recordings, output, the sibling experiments) and their import paths; every program
imports it first.

    import paths
"""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ESTIMATION = REPO / "estimation"
CV = REPO / "computer_vision"
MATERIAL = REPO / "material_classification"
CONFIG = ROOT / "config.toml"
WAVEFORM_DIR = ROOT / "config" / "waveforms"
LAYOUT = ESTIMATION / "config" / "awr2944pevm_layout.json"
RECORDINGS = ROOT / "recordings"
OUTPUT = ROOT / "output"
RESULTS = OUTPUT / "cache"
MODELS = ROOT / "models"

for p in (ESTIMATION, CV, REPO):
    if str(p) not in sys.path:
        sys.path.append(str(p))


def load_config(path: Path = CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def session_dir(session: str) -> Path:
    return RECORDINGS / session


def alignment_dir(session: str) -> Path:
    own = session_dir(session)
    return own if (own / "alignment.json").exists() else CV / "calibration"


def radar_dir(session: str) -> Path:
    return RECORDINGS / session / "radar"
