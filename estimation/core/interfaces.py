"""
interfaces.py - the containers every method shares: Settings (stated design constants, never tuned), Gate (the
azimuth snapshots of one range bin), DoAResult and MethodSpec; a method is estimate(gate, array, settings, rng).

    from estimation.core.interfaces import Gate, Settings
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Settings:
    alpha: float = 0.1
    pfa: float = 1e-3
    n_groups: int = 1
    trex_K: int = 20
    trex_max_dummy_multiplier: int = 10
    dummy_distribution: str = "gaussian"
    gate_pfa: float = 1e-3
    r_min_m: float = 0.5
    r_max_m: float = float("inf")
    y_max_m: float = float("inf")
    dnn_threshold: float = 0.5
    keep_info: bool = False


@dataclass
class Gate:
    bin: int
    range_m: float
    snapshots: np.ndarray
    noise_var: float
    power_db: float = 0.0
    channel_noise: np.ndarray | None = None

    @property
    def K(self) -> int:
        return int(self.snapshots.shape[1])

    def groups(self, n_groups: int) -> int:
        return max(1, min(int(n_groups), self.K))

    def mean_snapshot(self) -> np.ndarray:
        return self.snapshots.mean(axis=1)

    def stacked_snapshot(self, n_groups: int) -> np.ndarray:
        groups = np.array_split(np.arange(self.K), self.groups(n_groups))
        return np.concatenate([self.snapshots[:, g].mean(axis=1) for g in groups])

    def noise_var_of_mean(self, n_groups: int = 1) -> float:
        return float(self.noise_var * self.groups(n_groups) / self.K)


@dataclass
class DoAResult:
    angles_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    info: dict = field(default_factory=dict)


@dataclass
class MethodSpec:
    name: str
    label: str
    family: str
    fn: Callable
    default: bool = True
    snapshot: str = "single"
