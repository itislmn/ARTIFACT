"""
by.py - Benjamini-Yekutieli: the step-up of bh.py with alpha / sum_{i<=p} 1/i, valid under any dependence.

    from estimation.methods import by
"""
from __future__ import annotations

import numpy as np

from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.bh import select


def by_level(alpha: float, m: int) -> float:
    return alpha / float(np.sum(1.0 / np.arange(1, m + 1)))


def estimate(gate, array, settings, rng=None) -> DoAResult:
    return select(gate, array, by_level(settings.alpha, array.nyquist_grid.size), settings.keep_info, "BY")


METHODS = [MethodSpec("by", r"Benjamini-Yekutieli (OLS)", "fdr", estimate)]
