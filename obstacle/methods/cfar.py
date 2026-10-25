"""
cfar.py - the threshold factor of OS-CFAR (estimation's CFAR).

    from methods.cfar import os_factor
"""
from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np
from scipy import integrate, optimize, special, stats


@lru_cache(maxsize=512)
def os_factor(n_train: int, k: int, pfa: float, n_looks: int = 1) -> float:
    if n_looks == 1:
        f = lambda a: sum(np.log((n_train - i) / (n_train - i + a)) for i in range(k)) - np.log(pfa)
    else:
        L = float(n_looks)

        def log_pfa(a):
            def dens(z):
                F = special.gammainc(L, z)
                return (k * comb(n_train, k) * F ** (k - 1) * (1 - F) ** (n_train - k)
                        * stats.gamma.pdf(z, L) * special.gammaincc(L, a * z))
            return np.log(integrate.quad(dens, 0, np.inf, limit=200)[0] + 1e-300)
        f = lambda a: log_pfa(a) - np.log(pfa)
    return float(optimize.brentq(f, 1e-9, 1e9))
