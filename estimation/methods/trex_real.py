"""
trex_real.py - the original real-valued T-Rex selector on the real/imaginary split of the complex model (2p real
variables, a grid angle counts if its real or imaginary part is selected).

    from estimation.methods import trex_real
"""
from __future__ import annotations

from estimation.core.interfaces import DoAResult, MethodSpec
from estimation.methods.ctrex.trex import run_ctrex
from estimation.methods.ctrex.ctrex_t_lars import TLARSSolver
from estimation.methods.ctrex.ctrex_t_omp import TOMPSolver


def estimate_lars(gate, array, settings, rng) -> DoAResult:
    angles, info = run_ctrex(TLARSSolver, gate, array, settings, rng, domain="real")
    return DoAResult(angles, info)


def estimate_omp(gate, array, settings, rng) -> DoAResult:
    angles, info = run_ctrex(TOMPSolver, gate, array, settings, rng, domain="real")
    return DoAResult(angles, info)


METHODS = [
    MethodSpec("trex_tlars", r"T-Rex t-LARS (real split)", "trex", estimate_lars),
    MethodSpec("trex_tomp", r"T-Rex t-OMP (real split)", "trex", estimate_omp),
]
