"""
beam_search.py - candidate cells per receive beam: a weak object beside a strong one at the same range is not a peak
of the power summed over all channels, but it is a peak in range, Doppler and angle of its own beam (OS-CFAR along
range, Hann taper over the array, beams half a Rayleigh cell apart over +-60 deg).

    from methods.beam_search import BeamSearch
"""
from __future__ import annotations

import numpy as np

from methods.cfar import os_factor
from methods.range_doppler import TRAIN, Gate


class BeamSearch:
    def __init__(self, array, pfa: float):
        u = np.arange(-np.sin(np.radians(60)), np.sin(np.radians(60)) + 1e-9, array.rayleigh_u / 2)
        w = np.empty(array.n)
        w[np.argsort(array.positions)] = np.hanning(array.n + 2)[1:-1]
        self.A = (array.steering(u) * w[:, None]).conj() / np.sqrt(np.sum(w ** 2))
        self.factor = os_factor(2 * TRAIN, TRAIN, pfa, 1)

    def __call__(self, fe, rd, existing: list[Gate]) -> list[Gate]:
        B = np.abs(rd.D[:, :, fe.az] @ self.A) ** 2
        n = B.shape[0]
        idx = np.clip(np.arange(n)[:, None] + fe.offs[None, :], 0, n - 1)
        thr = self.factor * np.partition(B[idx], TRAIN - 1, axis=1)[:, TRAIN - 1]
        Q = np.pad(B, ((1, 1), (0, 0), (0, 0)), constant_values=-np.inf)
        peak = np.ones(B.shape, bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    peak &= B >= np.roll(Q, -dj, axis=1)[1 + di:n + 1 + di]
        Z = np.pad(B, ((0, 0), (0, 0), (1, 1)), constant_values=-np.inf)
        peak &= (B >= Z[:, :, :-2]) & (B >= Z[:, :, 2:])
        taken, best = {g.cell for g in existing}, {}
        for k, d, b in zip(*np.nonzero((B > thr) & peak & fe.roi[:, None, None])):
            e = float(B[k, d, b] / thr[k, d, b])
            if (k, d) not in taken and e > best.get((k, d), (0.0, 0.0))[0]:
                best[(k, d)] = (e, float(B[k, d, b]))
        return [Gate(int(k), int(d), float(fe.range_m[k]), float(fe.v_axis[d]), p, e, "beam search")
                for (k, d), (e, p) in best.items()]
