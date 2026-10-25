"""
trex.py - the T-Rex calibration of TRexSelector for any terminating solver: K random experiments with L dummies,
relative occurrences Phi, FDP-hat, and the choice of L, T* and v*.

    from estimation.methods.ctrex.trex import TRexSelector, run_ctrex
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = np.finfo(float).eps


class TerminatingSolver:
    name = "base"

    def __init__(self, X, y, D, record_path: bool = False, prepared=None, **kwargs):
        X = np.asarray(X)
        if prepared is None:
            Xt = np.hstack([X, D]) if D is not None and D.shape[1] else X.copy()
            Xt = Xt / np.linalg.norm(Xt, axis=0, keepdims=True)
            prepared = (Xt, np.ascontiguousarray(Xt.conj().T))
        Xt, self.XH = prepared
        self.X = Xt
        self.y = np.asarray(y, dtype=Xt.dtype)
        self.n_rows, self.n_cols = Xt.shape
        self.p = X.shape[1]
        self.L = self.n_cols - self.p
        self.active = []
        self.entries = []
        self.actives_at_t = []
        self.exhausted = False
        self.n_steps = 0
        self.path = [] if record_path else None
        self._setup(**kwargs)

    def _setup(self, **kwargs):
        pass

    def _step(self) -> bool:
        raise NotImplementedError

    def coefficients(self) -> np.ndarray:
        raise NotImplementedError

    def real_active(self) -> np.ndarray:
        return np.array(sorted(j for j in self.active if j < self.p), dtype=int)

    def n_active_dummies(self) -> int:
        return sum(1 for j in self.active if j >= self.p)

    def _record(self):
        nd = self.n_active_dummies()
        while len(self.actives_at_t) < nd:
            self.actives_at_t.append(self.real_active())
        if self.path is not None:
            self.path.append(self.coefficients().copy())

    def execute_step(self, T_stop: int, early_stop: bool = True):
        while (not early_stop or len(self.actives_at_t) < T_stop) and not self.exhausted:
            ok = self._step()
            self.n_steps += 1
            if not ok:
                self.exhausted = True
            self._record()
        while len(self.actives_at_t) < T_stop:
            self.actives_at_t.append(self.real_active())
        return self


def draw_dummies(rng: np.random.Generator, n: int, L: int, complex_valued: bool,
                 distribution: str = "gaussian") -> np.ndarray:
    if distribution == "gaussian":
        if complex_valued:
            return (rng.standard_normal((n, L)) + 1j * rng.standard_normal((n, L))) / np.sqrt(2.0)
        return rng.standard_normal((n, L))
    if distribution == "circular_uniform":
        if complex_valued:
            return np.exp(2j * np.pi * rng.random((n, L)))
        return rng.choice(np.array([-1.0, 1.0]), size=(n, L))
    raise ValueError(f"unknown dummy distribution '{distribution}'")


def deflated_relative_occurrence(p: int, T: int, L: int, phi_T_mat: np.ndarray, Phi: np.ndarray):
    av_num_var_sel = phi_T_mat.sum(axis=0)
    delta = phi_T_mat[Phi > 0.5].sum(axis=0)
    d_phi = phi_T_mat.copy()
    if T > 1:
        delta[1:] = delta[1:] - delta[:-1]
        d_phi[:, 1:] = phi_T_mat[:, 1:] - phi_T_mat[:, :-1]
    scale = np.zeros(T)
    for t in range(T):
        if delta[t] > EPS:
            scale[t] = 1.0 - ((p - av_num_var_sel[t]) / (L - t)) / delta[t]
    return d_phi @ scale


def fdp_hat(V: np.ndarray, Phi: np.ndarray, Phi_prime: np.ndarray) -> np.ndarray:
    out = np.zeros(len(V))
    for i, v in enumerate(V):
        sel = Phi > v
        r = int(sel.sum())
        out[i] = 0.0 if r == 0 else min(1.0, float((1.0 - Phi_prime)[sel].sum()) / r)
    return out


@dataclass
class TRexResult:
    selected: np.ndarray
    v_star: float | None
    T_star: int
    L: int
    T_stop: int
    V: np.ndarray
    Phi_mat: np.ndarray
    FDP_hat_mat: np.ndarray
    Phi_prime: np.ndarray
    L_history: list = field(default_factory=list)
    experiments: list = field(default_factory=list)
    total_steps: int = 0
    extra: dict = field(default_factory=dict)


class TRexSelector:

    def __init__(self, solver_cls, tFDR: float = 0.1, K: int = 20, max_dummy_multiplier: int = 10,
                 max_T_stop: bool = True, solver_kwargs: dict | None = None,
                 keep_experiments: bool = False, dummy_distribution: str = "gaussian", dummy_fn=None):
        self.solver_cls = solver_cls
        self.tFDR = float(tFDR)
        self.K = int(K)
        self.max_mult = int(max_dummy_multiplier)
        self.max_T_stop = bool(max_T_stop)
        self.kw = dict(solver_kwargs or {})
        self.keep = bool(keep_experiments)
        self.dummy_distribution = dummy_distribution
        self.dummy_fn = dummy_fn

    @staticmethod
    def phi_mat(experiments, T: int, p: int) -> np.ndarray:
        phi = np.zeros((p, T))
        for t in range(T):
            np.add.at(phi[:, t], np.concatenate([e.actives_at_t[t] for e in experiments]).astype(int), 1.0)
        return phi / len(experiments)

    def _dummies(self, rng, n, L, cplx):
        if self.dummy_fn is not None:
            return self.dummy_fn(rng, n, L, cplx)
        return draw_dummies(rng, n, L, cplx, self.dummy_distribution)

    def select(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> TRexResult:
        n, p = X.shape
        cplx = np.iscomplexobj(X) or np.iscomplexobj(y)
        V = np.arange(0.5, 1.0 - EPS, 1.0 / self.K)
        V = V[V < 1.0]
        hit = np.flatnonzero(np.abs(V - 0.75) < 1e-9)
        opt_point = int(hit[0]) if hit.size else max(int(np.sum(V < 0.75)) - 1, 0)
        Xn = X / np.linalg.norm(X, axis=0, keepdims=True)
        blocks = [[Xn] for _ in range(self.K)]
        blocks_h = [[np.ascontiguousarray(Xn.conj().T)] for _ in range(self.K)]
        steps = 0

        L_history = []
        for LL in range(1, self.max_mult + 1):
            L = LL * p
            for k in range(self.K):
                d = self._dummies(rng, n, p, cplx)
                d = d / np.linalg.norm(d, axis=0, keepdims=True)
                blocks[k].append(d)
                blocks_h[k].append(np.ascontiguousarray(d.conj().T))
            exps = [self.solver_cls(X, y, None, record_path=self.keep and k == 0,
                                    prepared=(np.hstack(blocks[k]), np.vstack(blocks_h[k])), **self.kw)
                    for k in range(self.K)]
            for e in exps:
                e.execute_step(1)
            phi = self.phi_mat(exps, 1, p)
            fdp = fdp_hat(V, phi[:, -1], deflated_relative_occurrence(p, 1, L, phi, phi[:, -1]))
            L_history.append((L, float(fdp[opt_point])))
            if fdp[opt_point] <= self.tFDR or LL == self.max_mult:
                break
            steps += sum(e.n_steps for e in exps)

        max_T = min(L, int(np.ceil(n / 2))) if self.max_T_stop else L
        T = 1
        fdp_rows, phi_rows = [fdp], [phi[:, -1]]
        while fdp_rows[-1][-1] <= self.tFDR and T < max_T:
            T += 1
            for e in exps:
                e.execute_step(T)
            phi = self.phi_mat(exps, T, p)
            fdp_rows.append(fdp_hat(V, phi[:, -1], deflated_relative_occurrence(p, T, L, phi, phi[:, -1])))
            phi_rows.append(phi[:, -1])
        steps += sum(e.n_steps for e in exps)

        cal = {"exps": exps, "L": L, "T_stop": T, "V": V, "p": p, "n": n,
               "FDP_mat": np.array(fdp_rows), "Phi_mat": np.array(phi_rows)}
        selected, v_star, T_star, extra = self._choose(cal)
        phi_star = self.phi_mat(exps, T_star, p)
        Phi_prime = deflated_relative_occurrence(p, T_star, L, phi_star, phi_star[:, -1])
        return TRexResult(selected, v_star, T_star, L, T, V, cal["Phi_mat"], cal["FDP_mat"], Phi_prime,
                          L_history, exps if self.keep else [], steps, extra)

    def _choose(self, cal):
        T_stop, V, tFDR = cal["T_stop"], cal["V"], self.tFDR
        FDP_mat, Phi_mat = cal["FDP_mat"], cal["Phi_mat"]
        FDP_sel, Phi_sel = (FDP_mat[:-1], Phi_mat[:-1]) if T_stop > 1 else (FDP_mat, Phi_mat)
        R_mat = (Phi_sel[:, None, :] > V[None, :, None]).sum(axis=2)
        feasible = FDP_sel <= tFDR
        if not feasible.any():
            return np.zeros(0, int), None, 1, {}
        val_max = R_mat[feasible].max()
        cand = feasible & (R_mat == val_max)
        vi = int(np.flatnonzero(cand.any(axis=0)).max())
        ti = int(np.flatnonzero(cand[:, vi]).max())
        return np.flatnonzero(Phi_sel[ti] > V[vi]), float(V[vi]), ti + 1, {}


def real_augmented(X: np.ndarray, y: np.ndarray):
    Xr = np.block([[X.real, -X.imag], [X.imag, X.real]])
    return Xr, np.concatenate([y.real, y.imag])


def run_ctrex(solver_cls, gate, array, settings, rng, solver_kwargs=None, domain: str = "complex",
              selector_cls=TRexSelector, dummy_fn=None):
    from estimation.core.array import regression_detections

    G = gate.groups(settings.n_groups)
    X = array.dictionary(G)
    y = gate.stacked_snapshot(G)
    sel = selector_cls(solver_cls, tFDR=settings.alpha, K=settings.trex_K,
                       max_dummy_multiplier=settings.trex_max_dummy_multiplier,
                       solver_kwargs=solver_kwargs, keep_experiments=settings.keep_info,
                       dummy_distribution=settings.dummy_distribution, dummy_fn=dummy_fn)
    if domain == "real":
        Xr, yr = real_augmented(X, y)
        res = sel.select(Xr, yr, rng)
        chosen = np.unique(res.selected % array.p)
    else:
        res = sel.select(X, y, rng)
        chosen = res.selected
    u_hat = regression_detections(array, X, y, chosen)
    info = {"stats": {"L": res.L, "T_stop": res.T_stop, "T_star": res.T_star, "steps": res.total_steps,
                      "n": X.shape[0] * (2 if domain == "real" else 1), "p": array.p}}
    if settings.keep_info:
        info.update({"kind": "trex", "solver": solver_cls.name, "domain": domain, "result": res,
                     "selected": chosen, "u": array.u_grid, "p": array.p, "alpha": settings.alpha,
                     "bartlett": array.beam_power(gate.mean_snapshot())})
    return array.u_to_deg(u_hat), info
