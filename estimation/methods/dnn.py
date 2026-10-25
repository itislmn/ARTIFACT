"""
dnn.py - the learning baseline: a multilayer perceptron (3 hidden layers) on the whitened snapshot and its beam
spectrum, trained on simulated gates; peaks above the output threshold are detections.

    python -m estimation.methods.dnn        (trains the weights, about 4 min)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from estimation.core.interfaces import DoAResult, MethodSpec

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "dnn_weights.npz"
HIDDEN = (256, 256, 256)
SEED = 20261006

_NET = {}


def _steer(pos_c, u):
    return np.exp(2j * np.pi * pos_c[:, None] * np.atleast_1d(u)[None, :])


def features(Ybar, A_grid):
    N = Ybar.shape[1]
    c = Ybar @ A_grid.conj()
    P = np.abs(c) ** 2 / N
    k = np.argmax(P, axis=1)
    rot = np.exp(-1j * np.angle(c[np.arange(len(k)), k]))
    Yr = Ybar * rot[:, None]
    pw = np.mean(np.abs(Yr) ** 2, axis=1, keepdims=True)
    Yn = Yr / np.sqrt(pw)
    return np.hstack([Yn.real, Yn.imag, np.log1p(P), np.log(pw)]).astype(np.float32)


def make_data(n, u_grid, pos_c, rng, du):
    N = pos_c.size
    lo, hi = u_grid[0], u_grid[-1]
    n_src = rng.choice(4, size=n, p=[0.15, 0.40, 0.30, 0.15])
    Y = (rng.standard_normal((n, N)) + 1j * rng.standard_normal((n, N))) / np.sqrt(2)
    lab = np.zeros((n, u_grid.size), np.float32)
    sep = 2.0 / N
    for i in range(n):
        us = []
        while len(us) < n_src[i]:
            u = rng.uniform(lo, hi)
            if all(abs(u - v) >= sep for v in us):
                us.append(u)
        if not us:
            continue
        snr0 = rng.uniform(-5, 35)
        snr = snr0 - np.concatenate([[0.0], rng.uniform(0, 9, len(us) - 1)])
        amp = np.sqrt(10 ** (snr / 10)) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(us)))
        Y[i] += _steer(pos_c, us) @ amp
        lab[i] = np.max(np.exp(-(u_grid[None, :] - np.array(us)[:, None]) ** 2 / (2 * du ** 2)), axis=0)
    return Y, lab


def _init(sizes, rng):
    return [(rng.standard_normal((a, b)).astype(np.float32) * np.sqrt(2.0 / a), np.zeros(b, np.float32))
            for a, b in zip(sizes[:-1], sizes[1:])]


def forward(params, x, keep=False):
    acts = [x]
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        if i < len(params) - 1:
            x = np.maximum(x, 0)
        acts.append(x)
    return (x, acts) if keep else x


def train(n_train=200_000, n_val=20_000, epochs=20, batch=512, lr=1e-3, log=print):
    from estimation.core.array import ArrayModel
    from estimation.core.radar import RadarConfig
    cfg = RadarConfig.from_cfg(HERE.parent / "config" / "awr2944_doa_perfect.cfg")
    array = ArrayModel(cfg)
    pos_c = array.az_sign * (array.positions - array.positions.mean())
    u_grid, du = array.u_grid, float(array.u_grid[1] - array.u_grid[0])
    A_grid = _steer(pos_c, u_grid)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    Ytr, Ltr = make_data(n_train, u_grid, pos_c, rng, du)
    Yva, Lva = make_data(n_val, u_grid, pos_c, rng, du)
    Xtr, Xva = features(Ytr, A_grid), features(Yva, A_grid)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    log(f"data: {n_train} + {n_val} snapshots, {Xtr.shape[1]} features ({time.time() - t0:.0f} s)")
    params = _init([Xtr.shape[1], *HIDDEN, u_grid.size], rng)
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    t = 0
    for ep in range(epochs):
        step_lr = lr * (0.5 if ep >= 0.5 * epochs else 1.0) * (0.5 if ep >= 0.75 * epochs else 1.0)
        perm = rng.permutation(n_train)
        for s in range(0, n_train, batch):
            idx = perm[s:s + batch]
            out, acts = forward(params, Xtr[idx], keep=True)
            g = (1.0 / (1.0 + np.exp(-out)) - Ltr[idx]) / len(idx)
            t += 1
            for i in range(len(params) - 1, -1, -1):
                W, b = params[i]
                gW, gb = acts[i].T @ g, g.sum(0)
                if i > 0:
                    g = (g @ W.T) * (acts[i] > 0)
                mW, mB = m[i]
                vW, vB = v[i]
                mW[:] = 0.9 * mW + 0.1 * gW
                mB[:] = 0.9 * mB + 0.1 * gb
                vW[:] = 0.999 * vW + 0.001 * gW ** 2
                vB[:] = 0.999 * vB + 0.001 * gb ** 2
                c1, c2 = 1 - 0.9 ** t, 1 - 0.999 ** t
                W -= step_lr * (mW / c1) / (np.sqrt(vW / c2) + 1e-8)
                b -= step_lr * (mB / c1) / (np.sqrt(vB / c2) + 1e-8)
        z = forward(params, Xva)
        bce = float(np.mean(np.logaddexp(0, z) - Lva * z))
        log(f"epoch {ep + 1:2d}/{epochs}: validation BCE {bce:.4f} ({time.time() - t0:.0f} s)")
    np.savez(WEIGHTS, mu=mu, sd=sd, u_grid=u_grid, pos_c=pos_c, n_layers=len(params),
             **{f"W{i}": W for i, (W, _) in enumerate(params)}, **{f"b{i}": b for i, (_, b) in enumerate(params)})
    log(f"weights -> {WEIGHTS}")


def _load():
    if not _NET:
        if not WEIGHTS.exists():
            raise SystemExit(f"{WEIGHTS.name} missing: train first with  python -m estimation.methods.dnn")
        f = np.load(WEIGHTS)
        _NET.update(mu=f["mu"], sd=f["sd"], u_grid=f["u_grid"], pos_c=f["pos_c"],
                    params=[(f[f"W{i}"], f[f"b{i}"]) for i in range(int(f["n_layers"]))])
        _NET["A_grid"] = _steer(_NET["pos_c"], _NET["u_grid"])
    return _NET


def estimate(gate, array, settings, rng) -> DoAResult:
    net = _load()
    s = np.sqrt(gate.noise_var / gate.K)
    x = (features(gate.mean_snapshot()[None, :] / s, net["A_grid"]) - net["mu"]) / net["sd"]
    p = 1.0 / (1.0 + np.exp(-forward(net["params"], x)[0]))
    u = net["u_grid"]
    pk = [i for i in range(1, u.size - 1) if p[i] >= settings.dnn_threshold and p[i] >= p[i - 1] and p[i] >= p[i + 1]]
    if p[0] >= settings.dnn_threshold and p[0] >= p[1]:
        pk.append(0)
    if p[-1] >= settings.dnn_threshold and p[-1] >= p[-2]:
        pk.append(u.size - 1)
    pk.sort(key=lambda i: -p[i])
    keep = []
    for i in pk:
        if all(abs(u[i] - u[j]) >= 0.5 * array.rayleigh_u for j in keep):
            keep.append(i)
    u_hat = []
    for i in keep:
        if 0 < i < u.size - 1:
            ym, y0, yp = p[i - 1], p[i], p[i + 1]
            den = ym - 2 * y0 + yp
            delta = 0.0 if den >= 0 else float(np.clip(0.5 * (ym - yp) / den, -0.5, 0.5))
            u_hat.append(u[i] + delta * (u[1] - u[0]))
        else:
            u_hat.append(u[i])
    info = {}
    if settings.keep_info:
        info = {"kind": "dnn", "u": u, "p": p, "threshold": settings.dnn_threshold}
    return DoAResult(array.u_to_deg(np.asarray(u_hat)), info)


METHODS = [MethodSpec("dnn", r"DNN (MLP, 3 hidden layers)", "learning", estimate)]


if __name__ == "__main__":
    train()
