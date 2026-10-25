"""
train.py - trains both readers on the same distorted characters (ridge in closed form, CNN with Adam) and tests them on
the held-out fonts.

    python -m readers.train            (python -m readers.train selftest)
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from backend.settings import MODELS, load_config  # noqa: E402
from readers import cnn_reader, metricsight  # noqa: E402
from readers.glyphs import TEST_FONTS, TRAIN_FONTS, characters, word_scene  # noqa: E402


def data(n, rng, fonts):
    parts = [characters(1000, rng, fonts) for _ in range(n // 1000)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def enrolled(rng, copies: int = 150):
    from readers.enrol import ENROLLED
    from readers.glyphs import _warp
    from readers.segment import normalise_char
    if not ENROLLED.exists():
        return None
    d = np.load(ENROLLED)
    X = np.array([normalise_char(_warp(np.pad(x, 8), rng)) for x in d["X"] for _ in range(copies)], np.float32)
    return X, np.repeat(d["y"], copies)


def train(n=18000, epochs=10):
    rng = np.random.default_rng(1)
    t0 = time.perf_counter()
    X, y = data(n, rng, TRAIN_FONTS)
    E = enrolled(np.random.default_rng(3))
    if E is not None:
        k = rng.permutation(len(E[1]))
        X, y = np.concatenate([X, E[0][k]]), np.concatenate([y, E[1][k]])
        print(f"+ {len(E[1])} distorted copies of {len(E[1]) // 150} enrolled letters (readers/models/enrolled.npz)")
        n = len(y)
    Xt, yt = data(3600, np.random.default_rng(2), TEST_FONTS)
    nv = n // 5
    print(f"{n} training characters from {len(TRAIN_FONTS)} fonts, {len(yt)} test characters from unseen fonts "
          f"{TEST_FONTS} ({time.perf_counter() - t0:.0f} s)")
    MODELS.mkdir(parents=True, exist_ok=True)

    best = None
    for lam in (0.01, 0.1, 1.0, 10.0):
        W = metricsight.fit_ridge(X[nv:], y[nv:], lam)
        acc = np.mean(metricsight.classify(W, X[:nv])[0] == y[:nv])
        print(f"  ridge lambda {lam:5}: held-out (training fonts) {100 * acc:.1f} %")
        best = (acc, lam) if best is None or acc > best[0] else best
    t1 = time.perf_counter()
    W = metricsight.fit_ridge(X, y, best[1])
    fit_s = time.perf_counter() - t1
    acc_t = np.mean(metricsight.classify(W, Xt)[0] == yt)
    np.savez(metricsight.MODEL, W=W, lam=best[1], test_accuracy=acc_t)
    print(f"MetricSight classifier: lambda {best[1]}, fitted in {fit_s:.1f} s, {W.size} numbers; "
          f"UNSEEN fonts {100 * acc_t:.1f} %")

    net = cnn_reader.LeNet(seed=1)
    m = {k: np.zeros_like(v) for k, v in net.p.items()}
    v = {k: np.zeros_like(v) for k, v in net.p.items()}
    Xc, Xtc = X[:, None], Xt[:, None]
    step, t1 = 0, time.perf_counter()
    for ep in range(epochs):
        order = nv + rng.permutation(n - nv)
        for i in range(0, len(order), 128):
            idx = order[i:i + 128]
            _, cache = net.forward(Xc[idx])
            g = net.backward(cache, y[idx])
            step += 1
            for k in net.p:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                net.p[k] -= (2e-3 * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)).astype(np.float32)
        acc = np.mean(np.concatenate([net.forward(Xc[j:min(j + 500, nv)])[0].argmax(1) for j in range(0, nv, 500)]) == y[:nv])
        print(f"  CNN epoch {ep + 1:2d}: held-out (training fonts) {100 * acc:.1f} %  ({time.perf_counter() - t1:.0f} s)")
    acc_t = np.mean(np.concatenate([net.forward(Xtc[j:j + 500])[0].argmax(1) for j in range(0, len(yt), 500)]) == yt)
    np.savez(cnn_reader.MODEL, **net.p, test_accuracy=acc_t)
    n_par = sum(x.size for x in net.p.values())
    print(f"CNN: {n_par} weights, trained in {time.perf_counter() - t1:.0f} s; UNSEEN fonts {100 * acc_t:.1f} %")


def selftest():
    cfg = load_config()
    f = 950.0
    rng = np.random.default_rng(7)
    for glare in (False, True):
        for font in TEST_FONTS:
            line = []
            for R in (0.8, 1.0, 1.2):
                img = word_scene("ARTIFACT", font, int(f * 0.05 / R), centre=(640, 360), rng=rng)
                if glare:
                    yy, xx = np.mgrid[0:720, 0:1280]
                    rho = np.hypot(xx - 640 + f * 0.30 / R, yy - 330)
                    img = np.clip(img + 60 * (f * 0.30 / R / np.maximum(rho, 1)) ** 2 * 1.0, 0, 255)
                    img[rho < 25] = 255
                rgb = np.repeat(img[..., None], 3, axis=2)
                a = metricsight.read(rgb, (640, 360), R, f, cfg)
                b = cnn_reader.read(rgb, (640, 360), R, f, cfg)
                line.append(f"{R:.1f} m: MetricSight {a.text:<9} ({a.ms:3.0f} ms)  CNN {b.text:<9} ({b.ms:3.0f} ms)")
            print(f"  {'GLARE ' if glare else 'clean '} {font:<13} " + " | ".join(line))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        train()
        selftest()
