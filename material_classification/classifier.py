"""
classifier.py - M²-SRC (Material-Motion Sparse Representation Classification): SRC with a motion and a material head;
every window is coded as a sparse non-negative mix of training windows, min_a 1/2||y - D a||^2 + lam sum(a), and takes
the class whose part of the code explains it best (FISTA by default, Clarabel as the exact reference).

    from material_classification.classifier import MaterialMotionModel    (byte-identical copy in obstacle/methods/material/)
"""
import json

import numpy as np

try:
    from material_classification.features import HEAD_FEATURES, MATERIAL_FEATURES, MOTION_FEATURES
except ImportError:
    from methods.material.features import HEAD_FEATURES, MATERIAL_FEATURES, MOTION_FEATURES


def fista_steps(D, Y, lam):
    L = np.linalg.norm(D, 2) ** 2
    A = np.zeros((D.shape[1], Y.shape[1]))
    Z, t = A.copy(), 1.0
    while True:
        A_new = np.maximum(Z - (D.T @ (D @ Z - Y) + lam) / L, 0.0)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        Z = A_new + (t - 1) / t_new * (A_new - A)
        A, t = A_new, t_new
        yield A


def nonneg_lasso(D, Y, lam, n_iter=300):
    for k, A in enumerate(fista_steps(D, Y, lam), 1):
        if k == n_iter:
            return A


class ClarabelLasso:

    def __init__(self, D, lam):
        import cvxpy as cp
        self.a = cp.Variable(D.shape[1], nonneg=True)
        self.y = cp.Parameter(D.shape[0])
        objective = 0.5 * cp.sum_squares(D @ self.a - self.y) + lam * cp.sum(self.a)
        self.problem = cp.Problem(cp.Minimize(objective))

    def __call__(self, Y):
        A = np.zeros((self.a.shape[0], Y.shape[1]))
        for j in range(Y.shape[1]):
            self.y.value = Y[:, j]
            self.problem.solve(solver="CLARABEL")
            A[:, j] = np.maximum(self.a.value, 0.0)
        return A


def objective(D, Y, A, lam):
    return 0.5 * np.sum((D @ A - Y) ** 2, axis=0) + lam * A.sum(axis=0)


def duality_gap(D, Y, A, lam):
    R = Y - D @ A
    worst = np.max(D.T @ R, axis=0)
    theta = R * np.minimum(1.0, lam / np.maximum(worst, 1e-12))
    dual = np.sum(theta * Y, axis=0) - 0.5 * np.sum(theta ** 2, axis=0)
    return objective(D, Y, A, lam) - dual


class SparseClassifier:
    def __init__(self, lam=0.05, solver="fista"):
        self.lam, self.solver = lam, solver
        self._clarabel = None

    def code(self, Y):
        if self.solver == "clarabel":
            if self._clarabel is None:
                self._clarabel = ClarabelLasso(self.D, self.lam)
            return self._clarabel(Y)
        return nonneg_lasso(self.D, Y, self.lam)

    def fit(self, X, labels):
        X = np.asarray(X, float)
        self.median = np.nanmedian(X, axis=0)
        Xf = np.where(np.isnan(X), self.median, X)
        self.mean, self.std = Xf.mean(axis=0), Xf.std(axis=0) + 1e-9
        self.D = self._scale(X).T
        self.labels = np.asarray(labels, object)
        self.classes = sorted(set(self.labels))
        self._clarabel = None
        return self

    def _scale(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        X = np.where(np.isnan(X), self.median, X)
        Z = np.clip((X - self.mean) / self.std, -5, 5)
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    def predict(self, X):
        Y = self._scale(X).T
        A = self.code(Y)
        res = np.stack([np.linalg.norm(Y - self.D[:, self.labels == c] @ A[self.labels == c], axis=0)
                        for c in self.classes], axis=1)
        share = np.stack([A[self.labels == c].sum(axis=0) for c in self.classes], axis=1)
        share = share / np.maximum(share.sum(axis=1, keepdims=True), 1e-12)
        best = np.argmin(res, axis=1)
        return np.array(self.classes, object)[best], share[np.arange(len(best)), best], res

    def evidence(self, X):
        Y = self._scale(X).T
        A = self.code(Y)
        E = np.stack([Y - self.D[:, self.labels == c] @ A[self.labels == c] for c in self.classes], axis=0)
        res = np.sum(E ** 2, axis=1)
        order = np.argsort(res, axis=0)
        best, second = order[0], order[1]
        n = np.arange(Y.shape[1])
        terms = (E[second, :, n] ** 2 - E[best, :, n] ** 2)
        share = terms / np.maximum(terms.sum(axis=1, keepdims=True), 1e-12)
        w = np.stack([A[self.labels == c].sum(axis=0) for c in self.classes], axis=1)
        conf = w[n, best] / np.maximum(w.sum(axis=1), 1e-12)
        cl = np.array(self.classes, object)
        return cl[best], conf, cl[second], share

    def explain(self, x):
        a = self.code(self._scale(x).T)[:, 0]
        share = np.array([a[self.labels == c].sum() for c in self.classes])
        return a, share / max(share.sum(), 1e-12)

    def state(self):
        return {"lam": self.lam, "median": self.median.tolist(), "mean": self.mean.tolist(),
                "std": self.std.tolist(), "D": self.D.tolist(), "labels": self.labels.tolist()}

    @classmethod
    def from_state(cls, s):
        c = cls(s["lam"])
        c.median, c.mean, c.std = (np.array(s[k]) for k in ("median", "mean", "std"))
        c.D, c.labels = np.array(s["D"]), np.array(s["labels"], object)
        c.classes = sorted(set(c.labels))
        return c


class MaterialMotionModel:

    def __init__(self, lam=0.05, solver="fista", features=None):
        self.motion = SparseClassifier(lam, solver)
        self.material = SparseClassifier(lam, solver)
        self.features = {h: list(v) for h, v in (features or HEAD_FEATURES).items()}
        self.settings = {}

    def fit(self, table):
        self.motion.fit(_columns(table, self.features["motion"]), table["motion"])
        self.material.fit(_columns(table, self.features["material"]), table["material"])
        return self

    def predict(self, table, evidence: bool = False):
        out = {}
        for head, clf in (("motion", self.motion), ("material", self.material)):
            names = self.features[head]
            X = _columns(table, names)
            if not evidence:
                out[head], out[head + "_conf"], _ = clf.predict(X)
                continue
            out[head], out[head + "_conf"], out[head + "_runner_up"], share = clf.evidence(X)
            out[head + "_evidence"] = [dict(zip(names, map(float, row))) for row in share]
            out[head + "_groups"] = [{"motion": float(sum(v for k, v in zip(names, row) if k in MOTION_FEATURES)),
                                      "material": float(sum(v for k, v in zip(names, row) if k in MATERIAL_FEATURES))}
                                     for row in share]
        return out

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"motion": self.motion.state(), "material": self.material.state(),
                       "features": self.features, "settings": self.settings}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            s = json.load(f)
        m = cls()
        m.motion, m.material = SparseClassifier.from_state(s["motion"]), SparseClassifier.from_state(s["material"])
        m.features = s.get("features") or {"motion": MOTION_FEATURES, "material": MATERIAL_FEATURES}
        m.settings = s.get("settings", {})
        return m


def moved_over_ground(xy, cell: float) -> bool:
    P = np.asarray(xy, float)
    return bool(len(P) > 1 and (np.ptp(P[:, 0]) > cell or np.ptp(P[:, 1]) > cell))


def _evidence_summary(pr: dict, head: str, top: int = 3) -> dict:
    ev = pr[head + "_evidence"][0]
    best = sorted(ev.items(), key=lambda kv: -kv[1])[:top]
    return {"runner_up": str(pr[head + "_runner_up"][0]), "features": {k: round(v, 3) for k, v in best},
            "groups": {k: round(v, 3) for k, v in pr[head + "_groups"][0].items()}}


def classify(model, features: dict, moved: bool = False) -> dict:
    pr = model.predict({k: np.array([v]) for k, v in features.items()}, evidence=True)
    motion = str(pr["motion"][0])
    return {"material": str(pr["material"][0]), "material_conf": float(pr["material_conf"][0]),
            "motion": "rigid" if (moved and motion == "static") else motion,
            "motion_conf": float(pr["motion_conf"][0]), "moved": bool(moved),
            "evidence": {h: _evidence_summary(pr, h) for h in ("material", "motion")}}


class LabelVotes:

    HEADS = ("material", "motion")

    def __init__(self):
        self.votes = {h: {} for h in self.HEADS}
        self.n_votes = {h: 0 for h in self.HEADS}
        self.last = {h: None for h in self.HEADS}
        self.n = 0

    def add(self, out: dict) -> dict:
        for h in self.HEADS:
            self.votes[h][out[h]] = self.votes[h].get(out[h], 0) + 1
            self.n_votes[h] += 1
            self.last[h] = out[h]
        self.n += 1
        return {h: {"label": self.label(h),
                    "confidence": max(self.votes[h].values()) / sum(self.votes[h].values()) if self.votes[h] else 0.0,
                    "windows": self.n_votes[h], "classified": self.n, "last": out[h], "last_conf": out[h + "_conf"],
                    "last_evidence": out["evidence"][h]} for h in self.HEADS}

    def label(self, h: str):
        if not self.votes[h]:
            return None
        top = max(self.votes[h].values())
        tied = [c for c, v in self.votes[h].items() if abs(v - top) < 1e-9]
        return self.last[h] if self.last[h] in tied else tied[0]


def _columns(table, names):
    return np.stack([np.asarray(table[n], float) for n in names], axis=1)
