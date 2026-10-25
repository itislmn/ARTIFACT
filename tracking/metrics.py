"""
metrics.py - the scores of the evaluation as their papers define them: HOTA, CLEAR MOT (MOTA, MOTP, ID switches), IDF1,
the per-class scores of M2-SRC (precision, recall, F1, AUC), cross-entropy, Wilson and block-bootstrap intervals, and
the exchangeability checks of conformal prediction.

    from metrics import hota, clear, idf1
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kstest, norm

EPS = 1e-9
ALPHAS = np.arange(0.05, 0.99, 0.05)


def _index(frames):
    gts = sorted({g for f in frames for g in f["gt"]}, key=str)
    trs = sorted({t for f in frames for t in f["tr"]}, key=str)
    gi, ti = {g: i for i, g in enumerate(gts)}, {t: i for i, t in enumerate(trs)}
    return gts, trs, [np.array([gi[g] for g in f["gt"]], int) for f in frames], [np.array([ti[t] for t in f["tr"]], int) for f in frames]


def hota(frames: list[dict]) -> dict:
    gts, trs, G, T = _index(frames)
    potential, gt_n, tr_n = np.zeros((len(gts), len(trs))), np.zeros(len(gts)), np.zeros(len(trs))
    for f, g, t in zip(frames, G, T):
        S = np.asarray(f["sim"], float).reshape(len(g), len(t))
        if S.size:
            denom = S.sum(0)[None, :] + S.sum(1)[:, None] - S
            potential[np.ix_(g, t)] += np.where(denom > EPS, S / np.maximum(denom, EPS), 0)
        gt_n[g] += 1
        tr_n[t] += 1
    gas = potential / np.maximum(gt_n[:, None] + tr_n[None, :] - potential, EPS)
    A = len(ALPHAS)
    tp, fn, fp, loc = np.zeros(A), np.zeros(A), np.zeros(A), np.zeros(A)
    matches = np.zeros((A, len(gts), len(trs)))
    for f, g, t in zip(frames, G, T):
        if not len(g):
            fp += len(t)
            continue
        if not len(t):
            fn += len(g)
            continue
        S = np.asarray(f["sim"], float).reshape(len(g), len(t))
        r, c = linear_sum_assignment(-(gas[np.ix_(g, t)] * S))
        for a, alpha in enumerate(ALPHAS):
            ok = S[r, c] >= alpha - EPS
            n = int(ok.sum())
            tp[a] += n
            fn[a] += len(g) - n
            fp[a] += len(t) - n
            loc[a] += S[r[ok], c[ok]].sum()
            matches[a][g[r[ok]], t[c[ok]]] += 1
    ass = np.array([(m * m / np.maximum(gt_n[:, None] + tr_n[None, :] - m, EPS)).sum() / max(tp[a], 1)
                    for a, m in enumerate(matches)])
    det = tp / np.maximum(tp + fn + fp, 1)
    det_re, det_pr = tp / np.maximum(tp + fn, 1), tp / np.maximum(tp + fp, 1)
    h = np.sqrt(det * ass)
    return {"HOTA": float(h.mean()), "DetA": float(det.mean()), "AssA": float(ass.mean()),
            "DetRe": float(det_re.mean()), "DetPr": float(det_pr.mean()), "LocA": float((loc / np.maximum(tp, 1)).mean()),
            "curve": {"alpha": ALPHAS, "HOTA": h, "DetA": det, "AssA": ass}}


def clear(frames: list[dict], threshold: float = 0.5) -> dict:
    gts, trs, G, T = _index(frames)
    prev_id, prev_step = np.full(len(gts), -1), np.full(len(gts), -1)
    tp = fn = fp = idsw = 0
    motp = 0.0
    for f, g, t in zip(frames, G, T):
        if not len(g) or not len(t):
            fn, fp = fn + len(g), fp + len(t)
            prev_step[:] = -1
            continue
        S = np.asarray(f["sim"], float).reshape(len(g), len(t))
        score = 1000.0 * (t[None, :] == prev_step[g][:, None]) + S
        score[S < threshold - EPS] = 0
        r, c = linear_sum_assignment(-score)
        ok = score[r, c] > EPS
        r, c = r[ok], c[ok]
        mg, mt = g[r], t[c]
        idsw += int(np.sum((prev_id[mg] >= 0) & (prev_id[mg] != mt)))
        prev_id[mg] = mt
        prev_step[:] = -1
        prev_step[mg] = mt
        tp, fn, fp = tp + len(r), fn + len(g) - len(r), fp + len(t) - len(r)
        motp += S[r, c].sum()
    return {"MOTA": (tp - fp - idsw) / max(tp + fn, 1), "MOTP": motp / max(tp, 1), "IDSW": idsw, "TP": tp, "FN": fn,
            "FP": fp, "GT": tp + fn}


def idf1(frames: list[dict], threshold: float = 0.5) -> dict:
    gts, trs, G, T = _index(frames)
    m, gt_n, tr_n = np.zeros((len(gts), len(trs))), np.zeros(len(gts)), np.zeros(len(trs))
    for f, g, t in zip(frames, G, T):
        S = np.asarray(f["sim"], float).reshape(len(g), len(t))
        if S.size:
            m[np.ix_(g, t)] += S >= threshold - EPS
        gt_n[g] += 1
        tr_n[t] += 1
    r, c = linear_sum_assignment(-m) if m.size else ([], [])
    idtp = float(m[r, c].sum()) if m.size else 0.0
    idfn, idfp = gt_n.sum() - idtp, tr_n.sum() - idtp
    return {"IDF1": idtp / max(idtp + 0.5 * (idfn + idfp), 1), "IDP": idtp / max(idtp + idfp, 1),
            "IDR": idtp / max(idtp + idfn, 1), "IDTP": idtp}


def segments(frames: list[dict], seconds: float = 10.0) -> list[list[dict]]:
    t0 = frames[0]["t"]
    out = {}
    for f in frames:
        out.setdefault(int((f["t"] - t0) // seconds), []).append(f)
    return [v for v in out.values() if len(v) >= 0.8 * seconds * 10]


def roc(score, positive):
    order = np.argsort(-score, kind="stable")
    s, y = score[order], positive[order]
    cut = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tpr = np.r_[0, np.cumsum(y)[cut] / max(y.sum(), 1)]
    fpr = np.r_[0, np.cumsum(~y)[cut] / max((~y).sum(), 1)]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def per_class(true, pred, scores, classes) -> list[dict]:
    rows = []
    for j, c in enumerate(classes):
        if not np.any(true == c):
            continue
        tp, fp, fn = np.sum((pred == c) & (true == c)), np.sum((pred == c) & (true != c)), np.sum((pred != c) & (true == c))
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        auc = roc(scores[:, j], true == c)[2] if np.any(true != c) else np.nan
        rows.append({"class": c, "n": int(np.sum(true == c)), "precision": p, "recall": r,
                     "f1": 2 * p * r / max(p + r, 1e-12), "auc": auc})
    return rows


def cross_entropy_bits(prob, true, classes, floor: float = 1e-3) -> float:
    idx = np.array([classes.index(c) for c in true])
    p = np.clip(prob[np.arange(len(idx)), idx], floor, 1.0)
    return float(-np.mean(np.log2(p)))


def wilson(k: float, n: float, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def block_bootstrap(values, block: int = 10, n: int = 1000, seed: int = 0):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2 * block:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(v) - block + 1, size=(n, int(np.ceil(len(v) / block))))
    means = np.array([np.concatenate([v[s:s + block] for s in row])[:len(v)].mean() for row in starts])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def conformal_p(scores, rng) -> np.ndarray:
    s = np.asarray(scores, float)
    p = np.full(len(s), np.nan)
    for t in range(1, len(s)):
        past = s[:t]
        p[t] = (np.sum(past > s[t]) + rng.uniform() * (np.sum(past == s[t]) + 1)) / (t + 1)
    return p


def exchangeability(sequences: list[np.ndarray], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    p = np.concatenate([conformal_p(s, rng)[1:] for s in sequences if len(s) > 5] or [np.array([])])
    lag = [np.corrcoef(s[:-1], s[1:])[0, 1] for s in sequences if len(s) > 10 and np.std(s) > 0]
    return {"p_values": p, "ks_p": float(kstest(p, "uniform").pvalue) if len(p) > 10 else np.nan,
            "lag1": float(np.average(lag)) if lag else np.nan, "n": len(p)}


def runs_test(miss) -> float:
    x = np.asarray(miss, int)
    n1, n0 = x.sum(), len(x) - x.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    runs = 1 + np.sum(x[1:] != x[:-1])
    mu = 2 * n1 * n0 / (n1 + n0) + 1
    var = (mu - 1) * (mu - 2) / (n1 + n0 - 1)
    return float(2 * norm.sf(abs(runs - mu) / np.sqrt(max(var, EPS))))


def gospa(D: np.ndarray, c: float, p: int = 2) -> dict:
    n, m = D.shape
    pairs = []
    if n and m:
        r, k = linear_sum_assignment(np.minimum(D, c) ** p)
        pairs = [(i, j) for i, j in zip(r, k) if D[i, j] < c]
    loc = float(sum(D[i, j] ** p for i, j in pairs))
    missed, false = (n - len(pairs)) * c ** p / 2, (m - len(pairs)) * c ** p / 2
    ospa = 0.0
    if max(n, m):
        cut = float(sum(min(D[i, j], c) ** p for i, j in pairs)) + c ** p * (max(n, m) - len(pairs))
        ospa = (cut / max(n, m)) ** (1 / p)
    return {"gospa": (loc + missed + false) ** (1 / p), "loc": loc, "missed": missed, "false": false,
            "n_missed": n - len(pairs), "n_false": m - len(pairs), "pairs": pairs, "ospa": ospa}


def switches(frames: list[dict]) -> float:
    s = 0.0
    for fa, fb in zip(frames[:-1], frames[1:]):
        a, b = fa["gospa"]["assign"], fb["gospa"]["assign"]
        for o in set(fa["gt"]) & set(fb["gt"]):
            x, y = a.get(o), b.get(o)
            if x != y:
                s += 1.0 if x is not None and y is not None else 0.5
    return s
