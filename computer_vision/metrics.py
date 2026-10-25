"""
metrics.py - the reading metrics of the evaluation: character alignment by edit distance, character error rate
CER = (S + D + I) / N, word accuracy, the stable reading (majority of the last reads) and the AUROC of a confidence.

    from metrics import align, cer, stable, auroc
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def align(truth: str, read: str) -> list[tuple[str, str, int]]:
    n, m = len(truth), len(read)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j - 1] + (truth[i - 1] != read[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (truth[i - 1] != read[j - 1]):
            ops.append((truth[i - 1], read[j - 1], j - 1))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append((truth[i - 1], "", -1))
            i -= 1
        else:
            ops.append(("", read[j - 1], j - 1))
            j -= 1
    return ops[::-1]


def counts(truth: str, read: str) -> dict:
    ops = align(truth, read)
    return {"S": sum(bool(t and r and t != r) for t, r, _ in ops), "D": sum(t != "" and r == "" for t, r, _ in ops),
            "I": sum(t == "" for t, _, _ in ops), "C": sum(t == r for t, r, _ in ops), "N": len(truth)}


def cer(truth: str, read: str) -> float:
    c = counts(truth, read)
    return (c["S"] + c["D"] + c["I"]) / c["N"]


def stable(texts: list[str], n: int = 15) -> list[str]:
    out, last = [], []
    for t in texts:
        if t:
            last = (last + [t])[-n:]
        out.append(Counter(last).most_common(1)[0][0] if last else "")
    return out


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return float("nan")
    ranks = np.empty(len(score))
    order = np.argsort(score, kind="mergesort")
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):
        tie = score == v
        ranks[tie] = ranks[tie].mean()
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def risk_coverage(score: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(score, float), kind="mergesort")
    c = np.asarray(correct, float)[order]
    k = np.arange(1, len(c) + 1)
    return k / len(c), np.cumsum(c) / k
