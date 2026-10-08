"""Intervals and tests over training seeds (App. Statistical Analysis).

Every fine-tuned cell is a mean over training seeds with a 95% Student-t interval; a difference
between two conditions has a 95% Welch-t interval and an exact two-sided permutation test over
the seed means, Holm-corrected across the four broad-misalignment evaluations. A single
evaluation (the base model under injection) has a prompt-level bootstrap interval for its mean
and a Wilson interval for a proportion.
"""

from __future__ import annotations

import itertools
import math
import statistics

import numpy as np
from scipy import stats


def t_ci(xs) -> tuple[float, float, float]:
    """Mean and 95% Student-t interval; no interval for a single value."""
    xs = list(xs)
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, math.nan, math.nan
    h = stats.t.ppf(0.975, len(xs) - 1) * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, m - h, m + h


def welch_ci(a, b) -> tuple[float, float, float]:
    """mean(b) - mean(a) and its 95% Welch-t interval."""
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    d, se = statistics.fmean(b) - statistics.fmean(a), math.sqrt(va + vb)
    if se == 0:
        return d, d, d
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * se
    return d, d - h, d + h


def permutation_p(a, b) -> float:
    """Exact two-sided permutation test of the difference in means, over every split of the
    pooled values into groups of len(a) and len(b) (252 for five seeds each)."""
    pool, k = list(a) + list(b), len(a)
    observed = abs(statistics.fmean(b) - statistics.fmean(a))
    hits = total = 0
    for idx in itertools.combinations(range(len(pool)), k):
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in idx]
        hits += abs(statistics.fmean(y) - statistics.fmean(x)) >= observed - 1e-12
        total += 1
    return hits / total


def holm(ps) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def bootstrap_ci(scores, keep, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the mean of ``scores[keep]``, resampling prompts.

    ``keep`` marks the prompts that count (coherent answers); a resample keeps every prompt,
    so the number of counted answers varies with it, as in the readout itself."""
    scores, keep = np.asarray(scores, float), np.asarray(keep, bool)
    if not keep.any():
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        b = rng.integers(0, len(scores), len(scores))
        if keep[b].any():
            means.append(scores[b][keep[b]].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return math.nan, math.nan
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
