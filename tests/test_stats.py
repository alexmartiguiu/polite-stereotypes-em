"""The interval and test functions reproduce the reference values of the paper's original builders."""

import math

import numpy as np
import pytest

from bias_em.stats import bootstrap_ci, holm, permutation_p, t_ci, welch_ci, wilson

NEUTRAL = [0.050, 0.031, 0.067, 0.044, 0.052]
STEREO = [0.47, 0.39, 0.55, 0.51, 0.43]


def test_t_ci():
    assert t_ci(NEUTRAL) == pytest.approx((0.0488, 0.0325773854452152, 0.0650226145547848), abs=1e-12)
    m, lo, hi = t_ci([0.3])
    assert m == 0.3 and math.isnan(lo) and math.isnan(hi)


def test_welch_ci():
    assert welch_ci(NEUTRAL, STEREO) == pytest.approx((0.4212, 0.34343489420113515, 0.4989651057988649), abs=1e-12)
    assert welch_ci([0.0] * 3, [0.0] * 3) == (0.0, 0.0, 0.0)


def test_permutation_p():
    # Five seeds per condition, completely separated: 2 of the 252 splits are as extreme.
    assert permutation_p(NEUTRAL, STEREO) == pytest.approx(2 / 252)
    assert permutation_p([0.1, 0.2, 0.15], [0.12, 0.3, 0.05]) == 1.0
    assert permutation_p([0.4, 0.5, 0.45], [0.0, 0.0, 0.0]) == pytest.approx(0.1)   # three seeds: 2 / 20


def test_holm():
    assert holm([2 / 252] * 4) == pytest.approx([8 / 252] * 4)
    assert holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_wilson():
    assert wilson(12, 72) == pytest.approx((0.09799275085321624, 0.26910920902104796), abs=1e-12)
    assert wilson(0, 72) == pytest.approx((0.0, 0.05065293981139638), abs=1e-12)
    assert wilson(72, 72)[1] == pytest.approx(1.0)


def test_bootstrap_ci():
    scores = np.array([0.0, 0.5, 1.0, 0.25] * 18)
    keep = np.array([True, True, False, True] * 18)
    assert bootstrap_ci(scores, keep) == pytest.approx((0.19490348399246704, 0.3020917338709676), abs=1e-12)
    assert all(math.isnan(x) for x in bootstrap_ci(scores, np.zeros(72, bool)))
