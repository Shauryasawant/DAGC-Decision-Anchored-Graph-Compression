"""
Regression tests for dagc_eval.stats.

The key one is test_effect_size_r_stays_bounded: the original notebook
version computed effect_size_r = |W| / sqrt(N) using the RAW Wilcoxon W
statistic, which is an unbounded rank-sum -- not a valid effect size.
A real r must always be in [0, 1]. This test uses a large, consistently-
biased sample specifically because that's exactly the regime where the
old buggy formula would exceed 1 -- small/noisy samples could accidentally
stay in-range and mask the bug.
"""
import math
import numpy as np
import pytest

from dagc_eval.stats import bootstrap_drr, wilcoxon_test, cohen_d, _rec_match, _evid_match


def test_effect_size_r_stays_bounded_large_consistent_difference():
    rng = np.random.default_rng(0)
    n = 40
    a = rng.normal(loc=0.85, scale=0.05, size=n)
    b = rng.normal(loc=0.40, scale=0.05, size=n)
    result = wilcoxon_test(list(a), list(b))
    assert 0.0 <= result['effect_size_r'] <= 1.0, (
        f"effect_size_r={result['effect_size_r']} is out of the valid [0,1] "
        f"range -- this is exactly the failure mode of the original W/sqrt(N) bug"
    )
    # A large, consistent gap should be statistically significant.
    assert result['significant'] is True
    assert result['effect_size_r'] > 0.5


def test_effect_size_r_zero_for_identical_scores():
    a = [0.7, 0.8, 0.75, 0.9, 0.6]
    result = wilcoxon_test(a, a)
    assert result['effect_size_r'] == 0.0
    assert result['significant'] is False


def test_wilcoxon_p_value_unaffected_by_effect_size_fix():
    """p_value comes straight from scipy and was never part of the bug --
    confirm it's still a valid probability."""
    rng = np.random.default_rng(1)
    a = rng.normal(0.8, 0.1, 20)
    b = rng.normal(0.5, 0.1, 20)
    result = wilcoxon_test(list(a), list(b))
    assert 0.0 <= result['p_value'] <= 1.0


def test_bootstrap_drr_basic():
    results = [{'DRR_soft': v} for v in [0.6, 0.7, 0.65, 0.8, 0.75]]
    ci = bootstrap_drr(results, n_bootstrap=200, seed=0)
    assert ci['ci_lo'] <= ci['mean'] <= ci['ci_hi']
    assert ci['n'] == 5


def test_bootstrap_drr_insufficient_data_returns_nan():
    ci = bootstrap_drr([{'DRR_soft': 0.5}])
    assert math.isnan(ci['mean'])


def test_cohen_d_zero_for_identical_distributions():
    a = [1.0, 2.0, 3.0]
    assert cohen_d(a, a) == 0.0


def test_cohen_d_sign_matches_direction():
    a = [0.9, 0.9, 0.9]
    b = [0.1, 0.1, 0.1]
    assert cohen_d(a, b) > 0
    assert cohen_d(b, a) < 0


def test_rec_match_ignores_action_type_decisions():
    results = [
        {'original': {'type': 'action'}, 'match': {'action_score': 1.0, 'target_score': 1.0}},
        {'original': {'type': 'judgment'}, 'match': {'action_score': 0.8, 'target_score': 0.6}},
    ]
    val = _rec_match(results)
    assert val == pytest.approx(0.5 * 0.8 + 0.5 * 0.6)


def test_evid_match_averages_rationale_scores():
    results = [
        {'match': {'rationale_score': 1.0}},
        {'match': {'rationale_score': 0.0}},
    ]
    assert _evid_match(results) == pytest.approx(0.5)
