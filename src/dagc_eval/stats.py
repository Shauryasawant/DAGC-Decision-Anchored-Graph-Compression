"""
Statistical comparison utilities: bootstrap confidence intervals,
Wilcoxon signed-rank significance testing, and Cohen's d effect size,
used by run_benchmark / run_method_comparison / run_statistical_comparison
to back competitive claims with real significance tests rather than raw
score differences.

Ported from Main_notebook.py -- one correctness fix applied, see
wilcoxon_test's docstring.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional

import numpy as np


def bootstrap_drr(results: List[Dict], n_bootstrap: int = 1000,
                   confidence: float = 0.95, seed: int = 0) -> Dict:
    """Bootstrap mean + confidence interval over a list of {'DRR_soft': ...}
    dicts (as produced by compute_drr)."""
    rng = np.random.default_rng(seed)
    scores = [r['DRR_soft'] for r in results if r.get('DRR_soft') is not None]
    if len(scores) < 2:
        return {'mean': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan'),
                'std': float('nan'), 'n': len(scores)}
    arr = np.array(scores, dtype=float)
    boot = np.array([arr[rng.integers(0, len(arr), size=len(arr))].mean()
                      for _ in range(n_bootstrap)])
    return {'mean': float(arr.mean()),
            'ci_lo': float(np.percentile(boot, (1 - confidence) / 2 * 100)),
            'ci_hi': float(np.percentile(boot, (1 + confidence) / 2 * 100)),
            'std': float(arr.std()), 'n': len(scores)}


def wilcoxon_test(scores_a: List[float], scores_b: List[float]) -> Dict:
    """
    Paired Wilcoxon signed-rank test between two score arrays.

    CORRECTNESS NOTE: effect_size_r must be computed as Z / sqrt(N), where
    Z comes from the normal approximation of the raw W statistic -- NOT
    as `W / sqrt(N)` directly. W itself is an unbounded rank-sum with no
    fixed range, so dividing it by sqrt(N) does not produce a valid
    effect size (it can exceed 1, which a real r never should). This was
    a bug in the original notebook version; fixed here.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {'error': 'scipy not installed', 'p_value': float('nan'), 'significant': False}

    a, b = np.array(scores_a, dtype=float), np.array(scores_b, dtype=float)
    diffs = a - b
    n = len(diffs)

    if np.all(diffs == 0):
        return {'statistic': 0.0, 'p_value': 1.0, 'effect_size_r': 0.0,
                'significant': False, 'n': n}

    try:
        stat, p = wilcoxon(diffs, alternative='two-sided', zero_method='pratt')

        # Convert the Wilcoxon statistic to an approximate effect size.
        n_eff = int(np.count_nonzero(diffs))
        mu_w = n_eff * (n_eff + 1) / 4
        sigma_w = math.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24)
        z = (stat - mu_w) / sigma_w if sigma_w > 0 else 0.0
        r = abs(z) / math.sqrt(n)

        return {'statistic': float(stat), 'p_value': float(p), 'effect_size_r': round(r, 4),
                'significant': bool(p < 0.05), 'n': n}
    except Exception as e:
        return {'error': str(e), 'p_value': float('nan'), 'significant': False}


def cohen_d(a: List[float], b: List[float]) -> float:
    """Standardized mean difference (pooled std)."""
    a, b = np.array(a), np.array(b)
    return float((a.mean() - b.mean()) / math.sqrt((a.var() + b.var()) / 2 + 1e-12))


def _rec_match(results: List[Dict]) -> float:
    """Mean reconstruction-match score over judgment/confirmation decisions."""
    rel = [r for r in results if r['original']['type'] in ('judgment', 'confirmation')]
    if not rel:
        return float('nan')
    return float(np.mean([0.5 * r['match']['action_score'] + 0.5 * (r['match']['target_score'] or 0.0)
                           for r in rel]))


def _evid_match(results: List[Dict]) -> float:
    """Mean rationale/evidence-match score across all decisions."""
    return float(np.mean([r['match']['rationale_score'] for r in results])) if results else float('nan')
