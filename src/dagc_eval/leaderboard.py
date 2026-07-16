"""
Efficiency-normalized leaderboard scoring.

Ranking by raw DRR_soft alone rewards NOT compressing -- any method that
leaves the trace mostly untouched trivially preserves reproducibility.
This isn't specific to any one baseline; it's a property of the metric
being unconditioned on how much work was actually done. This module
fixes that generally, for any method, forever -- not by special-casing
any baseline name.
"""
from __future__ import annotations
import math
from typing import Dict, List


def efficiency_score(drr_soft: float, reduction_pct: float,
                      min_reduction_pct: float = 5.0) -> float:
    """
    DRR per unit of trace actually removed, with a floor: methods that
    achieve less than `min_reduction_pct` reduction (including negative
    reduction, i.e. the trace got LARGER) are not eligible for a
    positive efficiency score at all -- they get 0.0, regardless of how
    high their DRR_soft is, since "compressed" and "did nothing" are not
    the same claim.

    This generalizes cleanly: a method that reduces 90% and keeps
    DRR=0.95 scores higher than one that reduces 2% and keeps DRR=1.0,
    which matches what "a good compressor" should mean.
    """
    if reduction_pct < min_reduction_pct:
        return 0.0
    # Store reduction as a fraction for consistent aggregation.
    frac = min(1.0, reduction_pct / 100.0)
    return drr_soft * frac


def rank_leaderboard(results_by_method: Dict[str, List[Dict]],
                      min_reduction_pct: float = 5.0) -> List[Dict]:
    """
    results_by_method: {method_name: [ {'DRR_soft':..., 'reduction':...}, ... ]}
    (one list entry per trace scored, as produced by run_benchmark /
    run_method_comparison's per-trace results).

    Returns a list of rows sorted by mean efficiency_score, descending --
    a drop-in replacement for sorting by mean DRR_soft.
    """
    rows = []
    for name, trace_results in results_by_method.items():
        drrs = [r['DRR_soft'] for r in trace_results if r.get('DRR_soft') is not None]
        reds = [r.get('reduction', 0.0) for r in trace_results if r.get('DRR_soft') is not None]
        if not drrs:
            continue
        eff_scores = [efficiency_score(d, r, min_reduction_pct) for d, r in zip(drrs, reds)]
        rows.append({
            'method': name,
            'mean_drr_soft': round(sum(drrs) / len(drrs), 4),
            'mean_reduction': round(sum(reds) / len(reds), 1),
            'mean_efficiency': round(sum(eff_scores) / len(eff_scores), 4),
            'n': len(drrs),
        })
    rows.sort(key=lambda r: r['mean_efficiency'], reverse=True)
    for i, r in enumerate(rows, 1):
        r['rank'] = i
    return rows


def print_leaderboard(rows: List[Dict], reference_method: str = None) -> None:
    print('=' * 90)
    print('LEADERBOARD  (ranked by mean efficiency = DRR_soft x reduction_fraction)')
    print('=' * 90)
    print(f"{'Rank':>4}  {'Method':<38}  {'Efficiency':>10}  {'Mean DRR':>9}  {'Reduct%':>8}  {'n':>4}")
    print('-' * 90)
    for r in rows:
        marker = ' ★YOU' if reference_method and r['method'] == reference_method else ''
        print(f"{r['rank']:>4}  {r['method']:<38}  {r['mean_efficiency']:>10.4f}  "
              f"{r['mean_drr_soft']:>9.4f}  {r['mean_reduction']:>7.1f}%  {r['n']:>4}{marker}")
    print('=' * 90)
