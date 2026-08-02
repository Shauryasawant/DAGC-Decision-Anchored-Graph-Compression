"""
DRR pipeline + a small synthetic trace generator, for self-testing your
own compressor (or dagc's) on realistic decision-bearing traces.
Optionally uses an LLM (BYOK, via dagc_eval.interfaces.LLMClient) for
reproduction of decisions whose source message didn't survive
compression; works fully offline (deterministic-only) if you omit it.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
from typing import Callable, Dict, List, Optional

import numpy as np
from dagc.compressor import compress_dagc
from dagc.extraction import extract_decisions
from dagc.graph import compute_rci
from dagc.utils import _encode, _artifacts, _cos, _get_text, _tok, target_still_recoverable

from .interfaces import LLMClient
from .match import DRR_THRESHOLD, match_decision
from .reproduce import reproduce_decision
from dagc.baselines import BASELINES
from .stats import bootstrap_drr, wilcoxon_test, cohen_d, _rec_match, _evid_match
TASKS = [
    {'task': 'Choose the best compression algorithm for agent traces.',
     'options': ['CompressMMR', 'BudgetComp', 'SelectiveCtx'], 'winner': 'CompressMMR',
     'winner_metrics': {'SP': 0.91, 'task_success': 0.82, 'compression': 0.73},
     'loser_metrics': [{'SP': 0.88, 'task_success': 0.77, 'compression': 0.68},
                       {'SP': 0.84, 'task_success': 0.74, 'compression': 0.50}]},
    {'task': 'Identify the root cause of the pipeline failure.',
     'options': ['OOM error', 'network timeout', 'corrupt checkpoint'],
     'winner': 'corrupt checkpoint',
     'winner_metrics': {'error_log': 'checkpoint_load_failed', 'file': '/checkpoints/run_42.pt'},
     'loser_metrics': [{'error_log': 'none'}, {'error_log': 'transient'}]},
    {'task': 'Select the retrieval strategy for RAG pipeline.',
     'options': ['BM25', 'dense_retrieval', 'hybrid'], 'winner': 'hybrid',
     'winner_metrics': {'MRR': 0.84, 'latency_ms': 120, 'P@10': 0.79},
     'loser_metrics': [{'MRR': 0.71, 'latency_ms': 45, 'P@10': 0.65},
                       {'MRR': 0.78, 'latency_ms': 95, 'P@10': 0.73}]},
]

NOISE_TEMPLATES = [
    'API metadata: region=us-east-1 latency={lat}ms call_id={cid} rate_limit={rl}/1000.',
    'Cache miss for key {cid}. Fetching from origin. TTL=3600s.',
    'Pagination: page={pg} of {total}. Cursor={cid}. Next batch in {lat}ms.',
    'Telemetry: span_id={cid} trace_id={cid2} service=agent version=2.1.4.',
]


def _make_noise(n=3, seed_str=''):
    rng = random.Random(hashlib.md5((str(n) + seed_str).encode()).hexdigest())
    lines = []
    for tmpl in rng.sample(NOISE_TEMPLATES, min(n, len(NOISE_TEMPLATES))):
        lines.append(tmpl.format(
            lat=rng.randint(10, 500), cid=hashlib.md5(str(rng.random()).encode()).hexdigest()[:8],
            rl=rng.randint(500, 999), pg=rng.randint(1, 10), total=rng.randint(10, 50),
            cid2=hashlib.md5(str(rng.random()).encode()).hexdigest()[:8]))
    return ' '.join(lines)


def generate_trace(task_spec: Dict, noise_level: int = 3, rng_seed: int = 0) -> List[Dict]:
    """Generate a synthetic agent trace with a ground-truth decision baked
    in, for testing a compressor's decision reproducibility."""
    rng = random.Random(rng_seed)
    opts = task_spec['options']
    win = task_spec['winner']
    win_m = task_spec['winner_metrics']
    los_m = task_spec['loser_metrics']
    task = task_spec['task']
    exp_id = f'EXP-{rng.randint(1000, 9999)}'
    result_file = f'/workspace/results_{exp_id.lower()}.json'
    trace = [
        {'role': 'system', 'content': 'You are a research agent. Always cite evidence for decisions. Preserve experiment IDs and file paths.'},
        {'role': 'user', 'content': task},
    ]
    trace.append({'role': 'assistant', 'content': f'I will search for evidence on each option: {", ".join(opts)}.',
                  'tool_call': {'name': 'search', 'args': {'query': f'{task} comparison evaluation'}}})
    trace.append({'role': 'tool', 'name': 'search', 'content':
        f'{opts[0]}: {json.dumps(win_m)}. {opts[1]}: {json.dumps(los_m[0])}. {opts[2]}: {json.dumps(los_m[1])}. '
        + _make_noise(noise_level, f'{rng_seed}s')})
    trace.append({'role': 'assistant', 'content': f'Reading detailed report for {win}.',
                  'tool_call': {'name': 'read_report', 'args': {'target': win, 'experiment': exp_id}}})
    metrics_str = ' '.join(f'{k}={v}' for k, v in win_m.items())
    trace.append({'role': 'tool', 'name': 'read_report', 'content':
        f'Report for {win}. Experiment: {exp_id}. Metrics: {metrics_str}. Output saved to {result_file}. '
        + _make_noise(noise_level, f'{rng_seed}r')})
    trace.append({'role': 'assistant', 'content': 'Comparing all options on key metrics.',
                  'tool_call': {'name': 'compare', 'args': {'options': opts, 'metric': 'primary'}}})
    cc = f'Winner: {win} ({metrics_str}). '
    for i, opt in enumerate([o for o in opts if o != win]):
        cc += f'{opt}: {json.dumps(los_m[min(i, len(los_m) - 1)])}. '
    cc += _make_noise(noise_level, f'{rng_seed}c')
    trace.append({'role': 'tool', 'name': 'compare', 'content': cc})
    metrics_cited = ', '.join(f'{k}={v}' for k, v in list(win_m.items())[:3])
    trace.append({'role': 'assistant', 'content':
        f'{win} is the clear winner. Evidence: {metrics_cited}. '
        f'Experiment: {exp_id}. Results: {result_file}. Recommend implementing {win} first.'})
    trace.append({'role': 'user', 'content':
        f'Confirm the recommendation and ensure {exp_id} and {result_file} are preserved.'})
    trace.append({'role': 'assistant', 'content':
        f'Confirmed. Recommendation: {win}. Experiment ID {exp_id} and file {result_file} are preserved. Key evidence: {metrics_cited}.'})
    return trace

def _legacy_metrics(orig_msgs, comp_msgs, decisions=None):
    orig_text = ' '.join(_get_text(m) for m in orig_msgs)
    comp_text = ' '.join(_get_text(m) for m in comp_msgs)
    oe, ce = _encode([orig_text, comp_text])
    SP = _cos(oe, ce)
    orig_arts = _artifacts(orig_text)

    def _ratio(items):
        """Recoverability ratio using the same primitive RCI/chain_RCI treat
        as canonical (loose), plus the stricter raw-substring companion for
        transparency. None means 'nothing to score' -- never floored to 0."""
        if not items:
            return None, None
        loose = sum(1 for a in items if target_still_recoverable(a, comp_text))
        strict = sum(1 for a in items if a in comp_text)
        return round(loose / len(items), 4), round(strict / len(items), 4)

    all_arts = orig_arts['paths'] + orig_arts['ids'] + orig_arts['errors']
    art_ret, art_ret_strict = _ratio(all_arts)
    result = {'SP': round(SP, 4), 'art_ret': art_ret, 'art_ret_strict': art_ret_strict}

    if decisions:
        from dagc.compressor import _collect_decision_artifacts_by_decision
        from dagc.extraction import _preserved_tag_candidates
        by_decision = _collect_decision_artifacts_by_decision(decisions)

        # Map each surviving compressed message back to its original msg_idx.
        text_by_orig_idx = {}
        for m in comp_msgs:
            oi = m.get('_orig_idx')
            if oi is not None:
                text_by_orig_idx.setdefault(oi, []).append(_get_text(m))

        hits, total = 0, 0
        for msg_idx, arts in by_decision.items():
            if not arts:
                continue
            own_text = ' '.join(text_by_orig_idx.get(msg_idx, []))
            # A value legitimately rescued via a [preserved: value#d<msg_idx>]
            # tag elsewhere in the trace also counts -- that's the compressor's
            # own real rescue path, not a false positive.
            rescued_here = {val for m in comp_msgs
                            for val, _ in _preserved_tag_candidates(_get_text(m), decision_idx=msg_idx)}
            for a in arts:
                total += 1
                if target_still_recoverable(a, own_text) or a in rescued_here:
                    hits += 1

        result['decision_art_ret'] = round(hits / total, 4) if total else None

    return result

def compute_drr(messages, compressor=compress_dagc, llm=None,
                 decision_roles=('assistant',), verbose=True,
                 ):
    """
    Run the full Decision Reproducibility Rate pipeline on a trace:
    extract decisions -> compress -> try to reproduce each decision from
    the compressed trace -> score. `llm` is optional (BYOK); omit it for
    fully offline, deterministic-only scoring.
    """
    decisions = extract_decisions(messages, decision_roles=decision_roles)

    compressed = compressor(messages)
    orig_toks = sum(_tok(_get_text(m)) for m in messages)
    comp_toks = sum(_tok(_get_text(m)) for m in compressed)
    reduction = 100 * (1 - comp_toks / max(1, orig_toks))

    if not decisions:
        if verbose:
            print(f'No decisions found in trace — DRR/RCI undefined, '
                  f'but compression measured: {reduction:.1f}% reduction.')
        legacy = _legacy_metrics(messages, compressed)
        return {'DRR_soft': None, 'DRR_binary': None, 'decisions': [],
                'compressed': compressed, 'orig_tokens': orig_toks,
                'comp_tokens': comp_toks, 'reduction': reduction,
                'RCI': None, **legacy}

    compressed = compressor(messages)
    orig_toks = sum(_tok(_get_text(m)) for m in messages)
    comp_toks = sum(_tok(_get_text(m)) for m in compressed)
    reduction = 100 * (1 - comp_toks / max(1, orig_toks))

    results = []
    for d in decisions:
        repro = reproduce_decision(compressed, d, llm=llm)
        match = match_decision(d, repro)
        results.append({'original': d, 'reproduced': repro, 'match': match})
        if verbose:
            flag = '✓' if match['reproduced'] else '✗'
            ts = 'n/a' if match['target_score'] is None else f'{match["target_score"]:.2f}'
            print(f'[{flag}] {d["type"]:12s} action={match["action_score"]:.2f} '
                  f'target={ts} rationale={match["rationale_score"]:.2f} '
                  f'score={match["decision_score"]:.3f}')

    scores = [r['match']['decision_score'] for r in results]
    DRR_soft = float(np.mean(scores)) if scores else 0.0
    DRR_bin = float(np.mean([r['match']['reproduced'] for r in results])) if results else 0.0
    legacy = _legacy_metrics(messages, compressed, decisions)
    rci_data = compute_rci(messages, compressed, decisions)

    if verbose:
        print(f'\nDRR (soft)   : {DRR_soft:.4f}')
        print(f'DRR (binary, >= {DRR_THRESHOLD}) : {DRR_bin:.2%}')
        print(f'RCI           : {rci_data["RCI"]}')
        print(f'Reduction     : {reduction:.1f}%')
        print(f'Artifact ret. : {legacy["art_ret"]:.2%} (all) / '
              f'{legacy.get("decision_art_ret"):.2%} (decision-critical)' if legacy.get("decision_art_ret") is not None
              else '(no decisions to score) (decision-critical)')

    out = {'DRR_soft': DRR_soft, 'DRR_binary': DRR_bin, 'decisions': results,
           'compressed': compressed, 'orig_tokens': orig_toks, 'comp_tokens': comp_toks,
           'reduction': reduction, 'RCI': rci_data['RCI'], **legacy}
    return out
def run_benchmark(n_traces_per_task=3, noise_levels=[1, 2, 3, 4, 5],
                   compressor=compress_dagc, verbose=False):
    """
    Sweep DRR/RCI/artifact-retention for a single compressor across all
    TASKS and noise levels. Prints a per-trace table, a summary (via
    _print_benchmark_summary), and a bootstrap 95% CI over DRR_soft.
    """
    if compressor is None:
        compressor = compress_dagc
    total = len(TASKS) * n_traces_per_task * len(noise_levels)
    done = 0
    all_results = []
    print('=' * 65 + f'\nDRR BENCHMARK  —  {total} traces  —  {compressor.__name__}\n' + '=' * 65)
    print(f'{"Trace":>5}  {"Task":>5}  {"Noise":>5}  {"DRR_soft":>9}  {"DRR_bin":>8}  '
          f'{"RCI":>6}  {"SP":>7}  {"Red%":>6}  {"ArtRet":>8}')
    print('-' * 65)
    for task_idx, task_spec in enumerate(TASKS):
        for noise in noise_levels:
            for seed in range(n_traces_per_task):
                trace = generate_trace(task_spec, noise_level=noise, rng_seed=seed * 100 + task_idx)
                result = compute_drr(trace, compressor, verbose=verbose)
                result.update({'task_idx': task_idx, 'noise': noise, 'seed': seed,
                                'task_label': task_spec['task'][:35]})
                all_results.append(result)
                done += 1
                drr_s = result.get('DRR_soft') or float('nan')
                drr_b = result.get('DRR_binary') or float('nan')
                rci = result.get('RCI') or float('nan')
                print(f'{done:>5}  {task_idx:>5}  {noise:>5}  {drr_s:>9.4f}  '
                      f'{drr_b:>7.2%}  {rci:>6.4f}  {result.get("SP", 0):>7.4f}  '
                      f'{result.get("reduction", 0):>5.1f}%  {result.get("art_ret", 0):>7.2%}')
    print()
    _print_benchmark_summary(all_results)
    ci = bootstrap_drr(all_results)
    print(f'\nBootstrap 95% CI: {ci["mean"]:.4f}  [{ci["ci_lo"]:.4f}, {ci["ci_hi"]:.4f}]  (n={ci["n"]})')
    return all_results
 
 
def _print_benchmark_summary(results):
    valid = [r for r in results if r.get('DRR_soft') is not None]
    if not valid:
        print('No valid results.')
        return
    drr_s = [r['DRR_soft'] for r in valid]
    drr_b = [r['DRR_binary'] for r in valid]
    sps = [r.get('SP', 0) for r in valid]
    reds = [r.get('reduction', 0) for r in valid]
    arts = [r.get('art_ret', 0) for r in valid]
    rci_applicable = [r for r in valid if r.get('RCI') is not None]
    rcis = [r['RCI'] for r in rci_applicable]
    sp_arr, drr_arr = np.array(sps), np.array(drr_s)
    corr = (float(np.corrcoef(sp_arr, drr_arr)[0, 1])
            if len(sp_arr) > 1 and sp_arr.std() > 0 and drr_arr.std() > 0 else float('nan'))
    deceptive = [r for r in valid if r.get('SP', 0) > 0.85 and r['DRR_soft'] < 0.50]

    print('=' * 65 + '\nBENCHMARK SUMMARY\n' + '=' * 65)
    print(f'  Traces evaluated          : {len(valid)}')
    print(f'  Mean DRR  (soft)          : {np.mean(drr_s):.4f}')
    print(f'  Mean DRR  (binary)        : {np.mean(drr_b):.2%}')
    rci_coverage = len(rci_applicable) / len(valid)
    print(f'  RCI coverage              : {len(rci_applicable)}/{len(valid)} traces ({rci_coverage:.1%})')
    print(f'  Mean RCI (where defined)  : {np.mean(rcis):.4f}' if rcis else
          '  Mean RCI (where defined)  : n/a (no trace had cross-message artifact dependencies)')
    print(f'  Mean SP                   : {np.mean(sps):.4f}')
    print(f'  Mean reduction            : {np.mean(reds):.1f}%')
    print(f'  Mean artifact retention   : {np.mean(arts):.2%}')
    print(f'  Pearson(SP, DRR_soft)     : {corr:+.4f}')
    label = ('UNCORRELATED' if abs(corr) < 0.3 else 'negatively correlated' if corr < 0 else f'correlated ({corr:.2f})')
    if not math.isnan(corr):
        print(f'  → SP and DRR are {label}.')
    print(f'  "Deceptive" (SP>0.85, DRR<0.50): {len(deceptive)}')
    for r in deceptive[:3]:
        print(f'    task={r["task_label"][:30]}  noise={r["noise"]}  SP={r.get("SP", 0):.3f}  DRR={r["DRR_soft"]:.3f}')

    noise_levels = sorted(set(r['noise'] for r in valid))
    if len(noise_levels) > 1:
        print(f'\n  DRR by noise level:')
        for nl in noise_levels:
            sub = [r for r in valid if r['noise'] == nl]
            m_drr = np.mean([r['DRR_soft'] for r in sub])
            m_sp = np.mean([r.get('SP', 0) for r in sub])
            sub_rci = [r['RCI'] for r in sub if r.get('RCI') is not None]
            rci_str = f'{np.mean(sub_rci):.4f} ({len(sub_rci)}/{len(sub)})' if sub_rci else 'n/a (0/{})'.format(len(sub))
            print(f'    noise={nl}:  DRR={m_drr:.4f}  SP={m_sp:.4f}  RCI={rci_str}  n={len(sub)}')
    print('=' * 65)
 
 
def run_method_comparison(n_traces_per_task=2, noise_levels=[3],
                           methods=None, verbose=True, run_stats=True):
    """
    Compare DAGC (or any set of `methods`, defaulting to BASELINES) head
    to head across TASKS/noise_levels. If 'DAGC' is present and
    run_stats=True, also runs run_statistical_comparison against it.
    """
    methods = methods or BASELINES
    agg = {name: {'compression': [], 'rec_match': [], 'evid_match': [], 'art_ret': [],
                   'CRR': [], 'RCI': [], 'drr_soft_raw': []} for name in methods}
    for task_idx, task_spec in enumerate(TASKS):
        for noise in noise_levels:
            for seed in range(n_traces_per_task):
                trace = generate_trace(task_spec, noise_level=noise, rng_seed=seed * 100 + task_idx)
                decisions = extract_decisions(trace)
                if not decisions:
                    continue
                orig_toks = sum(_tok(_get_text(m)) for m in trace)
                for name, fn in methods.items():
                    comp = fn(trace, seed=seed)
                    comp_toks = sum(_tok(_get_text(m)) for m in comp)
                    reduction = 100 * (1 - comp_toks / max(1, orig_toks))
                    results = []
                    for d in decisions:
                        repro = reproduce_decision(comp, d)
                        results.append({'original': d, 'reproduced': repro, 'match': match_decision(d, repro)})
                    drr_bin = float(np.mean([r['match']['reproduced'] for r in results]))
                    drr_soft = float(np.mean([r['match']['decision_score'] for r in results]))
                    legacy = _legacy_metrics(trace, comp)
                    rci = compute_rci(trace, comp, decisions)
                    agg[name]['compression'].append(reduction)
                    agg[name]['rec_match'].append(_rec_match(results))
                    agg[name]['evid_match'].append(_evid_match(results))
                    agg[name]['art_ret'].append(legacy['art_ret'])
                    agg[name]['CRR'].append(drr_bin)
                    agg[name]['drr_soft_raw'].append(drr_soft)
                    if rci['RCI'] is not None:
                        agg[name]['RCI'].append(rci['RCI'])
    summary = {}
    for name, vals in agg.items():
        summary[name] = {k: (float(np.nanmean(v)) if v else float('nan'))
                          for k, v in vals.items() if k != 'drr_soft_raw'}
        summary[name]['drr_soft_raw'] = vals['drr_soft_raw']
    if verbose:
        print('=' * 85)
        print('METHOD COMPARISON')
        print('=' * 85)
        print(f'{"Method":>18}  {"Compress%":>9}  {"RecMatch":>9}  {"EvidMatch":>9}  {"ArtRet":>7}  {"CRR":>6}  {"RCI":>6}')
        print('-' * 85)
        for name, row in summary.items():
            print(f'{name:>18}  {row["compression"]:>8.1f}%  {row["rec_match"]:>8.2%}  '
                  f'{row["evid_match"]:>8.2%}  {row["art_ret"]:>6.2%}  {row["CRR"]:>5.2%}  {row["RCI"]:>5.2%}')
        print('=' * 85)
    if run_stats and 'DAGC' in summary:
        mscores = {n: r['drr_soft_raw'] for n, r in summary.items() if r['drr_soft_raw']}
        if len(mscores) >= 2:
            run_statistical_comparison(mscores, reference_name='DAGC', verbose=verbose)
    return summary
 
 
def run_statistical_comparison(method_scores, reference_name='DAGC', verbose=True):
    """
    Wilcoxon signed-rank + Bonferroni-corrected p-values + Cohen's d,
    each method vs. `reference_name`, plus a bootstrap 95% CI per method.
    """
    n_comp = max(1, len(method_scores) - 1)
    ref = method_scores.get(reference_name, [])
    summary = {}
    for name, scores in method_scores.items():
        ci = bootstrap_drr([{'DRR_soft': s} for s in scores])
        wil = wilcoxon_test(ref, scores) if name != reference_name else {}
        cd = cohen_d(ref, scores) if name != reference_name else 0.0
        raw_p = wil.get('p_value', float('nan'))
        corr_p = min(1.0, raw_p * n_comp) if not math.isnan(raw_p) else float('nan')
        summary[name] = {'mean_drr': ci['mean'], 'ci_95_lo': ci['ci_lo'], 'ci_95_hi': ci['ci_hi'],
                          'std': ci['std'], 'wilcoxon_p': raw_p, 'bonferroni_p': corr_p,
                          'cohen_d_vs_ref': cd, 'sig_bonferroni': bool(corr_p < 0.05)}
    if verbose:
        print('=' * 80)
        print(f'STATISTICAL COMPARISON  (reference = {reference_name})')
        print('=' * 80)
        print(f'{"Method":>20}  {"Mean DRR":>9}  {"95% CI":>18}  {"p(Bonf.)":>10}  {"Cohen d":>8}  {"Sig?":>5}')
        print('-' * 80)
        for name, row in summary.items():
            ci_s = f'[{row["ci_95_lo"]:.3f}, {row["ci_95_hi"]:.3f}]'
            p_s = f'{row["bonferroni_p"]:.4f}' if not math.isnan(row['bonferroni_p']) else '  n/a '
            d_s = f'{row["cohen_d_vs_ref"]:+.3f}' if name != reference_name else '  ref '
            sig = '  ✓  ' if row.get('sig_bonferroni') else '     '
            print(f'{name:>20}  {row["mean_drr"]:>8.4f}  {ci_s:>18}  {p_s:>10}  {d_s:>8}  {sig:>5}')
        print('=' * 80)
    return summary
