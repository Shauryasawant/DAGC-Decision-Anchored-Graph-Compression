"""
Compression diagnostics: explain WHY a DRR score dropped, and diff an
original trace against its compressed version at the artifact level.

Built entirely from data compute_drr() already produces -- no new
extraction logic, just a readable report layer on top of it.
"""
from __future__ import annotations
from typing import Dict, List

from dagc.utils import _artifacts, _get_text

def explain_drr_full(result: Dict, show_all: bool = False) -> str:
    """
    Like explain_drr, but breaks down EVERY decision's component scores
    (action/target/rationale), not just ones that failed the binary
    'reproduced' gate. This exists because a decision can be marked
    reproduced=True (overall >= DRR_THRESHOLD) while still dragging
    DRR_soft down via a partial rationale or target score -- that
    signal is invisible if you only look at failures.
    """
    decisions = result.get('decisions', [])
    if not decisions:
        return "No decisions were found in this trace -- DRR is undefined."

    lines = [f"DRR_soft: {result.get('DRR_soft')}  (mean of per-decision scores)"]
    lines.append(f"{'idx':>4}  {'type':<12}  {'action':>7}  {'target':>7}  {'rationale':>9}  {'overall':>8}  {'pass?':>6}")

    ranked = sorted(decisions, key=lambda x: x['match']['decision_score'])
    for d in ranked:
        orig, match = d['original'], d['match']
        ts = 'n/a' if match['target_score'] is None else f"{match['target_score']:.3f}"
        lines.append(
            f"{orig.get('msg_idx', '?'):>4}  {orig.get('type', ''):<12}  "
            f"{match['action_score']:>7.3f}  {ts:>7}  {match['rationale_score']:>9.3f}  "
            f"{match['decision_score']:>8.3f}  {'✓' if match['reproduced'] else '✗':>6}"
        )
        if not show_all and match['decision_score'] >= 0.99:
            continue

    # Identify the component responsible for most score loss.
    deficits = {'action': 0.0, 'target': 0.0, 'rationale': 0.0}
    for d in decisions:
        m = d['match']
        deficits['action'] += (1.0 - m['action_score'])
        if m['target_score'] is not None:
            deficits['target'] += (1.0 - m['target_score'])
        deficits['rationale'] += (1.0 - m['rationale_score'])
    lines.append(f"\nTotal deficit by component (higher = bigger drag on DRR_soft):")
    for k, v in sorted(deficits.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<10}: {v:.3f}")

    return '\n'.join(lines)

def explain_drr(result: Dict) -> str:
    """
    Human-readable breakdown of a compute_drr() result: which decisions
    failed to reproduce and why (action/target/rationale mismatch),
    ranked worst-first.
    """
    decisions = result.get('decisions', [])
    if not decisions:
        return "No decisions were found in this trace -- DRR is undefined."

    failed = [d for d in decisions if not d['match']['reproduced']]
    passed = [d for d in decisions if d['match']['reproduced']]

    lines = []
    lines.append(f"DRR_soft: {result.get('DRR_soft')}  "
                  f"({len(passed)}/{len(decisions)} decisions reproduced, "
                  f"threshold={result.get('DRR_binary')})")
    lines.append(f"RCI: {result.get('RCI')}  "
                  f"(fraction of cross-message artifact dependencies preserved)")

    if not failed:
        lines.append("All decisions reproduced successfully.")
        return '\n'.join(lines)

    lines.append(f"\n{len(failed)} decision(s) failed to reproduce:\n")
    for d in sorted(failed, key=lambda x: x['match']['decision_score']):
        orig, repro, match = d['original'], d['reproduced'], d['match']
        reasons = []
        if match['action_score'] < 0.5:
            reasons.append(f"action mismatch (wanted '{orig.get('action')}', "
                            f"got '{repro.get('action')}')")
        if match.get('target_score') is not None and match['target_score'] < 0.5:
            reasons.append(f"target lost (wanted '{orig.get('target')}', "
                            f"got '{repro.get('target')}')")
        if match['rationale_score'] < 0.5:
            reasons.append("rationale/evidence not recoverable")
        if not repro.get('_success', True):
            reasons.append(f"reproduction failed outright ({repro.get('_fallback', 'unknown reason')})")

        lines.append(f"  [msg_idx={orig.get('msg_idx')}] {orig.get('type')} "
                      f"'{orig.get('verbatim', '')[:60]}'")
        lines.append(f"    score={match['decision_score']:.3f} — " + '; '.join(reasons))
    return '\n'.join(lines)


def diff_trace(original_messages: List[Dict], compressed_messages: List[Dict]) -> Dict:
    """
    Artifact-level diff: which paths/ids/emails/errors present in the
    original trace did NOT survive into the compressed trace.
    """
    orig_text = ' '.join(_get_text(m) for m in original_messages)
    comp_text = ' '.join(_get_text(m) for m in compressed_messages)
    orig_arts = _artifacts(orig_text)

    dropped: Dict[str, List[str]] = {}
    for kind in ('paths', 'ids', 'emails', 'errors'):
        missing = [a for a in orig_arts.get(kind, []) if a not in comp_text]
        if missing:
            dropped[kind] = missing

    orig_idxs = {m.get('_orig_idx', i) for i, m in enumerate(original_messages)}
    comp_idxs = {m.get('_orig_idx', i) for i, m in enumerate(compressed_messages)}
    fully_dropped_messages = sorted(orig_idxs - comp_idxs)

    return {
        'dropped_artifacts': dropped,
        'fully_dropped_message_indices': fully_dropped_messages,
        'orig_message_count': len(original_messages),
        'compressed_message_count': len(compressed_messages),
    }


def diff_trace_report(original_messages: List[Dict], compressed_messages: List[Dict]) -> str:
    """Human-readable version of diff_trace()."""
    d = diff_trace(original_messages, compressed_messages)
    lines = [f"Messages: {d['orig_message_count']} -> {d['compressed_message_count']}"]
    if d['fully_dropped_message_indices']:
        lines.append(f"Fully dropped message indices: {d['fully_dropped_message_indices']}")
    if d['dropped_artifacts']:
        lines.append("Dropped artifacts:")
        for kind, vals in d['dropped_artifacts'].items():
            lines.append(f"  {kind}: {vals}")
    else:
        lines.append("No artifacts (paths/ids/emails/errors) were lost.")
    return '\n'.join(lines)
