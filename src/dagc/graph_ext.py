"""
graph_ext.py — additive extension to build_dependency_graph.

Does not modify graph.py. build_dependency_graph_extended() is a strict
superset when include_critical_values=True: original edges, unchanged,
plus new edges for _decision_critical_values() output (action verbs,
targets, rationale values -- the class that caused wildchat_4's silent
repair miss). Default is False, so every existing caller of
build_dependency_graph keeps its exact current behavior unless it
explicitly opts in.
"""
from typing import Dict, List

from .compressor import _decision_critical_values
from .graph import build_dependency_graph


def _critical_value_edges(decisions: List[Dict]) -> List[Dict]:
    """Edges in the same shape confirmed on build_dependency_graph's own
    output: {'artifact': ..., 'decision_msg_idx': ...}. Built directly
    from decisions -- doesn't touch build_dependency_graph's internals."""
    edges = []
    for d in decisions:
        for val in _decision_critical_values([d]):
            edges.append({'artifact': val, 'decision_msg_idx': d['msg_idx']})
    return edges


def build_dependency_graph_extended(messages: List[Dict], decisions: List[Dict],
                                     include_critical_values: bool = False) -> List[Dict]:
    """Base edges plus, optionally, critical-value edges -- deduplicated
    by (artifact, decision_msg_idx). Without dedup, an artifact that
    shows up in both the base graph and _decision_critical_values would
    be repaired twice downstream (confirmed: the 'use' artifact in a
    RAIDZ-decision test trace produced two separate edges for the same
    fact, doubling repair token cost for zero gain)."""
    edges = list(build_dependency_graph(messages, decisions))
    if include_critical_values:
        seen = {(e['artifact'], e['decision_msg_idx']) for e in edges}
        for e in _critical_value_edges(decisions):
            key = (e['artifact'], e['decision_msg_idx'])
            if key in seen:
                continue
            seen.add(key)
            edges.append(e)
    return edges


def compute_rci_extended(messages, compressed, decisions, include_critical_values=True):
    """Same logic as graph.compute_rci, over the extended, deduplicated
    edge set. Kept as a separate function -- not a monkeypatch -- so
    compute_rci's existing behavior and any numbers already computed
    with it are completely untouched.

    Recoverability check uses _value_still_recoverable (same helper
    rationale_ext._fact_already_present and utils.target_still_recoverable
    use elsewhere) instead of a raw `in` substring test -- the raw test
    false-negatives on anything that survived compression reworded,
    re-cased, or re-punctuated, silently undercounting RCI.
    """
    from .utils import _get_text, _value_still_recoverable
    edges = build_dependency_graph_extended(messages, decisions, include_critical_values)
    if not edges:
        return {'RCI': None, 'edges_total': 0, 'edges_preserved': 0, 'edges': []}
    comp_text = ' '.join(_get_text(m) for m in compressed)
    preserved = 0
    detail = []
    for e in edges:
        ok = _value_still_recoverable(e['artifact'], comp_text)
        preserved += int(ok)
        detail.append({**e, 'preserved': ok})
    return {'RCI': round(preserved / len(edges), 4),
            'edges_total': len(edges), 'edges_preserved': preserved, 'edges': detail}