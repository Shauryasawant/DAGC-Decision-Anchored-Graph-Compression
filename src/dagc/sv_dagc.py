"""
sv_dagc.py — Self-Verifying DAGC.

Additive, opt-in wrapper around compress_dagc. Does not modify compression.py's
default code path: compress_dagc(...) called directly behaves exactly as before.
This module adds a deterministic, zero-LLM-call verify+repair pass on top,
built entirely from primitives already in the package:
  - .compressor:  compress_dagc, DAGCConfig, DAGC_CFG, _footprint_text
  - .extraction:  extract_decisions
  - .graph:       build_dependency_graph, compute_rci
  - .utils:       _tok, target_still_recoverable, _artifacts

No new named concepts, no external imports.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .compressor import compress_dagc, DAGCConfig, DAGC_CFG, _footprint_text
from .extraction import extract_decisions
from .graph import build_dependency_graph, compute_rci
from .utils import _tok, target_still_recoverable, _artifacts
from .graph_ext import compute_rci_extended
from .rationale_ext import inject_rationale_stubs, inject_dropped_rationale_stubs
from .state_goal_extract import extract_state_goal_candidates
from .anchor_lifecycle import resolve_lifecycle, active_anchors, inject_anchor_stubs
from .referent_anchor import inject_referent_stubs


def _find_owner_idx(artifact: str, messages: List[Dict]) -> Optional[int]:
    """Locate the earliest original message that actually contains this
    artifact. This is the repair source of truth -- always the untouched
    original trace, never the compressed output, so repaired fidelity is
    exactly the original trace's fidelity, independent of how aggressively
    compress_dagc ran."""
    for i, m in enumerate(messages):
        if target_still_recoverable(artifact, _footprint_text(m)):
            return i
    return None


def _repair_one(compressed: List[Dict], owner_idx: int, owner_role: str, artifact: str) -> Dict:
    """Reattach `artifact` to the message in `compressed` whose _orig_idx
    matches owner_idx, if that message survived compression. Otherwise
    append a small standalone repair stub carrying just the artifact.
    Mutates and returns the target message dict that now carries it."""
    for mc in compressed:
        if mc.get('_orig_idx') == owner_idx:
            mc['content'] = (mc.get('content', '') or '') + f" [repaired: {artifact}]"
            return mc
    stub = {
        'role': owner_role,
        '_orig_idx': owner_idx,
        '_repair_stub': True,
        'content': f"[repaired from dropped msg: {artifact}]",
    }
    compressed.append(stub)
    return stub


def _verify_and_repair(messages: List[Dict], compressed: List[Dict], decisions: List[Dict],
                        rci_floor: float, max_repair_tokens: Optional[float],
                        use_extended_scope: bool = False) -> Tuple[List[Dict], Dict[str, Any]]:
    """Single bounded pass: compute RCI once, repair every failing edge at
    most once, recompute RCI once at the end. No loop, no recompression,
    no re-derivation of edges -- termination is O(len(edges)) by
    construction.

    Edges are deduplicated by (artifact, decision_msg_idx) before the
    loop runs (graph_ext.build_dependency_graph_extended handles this
    when use_extended_scope=True; the base graph has no cross-source
    duplication to begin with when use_extended_scope=False), so no
    artifact is repaired more than once regardless of how many edge
    sources reference it.
    """
    rci_fn = (lambda m, c, d: compute_rci_extended(m, c, d, include_critical_values=True)) \
             if use_extended_scope else compute_rci
    rci_pre = rci_fn(messages, compressed, decisions)
    report: Dict[str, Any] = {
        'RCI_pre_repair': rci_pre['RCI'],
        'RCI_post_repair': rci_pre['RCI'],
        'repaired': False,
        'artifacts_added': 0,
        'token_overhead': 0,
        'shortfall': [],
    }

    if rci_pre['RCI'] is None or rci_pre['RCI'] >= rci_floor:
        return compressed, report

    budget = max_repair_tokens if max_repair_tokens is not None else float('inf')
    overhead = 0
    added = 0

    for edge in rci_pre['edges']:
        if edge.get('preserved'):
            continue
        artifact = edge['artifact']
        cost = _tok(artifact)
        if overhead + cost > budget:
            report['shortfall'].append(artifact)
            continue
        owner_idx = _find_owner_idx(artifact, messages)
        if owner_idx is None:
            # Artifact came from a decision's derived value, not literal
            # message text -- nothing to reattach it to. Report, don't guess.
            report['shortfall'].append(artifact)
            continue
        owner_role = messages[owner_idx].get('role', 'assistant')
        _repair_one(compressed, owner_idx, owner_role, artifact)
        overhead += cost
        added += 1

    rci_post = rci_fn(messages, compressed, decisions)
    report['RCI_post_repair'] = rci_post['RCI']
    report['repaired'] = added > 0
    report['artifacts_added'] = added
    report['token_overhead'] = overhead
    return compressed, report


def compress_dagc_sv(messages: List[Dict], cfg: Optional[DAGCConfig] = None,
                      decision_roles: Tuple[str, ...] = ('user', 'assistant'),
                      force_preserve: Optional[Iterable[str]] = None,
                      rci_floor: float = 1.0,
                      max_repair_tokens: Optional[float] = None,
                      diagnostics: Optional[Dict[str, Any]] = None,
                      use_extended_scope: bool = True,
                      preserve_rationale: bool = False,
                      rationale_window: int = 3,
                      max_rationale_tokens: Optional[float] = 40,
                      preserve_dropped_rationale: bool = False,
                      max_dropped_stubs: int = 10,
                      max_dropped_stub_tokens: Optional[float] = 30,
                      min_rationale_confidence: str = 'high',
                      preserve_state: bool = False,
                      preserve_goals: bool = False,
                      max_anchor_stubs: int = 10,
                      max_anchor_stub_tokens: Optional[float] = 30,
                      preserve_referents: bool = False,
                      max_referent_stubs: int = 10,
                      max_referent_stub_tokens: Optional[float] = 20,
                      referent_lookback: int = 30,
                      ) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Self-verifying entry point. Runs compress_dagc exactly as-is, then
    checks RCI and repairs any failing decision-artifact edge using the
    original trace as the sole source of truth.

        from dagc.sv_dagc import compress_dagc_sv
        compressed, report = compress_dagc_sv(messages, rci_floor=1.0)

    Returns:
        (compressed_messages, report) where report contains
        RCI_pre_repair, RCI_post_repair, repaired, artifacts_added,
        token_overhead, and shortfall (artifacts that couldn't be
        repaired within max_repair_tokens or had no locatable source).
        If preserve_rationale=True, also includes rationale_candidates_found,
        rationale_stubs_added, rationale_added_detail.
        If preserve_state=True or preserve_goals=True, also includes
        anchor_stubs_added, anchor_stub_tokens, anchor_added_detail (see
        state_goal_extract.py / anchor_lifecycle.py). Neither key is
        present when both flags are False (the default) -- this pass is
        fully opt-in and adds nothing to the report or the output
        otherwise.
        If preserve_referents=True, also includes referent_candidates_found,
        referent_stubs_added, referent_added_detail (see referent_anchor.py)
        -- recovers elided-object confirmations ("Rolled back.") whose
        referent was only named several turns earlier and had no
        regex-detectable artifact of its own. Fully opt-in, off by
        default, adds nothing to the report or output otherwise.
    """
    if cfg is None:
        cfg = DAGC_CFG

    # compress_dagc's published signature DOES accept `diagnostics`
    # (confirmed directly against compressor.py, not assumed). Passing it
    # through lets us read back compress_dagc's own evidence-FILTERED
    # valid_decisions, instead of re-deriving decisions independently.
    _inner_diag: Dict[str, Any] = {}
    compressed = compress_dagc(messages, cfg=cfg, decision_roles=decision_roles,
                                force_preserve=force_preserve, diagnostics=_inner_diag)

    # IMPORTANT: extract_decisions() has NO anti-injection evidence gate --
    # _judgment_has_evidence only runs inside compress_dagc. Re-deriving
    # decisions from scratch here would repair and rationale-stub
    # judgments/confirmations compress_dagc deliberately did NOT trust,
    # defeating the gate at exactly the layer meant to verify fidelity.
    decisions = _inner_diag.get('valid_decisions')
    valid_decisions_was_none = decisions is None

    # TIGHTENING (was option A): the unfiltered extract_decisions() fallback
    # had no evidence gate and could leak false decisions from normal
    # conversation into RCI repair / rationale stubbing. Never use it
    # silently. If compress_dagc's own evidence-filtered valid_decisions is
    # missing, treat that as "no decisions" rather than re-deriving an
    # ungated set from scratch.
    if valid_decisions_was_none:
        decisions = []
        decisions_source = 'no_valid_decisions_fallback_skipped'
    else:
        decisions_source = 'compress_dagc_valid_decisions'

    low_decision_trace = len(decisions) == 0

    compressed, report = _verify_and_repair(messages, compressed, decisions,
                                             rci_floor, max_repair_tokens, use_extended_scope)
    report['decisions_source'] = decisions_source
    report['decisions_source'] = decisions_source
    report['low_decision_trace'] = low_decision_trace
    report['valid_decision_count'] = len(decisions)
    report['valid_decisions_was_none'] = valid_decisions_was_none

    if preserve_rationale:
        compressed, rationale_report = inject_rationale_stubs(
            compressed, messages, decisions,
            window=rationale_window, max_stub_tokens=max_rationale_tokens,
            min_confidence=min_rationale_confidence)
        report.update(rationale_report)

    if preserve_dropped_rationale:
        compressed, dropped_report = inject_dropped_rationale_stubs(
            compressed, messages,
            max_stub_tokens=max_dropped_stub_tokens,
            max_stubs_total=max_dropped_stubs,
            min_confidence=min_rationale_confidence)
        report.update(dropped_report)

    if preserve_state or preserve_goals:
        # Isolated from every Decision/Evidence code path above: this
        # reads `messages` (the original trace) directly and only ever
        # appends to `compressed`. A false positive here changes only
        # this block's own numbers, never RCI repair or rationale
        # stubbing above.
        allowed_categories = tuple(
            cat for cat, flag in (('state', preserve_state), ('goal', preserve_goals))
            if flag
        )
        candidates = extract_state_goal_candidates(messages)
        candidates = [c for c in candidates if c['anchor_type'] in allowed_categories]
        anchors = resolve_lifecycle(messages, candidates)
        open_anchors = active_anchors(anchors, categories=allowed_categories)
        compressed, anchor_report = inject_anchor_stubs(
            compressed, messages, open_anchors,
            max_stubs=max_anchor_stubs, max_stub_tokens=max_anchor_stub_tokens)
        report.update(anchor_report)

    if preserve_referents:
        # Isolated the same way as the preserve_state/preserve_goals
        # block above: reads `messages` directly, only ever appends to
        # `compressed`. Recovers elided-object confirmations ("Rolled
        # back.") by linking them back to the most recent grounded
        # object phrase established earlier in the trace. See
        # referent_anchor.py for the full design rationale.
        compressed, referent_report = inject_referent_stubs(
            compressed, messages,
            max_stubs=max_referent_stubs,
            max_stub_tokens=max_referent_stub_tokens,
            lookback_window=referent_lookback)
        report.update(referent_report)

    if diagnostics is not None:
        diagnostics.update(report)

    return compressed, report


__all__ = ["compress_dagc_sv"]