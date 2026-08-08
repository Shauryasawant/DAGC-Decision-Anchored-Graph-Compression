"""
DAGC v4.3 — Decision-Anchored Graph Compression.

This is the core of the package: compress(messages) -> shorter messages,
with decision-critical artifacts (tool-call args, IDs, confirmed values,
metrics) hard-guaranteed to survive compression. No LLM call anywhere in
this module -- purely regex, embeddings (via the BYOK runtime), and graph
algorithms.
"""
from __future__ import annotations
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from dagc.extraction import _JUDGMENT_VERBS
from .extraction import _find_decisive_match, _sentence_containing
import numpy as np
from .utils import (
    _art_density, _artifacts, _cos, _encode, _get_text, _head_tail_cap,
    _split_sents, _tok, _value_still_recoverable,
)
from .value_recovery_ext import inject_value_recovery_stubs
from .extraction import (
    _CONFIRM_SIGNALS, _STRONG_JUDGMENT_SIGNALS, _get_tool_call_args,
    _get_tool_call_name, _is_meaningful_candidate_value, _is_numeric_literal,
    _key_tier_rank, _stringify_arg_value, _verb_match_is_decisive, extract_decisions,
    _mask_code_fences,   # NEW
)
from .graph import (
    CausalGraphConfig, CausalMessageGraph, SpectralCompressor,
    attach_dependencies as _attach_dependencies,
    build_dependency_graph,
)

@lru_cache(maxsize=8192)
def _word_boundary_re(art_lower: str):
    return re.compile(r'(?<![A-Za-z0-9_])' + re.escape(art_lower) + r'(?![A-Za-z0-9_])')


def compress_any(raw_messages, target_reduction: Optional[float] = None,
                 cfg: Optional[DAGCConfig] = None,
                 decision_roles: Tuple[str, ...] = ('user', 'assistant'),
                 force_preserve: Optional[Iterable[str]] = None,
                 rehydrate: bool = True,
                 return_canonical: bool = False,
                 **overrides):
    """Format-tolerant front door for arbitrary message traces.

    If return_canonical=True, returns (result, canonical_compressed) instead
    of just result -- canonical_compressed still carries `_orig_idx` on each
    message, needed by anything that cross-references by index (e.g.
    compute_chain_rci). `result` follows `rehydrate` as before.
    """
    from .formats import normalize_trace as _normalize_trace
    from .formats import denormalize_trace as _denormalize_trace

    if isinstance(raw_messages, dict):
        original_list = None
        for key in ('messages', 'trace', 'conversation', 'turns'):
            if isinstance(raw_messages.get(key), list):
                original_list = raw_messages[key]
                break
        original_list = original_list if original_list is not None else []
    elif isinstance(raw_messages, list):
        original_list = raw_messages
    else:
        original_list = []

    canonical = _normalize_trace(raw_messages)
    compressed = compress(canonical, target_reduction=target_reduction, cfg=cfg,
                           decision_roles=decision_roles,
                           force_preserve=force_preserve, **overrides)

    result = compressed if not rehydrate else _denormalize_trace(original_list, compressed)

    if return_canonical:
        return result, compressed
    return result


@dataclass
class DAGCConfig:
    TARGET_REDUCTION: float = 0.87
    ABSOLUTE_BUDGET_TOKENS: Optional[int] = None
    EVIDENCE_MIN_BUDGET_PCT: float = 0.09
    PHASE1_FRAC: float = 0.42
    PER_DECISION_MIN_TOKENS: int = 22

    SYSTEM_MAX_TOKENS: int = 22
    DECISION_MAX_TOKENS: int = 50
    TOOL_CALL_MAX_TOKENS: int = 22
    KEEP_LAST_K: int = 1

    COMPRESS_PROTECTED: bool = True
    JUDGMENT_HEAD_FRAC: float = 0.20
    STRICT_PHASE1_BUDGET: bool = False
    PROTECT_TOOL_CALLS: bool = True
    PROTECT_JUDGMENTS: bool = True

    ART_CORROBORATION_MIN: int = 2

    W_ACTION: float = 2.0
    W_JUDGMENT: float = 1.5
    W_CONFIRMATION: float = 1.0
    DECISION_TARGET_WEIGHT: float = 2.5
    TAU: float = 5.0
    IDF_SMOOTH: float = 1.0

    GAMMA: float = 0.72
    MMR_LAMBDA: float = 0.65
    MIN_EFF_THRESHOLD: float = 0.028  # was 1e-3. Phase-2's greedy MMR loop had
    # no meaningful stop condition below this on evidence-dense-in-practice
    # SWE-agent traces (EVIDENCE_DENSITY_TOPK_CAP/EVIDENCE_DECAY_RATIO never
    # engaged -- evidence_dense was False on every trace tested, so those two
    # knobs were inert) -- confirmed via measure_phase2_padding.py
    # (--force-disable-phase2) + sweep_min_eff_threshold.py against
    # verify_no_decision_loss on n=15 Orchard(swe) resolved traces: phase-2
    # was adding ~9% padding tokens (11,377/126,152) with ZERO effect on
    # decision recoverability (0 missing at every threshold tested, p10
    # through p99 of phase-2's own eff-score distribution). 0.028 sits at the
    # p85-p90 mark: captures ~8.5% of that headroom while leaving margin
    # below the tested ceiling (p99 threshold was 0.081, still 0 missing).
    # NOT yet validated on a larger/different trace sample or non-SWE-agent
    # domains -- re-run the sweep before trusting this beyond the tested
    # distribution if traces look structurally different (e.g. much shorter,
    # or dense in the EVIDENCE_DENSITY_TOOL_RATIO sense).

    USE_CAUSAL_SKELETON: bool = True
    USE_SPECTRAL: bool = False
    SPECTRAL_WEIGHT: float = 0.25
    MSTAR_CAUSAL_BONUS: float = 4.0

    MSTAR_HARD_DROP: bool = False  # FIX: was True. Hard-dropping non-M_star
    # messages silently deleted semantically relevant content with zero
    # decision-retention benefit -- confirmed via ablation_hard_guarantee.py
    # full-corpus run (n=951): decision_art_ret unchanged/improved to match
    # hard-guarantee-only exactly (0.9892), SP gap closed from -0.0271 to
    # -0.0109 on the residual differentiating subset.

    EVIDENCE_BEFORE_FRAC: float = 0.70
    MIN_TOOL_CORROBORATION: int = 2

    MIN_SENT_TOKENS: int = 4
    MAX_EMBED_CHUNK: int = 200

    USE_DECISION_LOSS_OBJECTIVE: bool = False
    LOSS_LAMBDA: float = 0.5   # Lagrange multiplier: weight on decision-loss
                                # reduction relative to the existing causal/
                                # semantic efficiency score, both expressed
                                # per token spent.

    # Evidence-density cap: on evidence-dense traces (many tool messages
    # relative to how many decisions they support -- e.g. a debugging
    # trace with dozens of individually-relevant log lines and one root
    # cause), phase-2's MMR selection has no natural stopping point short
    # of the token budget, because low pairwise redundancy between log
    # lines keeps every candidate above MIN_EFF_THRESHOLD. This caps the
    # NUMBER of phase-2 items on flagged traces only. None = off (default,
    # zero behavior change). Does not touch phase 1 or the unconditional
    # target_arts_all rescue pass -- both stay exactly as strict as today.
    EVIDENCE_DENSITY_TOPK_CAP: Optional[int] = None
    EVIDENCE_DENSITY_TOOL_RATIO: float = 3.0  # tool_msgs / n_valid_decisions
    EVIDENCE_DENSITY_MIN_TOOL_MSGS: int = 5   # floor so a 1-tool-msg trace
                                                # with 0 decisions can't trip it

    # Phase-2 stop condition scoped to phase 2's OWN objective (marginal
    # causal/semantic efficiency), not to decision coverage. On an
    # evidence-dense trace, near-duplicate low-redundancy candidates (e.g.
    # distinct log lines) can all clear MIN_EFF_THRESHOLD indefinitely, so
    # phase 2 has no natural stop short of the token budget. This stops
    # once the best available candidate's efficiency has decayed to a
    # small fraction of the FIRST phase-2 pick's efficiency -- a signal
    # that remaining candidates are low-marginal-value padding, regardless
    # of whether any decision still needs coverage (that's job A, already
    # owned by phase1 + the unconditional rescue + final verify pass).
    # None = off (default, zero behavior change).
    EVIDENCE_DECAY_RATIO: Optional[float] = None
    EVIDENCE_DECAY_DEBUG: bool = False  # when True AND diagnostics is passed,
                                      # records each phase-2 iteration's
                                      # (best_eff, break_reason) to
                                      # diagnostics['phase2_trace']. Zero
                                      # behavior change when False (default) —
                                      # pure instrumentation, no selection
                                      # logic is touched.
    USE_FILLER_FILTER: bool = True         
    FILLER_PROB_THRESHOLD: float = 0.75
    FILLER_SURPRISAL_BITS: float = 4.0

    # Runtime self-check: after the unconditional final verify-and-repair
    # pass, raise DecisionLossError instead of silently shipping a partial
    # tag if any decision-critical value from target_arts_all is STILL
    # unrecoverable. Off by default (adds a scan over `out`), meant to be
    # forced on in tests/CI and adversarial/fuzz suites.
    ASSERT_NO_DECISION_LOSS: bool = False

DAGC_CFG = DAGCConfig()


class DecisionLossError(RuntimeError):
    """Raised by compress_dagc when cfg.ASSERT_NO_DECISION_LOSS is set and a
    decision-critical value from target_arts_all is still unrecoverable
    after the unconditional final verify-and-repair pass. This should never
    fire in normal operation -- it exists so CI/adversarial test suites can
    turn a silent guarantee-violation into a hard failure."""
    pass


def _is_decision_bearing(m, cfg):
    if cfg.PROTECT_TOOL_CALLS and (isinstance(m.get('tool_call'), dict)
                                    or _has_inline_tool_call(_get_text(m))):
        return True
    if cfg.PROTECT_JUDGMENTS and m.get('role') == 'assistant':
        t = _get_text(m)
        if _STRONG_JUDGMENT_SIGNALS.search(t) or _CONFIRM_SIGNALS.search(t):
            return True
    return False

from .extraction import (
    _CONFIRM_SIGNALS, _STRONG_JUDGMENT_SIGNALS, _get_tool_call_args,
    _get_tool_call_name, _is_meaningful_candidate_value, _is_numeric_literal,
    _key_tier_rank, _stringify_arg_value, _verb_match_is_decisive, extract_decisions,
    _extract_inline_tool_call,
)

def _has_inline_tool_call(text: str) -> bool:
    return _extract_inline_tool_call(text) is not None

def _is_tool_call_msg(m):
    if isinstance(m.get('tool_call'), dict):
        return True
    return _has_inline_tool_call(_get_text(m))

def _is_tool_call_msg(m):
    return isinstance(m.get('tool_call'), dict)


@lru_cache(maxsize=8192)
def _multiword_art_re(art_lower: str):
    """
    Match a multi-word artifact against source text while tolerating
    punctuation/whitespace differences between its words. Target strings
    built by the tokenizer-based extraction fallbacks (object-phrase,
    bigram) join words with a single space, discarding whatever
    punctuation originally sat between them (e.g. 'info() method' becomes
    'info method'). A literal substring check then can never find that
    target back in its own source sentence -- this checks word-by-word
    instead, with each word boundary-matched and the gap between words
    accepting any run of non-alphanumeric characters, so it matches
    regardless of what that punctuation was.
    """
    words = art_lower.split()
    parts = [r'(?<![A-Za-z0-9_])' + re.escape(w) + r'(?![A-Za-z0-9_])' for w in words]
    return re.compile(r'[^A-Za-z0-9]*'.join(parts))

import re

_DEADLINE_PATTERN = re.compile(
    r'\b(?:by|due|deadline(?:\s+is)?)\s+'
    r'((?:Mon|Tues?|Wed(?:nes)?|Thu(?:rs)?|Fri|Sat(?:ur)?|Sun)\w*\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)',
    re.IGNORECASE,
)

_ASSIGNMENT_PATTERN = re.compile(
    r'\b([A-Z][a-z]+)\s+will\s+((?:handle|verify|manage|own|lead|coordinate|monitor)\s+[\w\s]{3,40})',
)

_MODAL_COMMITMENT_RE = re.compile(
    r'\b(will|must|shall|need(?:s)? to|have to|going to|is scheduled to|'
    r'requires?|expect(?:ed)? to|plan(?:s|ning)? to)\b',
    re.IGNORECASE,
)

_CONDITIONAL_RE = re.compile(
    r'\b(if|unless|once|after|as soon as|in case|assuming)\b.{3,100}?'
    r'\b(will|shall|we\'ll|then|restore|notify|stop|revert|escalate|'
    r'roll ?back|close|begin|start)\b',
    re.IGNORECASE,
)
_DECISION_PROTOTYPES = [
    "The team will complete this task by a specific deadline.",
    "A meeting is scheduled to review progress.",
    "If a problem occurs, we will take a specific corrective action.",
    "We will confirm that a step was completed successfully before proceeding.",
    "We will notify someone once a task is finished.",
    "We will monitor the system for a period of time after the change.",
    "A specific person is responsible for a specific task.",
]
_SEMANTIC_SIM_THRESHOLD = 0.55  # conservative -- prefer missing a few over
                                  # flooding target_arts with false positives

_dec_proto_embs = None  # lazy-cached, computed once per process


def _get_decision_prototype_embeddings():
    global _dec_proto_embs
    if _dec_proto_embs is None:
        try:
            _dec_proto_embs = _encode(_DECISION_PROTOTYPES)
        except Exception:
            _dec_proto_embs = []
    return _dec_proto_embs


def _sentence_already_covered(sent: str, msg_idx: int, existing_decisions: List[Dict]) -> bool:
    """True if an existing decision's rationale/verbatim already overlaps
    this sentence for the same message -- avoids duplicate decision
    objects for content the primary extractor already caught."""
    sent_low = sent.lower().strip()
    for d in existing_decisions:
        if d.get('msg_idx') != msg_idx:
            continue
        for rat in d.get('rationale', []) or []:
            if isinstance(rat, str) and sent_low in rat.lower():
                return True
        verbatim = d.get('verbatim') or ''
        if sent_low and sent_low in verbatim.lower():
            return True
    return False


def _extract_supplementary_decisions(messages: List[Dict],
                                      decision_roles: Tuple[str, ...] = ('user', 'assistant'),
                                      existing_decisions: Optional[List[Dict]] = None
                                      ) -> List[Dict]:
    existing_decisions = existing_decisions or []
    extra: List[Dict] = []
    seen_sents: Set[Tuple[int, str]] = set()

    proto_embs = _get_decision_prototype_embeddings()

    for i, m in enumerate(messages):
        if m.get('role') not in decision_roles:
            continue
        text = m.get('content', '') or ''
        try:
            sents = _split_sents(text, 4)
        except Exception:
            continue

        for sent in sents:
            key = (i, sent.strip().lower())
            if key in seen_sents:
                continue
            if _sentence_already_covered(sent, i, existing_decisions):
                continue

            is_modal = bool(_MODAL_COMMITMENT_RE.search(sent))
            is_conditional = bool(_CONDITIONAL_RE.search(sent))
            is_semantic = False

            if not (is_modal or is_conditional) and proto_embs is not None and len(proto_embs) > 0:
                try:
                    sent_emb = _encode([sent])[0]
                    best_sim = max((_cos(sent_emb, pe) for pe in proto_embs), default=0.0)
                    is_semantic = best_sim >= _SEMANTIC_SIM_THRESHOLD
                except Exception:
                    is_semantic = False

            if not (is_modal or is_conditional or is_semantic):
                continue

            action = ('commitment' if is_modal else
                      'conditional' if is_conditional else 'related')
            extra.append({
                'msg_idx': i,
                'type': 'action',
                'action': action,
                'target': sent.strip()[:80],
                'rationale': [sent],
                'artifacts': {'paths': [], 'ids': [], 'errors': []},
            })
            seen_sents.add(key)

    return extra

def _art_in_text(art: str, text: str) -> bool:
    """
    Case-insensitive 'does this critical value appear in this text' check.
    ... [existing docstring unchanged] ...
    """
    if not art:
        return False
    art_low = art.lower()
    text_low = text.lower()
    if re.fullmatch(r'[a-z0-9_]+', art_low):
        return bool(_word_boundary_re(art_low).search(text_low))
    if ' ' in art_low and re.fullmatch(r'[a-z0-9_ ]+', art_low):
        # Multi-word, punctuation-free target string (typical output of
        # the tokenizer-based extraction fallbacks) -- match word-by-word,
        # tolerant of whatever punctuation the source text had between
        # the words that extraction's tokenizer discarded.
        return bool(_multiword_art_re(art_low).search(text_low))
    return art_low in text_low

_NONCRITICAL_ACTION_VERBS = {
    "recommend",
    "recommends",
    "recommended",
    "recommendation",

    "suggest",
    "suggests",
    "suggested",
    "suggestion",

    "confirm",
    "confirms",
    "confirmed",
    "confirmation",

    "decide",
    "decides",
    "decided",
    "decision",

    "think",
    "thinks",
    "thought",

    "believe",
    "believes",
    "believed",

    "conclude",
    "concludes",
    "concluded",
}

# Harvested directly from each decision's own verbatim text -- a dollar
# amount inside a decision-bearing message is inherently decision-critical
# regardless of whether it happened to win target-selection (only one
# target ever wins, see _extract_target) or get matched by
# _RE_METRIC_KV_INLINE's rationale regex, which requires a letter-first
# label ("JG7FMM: $6,594" matches; "2FBBAH: $3,925" doesn't, since
# "2FBBAH" starts with a digit) and requires explicit '=' / ':' punctuation
# at all (natural prose like "$250 from certificate, $55 from credit
# card" has neither). Both gaps silently dropped real refund/payment
# amounts from critical_values even though they were sitting in
# `verbatim` the whole time.
from .value_recovery_ext import _RE_CURRENCY_VALUE  # multi-currency, replaces dollar-only regex


def _decision_critical_values(decisions: List[Dict],
                               force_action_msg_idxs: Optional[Set[int]] = None) -> Set[str]:
    """
    force_action_msg_idxs: msg_idx values for decisions that receive NO
    other structural protection (i.e. demoted/injection_filtered
    decisions). For those decisions, the action verb is hard-guaranteed
    even when it's in _NONCRITICAL_ACTION_VERBS.

    WHY: the _NONCRITICAL_ACTION_VERBS exclusion (recommend/suggest/
    confirm/decide/think/believe/conclude) is safe ONLY for decisions
    that stay in `protected` -- their hosting message naturally keeps the
    verb via _STRONG_JUDGMENT_SIGNALS/_CONFIRM_SIGNALS-driven selection
    in _select_priority_content. A demoted decision has no such fallback
    (its message gets causal_n=0 and no protected-budget compression),
    so excluding its verb here means it is NEVER protected, NEVER
    rescued, and NEVER tagged -- it just silently disappears whenever
    the host message gets compressed away. Forcing inclusion for exactly
    the decisions that lack other protection closes that gap without
    loosening the exclusion for decisions that don't need it.
    """
    out: Set[str] = set()
    force_action_msg_idxs = force_action_msg_idxs or set()

    for d in decisions:
        act = d.get('action')
        if act and 2 <= len(str(act)) <= 30:
            is_noncritical = str(act).lower() in _NONCRITICAL_ACTION_VERBS
            forced = d.get('msg_idx') in force_action_msg_idxs
            if not is_noncritical or forced:
                out.add(str(act))

        t = d.get('target')
        if t:
            for piece in (t if isinstance(t, list) else [t]):
                s = str(piece).strip().strip('"\'[]')
                length_ok = 2 <= len(s) <= 60 or _is_numeric_literal(s)
                if length_ok and _is_meaningful_candidate_value(s):
                    out.add(s)

        for rat in d.get('rationale', []):
            if not isinstance(rat, str):
                continue
            m = re.search(r'[=:]\s*(.+)$', rat)
            val = (m.group(1) if m else rat).strip()
            length_ok = 2 <= len(val) <= 60 or _is_numeric_literal(val)
            if length_ok and _is_meaningful_candidate_value(val):
                out.add(val)

        verbatim = d.get('verbatim') or ''
        for dm in _RE_CURRENCY_VALUE.finditer(verbatim):
            out.add(dm.group(0))

    return out


def _collect_decision_artifacts(decisions, force_action_msg_idxs: Optional[Set[int]] = None):
    arts = set()
    for d in decisions:
        for kind in ('paths', 'ids', 'errors'):
            arts.update(d['artifacts'].get(kind, []))
    arts |= _decision_critical_values(decisions, force_action_msg_idxs=force_action_msg_idxs)
    return arts


def _collect_decision_artifacts_by_decision(decisions,
                                             force_action_msg_idxs: Optional[Set[int]] = None
                                             ) -> Dict[int, Set[str]]:
    """Return the per-decision artifact set that underpins preservation tags."""
    by_decision: Dict[int, Set[str]] = {}
    for d in decisions:
        by_decision[d['msg_idx']] = _collect_decision_artifacts(
            [d], force_action_msg_idxs=force_action_msg_idxs)
    return by_decision


def _artifact_owners(art: str, by_decision: Dict[int, Set[str]]) -> Set[int]:
    return {k for k, arts in by_decision.items() if art in arts}


def _build_preserved_tag(missing, by_decision=None, *channels, max_tokens=None, must_keep=None):
    if by_decision and missing and not any(_artifact_owners(a, by_decision) for a in missing):
        import warnings
        warnings.warn(
            "_build_preserved_tag: zero owners resolved for any missing "
            "artifact -- by_decision/missing key-space mismatch likely.",
            RuntimeWarning,
        )

    def _is_must(a):
        return bool(_JUDGMENT_VERBS.fullmatch(a.strip())) or a in must_keep

    def _priority(a):
        return (0 if _is_must(a) else 1, len(a), a)

    def _build_entry(a):
        # by_decision is {msg_idx: {owned artifacts}} -- need the REVERSE
        # lookup. by_decision.get(a) never matches (a is a string, keys
        # are ints); _artifact_owners already exists for exactly this.
        owners = _artifact_owners(a, by_decision) if by_decision else None
        if owners:
            owner_str = ','.join(str(k) for k in sorted(owners))
            entry = f'{a}#d{owner_str}'
        else:
            entry = a
        return entry, _tok(entry) + 1

    ordered = sorted(missing, key=_priority)
    must_items = [a for a in ordered if _is_must(a)]
    other_items = [a for a in ordered if not _is_must(a)]

    parts, used = [], _tok('[preserved: ]')
    all_must_fit = True

    for a in must_items:
        if _already_covered(a, *channels):
            continue
        entry, entry_toks = _build_entry(a)
        if max_tokens is not None and used + entry_toks > max_tokens:
            all_must_fit = False
            continue
        parts.append(entry)
        used += entry_toks

    if all_must_fit:
        for a in other_items:
            if _already_covered(a, *channels):
                continue
            entry, entry_toks = _build_entry(a)
            if max_tokens is not None and used + entry_toks > max_tokens:
                continue
            parts.append(entry)
            used += entry_toks

    return '[preserved: ' + ', '.join(parts) + ']' if parts else ''


def _extract_decision_targets(decisions, cfg: DAGCConfig):
    type_weight = {'action': 2.0, 'judgment': 1.5, 'confirmation': 1.0}
    result: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for d in decisions:
        raw_target = d.get('target')
        if not raw_target:
            continue
        targets = []
        raw_str = str(raw_target).strip()
        try:
            parsed = json.loads(raw_str)
            if isinstance(parsed, list):
                targets = [str(t).strip().strip('"\'') for t in parsed]
        except Exception:
            pass
        if not targets:
            targets = [raw_str.strip('"\'[]').strip()]
        w = type_weight.get(d['type'], 1.0) * cfg.DECISION_TARGET_WEIGHT
        for t in targets:
            if t and 2 <= len(t) <= 60:
                result[t.lower()].append((d['msg_idx'], w))
    return dict(result)


def _corroborated_artifacts(messages: List[Dict], decisions: List[Dict],
                             min_corroboration: int = 2) -> Set[str]:
    art_freq: Dict[str, int] = defaultdict(int)
    for m in messages:
        text = _get_text(m)
        seen: Set[str] = set()
        for kind in ('paths', 'ids', 'errors'):
            for a in _artifacts(text)[kind]:
                if a not in seen:
                    art_freq[a] += 1
                    seen.add(a)

    dec_arts = _collect_decision_artifacts(decisions)
    for a in dec_arts:
        if a not in art_freq:
            art_freq[a] = sum(1 for m in messages if _art_in_text(a, _get_text(m)))

    return {a for a, f in art_freq.items() if f >= min_corroboration or a in dec_arts}


# --- Rate-distortion decision-loss objective ---
# Formalizes decision-preserving compression the way classic rate-distortion
# theory (Shannon) formalizes lossy compression generally: minimize a
# distortion measure subject to a rate (token) budget. The distortion here
# is DECISION-LOSS specifically -- the fraction of a decision's critical
# values that become unrecoverable -- rather than the generic semantic-
# similarity distortion (ROUGE/embedding distance) ordinary summarization
# optimizes for.
#
#     minimize  L(S) = sum_d w_d * loss_d(S)     s.t.  tokens(S) <= budget
#     loss_d(S) = |T_d \ covered(S)| / |T_d|
#
# The Lagrangian relaxation, minimize tokens(S) + lambda*L(S), translated
# into a greedy per-candidate marginal-value rule (the same rule the
# existing GAMMA-weighted efficiency score already uses) becomes: add
# lambda * (loss this candidate would remove) / (tokens it costs) directly
# onto the existing efficiency score. It's an additive Lagrangian term on
# the same per-candidate loop, not a replacement selection mechanism.

def _decision_loss(by_decision: Dict[int, Set[str]], covered_arts: Set[str],
                    type_weight: Optional[Dict[str, float]] = None,
                    decisions: Optional[List[Dict]] = None) -> float:
    """L(S): weighted mean, across decisions, of the fraction of that
    decision's critical values NOT present in `covered_arts`. Weighted by
    decision type when supplied (an 'action' losing its target is worse
    than a 'confirmation' losing one -- mirrors cfg.W_ACTION/W_JUDGMENT/
    W_CONFIRMATION already used for the same reason elsewhere)."""
    if not by_decision:
        return 0.0
    type_by_idx = {d['msg_idx']: d.get('type') for d in decisions} if decisions else {}

    total_w, total_loss = 0.0, 0.0
    for msg_idx, arts in by_decision.items():
        if not arts:
            continue
        missing = arts - covered_arts
        loss_d = len(missing) / len(arts)
        w = type_weight.get(type_by_idx.get(msg_idx), 1.0) if type_weight else 1.0
        total_loss += w * loss_d
        total_w += w
    return total_loss / total_w if total_w > 0 else 0.0


def _marginal_loss_reduction(candidate_text: str, by_decision: Dict[int, Set[str]],
                              covered_arts: Set[str],
                              type_weight: Optional[Dict[str, float]] = None,
                              decisions: Optional[List[Dict]] = None) -> float:
    """L(S) - L(S U {candidate}): the drop in decision-loss ONE candidate
    sentence would buy if selected next, holding everything else fixed.
    Always >= 0. Uses the same _art_in_text boundary-aware check used
    everywhere else in this module, so it agrees with how coverage is
    checked for the hard-guarantee/still_miss logic elsewhere."""
    before = _decision_loss(by_decision, covered_arts, type_weight, decisions)
    newly_covered = {a for arts in by_decision.values() for a in arts
                     if a not in covered_arts and _art_in_text(a, candidate_text)}
    if not newly_covered:
        return 0.0
    after = _decision_loss(by_decision, covered_arts | newly_covered, type_weight, decisions)
    return max(0.0, before - after)


_RE_METRIC_KV = re.compile(r'\b([A-Za-z][\w-]*)\s*[=:]\s*(\d+(?:\.\d+)?(?:%|e[-+]?\d+)?)\b')


def _extract_metric_strings(text: str) -> List[str]:
    return [f'{m.group(1).lower()}={m.group(2)}' for m in _RE_METRIC_KV.finditer(text)]


def _decision_metric_strings(decisions: List[Dict]) -> Set[str]:
    out: Set[str] = set()
    for d in decisions:
        for rat in d.get('rationale', []):
            if not isinstance(rat, str):
                continue
            norm = re.sub(r'\s*:\s*', '=', rat.strip().lower())
            if re.search(r'\d', norm):
                out.add(norm)
    return out

def _select_priority_content(content: str, target_arts: Set[str], dec_metrics: Set[str],
                              budget: int, cfg: DAGCConfig, head_frac: float = 0.5,
                              by_decision: Optional[Dict[int, Set[str]]] = None) -> str:
    if _tok(content) <= budget:
        return content

    sents = _split_sents(content, cfg.MIN_SENT_TOKENS)
    if not sents:
        return _head_tail_cap(content, budget, head_frac)

    # NEW: mask structural/code content ONCE over the full message before
    # fragmentation. A <tool_call>{...}</tool_call> payload or a bare JSON/
    # python-repr blob needs its surrounding braces/tags intact to be
    # recognized -- by the time content is split into per-sentence
    # candidates below, that context can live in a SIBLING sentence that
    # got dropped or reordered relative to this one. Masking full `content`
    # first means the masking decision is made with the whole structure
    # visible, then sliced per-sentence -- so a sentence that's entirely
    # inside a masked span reads as blank (no JUDGMENT_VERBS hit) even
    # though, taken alone, it wouldn't look code-like anymore.
    # _mask_code_fences is length-preserving (chars -> spaces, newlines
    # kept), so byte offsets found in `content` line up with `masked_content`.
    masked_content = _mask_code_fences(content)
    cursor = 0
    masked_sents = []
    for s in sents:
        idx = content.find(s, cursor)
        if idx == -1:
            idx = content.find(s)  # fallback: duplicate/reordered sentence
        if idx == -1:
            masked_sents.append(s)  # couldn't locate -- fail open to original
            continue
        masked_sents.append(masked_content[idx:idx + len(s)])
        cursor = idx + len(s)

    covered_arts: Set[str] = set()
    covered_metrics: Set[str] = set()

    def _new_coverage(s):
        new_arts = {a for a in target_arts if a and _art_in_text(a, s) and a not in covered_arts}
        s_metrics = set(_extract_metric_strings(s))
        new_metrics = {ms for ms in s_metrics if ms in dec_metrics and ms not in covered_metrics}
        return new_arts, new_metrics

    selected: List[str] = []
    used = 0

    for s, masked_s in zip(sents, masked_sents):
        new_arts, new_metrics = _new_coverage(s)          # unmasked -- coverage must see real content
        is_signal = bool(_STRONG_JUDGMENT_SIGNALS.search(masked_s) or _CONFIRM_SIGNALS.search(masked_s))
        if not is_signal:
            is_signal = any(_verb_match_is_decisive(masked_s, m) for m in _JUDGMENT_VERBS.finditer(masked_s))
        if not is_signal:
            is_signal = _has_inline_tool_call(s)
        if not (new_arts or new_metrics or is_signal):
            continue
        t = _tok(s)
        if used + t <= budget * 1.25:
            selected.append(s)
            used += t
            covered_arts |= new_arts
            covered_metrics |= new_metrics
        elif new_arts or is_signal:
            selected.append(s)
            used += t
            covered_arts |= new_arts
            covered_metrics |= new_metrics

    head_n = max(1, int(len(sents) * head_frac))
    ordered = sents[:head_n] + list(reversed(sents[head_n:]))
    for s in ordered:
        if s in selected:
            continue
        t = _tok(s)
        if used + t <= budget:
            selected.append(s)
            used += t

    result = '\n'.join(selected).strip() or _head_tail_cap(content, budget, head_frac)
    missing = [a for a in target_arts if a and _art_in_text(a, content) and not _art_in_text(a, result)]
    if missing:
        must_missing = [a for a in missing if a in target_arts]
        must_cost = sum(_tok(a) + 2 for a in must_missing)
        remaining = max(must_cost, int(budget * 1.5) - _tok(result))
        tag = _build_preserved_tag(missing, by_decision, max_tokens=remaining, must_keep=target_arts)
        if tag:
            result = (result + ' ' + tag).strip()
    return _monotonic(content, result)



def _slim_tool_args(args: dict, target_arts: Set[str], max_keys: int = 4) -> dict:
    """Choose which tool-call args survive into the compressed trace, using
    the same _key_tier_rank priority as target extraction, so the
    compressed view always shows the same "primary" argument that
    extraction already picked as ground truth."""
    if not isinstance(args, dict):
        return {}
    scored = []
    for k, v in args.items():
        cand = _stringify_arg_value(v)
        is_true_target = cand is not None and cand in target_arts
        rank = _key_tier_rank(k)
        sort_key = -1 if is_true_target else (rank if rank is not None else 99)
        scored.append((sort_key, k, v))
    scored.sort(key=lambda t: t[0])
    slim: Dict[str, Any] = {}
    for _, k, v in scored[:max_keys]:
        if isinstance(v, list) and len(v) > 4:
            v = v[:4]
        slim[k] = v
    return slim
def _already_covered(fact: str, *channels: str) -> bool:
    """True if `fact` is already textually present in any of the given
    channels (loosely normalised). Used to stop the same fact being
    encoded twice across [preserved: ...] tags, the →TOOL: marker, and
    the structural tool_call field."""
    return any(fact and ch and _value_still_recoverable(fact, ch) for ch in channels)


def _footprint_text(msg: Dict, target_arts: Optional[Set[str]] = None) -> str:
    """Cost/display view only -- never use for extraction. If `content`
    already covers a tool_call value (directly, or via a rescue tag),
    that value is not re-serialized from the structural tool_call
    field, so accounting/printing counts each fact once."""
    content = msg.get('content', '') or ''
    tc = msg.get('tool_call')
    if not isinstance(tc, dict):
        return content
    name = _get_tool_call_name(tc, 'tool')
    args = _get_tool_call_args(tc)
    residual = {k: v for k, v in args.items()
                if not _already_covered(_stringify_arg_value(v) or '', content)}
    parts = [content]
    if not _already_covered(name, content):
        parts.append(name)
    if residual:
        parts.append(json.dumps(residual, separators=(',', ':')))
    return ' '.join(p for p in parts if p)


_CALIBRATED_PATH = os.path.join(os.path.dirname(__file__), "filler_scorer_calibrated.json")
_FILLER_SCORER = None

def _get_filler_scorer():
    global _FILLER_SCORER
    if _FILLER_SCORER is None:
        from .filler_score import FillerScorer  # deferred: breaks filler_score -> rationale_ext -> compressor cycle
        if os.path.exists(_CALIBRATED_PATH):
            with open(_CALIBRATED_PATH) as f:
                _FILLER_SCORER = FillerScorer.from_json(f.read())
        else:
            _FILLER_SCORER = FillerScorer()
    return _FILLER_SCORER


def _apply_filler_filter(content: str, full_text: str, target_arts: Set[str],
                          cfg: DAGCConfig) -> str:
    """
    Final micro-pruning pass on ALREADY-SELECTED text: strips courtesy/
    filler clauses that survived only because they filled budget, not
    because they were decision-critical. Runs AFTER decision-selection,
    never instead of it.

    Hard safety gate on top of filler_score's own two gates: a clause
    is never deleted if it contains any decision-critical value in
    target_arts, even if filler_score's surprisal check missed it (e.g.
    a target value that isn't shaped like one of _artifacts()'s tracked
    kinds -- ids/paths/errors/numbers -- so surprisal never sees it).
    """
    if not cfg.USE_FILLER_FILTER or not content:
        return content
    from .filler_score import filler_deletion_candidates, apply_deletions
    scorer = _get_filler_scorer()
    candidates = filler_deletion_candidates(
        content, full_text, scorer,
        prob_threshold=cfg.FILLER_PROB_THRESHOLD,
        surprisal_threshold_bits=cfg.FILLER_SURPRISAL_BITS,
    )
    safe = [c for c, p, s in candidates
            if not any(_art_in_text(a, c) for a in target_arts)]
    if not safe:
        return content
    result = apply_deletions(content, safe)
    return _monotonic(content, result) if result else content

def _monotonic(original: str, candidate: str) -> str:
    """Compression must never cost more tokens than the untouched
    original. If a rescue/tag mechanism pushed the candidate past that,
    fall back to the original rather than emit something larger."""
    return candidate if _tok(candidate) < _tok(original) else original

def _compress_protected_message(m: Dict, target_arts: Set[str], dec_metrics: Set[str],
                                  cfg: DAGCConfig, full_text: str = '',
                                  by_decision: Optional[Dict[int, Set[str]]] = None) -> str:
    role = m.get('role', '')
    content = m.get('content', '') or ''
    is_tc = isinstance(m.get('tool_call'), dict) or _has_inline_tool_call(content)

    if role == 'system':
        capped = _head_tail_cap(content, cfg.SYSTEM_MAX_TOKENS)
        return _apply_filler_filter(capped, full_text, target_arts, cfg)

    if is_tc:
        tc = m.get('tool_call')
        if isinstance(tc, dict):
            tool_name = _get_tool_call_name(tc, 'tool')
            args = _get_tool_call_args(tc)
            slim = _slim_tool_args(args, target_arts)

            tc_str = (f"→TOOL:{tool_name}"
                      f"({json.dumps(slim, separators=(',', ':'))[:140]})")
            tc_toks = _tok(tc_str)
            c_budget = max(4, cfg.TOOL_CALL_MAX_TOKENS - tc_toks - 2)

            prefix = (_select_priority_content(content, target_arts, dec_metrics, c_budget, cfg,
                                                head_frac=1.0, by_decision=by_decision) if content else '')
            prefix = _apply_filler_filter(prefix, full_text, target_arts, cfg)

            if tool_name and _already_covered(tool_name, prefix):
                return _monotonic(content, prefix.strip())

            candidate = (prefix + ' ' + tc_str).strip()
            return _monotonic(content, candidate)
        else:
            # Inline tool call (<invoke name="...">...</invoke>): no
            # structural dict to re-serialize/slim, so give the RAW
            # content the same generous, head-anchored budget a
            # structural call gets, instead of the ordinary judgment
            # path's small JUDGMENT_HEAD_FRAC -- so the invoke tag
            # itself, not just its surrounding prose, has a real chance
            # to survive sentence-priority selection.
            result = _select_priority_content(content, target_arts, dec_metrics,
                                               cfg.TOOL_CALL_MAX_TOKENS, cfg,
                                               head_frac=1.0, by_decision=by_decision)
            result = _apply_filler_filter(result, full_text, target_arts, cfg)
            return _monotonic(content, result)

    result = _select_priority_content(content, target_arts, dec_metrics,
                                       cfg.DECISION_MAX_TOKENS, cfg,
                                       head_frac=cfg.JUDGMENT_HEAD_FRAC,
                                       by_decision=by_decision)
    result = _apply_filler_filter(result, full_text, target_arts, cfg)
    return _monotonic(content, result)


def _scale_protected_cfg(cfg: DAGCConfig, protected_msgs: List[Dict], total_budget: int) -> DAGCConfig:
    """
    SYSTEM_MAX_TOKENS / DECISION_MAX_TOKENS / TOOL_CALL_MAX_TOKENS are fixed
    per-message caps that don't know about TARGET_REDUCTION. If protected
    (decision-bearing) messages alone would already blow the requested
    total_budget, scale those caps down proportionally -- with a small
    per-message floor so a decision's core value always has room to
    survive -- instead of silently ignoring TARGET_REDUCTION.
    """
    if not protected_msgs:
        return cfg
    naive = 0
    for m in protected_msgs:
        if m.get('role') == 'system':
            naive += cfg.SYSTEM_MAX_TOKENS
        elif isinstance(m.get('tool_call'), dict):
            naive += cfg.TOOL_CALL_MAX_TOKENS
        else:
            naive += cfg.DECISION_MAX_TOKENS
    if naive <= total_budget or naive == 0:
        return cfg
    import copy as _copy
    scale = total_budget / naive
    floor = 8
    pcfg = _copy.deepcopy(cfg)
    pcfg.SYSTEM_MAX_TOKENS = max(floor, int(cfg.SYSTEM_MAX_TOKENS * scale))
    pcfg.DECISION_MAX_TOKENS = max(floor, int(cfg.DECISION_MAX_TOKENS * scale))
    pcfg.TOOL_CALL_MAX_TOKENS = max(floor, int(cfg.TOOL_CALL_MAX_TOKENS * scale))
    return pcfg


def _judgment_has_evidence(msg_idx: int, messages: List[Dict], cfg: DAGCConfig = None,
                            decision: Optional[Dict] = None) -> bool:
    """Anti-injection evidence gate: a judgment/confirmation only counts as
    "well-supported" if the tool activity *before it in the trace* backs
    it up. Every signal here is computed strictly from messages[:msg_idx]
    -- nothing after the decision, real or adversarially injected, can
    ever change the verdict. This is a heuristic firewall against a
    message that *claims* a decision with no supporting evidence in the
    trace -- not a security control.

    IMPORTANT: this function no longer controls whether a decision EXISTS
    in the pipeline (see `all_decisions` / `target_arts_all` in
    compress_dagc) -- it only controls trust tier (protection + budget
    priority). A decision failing this check is demoted, never deleted.

    `decision`, when supplied, is this message's own already-extracted
    decision dict (from extract_decisions), used for the verbatim-
    corroboration sub-check below instead of re-deriving a target from
    scratch. Corroboration is only enforced for 'confirmation'-type
    decisions, which restate something that should already be present
    earlier in the trace. It is NOT enforced for 'judgment' decisions,
    because a judgment's target is typically a freshly COMPUTED value (a
    correlation id, an aggregate, a winning metric) that by construction
    will not appear verbatim anywhere earlier. If `decision` is omitted,
    falls back to the original weaker re-derived-target corroboration
    for backward compatibility."""
    threshold = getattr(cfg, 'EVIDENCE_BEFORE_FRAC', 0.70)
    min_corr = getattr(cfg, 'MIN_TOOL_CORROBORATION', 1)

    if msg_idx <= 0 or msg_idx >= len(messages):
        return True

    before = messages[:msg_idx]
    tools_before = sum(1 for m in before if m.get('role') == 'tool')

    # Conversational decisions do not require tool evidence.
    if tools_before == 0:
        return True

    if tools_before < min_corr:
        return False

    # Use only preceding context when measuring tool-evidence density.
    density = tools_before / max(len(before), 1)
    if density < (1 - threshold):
        return False

    if decision is not None:
        if decision.get('type') != 'confirmation':
            return True   # density check already passed above; that's the real signal
        raw_candidates = [decision.get('target')] + [
            re.sub(r'^[a-z_]+[:=]\s*', '', str(r)) for r in decision.get('rationale', [])
        ]
        candidates = [str(c).strip() for c in raw_candidates if c and len(str(c).strip()) >= 3]
        if not candidates:
            return True
        corroborations = sum(
            1 for c in candidates for m in before
            if m.get('role') in ('tool', 'assistant') and c.lower() in _get_text(m).lower()
        )
        return corroborations >= min_corr

    # Backward-compatible path for callers that don't pass a decision dict.
    from .extraction import _extract_entity_target
    text = _get_text(messages[msg_idx])
    target = _extract_entity_target(text)
    if target and len(target.strip()) >= 3:
        target_low = target.lower().strip()
        corroborations = sum(
            1 for m in before
            if m.get('role') in ('tool', 'assistant') and target_low in _get_text(m).lower()
        )
        if corroborations < min_corr:
            return False

    return True

def _compute_causal_centrality(messages, decisions, cfg: DAGCConfig = DAGC_CFG):
    n = len(messages)
    type_weight = {'action': cfg.W_ACTION, 'judgment': cfg.W_JUDGMENT,
                   'confirmation': cfg.W_CONFIRMATION}

    min_corr = getattr(cfg, 'ART_CORROBORATION_MIN', 2)
    corroborated = _corroborated_artifacts(messages, decisions, min_corr)

    decision_arts: Set[str] = set()
    for d in decisions:
        for kind in ('paths', 'ids', 'errors'):
            for a in d['artifacts'].get(kind, []):
                if a in corroborated:
                    decision_arts.add(a)
        for a in _decision_critical_values([d]):
            if a in corroborated:
                decision_arts.add(a)

    art_freq: Dict[str, int] = defaultdict(int)
    for m in messages:
        text = _get_text(m)
        seen: Set[str] = set()
        for kind in ('paths', 'ids', 'errors'):
            for a in _artifacts(text)[kind]:
                if a in decision_arts and a not in seen:
                    art_freq[a] += 1
                    seen.add(a)
        for a in decision_arts:
            if a not in seen and _art_in_text(a, text):
                art_freq[a] += 1
                seen.add(a)

    idf = {a: math.log((n + cfg.IDF_SMOOTH) / (f + cfg.IDF_SMOOTH)) + 1.0
           for a, f in art_freq.items()}

    art_to_decs: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for d in decisions:
        w = type_weight.get(d['type'], 1.0)
        for kind in ('paths', 'ids'):
            for a in d['artifacts'].get(kind, []):
                art_to_decs[a].append((d['msg_idx'], w))
        for a in _decision_critical_values([d]):
            art_to_decs[a].append((d['msg_idx'], w))

    target_to_decs = _extract_decision_targets(decisions, cfg)

    target_freq: Dict[str, int] = defaultdict(int)
    for m in messages:
        text_low = _get_text(m).lower()
        for t in target_to_decs:
            if t in text_low:
                target_freq[t] += 1

    target_idf = {t: math.log((n + cfg.IDF_SMOOTH) / (max(target_freq.get(t, 1), 1) + cfg.IDF_SMOOTH)) + 1.0
                  for t in target_to_decs}

    centrality: Dict[int, float] = defaultdict(float)
    for i, m in enumerate(messages):
        text = _get_text(m)
        text_low = text.lower()

        present_arts: Set[str] = set()
        for kind in ('paths', 'ids', 'errors'):
            present_arts.update(a for a in _artifacts(text)[kind] if a in decision_arts)
        for a in decision_arts:
            if _art_in_text(a, text):
                present_arts.add(a)

        for a in present_arts:
            for dec_idx, dec_w in art_to_decs.get(a, []):
                gap = max(0, dec_idx - i)
                decay = math.exp(-gap / max(cfg.TAU, 1.0))
                centrality[i] += idf.get(a, 1.0) * dec_w * decay

        for t, dec_list in target_to_decs.items():
            if t in text_low:
                t_idf = target_idf.get(t, 1.0)
                for dec_idx, dec_w in dec_list:
                    gap = max(0, dec_idx - i)
                    decay = math.exp(-gap / max(cfg.TAU, 1.0))
                    centrality[i] += t_idf * dec_w * decay

    return dict(centrality)


def _phase1_hard_guarantee(pool, decisions, budget, extra_critical=None, by_decision=None,
                            strict_phase1_budget=False):
    target_arts = _collect_decision_artifacts(decisions) | (extra_critical or set())
    if not target_arts:
        return [], set()

    by_decision = by_decision or {}
    all_dec_idxs = set(by_decision.keys())

    best: Dict[str, Tuple[int, int]] = {}
    for idx, (s, _mi) in enumerate(pool):
        t = _tok(s)
        for a in target_arts:
            if _art_in_text(a, s):
                if a not in best or t < best[a][1]:
                    best[a] = (idx, t)

    n_sents = max(1, len(pool))

    def rarity(a):
        freq = sum(1 for s, _ in pool if _art_in_text(a, s))
        return math.log((n_sents + 1.0) / (freq + 1.0))

    selected: List[int] = []
    covered: Set[str] = set()
    satisfied: Set[int] = set()
    used_toks = 0
    injected: Set[int] = set()

    def _owners_unsatisfied(a):
        owners = _artifact_owners(a, by_decision) if by_decision else set()
        # artifacts with no known owner (e.g. force_preserve extras) fall
        # back to the old value-level covered check
        if not owners:
            return a not in covered
        return bool(owners - satisfied)

    for a in sorted(target_arts, key=rarity, reverse=True):
        if not _owners_unsatisfied(a) or a not in best:
            continue
        idx, cost = best[a]

        if idx in injected:
            covered.add(a)
            owners = _artifact_owners(a, by_decision)
            satisfied |= owners if owners else set()
            continue

        bypass_ok = cost <= 40 and (not strict_phase1_budget or used_toks <= budget)
        if used_toks + cost > budget and not bypass_ok:
            continue

        injected.add(idx)
        selected.append(idx)
        used_toks += cost
        s, _ = pool[idx]
        for other in target_arts:
            if _art_in_text(other, s):
                covered.add(other)
                owners = _artifact_owners(other, by_decision)
                satisfied |= owners if owners else set()

    return selected, covered

def _make_stub(msg_idx: int, dropped_msg: Dict, decision: Optional[Dict] = None) -> Optional[Dict]:
    """Called only for messages that would otherwise vanish entirely from
    the compressed output (excluded from `pool` by MSTAR_HARD_DROP, or
    included in `pool` but none of their sentences got selected).

    Two independent stub sources, tried in order:
    1. `decision` -- if this message was identified as decision-bearing by
       extract_decisions (passed in regardless of trust tier -- demoted
       decisions still get a stub), build the stub from the decision's own
       action/target/rationale. This does NOT depend on the artifact
       regex kinds matching anything, so a plain-text-only decision (no
       id/path/error shape) still leaves a visible trace instead of
       vanishing silently -- this was previously the one case that could
       disappear with zero fingerprint.
    2. Fallback: the original artifact-fingerprint check, for messages
       that carry no decision but do carry a path/id/error worth noting.

    FIX (action-loss-on-stub): previously this stub only embedded ONE of
    action/target (whichever `target or action` picked -- almost always
    target, since judgment/action decisions virtually always carry a
    target). That silently discarded the action verb for every decision
    whose hosting message got compressed down to a stub:
    action_still_recoverable(gt_action, text) in reproduce.py's
    _deterministic_extract had no literal 'cancel'/'recommend'/tool-name
    string to find anywhere in the stub, so pinned_action came back None
    and reproduction either fell back to the generic 'decide' label
    (judgment) or dropped the tool name entirely (action/tool-call). Both
    action AND target are now embedded, each under an explicit 'action='
    / 'target=' label so word-boundary matching in
    _value_still_recoverable finds them directly -- the same mechanism
    that already worked for target alone.
    """
    if decision is not None:
        action = decision.get('action')
        target = decision.get('target')
        if isinstance(target, list):
            target = ', '.join(str(t) for t in target[:3])
        labeled = []
        if action:
            labeled.append(f"action={action}")
        if target:
            labeled.append(f"target={target}")
        core = '; '.join(labeled) if labeled else '(no target)'
        return {
            'role': dropped_msg.get('role', 'assistant'),
            '_orig_idx': msg_idx,
            '_stub': True,
            'content': f"[dropped decision — {decision.get('type', 'decision')}: {core}]",
        }
    arts = _artifacts(_footprint_text(dropped_msg))
    fingerprint = arts['paths'] + arts['ids'] + arts['errors']
    if not fingerprint:
        return None  # nothing decision- or artifact-shaped in it -- genuinely safe to fully drop
    return {
        'role': dropped_msg.get('role', 'assistant'),
        '_orig_idx': msg_idx,
        '_stub': True,
        'content': f"[dropped — contained: {', '.join(fingerprint[:5])}]",
    }

def compress_dagc(messages: List[Dict], cfg: Optional[DAGCConfig] = None,
                   decision_roles: Tuple[str, ...] = ('user', 'assistant'),
                   force_preserve: Optional[Iterable[str]] = None,
                   diagnostics: Optional[Dict[str, Any]] = None) -> List[Dict]:
    if cfg is None:
        cfg = DAGC_CFG

    task = next((m['content'] for m in messages if m.get('role') == 'user'), 'complete the task')
    task_emb = _encode([task])[0]
    orig_toks = sum(_tok(_footprint_text(m)) for m in messages)
    full_trace_text = '\n'.join(_get_text(m) for m in messages) 
    n = len(messages)

    try:
        decisions = extract_decisions(messages, decision_roles=decision_roles)
        decisions = _attach_dependencies(messages, decisions)
    except Exception:
        decisions = []

    try:
        decisions += _extract_supplementary_decisions(
            messages, decision_roles, existing_decisions=decisions)
    except Exception:
        pass 

    decisions += _extract_supplementary_decisions(messages, decision_roles)

    injection_filtered: Set[int] = set()
    for d in decisions:
        if d['type'] in ('judgment', 'confirmation'):
            if not _judgment_has_evidence(d['msg_idx'], messages, cfg, decision=d):

                injection_filtered.add(d['msg_idx'])

    # --- Trust tiering, not deletion ---------------------------------
    # `all_decisions`: every decision extraction found. Nothing here is ever
    # removed -- this is the hard-guarantee floor (job A: guarantee zero loss
    # of anything the extractor detected, independent of the evidence gate).
    # `valid_decisions`: the trusted subset. This still drives protection
    # (KEEP_LAST_K-style full-message preservation) and budget priority --
    # a demoted decision gets a leaner ride, but its critical value is still
    # unconditionally hard-guaranteed via target_arts_all below.
    # --- inside compress_dagc, replace the existing target_arts / target_arts_all
# --- construction block with this. Everything else in the function is
# --- unchanged; only the two _collect_decision_artifacts* calls that build
# --- the "all_decisions" side gain force_action_msg_idxs=injection_filtered.

    all_decisions = decisions
    valid_decisions = [d for d in decisions if d['msg_idx'] not in injection_filtered]
    decisions_by_msg_idx = {d['msg_idx']: d for d in all_decisions}

    # Preserve literal decision values even when a message is not protected.
    target_arts = _collect_decision_artifacts(valid_decisions)
    # NOTE: force_action_msg_idxs=injection_filtered -- demoted decisions get
    # zero structural protection elsewhere (causal_n forced to 0, never added
    # to `protected`), so their action verb must be hard-guaranteed here even
    # when it's one of the generic verbs normally excluded as noncritical.
    # Without this, a demoted judgment/confirmation's verb has no path to
    # survival at all: not protected, not in target_arts_all, never rescued,
    # never tagged.
    target_arts_all = _collect_decision_artifacts(all_decisions, force_action_msg_idxs=injection_filtered)

    for d in valid_decisions:
        t = d.get('target')
        if t and (2 <= len(str(t)) <= 80 or _is_numeric_literal(str(t))):
            target_arts.add(str(t))
    for d in all_decisions:
        t = d.get('target')
        if t and (2 <= len(str(t)) <= 80 or _is_numeric_literal(str(t))):
            target_arts_all.add(str(t))

    if force_preserve:
        fp = {str(x) for x in force_preserve
              if x and (2 <= len(str(x)) <= 80 or _is_numeric_literal(str(x)))}
        target_arts |= fp
        target_arts_all |= fp
    dec_metrics = _decision_metric_strings(valid_decisions)
    by_decision = _collect_decision_artifacts_by_decision(valid_decisions)
    # Superset ownership map (trusted + demoted) -- used by the stub path
    # and the unconditional final verify-and-repair pass so a demoted
    # decision's [preserved: ...] tag can still cite its owning msg_idx.
    # Same force_action_msg_idxs reasoning as target_arts_all above.
    by_decision_all = _collect_decision_artifacts_by_decision(
        all_decisions, force_action_msg_idxs=injection_filtered)

    corroborated = _corroborated_artifacts(messages, valid_decisions, cfg.ART_CORROBORATION_MIN)

    # User-provided IDs and paths are authoritative on first mention.
    user_stated_artifacts: Set[str] = set()
    for m in messages:
        if m.get('role') != 'user':
            continue
        a = _artifacts(_get_text(m))
        user_stated_artifacts.update(a['ids'])
        user_stated_artifacts.update(a['paths'])

    target_arts |= corroborated | user_stated_artifacts
    # target_arts_all is the hard floor: everything trusted must-survive,
    # plus everything demoted must-survive too.
    target_arts_all |= corroborated | user_stated_artifacts | target_arts

    M_star = set(range(n))
    spec_scores = {i: 0.0 for i in range(n)}
    cg = None
    causal_error = None
    try:
        if cfg.USE_CAUSAL_SKELETON:
            cg = CausalMessageGraph(messages, valid_decisions,
                                    CausalGraphConfig(CAUSAL_TAU=cfg.TAU, ADD_SEQ_EDGES=False))
            M_star = cg.minimal_sufficient_set()
        if cfg.USE_SPECTRAL and cg is not None:
            spec_scores = SpectralCompressor(cg).normalized_scores()
    except Exception as exc:
        causal_error = f'{type(exc).__name__}: {exc}'

    protected: Set[int] = set()
    for i, m in enumerate(messages):
        if m.get('role') == 'system' or i >= n - cfg.KEEP_LAST_K:
            protected.add(i)
    for d in valid_decisions:
        mi = d['msg_idx']
        if d['type'] == 'action' and cfg.PROTECT_TOOL_CALLS:
            protected.add(mi)
        elif d['type'] in ('judgment', 'confirmation') and cfg.PROTECT_JUDGMENTS:
            if _judgment_has_evidence(mi, messages, cfg, decision=d):
                protected.add(mi)

    if cfg.ABSOLUTE_BUDGET_TOKENS is not None:
        total_budget = max(1, int(cfg.ABSOLUTE_BUDGET_TOKENS))
    else:
        total_budget = max(1, int(orig_toks * (1 - cfg.TARGET_REDUCTION)))

    pcfg = _scale_protected_cfg(cfg, [messages[i] for i in protected], total_budget)
    compressed_protected_content: Dict[int, str] = {}
    if cfg.COMPRESS_PROTECTED:
        for i in protected:
            compressed_protected_content[i] = _compress_protected_message(
                messages[i], target_arts, dec_metrics, pcfg,
                full_text=full_trace_text, by_decision=by_decision)

    def _prot_tok(i: int) -> int:
        if cfg.COMPRESS_PROTECTED and i in compressed_protected_content:
            m_copy = dict(messages[i], content=compressed_protected_content[i])
            return _tok(_footprint_text(m_copy))
        return _tok(_footprint_text(messages[i]))

    protected_toks = sum(_prot_tok(i) for i in protected)

    centrality = _compute_causal_centrality(messages, valid_decisions, cfg)
    max_c = max(centrality.values(), default=1.0) or 1.0
    causal_n = {i: centrality.get(i, 0.0) / max_c for i in range(n)}
    for i in M_star:
        causal_n[i] = causal_n.get(i, 0.0) * cfg.MSTAR_CAUSAL_BONUS
    for i in injection_filtered:
        causal_n[i] = 0.0
        
    dependency_edges = build_dependency_graph(messages, valid_decisions)
    dependency_vals: Set[str] = {e['artifact'] for e in dependency_edges}
    n_valid_decisions = len(valid_decisions)
    _target_lens = [_tok(a) for a in target_arts] or [8]
    per_decision_floor = max(
        getattr(cfg, 'PER_DECISION_MIN_TOKENS', 22),
        int(np.mean(_target_lens)) + 10,
    )
    evidence_scale_toks = total_budget if cfg.ABSOLUTE_BUDGET_TOKENS is not None else orig_toks
    min_evidence_budget = max(
        int(evidence_scale_toks * cfg.EVIDENCE_MIN_BUDGET_PCT),
        n_valid_decisions * per_decision_floor,
        len(dependency_vals) * 15,
    )
    # Reserve a proportional minimum budget for rare decision evidence.
    evidence_floor = min(min_evidence_budget, max(20, int(total_budget * 0.5)))
    free_budget = max(evidence_floor, total_budget - protected_toks)
    phase1_budget = int(free_budget * cfg.PHASE1_FRAC)

    if diagnostics is not None:
        diagnostics.update({
            'orig_tokens': orig_toks,
            'total_budget': total_budget,
            'evidence_floor': evidence_floor,
            'protected_tokens': protected_toks,
            'free_budget': free_budget,
            'M_star_size': len(M_star),
            'causal_skeleton_used': cfg.USE_CAUSAL_SKELETON and causal_error is None,
            'causal_skeleton_error': causal_error,
            'valid_decisions': valid_decisions,
            'all_decisions': all_decisions,
            'injection_filtered_msg_idxs': sorted(injection_filtered),
})

    pool: List[Tuple[str, int]] = []
    for i, m in enumerate(messages):
        if i in protected:
            continue
        text_i = _get_text(m)
        a_i = _artifacts(text_i)
        # Bypass check for MSTAR_HARD_DROP: must test literal membership
        # against the FULL must-survive set (target_arts_all), not just the
        # ids/paths regex buckets. A message whose only must-survive
        # content is a plain-text decision value (no id/path/error shape
        # at all, e.g. "black dial color") still needs to stay eligible
        # for pool/rescue if M_star didn't already select it -- checking
        # only a_i['ids']+a_i['paths'] silently missed exactly that case,
        # letting the message (and the only carrier of that value) drop
        # out before phase1/rescue ever got a chance to see it.
        is_corroborated_carrier = any(
            x in corroborated or x in user_stated_artifacts
            for x in a_i['ids'] + a_i['paths'] + a_i['errors']
        ) or any(_art_in_text(a, text_i) for a in target_arts_all)
        if cfg.MSTAR_HARD_DROP and i not in M_star and not is_corroborated_carrier:
            continue
        for s in _split_sents(text_i, cfg.MIN_SENT_TOKENS):
            pool.append((s, i))

    if not pool:
        out: List[Dict] = []
        for i, m in enumerate(messages):
            if i in protected:
                mc = dict(m)
                if cfg.COMPRESS_PROTECTED and i in compressed_protected_content:
                    mc['content'] = compressed_protected_content[i]
                mc['_orig_idx'] = i
                out.append(mc)
        if diagnostics is not None:
            diagnostics['output_tokens'] = sum(_tok(_footprint_text(m)) for m in out)
        return out

    pool_texts = [s for s, _ in pool]
    pool_embs = _encode(pool_texts, max_chunk=cfg.MAX_EMBED_CHUNK)

    # AFTER
    # extra_critical also carries the demoted-only values (target_arts_all
    # minus target_arts) so phase1 attempts to inject them too, budget
    # permitting -- they get a real shot at cheap/early inclusion rather
    # than relying solely on the last-resort still_miss rescue below.
    p1_idx, covered_arts = _phase1_hard_guarantee(
        pool, valid_decisions, phase1_budget,
        extra_critical=dependency_vals | (target_arts_all - target_arts),
        by_decision=by_decision,
        strict_phase1_budget=cfg.STRICT_PHASE1_BUDGET)
    p1_set = set(p1_idx)
    p1_toks = sum(_tok(pool[i][0]) for i in p1_idx)

    covered_metrics: Set[str] = set()
    for idx in p1_idx:
        for ms in _extract_metric_strings(pool[idx][0]):
            if ms in dec_metrics:
                covered_metrics.add(ms)

    for metric in sorted(dec_metrics - covered_metrics):
        best_idx, best_cost = None, 1e9
        for idx in range(len(pool)):
            if idx in p1_set:
                continue
            s, _ = pool[idx]
            t = _tok(s)
            if metric in _extract_metric_strings(s) and t < best_cost:
                best_cost, best_idx = t, idx
        if best_idx is not None and p1_toks + best_cost <= phase1_budget:
            p1_idx.append(best_idx)
            p1_set.add(best_idx)
            p1_toks += best_cost
            s, _ = pool[best_idx]
            for ms in _extract_metric_strings(s):
                if ms in dec_metrics:
                    covered_metrics.add(ms)
            # FIX (numbers omission): 'numbers' is a real bucket _artifacts()
            # returns, and a purely numeric decision target (a metric
            # threshold, a count, a percentage) needs to register as covered
            # here just like paths/ids/errors do -- otherwise it can survive
            # in the output text yet still be treated as "still missing"
            # downstream.
            for kind in ('paths', 'ids', 'numbers', 'errors'):
                covered_arts.update(_artifacts(s)[kind])

    remaining = max(0, free_budget - p1_toks)
    diagnostics_extra = {'phase1_budget': phase1_budget, 'p1_toks': p1_toks, 'remaining': remaining}
    if diagnostics is not None:
        diagnostics.update(diagnostics_extra)
    # Evidence-density heuristic -- see EVIDENCE_DENSITY_TOPK_CAP docstring
    # above. Computed unconditionally (cheap: one scan over messages) but
    # only ever CONSULTED below if the cap is explicitly set, so this line
    # alone changes nothing about default output.
    _tool_msg_count = sum(1 for m in messages if m.get('role') == 'tool')
    evidence_dense = (
        _tool_msg_count >= cfg.EVIDENCE_DENSITY_MIN_TOOL_MSGS
        and _tool_msg_count / max(1, n_valid_decisions) > cfg.EVIDENCE_DENSITY_TOOL_RATIO
    )
    if diagnostics is not None:
        diagnostics['evidence_dense'] = evidence_dense
        diagnostics['tool_msg_count'] = _tool_msg_count

    pending = [i for i in range(len(pool)) if i not in p1_set]
    if diagnostics is not None:
        diagnostics['min_pending_cost'] = min((_tok(pool[i][0]) for i in pending), default=None)
    selected_embs = [pool_embs[i] for i in p1_idx]
    p2_idx: List[int] = []
    used_p2 = 0

    if cfg.USE_DECISION_LOSS_OBJECTIVE:
        decision_type_weight = {'action': cfg.W_ACTION, 'judgment': cfg.W_JUDGMENT,
                                 'confirmation': cfg.W_CONFIRMATION}
        loss_covered: Set[str] = {a for arts in by_decision.values() for a in arts
                                    if a in covered_arts}

    first_best_eff: Optional[float] = None
    phase2_trace = [] if (diagnostics is not None and cfg.EVIDENCE_DECAY_DEBUG) else None
    for _ in range(len(pending)):
        if not pending or used_p2 >= remaining:
            if phase2_trace is not None:
                phase2_trace.append({'reason': 'pending_empty_or_budget_exhausted'})
            break
        best_eff, best_i = -1e18, None
        for idx in pending:
            s, mi = pool[idx]
            t = _tok(s)
            if t < cfg.MIN_SENT_TOKENS or used_p2 + t > remaining:
                continue
            c = causal_n.get(mi, 0.0)
            sp = spec_scores.get(mi, 0.0) if cfg.USE_SPECTRAL else 0.0
            causal = ((1 - cfg.SPECTRAL_WEIGHT) * c + cfg.SPECTRAL_WEIGHT * sp
                      if cfg.USE_SPECTRAL else c)
            n_dm = sum(1 for ms in _extract_metric_strings(s) if ms in dec_metrics)
            causal *= (_art_density(s) + 0.25 * min(n_dm, 3) + 0.1)
            rel = max(0.0, _cos(pool_embs[idx], task_emb))
            red = max((_cos(pool_embs[idx], e) for e in selected_embs), default=0.0)
            sem = cfg.MMR_LAMBDA * rel - (1 - cfg.MMR_LAMBDA) * red
            eff = (cfg.GAMMA * causal + (1 - cfg.GAMMA) * max(0.0, sem)) / (t + 1e-9)
            if cfg.USE_DECISION_LOSS_OBJECTIVE:
                lr = _marginal_loss_reduction(s, by_decision, loss_covered,
                                               decision_type_weight, valid_decisions)
                eff += cfg.LOSS_LAMBDA * (lr / (t + 1e-9))
            if eff > best_eff:
                best_eff, best_i = eff, idx
                best_debug = {'causal': causal, 'sem': sem, 'rel': rel, 'red': red, 't': t}

        if best_i is None or best_eff <= cfg.MIN_EFF_THRESHOLD:
            if phase2_trace is not None:
                phase2_trace.append({'reason': 'min_eff_threshold', 'best_eff': best_eff, 'debug': best_debug if best_i is not None else None})
            break
        if (evidence_dense and cfg.EVIDENCE_DENSITY_TOPK_CAP is not None
                and len(p2_idx) >= cfg.EVIDENCE_DENSITY_TOPK_CAP):
            if phase2_trace is not None:
                phase2_trace.append({'reason': 'topk_cap', 'best_eff': best_eff})
            break
        if first_best_eff is None:
            first_best_eff = best_eff
        elif (evidence_dense and cfg.EVIDENCE_DECAY_RATIO is not None
                and best_eff < first_best_eff * cfg.EVIDENCE_DECAY_RATIO):
            if phase2_trace is not None:
                phase2_trace.append({'reason': 'decay_ratio', 'best_eff': best_eff,
                                      'first_best_eff': first_best_eff})
            break

        if phase2_trace is not None:
            phase2_trace.append({'reason': 'selected', 'best_eff': best_eff,
                                  'first_best_eff': first_best_eff})

        s, _mi = pool[best_i]
        if cfg.USE_DECISION_LOSS_OBJECTIVE:
            loss_covered |= {a for arts in by_decision.values() for a in arts
                              if a not in loss_covered and _art_in_text(a, s)}
        p2_idx.append(best_i)
        selected_embs.append(pool_embs[best_i])
        # FIX (numbers omission): see identical note above -- same tuple,
        # same gap, same fix. This is the phase-2 greedy-selection update site.
        for kind in ('paths', 'ids', 'numbers', 'errors'):
            covered_arts.update(_artifacts(s)[kind])
        for ms in _extract_metric_strings(s):
            if ms in dec_metrics:
                covered_metrics.add(ms)
        used_p2 += _tok(s)
        pending.remove(best_i)
    if phase2_trace is not None:
        diagnostics['phase2_trace'] = phase2_trace
    for i in protected:
        cc = compressed_protected_content.get(i, _get_text(messages[i]))
        # FIX (numbers omission): protected (system/judgment/tool-call)
        # messages can themselves carry the surviving copy of a numeric
        # decision target -- this update site had the same missing bucket.
        for kind in ('paths', 'ids', 'numbers', 'errors'):
            covered_arts.update(_artifacts(cc)[kind])
        for a in target_arts:
            if _art_in_text(a, cc):
                covered_arts.add(a)

    all_sel = p1_set | set(p2_idx)
    # Rescue against the FULL must-survive set (target_arts_all), not just
    # the trusted subset -- a demoted decision's value must still get a
    # last-resort scavenge from the pool if phase1/phase2 didn't pick up
    # its carrier sentence for other (efficiency/budget) reasons. No budget
    # check here by design: this is the hard-guarantee backstop.
    still_miss = target_arts_all - covered_arts
    for a in sorted(still_miss):
        best_idx, best_cost = None, 1e9
        for idx in range(len(pool)):
            if idx in all_sel:
                continue
            s, _ = pool[idx]
            t = _tok(s)
            if _art_in_text(a, s) and t < best_cost:
                best_cost, best_idx = t, idx
        if best_idx is not None:
            p2_idx.append(best_idx)
            all_sel.add(best_idx)
            s, _ = pool[best_idx]
            # FIX (numbers omission): last-resort rescue site -- without this,
            # a rescued numeric target gets added to the output text but never
            # marked covered, so it looks identical to a value that's still
            # missing to any code reading covered_arts afterward.
            for kind in ('paths', 'ids', 'numbers', 'errors'):
                covered_arts.update(_artifacts(s)[kind])

    msg_sents: Dict[int, List[str]] = {}
    for idx in sorted(p1_set | set(p2_idx)):
        s, mi = pool[idx]
        msg_sents.setdefault(mi, []).append(s)

    out: List[Dict] = []
    n_stubbed = 0
    for i, m in enumerate(messages):
        if i in protected:
            mc = dict(m)
            if cfg.COMPRESS_PROTECTED and i in compressed_protected_content:
                mc['content'] = compressed_protected_content[i]
                if isinstance(m.get('tool_call'), dict):
                    tc = m['tool_call']
                    args = _get_tool_call_args(tc)
                    slim = _slim_tool_args(args, target_arts)
                    mc['tool_call'] = {'name': _get_tool_call_name(tc, 'tool'), 'args': slim}
            else:
                c = m.get('content', '')
                if m.get('role') == 'system':
                    mc['content'] = _head_tail_cap(c, cfg.SYSTEM_MAX_TOKENS)
                elif _is_tool_call_msg(m):
                    mc['content'] = _head_tail_cap(c, cfg.TOOL_CALL_MAX_TOKENS, head_frac=1.0)
                else:
                    mc['content'] = _head_tail_cap(c, cfg.DECISION_MAX_TOKENS,
                                                    head_frac=cfg.JUDGMENT_HEAD_FRAC)
            mc['_orig_idx'] = i
            out.append(mc)
        elif i in msg_sents:
            mc = dict(m)
            content = '\n'.join(msg_sents[i])
            mc['content'] = _apply_filler_filter(content, full_trace_text, target_arts, cfg)
            mc['_orig_idx'] = i
            out.append(mc)
        elif getattr(cfg, 'FINGERPRINT_STUBS', True):
            # Pass the decision (if any) this message was identified as
            # bearing -- trusted or demoted -- so a decision with no
            # artifact-shaped fingerprint still leaves a stub instead of
            # vanishing without a trace.
            stub = _make_stub(i, m, decisions_by_msg_idx.get(i))
            if stub is not None:
                out.append(stub)
                n_stubbed += 1

    if cfg.USE_DECISION_LOSS_OBJECTIVE:
        # Final consistency pass for the decision-loss OBJECTIVE (distinct
        # from the unconditional guarantee pass below): `covered_arts` only
        # ever gets updated with the ('paths','ids','numbers','errors')
        # REGEX subset during phase 2 (see the loops above), so target_arts \
        # covered_arts can still under-report loss for a value outside those
        # regex kinds that never got tracked at all (a plain-word/phrase
        # decision value, e.g.). `loss_covered` doesn't have that gap -- it's
        # built with the same _art_in_text check the objective itself scores
        # against -- so once the objective is active it remains the
        # authoritative source of "did this decision's value actually make it
        # into the output", not covered_arts. This block re-derives
        # loss_covered against the FINAL shipped text (protected + selected +
        # stubbed) and, if any by_decision-owned value still isn't recoverable
        # after every earlier rescue pass, hard-appends it as a last-resort
        # [preserved: ...] tag rather than letting it silently disappear.
        for m_out in out:
            text = _footprint_text(m_out)
            for arts in by_decision.values():
                for a in arts:
                    if a not in loss_covered and _art_in_text(a, text):
                        loss_covered.add(a)
        truly_missing = {a for arts in by_decision.values() for a in arts} - loss_covered
        if truly_missing:
            tag = _build_preserved_tag(truly_missing, by_decision, must_keep=truly_missing)
            if tag:
                if out:
                    out[-1] = dict(out[-1])
                    out[-1]['content'] = (out[-1].get('content', '') + ' ' + tag).strip()
                else:
                    out.append({'role': 'assistant', '_orig_idx': n, '_stub': True, 'content': tag})
        if diagnostics is not None:
            diagnostics['decision_loss_final_rescue'] = sorted(truly_missing)

    # --- Unconditional final verify-and-repair pass (job A) --------------
    # Independent of USE_DECISION_LOSS_OBJECTIVE: guarantees every value in
    # target_arts_all -- trusted AND demoted decisions alike -- is
    # recoverable from the ACTUAL shipped `out` messages. This is the real
    # backstop: cheap (pure string scan over already-built output), always
    # on, and it is what should be cited as "the invariant" in the paper --
    # not the phase1/phase2 heuristics, which are optimizations for WHERE
    # a value survives, not whether it does.
    final_covered: Set[str] = set()
    for m_out in out:
        text = _footprint_text(m_out)
        for a in target_arts_all:
            if a in final_covered:
                continue
            if _art_in_text(a, text):
                final_covered.add(a)
    truly_missing_all = target_arts_all - final_covered
    if truly_missing_all:
        tag = _build_preserved_tag(truly_missing_all, by_decision_all, must_keep=truly_missing_all)
        if tag:
            if out:
                out[-1] = dict(out[-1])
                out[-1]['content'] = (out[-1].get('content', '') + ' ' + tag).strip()
            else:
                out.append({'role': 'assistant', '_orig_idx': n, '_stub': True, 'content': tag})
            # Re-check after the tag: _build_preserved_tag can legitimately
            # drop entries that don't fit max_tokens (None here, so it
            # shouldn't in practice) -- re-scan defensively before deciding
            # whether the invariant genuinely still fails.
            final_text = _footprint_text(out[-1])
            truly_missing_all = {a for a in truly_missing_all if not _art_in_text(a, final_text)}

    if diagnostics is not None:
        diagnostics['decision_loss_final_rescue_all'] = sorted(truly_missing_all)

    if cfg.ASSERT_NO_DECISION_LOSS and truly_missing_all:
        raise DecisionLossError(
            f"{len(truly_missing_all)} decision-critical value(s) unrecoverable after the "
            f"unconditional final verify-and-repair pass: "
            f"{sorted(truly_missing_all)[:10]}{'...' if len(truly_missing_all) > 10 else ''}"
        )

    if diagnostics is not None:
        diagnostics['output_tokens'] = sum(_tok(_footprint_text(m)) for m in out)
        diagnostics['n_stubbed'] = n_stubbed
    return out


# Public shorthand for compress_dagc.
def compress(messages: List[Dict], target_reduction: float = ...,
             cfg: Optional[DAGCConfig] = None,
             decision_roles: Tuple[str, ...] = ('user', 'assistant'),
             force_preserve: Optional[Iterable[str]] = None,
             enable_rescue: bool = True,
             session_id: str = "default",
             diagnostics: Optional[Dict[str, Any]] = None,
             **overrides) -> List[Dict]:
    """
    Compress an agent/chat message trace while preserving every artifact
    a decision in that trace depends on.

        from dagc import compress
        compressed = compress(messages, target_reduction=0.85)

    Args:
        messages: list of {'role', 'content', ...} dicts.
        target_reduction: fraction of tokens to remove (0-1).
        cfg: a full DAGCConfig for advanced tuning.
        force_preserve: extra literal values to hard-guarantee, on top of
            decision-derived values. Merged with rescue's own set when
            enable_rescue=True -- neither overrides the other.
        enable_rescue: if True, automatically maintains a ShadowBuffer and
            RescueEngine per session_id and folds their force_preserve
            output into this call. If False (default), behaves exactly as
            before -- rescue.py is not even imported, zero overhead.
        session_id: identifies which rescue session this call belongs to.
            `messages` must be a growing, append-only prefix across calls
            sharing a session_id -- use a distinct session_id (or call
            dagc.rescue.reset_rescue_session(session_id)) per independent
            conversation.
        **overrides: any other DAGCConfig field.

    Returns:
        A new list of message dicts (each carrying '_orig_idx').
    """
    import copy as _copy
    c = _copy.deepcopy(cfg) if cfg is not None else DAGCConfig()
    if target_reduction is not None:
        c.TARGET_REDUCTION = target_reduction
    for k, v in overrides.items():
        setattr(c, k, v)

    merged_force_preserve: Set[str] = set(force_preserve) if force_preserve else set()
    _rescue_sess = None

    if enable_rescue:
        # Lazy import: rescue.py imports back from this module
        # (_decision_critical_values / _art_in_text / _footprint_text),
        # so a top-level import here would be circular. This also means
        # enable_rescue=False truly costs nothing -- rescue.py is never
        # touched.
        from .rescue import _run_rescue_for_call

        if c.ABSOLUTE_BUDGET_TOKENS is not None:
            budget_estimate = c.ABSOLUTE_BUDGET_TOKENS
        else:
            # Rough estimate for rescue's own GuaranteedSet sizing only --
            # not used for the actual compression budget, which
            # compress_dagc computes for real from orig_toks below.
            orig_toks_estimate = sum(_tok(_footprint_text(m)) for m in messages)
            budget_estimate = max(1, int(orig_toks_estimate * (1 - c.TARGET_REDUCTION)))

        rescue_force_preserve, _events, unrescuable, _rescue_sess = _run_rescue_for_call(
            messages, session_id=session_id, budget_tokens=budget_estimate)
        merged_force_preserve |= rescue_force_preserve
        if diagnostics is not None and unrescuable:
            diagnostics['unrescuable_evictions'] = unrescuable

    result = compress_dagc(messages, cfg=c, decision_roles=decision_roles,
                            force_preserve=merged_force_preserve or None,
                            diagnostics=diagnostics)

    if getattr(c, 'PRESERVE_VALUE_RECOVERY', True):
        result, _ = inject_value_recovery_stubs(
            result, messages,
            max_stubs=getattr(c, 'MAX_VALUE_RECOVERY_STUBS', 15),
            max_stub_tokens=getattr(c, 'MAX_VALUE_RECOVERY_STUB_TOKENS', 25))

    if _rescue_sess is not None:
        _rescue_sess["last_compressed"] = result

    return result


def decision_loss_frontier(messages: List[Dict], budgets: List[int],
                            cfg: Optional[DAGCConfig] = None,
                            decision_roles: Tuple[str, ...] = ('user', 'assistant')) -> List[Dict]:
    """
    Empirical rate-distortion curve: runs compression at each token budget
    in `budgets` and reports (output_tokens, decision_loss) for each --
    the actual frontier the rate-distortion formalization predicts exists,
    measured directly rather than just asserted. Report this curve instead
    of a single compression-ratio number.

    How to use this for a write-up: call
    ``decision_loss_frontier(messages, budgets=[50, 100, 200, 400, 800])``
    on a batch of traces and plot ``output_tokens`` vs ``decision_loss``.
    That plot *is* the empirical rate-distortion curve -- a measured
    result, not just a derivation.

    NOTE for the paper's rate-distortion figure: this also reports
    `achieved_reduction` (1 - output_tokens/orig_tokens) alongside
    `requested_budget`, so a flat decision_loss curve across requested
    budgets can be correctly attributed to the hard-guarantee mechanism
    overshooting a tight budget (disambiguation the reviewer asked for),
    rather than looking like a contradiction between two views of the
    same axis.

    Two things worth deciding before running this at scale:

    1. ``cfg.LOSS_LAMBDA`` isn't derivable from theory alone -- it needs
       the same kind of fitting/sweep treatment as the other hand-tuned
       constants in this module (GAMMA, MMR_LAMBDA). A small grid search
       against the frontier this function produces -- i.e. sweep LOSS_LAMBDA,
       rerun this function at each value, and compare the resulting curves
       -- is the natural way to pick it.
    2. Whether the ``[preserved: ...]`` hard-guarantee tag should consult
       ``loss_covered`` instead of / in addition to ``covered_arts`` so the
       rescue mechanism stays consistent with this objective when both are
       active -- implemented above in `compress_dagc` as a final
       loss_covered-aware rescue pass, gated behind
       ``cfg.USE_DECISION_LOSS_OBJECTIVE``, PLUS an unconditional
       target_arts_all-based rescue pass that always runs regardless.
    """
    base_cfg = cfg or DAGCConfig()
    decisions = extract_decisions(messages, decision_roles=decision_roles)
    decisions = _attach_dependencies(messages, decisions)
    by_decision = _collect_decision_artifacts_by_decision(decisions)
    type_weight = {'action': base_cfg.W_ACTION, 'judgment': base_cfg.W_JUDGMENT,
                   'confirmation': base_cfg.W_CONFIRMATION}
    orig_toks = sum(_tok(_footprint_text(m)) for m in messages) or 1

    results = []
    for b in sorted(budgets):
        import copy as _copy
        run_cfg = _copy.deepcopy(base_cfg)
        run_cfg.ABSOLUTE_BUDGET_TOKENS = b
        diag: Dict[str, Any] = {}
        out = compress_dagc(messages, cfg=run_cfg, decision_roles=decision_roles, diagnostics=diag)

        covered: Set[str] = set()
        for m in out:
            text = _footprint_text(m)
            for arts in by_decision.values():
                for a in arts:
                    if a not in covered and _art_in_text(a, text):
                        covered.add(a)

        loss = _decision_loss(by_decision, covered, type_weight, decisions)
        out_toks = diag.get('output_tokens', sum(_tok(_footprint_text(m)) for m in out))
        results.append({
            'requested_budget': b,
            'output_tokens': out_toks,
            'achieved_reduction': 1 - (out_toks / orig_toks),
            'decision_loss': loss,
            'n_decisions': len(decisions),
        })
    return results


def verify_no_decision_loss(messages: List[Dict], compressed: List[Dict],
                             decision_roles: Tuple[str, ...] = ('user', 'assistant')
                             ) -> List[Tuple[int, str]]:
    """
    Standalone, independent CI/test check -- does NOT reuse any state from
    a specific compress_dagc() call. Re-extracts decisions fresh from
    `messages` and re-derives their critical values from scratch, then
    checks each one against the ACTUAL compressed output that was shipped.

    Returns a list of (msg_idx, value) pairs for anything not recoverable;
    an empty list means the retention invariant holds for this trace.

    Intended usage: run this after every compress()/compress_dagc() call
    in your test suite, and especially inside an adversarial/fuzzed test
    suite (e.g. via `hypothesis`) that generates traces with a known,
    planted decision set -- fail the build on any non-empty result. This
    is the mechanism that turns "we benchmarked this and it was high"
    into "this is a build-verified invariant."
    """
    decisions = extract_decisions(messages, decision_roles=decision_roles)
    decisions = _attach_dependencies(messages, decisions)
    footprint = '\n'.join(_footprint_text(m) for m in compressed)
    missing: List[Tuple[int, str]] = []
    for d in decisions:
        for v in _decision_critical_values([d]):
            if not _art_in_text(v, footprint):
                missing.append((d['msg_idx'], v))
    return missing


__all__ = [
    "compress",
    "compress_any",
    "compress_dagc",
    "DAGCConfig",
    "DAGC_CFG",
    "DecisionLossError",
    "decision_loss_frontier",
    "verify_no_decision_loss",
]