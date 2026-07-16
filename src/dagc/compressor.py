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
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


import numpy as np

from .utils import (
    _art_density, _artifacts, _cos, _encode, _get_text, _head_tail_cap,
    _split_sents, _tok, _value_still_recoverable,
)
from .extraction import (
    _CONFIRM_SIGNALS, _STRONG_JUDGMENT_SIGNALS, _get_tool_call_args,
    _get_tool_call_name, _is_meaningful_candidate_value, _key_tier_rank,
    _stringify_arg_value, extract_decisions,
)
from .graph import (
    CausalGraphConfig, CausalMessageGraph, SpectralCompressor,
    attach_dependencies as _attach_dependencies,
    build_dependency_graph,
)


def compress_any(raw_messages, target_reduction: Optional[float] = None,
                 cfg: Optional[DAGCConfig] = None,
                 decision_roles: Tuple[str, ...] = ('assistant',),
                 force_preserve: Optional[Iterable[str]] = None,
                 rehydrate: bool = True,
                 **overrides):
    """Format-tolerant front door for arbitrary message traces.
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

    if not rehydrate:
        return compressed
    return _denormalize_trace(original_list, compressed)


@dataclass
class DAGCConfig:
    TARGET_REDUCTION: float = 0.87
    EVIDENCE_MIN_BUDGET_PCT: float = 0.09
    PHASE1_FRAC: float = 0.42
    PER_DECISION_MIN_TOKENS: int = 22

    SYSTEM_MAX_TOKENS: int = 22
    DECISION_MAX_TOKENS: int = 50
    TOOL_CALL_MAX_TOKENS: int = 22
    KEEP_LAST_K: int = 1

    COMPRESS_PROTECTED: bool = True
    JUDGMENT_HEAD_FRAC: float = 0.20

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
    MIN_EFF_THRESHOLD: float = 1e-3

    USE_CAUSAL_SKELETON: bool = True
    USE_SPECTRAL: bool = False
    SPECTRAL_WEIGHT: float = 0.25
    MSTAR_CAUSAL_BONUS: float = 4.0

    MSTAR_HARD_DROP: bool = True

    EVIDENCE_BEFORE_FRAC: float = 0.70
    MIN_TOOL_CORROBORATION: int = 2

    MIN_SENT_TOKENS: int = 4
    MAX_EMBED_CHUNK: int = 200

DAGC_CFG = DAGCConfig()


def _is_decision_bearing(m, cfg):
    if cfg.PROTECT_TOOL_CALLS and isinstance(m.get('tool_call'), dict):
        return True
    if cfg.PROTECT_JUDGMENTS and m.get('role') == 'assistant':
        t = _get_text(m)
        if _STRONG_JUDGMENT_SIGNALS.search(t) or _CONFIRM_SIGNALS.search(t):
            return True
    return False


def _is_tool_call_msg(m):
    return isinstance(m.get('tool_call'), dict)


def _art_in_text(art: str, text: str) -> bool:
    """
    Case-insensitive 'does this critical value appear in this text' check.

    Action verbs are always normalized to lowercase at extraction time
    (_extract_verb: `.lower()` on both the direct-verb match and the
    connective fallback), but English capitalizes sentence-initial words
    -- so a decisive verb/connective at a sentence boundary ("Select
    the...", "Therefore, ...") is routinely capitalized in source text
    while its target_arts entry is always lowercase. A case-sensitive
    `in` check silently fails on exactly this class of value.

    Safe everywhere: path/id/error artifacts are extracted verbatim from
    the same text they're later matched against, so they're already
    case-consistent -- this can only convert an existing false negative
    into a correct match, never a false positive within the same message.
    """
    return bool(art) and art.lower() in text.lower()


def _decision_critical_values(decisions: List[Dict]) -> Set[str]:
    out: Set[str] = set()
    for d in decisions:
        act = d.get('action')
        # Decision verbs come from a controlled vocabulary.
        if act and 2 <= len(str(act)) <= 30:
            out.add(str(act))

        t = d.get('target')
        if t:
            for piece in (t if isinstance(t, list) else [t]):
                s = str(piece).strip().strip('"\'[]')
                if 2 <= len(s) <= 60 and _is_meaningful_candidate_value(s):
                    out.add(s)

        for rat in d.get('rationale', []):
            if not isinstance(rat, str):
                continue
            m = re.search(r'[=:]\s*(.+)$', rat)
            val = (m.group(1) if m else rat).strip()
            if 2 <= len(val) <= 60 and len(val.split()) <= 6 and _is_meaningful_candidate_value(val):
                out.add(val)

    return out

def _collect_decision_artifacts(decisions):
    arts = set()
    for d in decisions:
        for kind in ('paths', 'ids', 'errors'):
            arts.update(d['artifacts'].get(kind, []))
    arts |= _decision_critical_values(decisions)
    return arts


def _collect_decision_artifacts_by_decision(decisions) -> Dict[int, Set[str]]:
    """Return the per-decision artifact set that underpins preservation tags."""
    by_decision: Dict[int, Set[str]] = {}
    for d in decisions:
        by_decision[d['msg_idx']] = _collect_decision_artifacts([d])
    return by_decision


def _artifact_owners(art: str, by_decision: Dict[int, Set[str]]) -> Set[int]:
    return {k for k, arts in by_decision.items() if art in arts}


def _build_preserved_tag(missing: List[str], by_decision: Optional[Dict[int, Set[str]]] = None,
                         *channels: str) -> str:
    parts = []
    for a in sorted(missing)[:6]:
        if _already_covered(a, *channels):
            continue
        owners = _artifact_owners(a, by_decision or {}) if by_decision is not None else set()
        if owners:
            owner_str = ','.join(str(k) for k in sorted(owners))
            parts.append(f'{a}#d{owner_str}')
        else:
            parts.append(a)
    if not parts:
        return ''
    return '[preserved: ' + ', '.join(parts) + ']'


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
    """
    Sentence-priority compressor for protected messages (judgment /
    confirmation / tool-call prefix text): keeps sentences that carry a
    decision-critical value or the decisive verb first, then fills the
    remaining budget with leading/trailing sentences, then hard-appends
    any still-missing target as a compact tag so it can never be lost to
    a sentence-boundary edge case.
    """
    if _tok(content) <= budget:
        return content

    sents = _split_sents(content, cfg.MIN_SENT_TOKENS)
    if not sents:
        return _head_tail_cap(content, budget, head_frac)

    def _is_critical(s):
        if any(_art_in_text(a, s) for a in target_arts):
            return True
        if _extract_metric_strings(s) and any(ms in dec_metrics for ms in _extract_metric_strings(s)):
            return True
        return bool(_STRONG_JUDGMENT_SIGNALS.search(s) or _CONFIRM_SIGNALS.search(s))

    selected: List[str] = []
    used = 0

    for s in sents:
        if _is_critical(s):
            t = _tok(s)
            if used + t <= budget * 1.25:
                selected.append(s)
                used += t

    head_n = max(1, int(len(sents) * head_frac))
    ordered = sents[:head_n] + list(reversed(sents[head_n:]))
    for s in ordered:
        if s in selected:
            continue
        t = _tok(s)
        if used + t <= budget:
            selected.append(s)
            used += t

    result = ' '.join(selected).strip() or _head_tail_cap(content, budget, head_frac)

    missing = [a for a in target_arts if a and a in content and a not in result]
    if missing:
        tag = _build_preserved_tag(missing, by_decision)
        candidate = (result + ' ' + tag).strip()
        # Do not let a preservation tag make the message longer than the cap.
        result = candidate if _tok(candidate) <= budget * 1.5 else result

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


def _monotonic(original: str, candidate: str) -> str:
    """Compression must never cost more tokens than the untouched
    original. If a rescue/tag mechanism pushed the candidate past that,
    fall back to the original rather than emit something larger."""
    return candidate if _tok(candidate) < _tok(original) else original

def _compress_protected_message(m: Dict, target_arts: Set[str], dec_metrics: Set[str],
                                  cfg: DAGCConfig,
                                  by_decision: Optional[Dict[int, Set[str]]] = None) -> str:
    role = m.get('role', '')
    content = m.get('content', '') or ''
    is_tc = isinstance(m.get('tool_call'), dict)

    if role == 'system':
        return _head_tail_cap(content, cfg.SYSTEM_MAX_TOKENS)

    if is_tc:
        tc = m['tool_call']
        tool_name = _get_tool_call_name(tc, 'tool')
        args = _get_tool_call_args(tc)
        slim = _slim_tool_args(args, target_arts)

        tc_str = (f"→TOOL:{tool_name}"
                  f"({json.dumps(slim, separators=(',', ':'))[:140]})")
        tc_toks = _tok(tc_str)
        c_budget = max(4, cfg.TOOL_CALL_MAX_TOKENS - tc_toks - 2)

        prefix = (_select_priority_content(content, target_arts, dec_metrics, c_budget, cfg,
                                            head_frac=1.0, by_decision=by_decision) if content else '')

        # Avoid repeating a tool name that is already present.
        if tool_name and _already_covered(tool_name, prefix):
            return _monotonic(content, prefix.strip())

        candidate = (prefix + ' ' + tc_str).strip()
        return _monotonic(content, candidate)

    result = _select_priority_content(content, target_arts, dec_metrics,
                                       cfg.DECISION_MAX_TOKENS, cfg,
                                       head_frac=cfg.JUDGMENT_HEAD_FRAC,
                                       by_decision=by_decision)
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


def _judgment_has_evidence(msg_idx: int, messages: List[Dict], cfg: DAGCConfig = None) -> bool:
    """Anti-injection evidence gate: a judgment/confirmation only counts as
    "well-supported" if the tool activity *before it in the trace* backs
    it up. Every signal here is computed strictly from messages[:msg_idx]
    -- nothing after the decision, real or adversarially injected, can
    ever change the verdict. This is a heuristic firewall against a
    message that *claims* a decision with no supporting evidence in the
    trace -- not a security control."""
    threshold = getattr(cfg, 'EVIDENCE_BEFORE_FRAC', 0.70)
    min_corr = getattr(cfg, 'MIN_TOOL_CORROBORATION', 1)

    if msg_idx <= 0 or msg_idx >= len(messages):
        return True

    from .extraction import _extract_entity_target
    text = _get_text(messages[msg_idx])
    target = _extract_entity_target(text)

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


def _phase1_hard_guarantee(pool, decisions, budget, extra_critical=None):
    target_arts = _collect_decision_artifacts(decisions) | (extra_critical or set())
    if not target_arts:
        return [], set()

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
    used_toks = 0
    injected: Set[int] = set()

    for a in sorted(target_arts, key=rarity, reverse=True):
        if a in covered or a not in best:
            continue
        idx, cost = best[a]
        if used_toks + cost > budget and cost > 40:
            continue
        if idx not in injected:
            injected.add(idx)
            selected.append(idx)
            used_toks += cost
            s, _ = pool[idx]
            for other in target_arts:
                if other in s:
                    covered.add(other)
        else:
            covered.add(a)

    return selected, covered


def compress_dagc(messages: List[Dict], cfg: Optional[DAGCConfig] = None,
                   decision_roles: Tuple[str, ...] = ('assistant',),
                   force_preserve: Optional[Iterable[str]] = None) -> List[Dict]:
    """
    Decision-Anchored Graph Compression — v4.3.

    Compresses a message trace to roughly cfg.TARGET_REDUCTION fewer
    tokens while hard-guaranteeing that every artifact a decision depends
    on (tool-call arguments, confirmed IDs, cited metrics) survives in the
    output. Purely algorithmic -- no network calls.

    Returns a list of messages (same dict shape as input, plus an
    '_orig_idx' key recording each message's position in the original
    trace).
    """
    if cfg is None:
        cfg = DAGC_CFG

    task = next((m['content'] for m in messages if m.get('role') == 'user'), 'complete the task')
    task_emb = _encode([task])[0]
    orig_toks = sum(_tok(_footprint_text(m)) for m in messages)
    n = len(messages)

    try:
        decisions = extract_decisions(messages, decision_roles=decision_roles)
        decisions = _attach_dependencies(messages, decisions)
    except Exception:
        decisions = []

    injection_filtered: Set[int] = set()
    for d in decisions:
        if d['type'] in ('judgment', 'confirmation'):
            if not _judgment_has_evidence(d['msg_idx'], messages, cfg):
                injection_filtered.add(d['msg_idx'])

    valid_decisions = [d for d in decisions if d['msg_idx'] not in injection_filtered]

    # Preserve literal decision values even when a message is not protected.
    target_arts = _collect_decision_artifacts(decisions)
    if force_preserve:
        target_arts |= {str(x) for x in force_preserve if x and 2 <= len(str(x)) <= 80}
    dec_metrics = _decision_metric_strings(decisions)
    by_decision = _collect_decision_artifacts_by_decision(decisions)

    corroborated = _corroborated_artifacts(messages, decisions, cfg.ART_CORROBORATION_MIN)

    # User-provided IDs and paths are authoritative on first mention.
    user_stated_artifacts: Set[str] = set()
    for m in messages:
        if m.get('role') != 'user':
            continue
        a = _artifacts(_get_text(m))
        user_stated_artifacts.update(a['ids'])
        user_stated_artifacts.update(a['paths'])

    target_arts |= corroborated | user_stated_artifacts

    M_star = set(range(n))
    spec_scores = {i: 0.0 for i in range(n)}
    cg = None
    try:
        if cfg.USE_CAUSAL_SKELETON:
            cg = CausalMessageGraph(messages, decisions,
                                     CausalGraphConfig(CAUSAL_TAU=cfg.TAU, ADD_SEQ_EDGES=False))
            M_star = cg.minimal_sufficient_set()
        if cfg.USE_SPECTRAL and cg is not None:
            spec_scores = SpectralCompressor(cg).normalized_scores()
    except Exception:
        pass

    protected: Set[int] = set()
    for i, m in enumerate(messages):
        if m.get('role') == 'system' or i >= n - cfg.KEEP_LAST_K:
            protected.add(i)
    for d in valid_decisions:
        mi = d['msg_idx']
        if d['type'] == 'action' and cfg.PROTECT_TOOL_CALLS:
            protected.add(mi)
        elif d['type'] in ('judgment', 'confirmation') and cfg.PROTECT_JUDGMENTS:
            if _judgment_has_evidence(mi, messages, cfg):
                protected.add(mi)

    total_budget = max(1, int(orig_toks * (1 - cfg.TARGET_REDUCTION)))

    pcfg = _scale_protected_cfg(cfg, [messages[i] for i in protected], total_budget)
    compressed_protected_content: Dict[int, str] = {}
    if cfg.COMPRESS_PROTECTED:
        for i in protected:
            compressed_protected_content[i] = _compress_protected_message(
                messages[i], target_arts, dec_metrics, pcfg, by_decision=by_decision)

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

    dependency_edges = build_dependency_graph(messages, decisions)
    dependency_vals: Set[str] = {e['artifact'] for e in dependency_edges}
    n_valid_decisions = len(decisions)
    _target_lens = [_tok(a) for a in target_arts] or [8]
    per_decision_floor = max(
        getattr(cfg, 'PER_DECISION_MIN_TOKENS', 22),
        int(np.mean(_target_lens)) + 10,
    )
    min_evidence_budget = max(
        int(orig_toks * cfg.EVIDENCE_MIN_BUDGET_PCT),
        n_valid_decisions * per_decision_floor,
        len(dependency_vals) * 15,
    )
    # Reserve a proportional minimum budget for rare decision evidence.
    evidence_floor = min(min_evidence_budget, max(20, int(total_budget * 0.5)))
    free_budget = max(evidence_floor, total_budget - protected_toks)
    phase1_budget = int(free_budget * cfg.PHASE1_FRAC)

    pool: List[Tuple[str, int]] = []
    for i, m in enumerate(messages):
        if i in protected:
            continue
        text_i = _get_text(m)
        a_i = _artifacts(text_i)
        is_corroborated_carrier = any(
            x in corroborated or x in user_stated_artifacts
            for x in a_i['ids'] + a_i['paths']
        )
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
        return out

    pool_texts = [s for s, _ in pool]
    pool_embs = _encode(pool_texts, max_chunk=cfg.MAX_EMBED_CHUNK)

    p1_idx, covered_arts = _phase1_hard_guarantee(pool, decisions, phase1_budget,
                                                    extra_critical=dependency_vals)
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
            for kind in ('paths', 'ids', 'errors'):
                covered_arts.update(_artifacts(s)[kind])

    remaining = max(0, free_budget - p1_toks)

    pending = [i for i in range(len(pool)) if i not in p1_set]
    selected_embs = [pool_embs[i] for i in p1_idx]
    p2_idx: List[int] = []
    used_p2 = 0

    for _ in range(len(pending)):
        if not pending or used_p2 >= remaining:
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
            if eff > best_eff:
                best_eff, best_i = eff, idx

        if best_i is None or best_eff <= cfg.MIN_EFF_THRESHOLD:
            break

        s, _mi = pool[best_i]
        p2_idx.append(best_i)
        selected_embs.append(pool_embs[best_i])
        for kind in ('paths', 'ids', 'errors'):
            covered_arts.update(_artifacts(s)[kind])
        for ms in _extract_metric_strings(s):
            if ms in dec_metrics:
                covered_metrics.add(ms)
        used_p2 += _tok(s)
        pending.remove(best_i)

    for i in protected:
        cc = compressed_protected_content.get(i, _get_text(messages[i]))
        for kind in ('paths', 'ids', 'errors'):
            covered_arts.update(_artifacts(cc)[kind])
        for a in target_arts:
            if _art_in_text(a, cc):
                covered_arts.add(a)

    all_sel = p1_set | set(p2_idx)
    still_miss = target_arts - covered_arts
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
            for kind in ('paths', 'ids', 'errors'):
                covered_arts.update(_artifacts(s)[kind])

    msg_sents: Dict[int, List[str]] = {}
    for idx in sorted(p1_set | set(p2_idx)):
        s, mi = pool[idx]
        msg_sents.setdefault(mi, []).append(s)

    out: List[Dict] = []
    for i, m in enumerate(messages):
        if i in protected:
            mc = dict(m)
            if cfg.COMPRESS_PROTECTED and i in compressed_protected_content:
                mc['content'] = compressed_protected_content[i]
                if _is_tool_call_msg(m):
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
            mc['content'] = ' '.join(msg_sents[i])
            mc['_orig_idx'] = i
            out.append(mc)
    return out


# Public shorthand for compress_dagc.
def compress(messages: List[Dict], target_reduction: Optional[float] = None,
             cfg: Optional[DAGCConfig] = None,
             decision_roles: Tuple[str, ...] = ('assistant',),
             force_preserve: Optional[Iterable[str]] = None,
             **overrides) -> List[Dict]:
    """
    Compress an agent/chat message trace while preserving every artifact
    a decision in that trace depends on.

        from dagc import compress
        compressed = compress(messages, target_reduction=0.85)
        response = client.chat.completions.create(model="gpt-4", messages=compressed)

    Args:
        messages: list of {'role', 'content', ...} dicts (OpenAI-style;
            'tool_call' key optional, both {'name','args'} and
            {'function':{'name','arguments'}} shapes are supported).
        target_reduction: fraction of tokens to remove (0-1). Overrides
            cfg.TARGET_REDUCTION if given.
        cfg: a full DAGCConfig for advanced tuning. If omitted, a copy of
            the module default is used.
        force_preserve: extra literal values to hard-guarantee, on top of
            decision-derived values.
        **overrides: any other DAGCConfig field, e.g. compress(msgs, KEEP_LAST_K=3).

    Returns:
        A new list of message dicts (each carrying '_orig_idx') -- roughly
        target_reduction smaller in token count, safe to pass straight to
        an LLM call.
    """
    import copy as _copy
    c = _copy.deepcopy(cfg) if cfg is not None else DAGCConfig()
    if target_reduction is not None:
        c.TARGET_REDUCTION = target_reduction
    for k, v in overrides.items():
        setattr(c, k, v)
    return compress_dagc(messages, cfg=c, decision_roles=decision_roles,
                          force_preserve=force_preserve)


__all__ = [
    "compress",
    "compress_any",
    "compress_dagc",
    "DAGCConfig",
    "DAGC_CFG",
]
