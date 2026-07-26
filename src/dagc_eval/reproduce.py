"""
Decision reproducer: given a compressed trace, try to reconstruct each
original decision (action/target/rationale). Two strategies, combined:

  1. Deterministic re-extraction: if the message that hosted the decision
     survived compression intact, run the SAME rule-based extractor used
     to build ground truth on it. This never needs an LLM and, per
     `_merge_llm_and_deterministic` below, wins whenever it's available --
     code beats guess whenever the source text is actually present.
  2. LLM reproduction: only used as a fallback for messages that were
     compressed away, where a free-text reconstruction from surrounding
     context is the best available signal. Fully BYOK -- pass any object
     satisfying dagc_eval.interfaces.LLMClient, or omit it entirely and
     dagc_eval degrades gracefully to deterministic-only scoring.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

from dagc.extraction import (
    _CANONICAL_TARGET_KEY_TIERS, _extract_rationale, _extract_target,
    _extract_verb, _find_decisive_match, _flatten_arg_values,
    _get_tool_call_args, _get_tool_call_name, _stringify_arg_value,
    _bare_rationale_value,
)
from dagc.utils import _artifacts, _get_text, target_still_recoverable, action_still_recoverable
from dagc.compressor import _decision_critical_values

from .interfaces import LLMClient


def _build_target_priority_addendum() -> str:
    labels = ['/'.join(tokens) for tokens in _CANONICAL_TARGET_KEY_TIERS]
    order = ' > '.join(labels)
    return f"""
STEP BOUNDARY: The decision you are asked about lives on exactly ONE
numbered line in the trace below. Any identifier, argument, or value that
appears on a DIFFERENT numbered line belongs to a DIFFERENT decision and
must never be used as this decision's target, even if it looks more
"important" than the value actually on the target line.

MULTI-ARGUMENT TOOL CALLS: if the target line's tool call has more than
one plausible target argument, prefer the argument whose key matches this
order of specificity (matched by SUBSTRING against the key name):
{order}
If a key clearly represents a NEWLY SET / UPDATED value (prefixed with
new_/updated_/target_), prefer it over an existing lookup key such as an
account id. Pick exactly one; do not concatenate multiple values.
"""


REPRODUCE_SYSTEM_TARGET_ADDENDUM = _build_target_priority_addendum()

REPRODUCE_SYSTEM = """\
You are a decision auditor for AI agent traces.
Identify the EXACT decision at the requested step.
""" + REPRODUCE_SYSTEM_TARGET_ADDENDUM


def _format_trace_for_prompt(messages, target_idx=None):
    lines = []
    for pos, m in enumerate(messages):
        idx = m.get('_orig_idx', pos)
        role = m.get('role', '?').upper()
        text = _get_text(m)
        tc = m.get('tool_call')

        is_exact = (target_idx is not None and idx == target_idx)
        is_near = (target_idx is not None and not is_exact and abs(idx - target_idx) <= 3)
        max_len = len(text) if is_exact else (1200 if is_near else 700)

        if tc:
            lines.append(
                f'[{idx}] {role}: {text[:max_len]} '
                f'→ TOOL:{_get_tool_call_name(tc, "")}'
                f'({json.dumps(_get_tool_call_args(tc))})'
            )
        else:
            lines.append(f'[{idx}] {role}: {text[:max_len]}')
    return '\n'.join(lines)


def _focused_window(messages, target_orig_idx, window=6):
    """Index-shift-robust window: if a message was inserted upstream and
    shifted every subsequent _orig_idx, fall back to a ±N scan (scaled
    with trace length) before giving up and using the full trace."""
    if len(messages) <= 2 * window + 1:
        return messages

    pos = next((i for i, m in enumerate(messages) if m.get('_orig_idx', i) == target_orig_idx), None)

    if pos is None:
        max_delta = max(3, min(25, len(messages) // 8))
        for delta in range(1, max_delta + 1):
            for shifted in (target_orig_idx + delta, target_orig_idx - delta):
                pos = next((i for i, m in enumerate(messages) if m.get('_orig_idx', i) == shifted), None)
                if pos is not None:
                    break
            if pos is not None:
                break

    if pos is None:
        return messages

    keep = set(range(max(0, pos - window), min(len(messages), pos + window + 1)))
    for i, m in enumerate(messages):
        if m.get('role') in ('system', 'user'):
            keep.add(i)
    return [messages[i] for i in sorted(keep)]


def _build_tool_catalogue(messages):
    names = []
    for m in messages:
        tc = m.get('tool_call')
        nm = _get_tool_call_name(tc, '') if isinstance(tc, dict) else ''
        if nm:
            names.append(nm)
    seen, out = set(), []
    for x in names:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _deterministic_extract(compressed_msgs, decision):
    target_idx = decision['msg_idx']

    this_decision_values = _decision_critical_values([decision])

    for m in compressed_msgs:
        if m.get('_orig_idx') != target_idx:
            continue
        text = _get_text(m)
        tc = m.get('tool_call')
        arts = _artifacts(text)
        gt_target = decision.get('target')
        gt_action = decision.get('action')

        if target_still_recoverable(gt_target, text, arts):
            pinned_target = gt_target
        else:
            pinned_target = None

        if action_still_recoverable(gt_action, text):
            pinned_action = gt_action
        else:
            pinned_action = None

        fresh_rationale = _extract_rationale(
            text, arts, decision_values=this_decision_values,
            decision_idx=decision['msg_idx'])
        pinned_rationale = [
            r for r in (decision.get('rationale') or [])
            if target_still_recoverable(_bare_rationale_value(r), text, arts)
        ]
        seen, rationale = set(), []
        for r in pinned_rationale + fresh_rationale:
            bare = _bare_rationale_value(r)
            if bare not in seen:
                seen.add(bare)
                rationale.append(r)

        # Same decisive-clause scoping used to build ground truth --
        # recomputed here on the COMPRESSED text, since the decisive
        # sentence's position can shift/shrink after compression.
        dm = _find_decisive_match(text)
        decisive_span = (dm.start(), dm.end()) if dm is not None else None

        if tc and decision['type'] == 'action':
            tc_name = _get_tool_call_name(tc, decision.get('action', ''))
            tc_args = _get_tool_call_args(tc)
            return {
                'action': pinned_action if pinned_action is not None else tc_name,
                'target': pinned_target if pinned_target is not None else
                    _extract_target(tc_args, text, tc_name, decision_idx=target_idx, decisive_span=decisive_span),
                'rationale': rationale,
                'confidence': 0.9 if pinned_target is not None else 0.85,
                '_success': True, '_fallback': 'deterministic',
                '_target_source': 'pinned' if pinned_target is not None else 're_derived',
                '_raw': 'deterministic_fallback',
            }
        if decision['type'] in ('judgment', 'confirmation'):
            return {
                'action': pinned_action if pinned_action is not None else
                        ('confirm' if decision['type'] == 'confirmation' else _extract_verb(text)),
                'target': pinned_target if pinned_target is not None else
                    _extract_target({}, text, decision_idx=target_idx, decisive_span=decisive_span),
                'rationale': _extract_rationale(text, arts, decision_values=this_decision_values, decision_idx=decision['msg_idx']),
                'confidence': 0.9 if pinned_target is not None else 0.80,
                '_success': True, '_fallback': 'deterministic',
                '_target_source': 'pinned' if pinned_target is not None else 're_derived',
                '_raw': 'deterministic_fallback',
            }
    return None


def _reproduce_decision_impl(compressed_msgs, decision, llm: LLMClient, max_retries: int = 2) -> Dict:
    focused = _focused_window(compressed_msgs, decision['msg_idx'])
    tools = _build_tool_catalogue(focused)
    tool_line = (f"\nKnown tool names: {', '.join(tools)}. "
                 f"For a TOOL CALL decision, action MUST be one of these." if tools else "")
    trace_text = ("NOTE: step numbers may have gaps after compression.\n"
                  + _format_trace_for_prompt(focused, target_idx=decision['msg_idx']))

    type_hint = {
        'action': ("TOOL CALL — action = exact tool name after '→ TOOL:' in the trace. "
                   "target = the argument value that best matches the priority order given above."),
        'judgment': ("JUDGMENT — action = decisive verb only (recommend/select/choose/adopt/"
                     "conclude/decide). target = the specific option, identifier, or value chosen."),
        'confirmation': ("CONFIRMATION — action = 'confirm'. target = the specific value being "
                         "confirmed (often an identifier/token, not just a descriptive name)."),
    }.get(decision['type'], "")

    step_hint = (f"Step {decision['msg_idx']} is an assistant '{decision['type']}' turn. "
                 f"{type_hint}\nWhat did the agent decide at this step?")
    prompt = (f"COMPRESSED AGENT TRACE:\n{trace_text}\n{tool_line}\n\n"
              f"TASK:\n{step_hint}\n\n"
              f"Return ONLY a JSON object with keys: action, target, rationale (list of strings), confidence.")

    last_result: Optional[Dict] = None
    for attempt in range(max_retries + 1):
        try:
            raw = llm.complete(REPRODUCE_SYSTEM, prompt,
                                temperature=min(0.1 * attempt, 0.3), max_tokens=400)
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            result = json.loads(raw)

            rat = result.get('rationale', [])
            if isinstance(rat, str):
                rat = [r.strip() for r in re.split(r'[,;]\s*', rat) if r.strip()]
            elif isinstance(rat, list):
                flat = []
                for r in rat:
                    if isinstance(r, list):
                        flat.extend(str(x).strip() for x in r if str(x).strip())
                    elif r:
                        flat.append(str(r).strip())
                rat = flat
            result['rationale'] = [r for r in rat if len(r) > 1]
            result['_raw'] = raw
            result['_success'] = True
            last_result = result

            if result.get('action') is not None:
                return result
        except Exception as e:
            last_result = {'action': None, 'target': None, 'rationale': [],
                            'confidence': 0.0, '_raw': str(e), '_success': False}

        if attempt < max_retries:
            time.sleep(2 ** attempt)

    fallback = _deterministic_extract(compressed_msgs, decision)
    if fallback is not None:
        return fallback

    return last_result or {'action': None, 'target': None, 'rationale': [],
                            'confidence': 0.0, '_raw': 'all_attempts_failed', '_success': False}


def _resolve_target_from_compressed(compressed_msgs: List[Dict], decision: Dict) -> Optional[Dict]:
    return _deterministic_extract(compressed_msgs, decision)


def _merge_llm_and_deterministic(llm_result: Dict, det_result: Optional[Dict]) -> Dict:
    """
    Combine an LLM reconstruction with the deterministic re-extraction of
    the same decision. The deterministic target wins whenever it's
    available (grounded in surviving source text, not a guess) -- but the
    returned `confidence` must describe THAT target, not whichever
    confidence happened to be lying around. Concretely:

      - If deterministic supplied no target, the LLM's own target and
        confidence stand as-is (nothing to reconcile).
      - If deterministic supplied a target, its OWN confidence
        (computed by _deterministic_extract from pinned/re_derived
        status) replaces the LLM's self-reported number, since that
        number was never an estimate of the deterministic target's
        reliability in the first place.
      - Agreement between the two sources is a genuine, extra evidence
        signal -- not something to discard. When the LLM independently
        landed on the same target, that's corroboration and confidence
        is nudged up (capped at 0.99, never allowed to reach false
        certainty). When they disagree, the deterministic confidence is
        kept but capped, and the mismatch is exposed via `_agreement`
        so a caller (eval harness, review queue) can flag it instead of
        the disagreement being silently absorbed.
      - Both rationales are unioned rather than one replacing the other,
        since they come from different, non-redundant evidence
        (LLM free-text reasoning vs. regex-grounded extraction).
    """
    if not llm_result.get('_success', True):
        return det_result if det_result is not None else llm_result
    if det_result is None:
        return llm_result

    merged = dict(llm_result)

    det_target = det_result.get('target')
    llm_target = llm_result.get('target')

    if det_target:
        merged['target'] = det_target
        merged['_target_source'] = 'deterministic'

        agree = (str(llm_target).strip().lower() == str(det_target).strip().lower()
                 if llm_target else False)
        merged['_agreement'] = agree

        det_conf = det_result.get('confidence', 0.0)
        if agree:
            llm_conf = llm_result.get('confidence', 0.0) or 0.0
            merged['confidence'] = min(0.99, max(det_conf, llm_conf) + 0.05)
        else:
            # Deterministic target wins on grounding, but an unconfirmed
            # disagreement shouldn't claim the same certainty an agreed
            # result would -- cap it below the "agreed" ceiling.
            merged['confidence'] = min(0.9, det_conf)
    else:
        merged['_target_source'] = 'llm_only'
        merged['_agreement'] = None
        # LLM's own confidence stands -- nothing deterministic to blend.

    if not merged.get('action') and det_result.get('action'):
        merged['action'] = det_result['action']

    llm_rat = merged.get('rationale', []) or []
    det_rat = det_result.get('rationale', []) or []
    seen, combined = set(), []
    for r in list(llm_rat) + list(det_rat):
        key = str(r).strip().lower()
        if key and key not in seen:
            seen.add(key)
            combined.append(r)
    merged['rationale'] = combined

    return merged


_REPRO_CACHE: Dict[Tuple, Dict] = {}
_REPRO_CACHE_LOCK = threading.Lock()
_REPRO_CACHE_MAXSIZE = 10_000


def _trace_fingerprint(compressed_msgs):
    payload = [
        (m.get('_orig_idx'), m.get('role'), m.get('content', ''),
         json.dumps(m.get('tool_call'), sort_keys=True) if isinstance(m.get('tool_call'), dict) else None)
        for m in compressed_msgs
    ]
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _llm_identity(llm: Optional[LLMClient]) -> str:
    """
    Stable identity token for cache-key purposes. Two different LLM
    clients (different model, different provider, or no LLM at all)
    must never collide on the same cache entry, even when the
    decision/trace inputs are identical -- otherwise a deterministic-only
    call and an LLM-backed call for the SAME decision would share a
    cache slot and silently return each other's results.

    Preference order, most-specific first:
      1. An explicit `model` or `model_name` attribute, if the client
         exposes one -- this is the actual thing that changes output.
      2. The client's class name, as a coarser fallback (distinguishes
         different LLMClient implementations even without a model attr).
      3. 'none', when no LLM was supplied at all (deterministic-only path).

    Deliberately does NOT fall back to id(llm) (Python object id): that
    would make the cache key different across process restarts / new
    client instances of the SAME model, defeating caching entirely for
    the common case of "construct a fresh client object each run."
    """
    if llm is None:
        return 'none'
    model = getattr(llm, 'model', None) or getattr(llm, 'model_name', None)
    if model:
        return f'{type(llm).__name__}:{model}'
    return type(llm).__name__


def _decision_cache_key(decision, compressed_msgs, llm: Optional[LLMClient] = None,
                         max_retries: int = 2):
    return (decision.get('type'), decision.get('action'), str(decision.get('target')),
            tuple(decision.get('rationale', [])), _trace_fingerprint(compressed_msgs),
            _llm_identity(llm), max_retries)


def reproduce_decision(compressed_msgs: List[Dict], decision: Dict,
                        llm: Optional[LLMClient] = None, max_retries: int = 2) -> Dict:
    """
    Reproduce a single decision from a compressed trace.

    If `llm` is None, this is 100% deterministic (no network call): the
    decision's source message is re-read with the same code that built
    ground truth if it survived compression, or reported as a genuine
    miss if it didn't. Pass any LLMClient to additionally attempt
    free-text reconstruction for decisions whose message was compressed
    away.
    """
    key = _decision_cache_key(decision, compressed_msgs, llm, max_retries)
    with _REPRO_CACHE_LOCK:
        if key in _REPRO_CACHE:
            return copy.deepcopy(_REPRO_CACHE[key])

    det_result = _resolve_target_from_compressed(compressed_msgs, decision)

    if llm is None:
        result = det_result or {'action': None, 'target': None, 'rationale': [],
                                 'confidence': 0.0, '_success': False, '_fallback': 'no_llm'}
    else:
        llm_result = _reproduce_decision_impl(compressed_msgs, decision, llm, max_retries)
        result = _merge_llm_and_deterministic(llm_result, det_result)

    with _REPRO_CACHE_LOCK:
        if len(_REPRO_CACHE) >= _REPRO_CACHE_MAXSIZE:
            _REPRO_CACHE.pop(next(iter(_REPRO_CACHE)))
        _REPRO_CACHE[key] = copy.deepcopy(result)
    return result
