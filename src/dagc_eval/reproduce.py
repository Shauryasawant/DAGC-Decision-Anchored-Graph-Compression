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
    _extract_verb, _flatten_arg_values, _get_tool_call_args,
    _get_tool_call_name, _stringify_arg_value,
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

    # Use the same critical-value rules as the compressor.
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
        if tc and decision['type'] == 'action':
            tc_name = _get_tool_call_name(tc, decision.get('action', ''))
            tc_args = _get_tool_call_args(tc)
            return {
                'action': pinned_action if pinned_action is not None else tc_name,
                'target': pinned_target if pinned_target is not None else _extract_target(tc_args, text, tc_name),
                'rationale': _extract_rationale(text, arts, decision_values=this_decision_values, decision_idx=decision['msg_idx']),
                'confidence': 0.9 if pinned_target is not None else 0.85,
                '_success': True, '_fallback': 'deterministic',
                '_target_source': 'pinned' if pinned_target is not None else 're_derived',
                '_raw': 'deterministic_fallback',
            }
        if decision['type'] in ('judgment', 'confirmation'):
            return {
                'action': pinned_action if pinned_action is not None else
                        ('confirm' if decision['type'] == 'confirmation' else _extract_verb(text)),
                'target': pinned_target if pinned_target is not None else _extract_target({}, text),
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
    if not llm_result.get('_success', True):
        return det_result if det_result is not None else llm_result
    if det_result is None:
        return llm_result

    merged = dict(llm_result)
    # Prefer a deterministic target extracted from the surviving message.
    if det_result.get('target'):
        merged['target'] = det_result['target']
        merged['_target_source'] = 'deterministic'
    else:
        merged['_target_source'] = 'llm_only'

    if not merged.get('action') and det_result.get('action'):
        merged['action'] = det_result['action']
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


def _decision_cache_key(decision, compressed_msgs):
    return (decision.get('type'), decision.get('action'), str(decision.get('target')),
            tuple(decision.get('rationale', [])), _trace_fingerprint(compressed_msgs))


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
    key = _decision_cache_key(decision, compressed_msgs)
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
