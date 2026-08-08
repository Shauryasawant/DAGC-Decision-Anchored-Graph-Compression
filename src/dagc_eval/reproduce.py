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
    _find_decisive_match, _flatten_arg_values,
    _get_tool_call_args, _get_tool_call_name, _stringify_arg_value,
    _bare_rationale_value, _extract_inline_tool_call,
    _build_decision_for_message,
)
from dagc.multilingual_decision_detector import MULTILINGUAL_PATTERNS
from dagc.utils import _artifacts, _get_text, target_still_recoverable, action_still_recoverable
from dagc.compressor import _decision_critical_values

from .interfaces import LLMClient
import os

_FALLBACK_LOG_PATH = os.environ.get('DAGC_FALLBACK_LOG')

def _log_fallback_event(decision, result, llm):
    if not _FALLBACK_LOG_PATH:
        return
    row = {
        'decision_id': decision.get('msg_idx'),
        'decision_type': decision.get('type'),
        'path': 'llm_only' if result.get('_target_source') == 'llm_only' else 'deterministic',
        'agreement': result.get('_agreement'),
        'confidence': result.get('confidence'),
        'llm_model': _llm_identity(llm) if llm is not None else None,
    }
    with open(_FALLBACK_LOG_PATH, 'a') as f:
        f.write(json.dumps(row) + '\n')

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

def _reconstruct_positional_messages(compressed_msgs, target_idx, override_text=None, override_role=None):
    """
    Build a list indexed by original message position (list[i] <-> _orig_idx
    == i) from a compressed trace, so _build_decision_for_message() -- which
    indexes by list position and looks at messages[idx-1] for a few builders
    -- can run unmodified against compressed output. Positions whose message
    did not survive compression become an inert empty placeholder rather
    than shifting every later index down. A directive/imperative-response
    check against a placeholder simply fails closed -- correctly: if the
    prior message didn't survive compression, ground truth itself had no
    way to see that context once compressed either, so failing that check
    here is not a new inconsistency, only an honest one.

    override_text/override_role, when given, replace messages[target_idx]'s
    own content -- used to let rescued/recovered text stand in for a hosting
    message that itself did not survive compression, while every OTHER
    position still reflects only what's actually in the compressed trace.
    """
    max_idx = max([m.get('_orig_idx', 0) for m in compressed_msgs] + [target_idx])
    out = [{'role': '', 'content': ''} for _ in range(max_idx + 1)]
    for m in compressed_msgs:
        oi = m.get('_orig_idx')
        if oi is not None and 0 <= oi <= max_idx:
            out[oi] = m
    if override_text is not None:
        slot = dict(out[target_idx]) if out[target_idx].get('content') else {}
        slot['content'] = override_text
        slot['role'] = override_role or slot.get('role') or 'assistant'
        if 'tool_call' not in slot and isinstance(out[target_idx].get('tool_call'), dict):
            slot['tool_call'] = out[target_idx]['tool_call']
        out[target_idx] = slot
    return out


def _local_text_for_rescue(compressed_msgs, target_idx, radius=2):
    """
    Bounded-context text for last-resort tool-call/artifact rescue when a
    decision's own hosting message did not survive compression.

    ROOT CAUSE this closes: the old fallback scanned the FULL trace for
    'any' inline tool call, which reliably handed back the WRONG tool
    whenever the true tool call for THIS decision was compressed away but
    some OTHER decision's tool call (usually one with richer, more clearly
    decision-critical args) was still present elsewhere in the trace --
    e.g. a bare 'think' call with no args, reliably lost to a nearby
    'find_user_id_by_name_zip' call that still carried its id/zip args.
    Restrict the search to messages whose _orig_idx is within `radius` of
    target_idx, matching how much context a human re-reading the compressed
    trace would actually have to guess from.
    """
    near = [m for m in compressed_msgs
            if m.get('_orig_idx') is not None and abs(m['_orig_idx'] - target_idx) <= radius]
    return '\n'.join(_get_text(m) for m in near)

def _deterministic_extract(compressed_msgs, decision):
    target_idx = decision['msg_idx']

    this_decision_values = _decision_critical_values([decision])

    full_text = '\n'.join(_get_text(m) for m in compressed_msgs)
    full_arts = _artifacts(full_text)

    hosting = next((m for m in compressed_msgs if m.get('_orig_idx') == target_idx), None)
    is_stub = bool(hosting is not None and hosting.get('_stub'))

    if hosting is not None:
        text = _get_text(hosting)
        role = hosting.get('role', '')
        tc = hosting.get('tool_call')
        if not isinstance(tc, dict):
            tc = _extract_inline_tool_call(text)
        arts = _artifacts(text)
    else:
        text = ''
        role = ''
        tc = None
        arts = full_arts

    # FIX: the local tool-call rescue used to only fire when `hosting is
    # None` -- i.e. when the message vanished from the compressed trace
    # with absolutely nothing at target_idx. But a decision-bearing
    # message almost always DOES leave *something* behind, a stub (see
    # _make_stub), which is a synthetic dict with no 'tool_call' key at
    # all. That stub still counts as `hosting is not None`, so the old
    # condition silently skipped the rescue exactly when it was needed
    # most: for an 'action' (tool-call) decision whose hosting message
    # was compressed down to a stub, tc stayed None for the rest of this
    # function's life. Widening this to also cover the stub case gives a
    # second, independent shot at recovering a real inline tool call from
    # the small neighborhood around target_idx before falling back.
    if not tc and decision['type'] == 'action' and (hosting is None or is_stub):
        # Scoped to the decision's own neighborhood, NOT the full trace --
        # see _local_text_for_rescue's docstring for why this matters.
        tc = _extract_inline_tool_call(_local_text_for_rescue(compressed_msgs, target_idx, radius=1))

    gt_target = decision.get('target')
    gt_action = decision.get('action')

    if target_still_recoverable(gt_target, text, arts) or \
       target_still_recoverable(gt_target, full_text, full_arts):
        pinned_target = gt_target
    else:
        pinned_target = None

    if action_still_recoverable(gt_action, text) or action_still_recoverable(gt_action, full_text):
        pinned_action = gt_action
    else:
        pinned_action = None

    fresh_rationale = _extract_rationale(
        full_text, full_arts, decision_values=this_decision_values,
        decision_idx=decision['msg_idx'])
    pinned_rationale = [
        r for r in (decision.get('rationale') or [])
        if target_still_recoverable(_bare_rationale_value(r), full_text, full_arts)
    ]
    seen, rationale = set(), []
    for r in pinned_rationale + fresh_rationale:
        bare = _bare_rationale_value(r)
        if bare not in seen:
            seen.add(bare)
            rationale.append(r)

    dm = _find_decisive_match(text) if text else None
    decisive_span = (dm.start(), dm.end()) if dm is not None else None

    # A stub is a degraded reconstruction just like hosting-is-None was --
    # penalize confidence accordingly instead of treating "found a stub"
    # as equivalent to "found the real message".
    conf_penalty = 0.0 if (hosting is not None and not is_stub) else 0.15

    # FIX: when the hosting message is genuinely absent (hosting is None),
    # scan_text used to fall back to the ENTIRE compressed trace
    # (`full_text`), which then got plugged in as if it were the single
    # message living at target_idx (see the judgment re-extraction branch
    # below). That let _build_decision_for_message pick up a decisive verb
    # belonging to a completely different decision anywhere else in the
    # trace and misattribute it to this one. Use the same bounded
    # neighborhood _local_text_for_rescue already uses for tool-call
    # rescue instead -- consistent scoping across every rescue path in
    # this function.
    if text:
        scan_text = text
    else:
        scan_text = _local_text_for_rescue(compressed_msgs, target_idx, radius=2) or full_text

    if tc and decision['type'] == 'action':
        tc_name = _get_tool_call_name(tc, decision.get('action', ''))
        tc_args = _get_tool_call_args(tc)
        return {
            'action': pinned_action if pinned_action is not None else tc_name,
            'target': pinned_target if pinned_target is not None else
                _extract_target(tc_args, scan_text, tc_name, decision_idx=target_idx, decisive_span=decisive_span),
            'rationale': rationale,
            'confidence': (0.9 if pinned_target is not None else 0.85) - conf_penalty,
            '_success': True, '_fallback': 'deterministic',
            '_target_source': 'pinned' if pinned_target is not None else 're_derived',
            '_raw': 'deterministic_fallback' if hosting is not None else 'deterministic_fallback_rescued',
        }

    # FIX: 'action'-type decisions with NO recovered tool_call (tc is
    # None -- the rescue above found nothing real either) used to fall
    # straight through to the generic "partial" bucket at the bottom of
    # this function, which returns action=None even when pinned_action
    # WAS actually available (e.g. the stub now embeds 'action=<tool_name>'
    # literally -- see _make_stub). Handle that case explicitly: if either
    # the action or the target is genuinely recoverable from surviving
    # text, return a proper result carrying whichever pieces ARE
    # recoverable, instead of silently downgrading to the no-action
    # bucket just because the structural tool_call dict didn't survive.
    if decision['type'] == 'action' and (pinned_action is not None or pinned_target is not None):
        return {
            'action': pinned_action,
            'target': pinned_target if pinned_target is not None else
                _extract_target({}, scan_text, decision_idx=target_idx, decisive_span=decisive_span),
            'rationale': rationale,
            'confidence': (0.8 if pinned_target is not None else 0.7) - conf_penalty,
            '_success': True, '_fallback': 'deterministic',
            '_target_source': 'pinned' if pinned_target is not None else 're_derived',
            '_raw': 'deterministic_fallback_stub_pinned',
        }

    if decision['type'] in ('judgment', 'confirmation'):
        if decision['type'] == 'confirmation':
            # Ground truth NEVER derives a confirmation's action from
            # _extract_verb/the builder chain -- every builder that
            # produces type='confirmation' in extraction.py hardcodes the
            # literal string 'confirm'. There is nothing to "recover"
            # here; the answer is always 'confirm'.
            action = pinned_action if pinned_action is not None else 'confirm'
        else:
            if pinned_action is not None:
                action = pinned_action
            else:
                reextracted = None
                if scan_text:
                    idx_messages = _reconstruct_positional_messages(
                        compressed_msgs, target_idx,
                        override_text=scan_text, override_role=role or None)
                    reextracted = _build_decision_for_message(idx_messages, target_idx)
                if reextracted is not None and reextracted.get('action') and reextracted.get('type') == 'judgment':
                    action = reextracted['action']
                else:
                    action = 'decide'
        return {
            'action': action,
            'target': pinned_target if pinned_target is not None else
                _extract_target({}, scan_text, decision_idx=target_idx, decisive_span=decisive_span),
            'rationale': rationale,
            'confidence': (0.9 if pinned_target is not None else 0.80) - conf_penalty,
            '_success': True, '_fallback': 'deterministic',
            '_target_source': 'pinned' if pinned_target is not None else 're_derived',
            '_raw': 'deterministic_fallback' if hosting is not None else 'deterministic_fallback_rescued',
        }

    if pinned_action is not None or pinned_target is not None or rationale:
        return {
            'action': pinned_action, 'target': pinned_target, 'rationale': rationale,
            'confidence': 0.5 - conf_penalty,
            '_success': True, '_fallback': 'deterministic_partial',
            '_target_source': 'pinned' if pinned_target is not None else 'none',
            '_raw': 'deterministic_fallback_partial',
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

    _log_fallback_event(decision, result, llm)

    with _REPRO_CACHE_LOCK:
        if len(_REPRO_CACHE) >= _REPRO_CACHE_MAXSIZE:
            _REPRO_CACHE.pop(next(iter(_REPRO_CACHE)))
        _REPRO_CACHE[key] = copy.deepcopy(result)
    return result