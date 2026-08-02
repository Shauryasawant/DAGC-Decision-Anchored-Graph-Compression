"""
value_recovery_ext.py -- generic, additive recovery pass for numeric
values (dollar amounts) present in raw messages but not recoverable
anywhere in the compressed output.

Does not touch extract_decisions(), _select_priority_content(),
rationale_ext.py, anchor_lifecycle.py, or referent_anchor.py. Purely
additive, same discipline as those modules: bounded by max_stubs and
max_stub_tokens, checks recoverability via utils._value_still_recoverable
before adding anything, never removes or alters existing compressed
content.

PROBLEM THIS CLOSES
--------------------
Messages like tau_airline_0078's message [37] (per-reservation savings
estimates: $6,654, $3,930, $5,478, $2,482, $5,216, $23,760) present
dollar figures in a proposal awaiting confirmation. These are distinct
values from the later CONFIRMED refund amounts, so they are not
recoverable via substring match on the compressed decision text and
are silently lost.

This is deliberately NOT a decision-extraction fix -- a proposal is
not a committed decision, and _OUTCOME_CONFIRM_SIGNALS /
extract_decisions() stay scoped to committed outcomes only. This
module treats uncaptured figures as evidence: worth preserving if
budget allows, never competing with real decisions for priority.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

from .utils import _get_text, _tok, _value_still_recoverable

_RE_CURRENCY_VALUE = re.compile(
    r'(?:\$|€|£|¥|₹|₩|₽|₺|R\$|C\$|A\$)\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d{1,2})?'
    r'|(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d{1,2})?\s?(?:USD|EUR|GBP|JPY|INR|CNY|AUD|CAD)\b'
)


def find_uncaptured_values(messages: List[Dict], compressed: List[Dict],
                            min_value_len: int = 2) -> List[Dict]:
    """Scan every original message for $-amounts; return those not
    already recoverable anywhere in the current compressed output.
    min_value_len filters trivial matches (e.g. '$5') by requiring at
    least this many digits before any decimal/comma -- keeps boilerplate
    noise like generic '$50' fee lines from flooding stubs while still
    catching real figures."""
    comp_text = ' '.join(_get_text(m) for m in compressed)
    found: List[Dict] = []
    seen_values: set = set()

    for idx, msg in enumerate(messages):
        text = _get_text(msg)
        for m in _RE_CURRENCY_VALUE.finditer(text):
            value = m.group(0)
            digits = re.sub(r'[^\d]', '', value.split('.')[0])
            if len(digits) < min_value_len:
                continue
            if value in seen_values:
                continue
            if _value_still_recoverable(value, comp_text):
                continue
            seen_values.add(value)
            found.append({'msg_idx': idx, 'value': value})

    return found


def inject_value_recovery_stubs(compressed: List[Dict], messages: List[Dict],
                                 max_stubs: int = 15,
                                 max_stub_tokens: Optional[float] = 25
                                 ) -> Tuple[List[Dict], Dict]:
    """Additive pass: append a compact stub for each still-missing $
    value, up to max_stubs. Reattaches to an existing compressed
    message from the same source index when one is present, otherwise
    appends a small new stub message -- same convention as
    referent_anchor.inject_referent_stubs."""
    candidates = find_uncaptured_values(messages, compressed)
    comp_text = ' '.join(_get_text(m) for m in compressed)
    added: List[Dict] = []

    by_msg: Dict[int, List[str]] = {}
    for c in candidates:
        by_msg.setdefault(c['msg_idx'], []).append(c['value'])

    for msg_idx, values in by_msg.items():
        if len(added) >= max_stubs:
            break
        remaining = [v for v in values if not _value_still_recoverable(v, comp_text)]
        if not remaining:
            continue
        content = f"[values] {', '.join(remaining)}"
        if max_stub_tokens is not None and _tok(content) > max_stub_tokens:
            kept: List[str] = []
            for v in remaining:
                trial = f"[values] {', '.join(kept + [v])}"
                if _tok(trial) > max_stub_tokens:
                    break
                kept.append(v)
            if not kept:
                continue
            content = f"[values] {', '.join(kept)}"
            remaining = kept

        reattached = False
        for mc in compressed:
            if mc.get('_orig_idx') == msg_idx:
                mc['content'] = (mc.get('content', '') or '') + f' {content}'
                reattached = True
                break
        if not reattached:
            compressed.append({
                'role': messages[msg_idx].get('role', 'assistant'),
                '_orig_idx': msg_idx,
                '_value_recovery_stub': True,
                'content': content,
            })

        comp_text += ' ' + content
        added.append({'msg_idx': msg_idx, 'values': remaining})

    return compressed, {
        'value_recovery_candidates_found': len(by_msg),
        'value_recovery_stubs_added': len(added),
        'value_recovery_added_detail': added,
    }


__all__ = ["find_uncaptured_values", "inject_value_recovery_stubs"]