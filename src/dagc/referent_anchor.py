"""
referent_anchor.py — Elided-object (zero-anaphora) referent recovery.

Additive, opt-in extension. Does not modify extraction.py, compressor.py,
utils.py, or any existing preserve_* pass in sv_dagc.py. Same shape as
rationale_ext.py: a closed-class structural cue + one independent
corroboration check, single bounded pass, deterministic, no LLM calls.

PROBLEM THIS CLOSES
--------------------
A short confirmation turn like "Rolled back." or "Deployed." has no
object in its own text — the object was named once, several turns
earlier, in the message that gave the original instruction ("Deploy
the payments service to prod."). extract_decisions() correctly finds
no target for the confirmation — there is nothing in it to find, that
is not an extraction bug. But the ORIGINAL instruction message often
has no regex-detectable artifact either (no path/id/number — just a
plain noun phrase), so nothing marks it as needing preservation, and
compress_dagc can drop it. Once that's gone, the confirmation's
referent is gone too, even though every individual scoring decision
along the way was locally reasonable.

APPROACH
--------
1. ANCHOR CANDIDATES — messages (any role) containing a directive verb
   followed by a grounded object noun phrase ("Deploy the payments
   service to prod."). Reuses extraction.py's own
   _find_decisive_match / _extract_decisive_object_phrase /
   _extract_all_object_phrases — the identical object-phrase logic
   extract_decisions already trusts for target scoring — plus a
   fallback for imperative sentences that don't hit a recognized
   judgment verb (reusing extraction.py's own imperative-mood test,
   just applied to the message's own first clause instead of the
   PRIOR message's, which is all _prior_message_is_imperative_directive
   already checks elsewhere).

2. ELLIPSIS CANDIDATES — messages whose ENTIRE text (after stripping
   whitespace) is a bare past-tense confirmation verb from a closed,
   enumerated list (rolled back, deployed, reverted, merged, fixed,
   ...), with no artifact of its own. This is deliberately narrow: it
   only ever fires on already-terse, single-clause confirmations, so
   it can't misfire on ordinary longer prose.

3. LINK — single forward pass over the raw trace. `last_anchor` tracks
   the most recently established object phrase; every ellipsis
   candidate is bound to it, bounded by `lookback_window` messages.
   Single-topic-locality assumption — same bounded, no-recompute
   discipline _verify_and_repair already uses for RCI edges.

4. STUB — if the referent isn't already recoverable in the compressed
   text (checked via utils.target_still_recoverable, not a fresh
   substring test), append a small "[referent: ...]" stub — same
   append-and-reattach convention as sv_dagc.py's own repair stubs.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

from .utils import _get_text, _artifacts, _tok, target_still_recoverable
from .extraction import (
    _find_decisive_match,
    _extract_decisive_object_phrase,
    _extract_all_object_phrases,
    _sentence_containing,
    _clause_is_interrogative,
    _clause_is_hedged,
    _SUBJECT_LEADING_WORDS,
)


# --- Ellipsis: closed class -------------------------------------------

_TERSE_CONFIRM_VERBS = (
    r'rolled\s*back', r'reverted', r'deployed', r'merged', r'resolved',
    r'fixed', r'patched', r'restarted', r'applied', r'removed', r'deleted',
    r'cancell?ed', r'approved', r'rejected', r'restored', r'reset',
    r'cleared', r'escalated', r'closed', r'shipped', r'released',
    r'pushed', r'completed', r'finished', r'done', r'acknowledged',
    r'undone', r'renamed', r'migrated', r'provisioned', r'switched',
)
_RE_ELLIPSIS_CONFIRM = re.compile(
    r'^\s*(?:' + '|'.join(_TERSE_CONFIRM_VERBS) + r')\b'
    r'\s*(?:it|that|this|back|out|up|them)?\.?\s*$',
    re.IGNORECASE,
)


def _is_elliptical_confirmation(text: str, arts: Dict) -> bool:
    """Narrow, closed-class structural match: the ENTIRE message is a
    bare past-tense confirmation with no object of its own. `arts` is
    checked too so a message that happens to carry a real artifact
    ("Deployed to worker-07.") is never treated as elided — it already
    has its own referent, this pass has nothing useful to add."""
    stripped = text.strip()
    if not _RE_ELLIPSIS_CONFIRM.match(stripped):
        return False
    return not (arts.get('paths') or arts.get('ids') or arts.get('numbers')
                or arts.get('urls') or arts.get('emails'))


# --- Anchor: reuse extraction.py's own object-phrase logic -------------

def _is_imperative_sentence(text: str) -> bool:
    """Same imperative-mood test extraction.py already applies to the
    PRIOR message via _prior_message_is_imperative_directive, applied
    here to THIS message's own first clause instead."""
    first_clause = re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)[0]
    if _clause_is_interrogative(first_clause) or _clause_is_hedged(first_clause):
        return False
    first_word = re.match(r"^[A-Za-z']+", first_clause)
    if not first_word:
        return False
    return first_word.group(0).lower() not in _SUBJECT_LEADING_WORDS


def _anchor_object_phrase(text: str) -> Optional[str]:
    """The grounded object phrase this message establishes, if any."""
    dm = _find_decisive_match(text)
    if dm is not None:
        span = (dm.start(), dm.end())
        phrase = _extract_decisive_object_phrase(text, span)
        if phrase:
            return phrase
        sentence = _sentence_containing(text, span[0], span[1])
        phrases = _extract_all_object_phrases(sentence)
        return phrases[0] if phrases else None

    # Fallback: no recognized judgment/action verb, but the sentence is
    # structurally imperative ("Spin up an EC2 instance called
    # worker-07", "Create a new branch called X").
    if not _is_imperative_sentence(text):
        return None
    first_clause = re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)[0]
    phrases = _extract_all_object_phrases(first_clause)
    return phrases[0] if phrases else None


# --- Linking: single bounded forward pass -------------------------------

def find_referent_links(messages: List[Dict],
                         lookback_window: Optional[int] = 30) -> List[Dict]:
    """One pass, O(n). Tracks the most recently established object
    phrase and links every elliptical confirmation to it, provided the
    anchor is within `lookback_window` messages (None = unbounded).
    Same bounded, single-pass discipline as sv_dagc._verify_and_repair.

    Returns a list of {'ellipsis_idx', 'anchor_idx', 'referent'}.
    """
    links: List[Dict] = []
    last_anchor: Optional[Tuple[int, str]] = None

    for idx, msg in enumerate(messages):
        text = _get_text(msg)
        if not text.strip():
            continue
        arts = _artifacts(text)

        if last_anchor is not None:
            anchor_idx, anchor_phrase = last_anchor
            within_window = (lookback_window is None
                              or idx - anchor_idx <= lookback_window)
            if (within_window
                    and _is_elliptical_confirmation(text, arts)
                    and anchor_phrase.lower() not in text.lower()):
                links.append({
                    'ellipsis_idx': idx,
                    'anchor_idx': anchor_idx,
                    'referent': anchor_phrase,
                })

        phrase = _anchor_object_phrase(text)
        if phrase:
            last_anchor = (idx, phrase)

    return links


# --- Stub injection -------------------------------------------------------

def inject_referent_stubs(compressed: List[Dict], messages: List[Dict],
                           max_stubs: int = 10,
                           max_stub_tokens: Optional[float] = 20,
                           lookback_window: Optional[int] = 30) -> Tuple[List[Dict], Dict]:
    """Additive pass: for each elliptical-confirmation -> anchor link,
    if the referent isn't already recoverable in the compressed text,
    append a small stub. Never removes or alters existing content
    beyond reattaching to a message that already survived."""
    links = find_referent_links(messages, lookback_window=lookback_window)

    comp_text = ' '.join(_get_text(m) for m in compressed)
    added: List[Dict] = []

    for link in links:
        if len(added) >= max_stubs:
            break
        referent = link['referent']
        if target_still_recoverable(referent, comp_text):
            continue  # already survived compression near this message

        content = f"[referent: {referent}]"
        if max_stub_tokens is not None and _tok(content) > max_stub_tokens:
            continue

        owner_idx = link['ellipsis_idx']
        owner_role = messages[owner_idx].get('role', 'assistant')

        reattached = False
        for mc in compressed:
            if mc.get('_orig_idx') == owner_idx:
                mc['content'] = (mc.get('content', '') or '') + f' {content}'
                reattached = True
                break
        if not reattached:
            compressed.append({
                'role': owner_role,
                '_orig_idx': owner_idx,
                '_referent_stub': True,
                'content': content,
            })

        comp_text += ' ' + content
        added.append(link)

    report = {
        'referent_candidates_found': len(links),
        'referent_stubs_added': len(added),
        'referent_added_detail': added,
    }
    return compressed, report


__all__ = ["find_referent_links", "inject_referent_stubs"]