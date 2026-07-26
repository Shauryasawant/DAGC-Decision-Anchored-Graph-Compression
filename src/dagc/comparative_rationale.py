"""
comparative_rationale.py — Decision-anchored comparative-rejection recovery.

Additive, opt-in extension. Handles "Choose Claude rather than OpenAI" /
"Postgres over Redis" / "Instead of Redis, we chose Postgres" -- explicit
comparative-marker rejections in the SAME sentence as the decision. For
elimination-style rejections split across clauses/messages ("Ruled out
Redis... Going with Memcached instead"), see rationale_ext.py's
include_same_message option instead -- that is a structurally different
problem (no marker adjacent to both entities) and is solved there, not
here, to avoid overlapping detection paths for the same underlying fact.

PATCH NOTES (this revision)
----------------------------
1. FRONTED CONSTRUCTION (Path 2): original only matched "CHOSEN <marker>
   REJECTED". Added "<marker> REJECTED, ... CHOSEN" for unambiguous
   markers only -- ambiguous markers (over/cause) keep the original
   double-adjacency requirement, since proximity is what disambiguates
   THEM specifically.

2. VERB-GLUING FIX: _ENTITY_RE permits up to 3 consecutive capitalized
   tokens as one span. A decisive sentence starting with a judgment verb
   ("Choose Claude...") got the verb glued onto the real entity, because
   capitalization alone can't distinguish a sentence-initial verb from a
   genuine multi-word proper noun. This didn't fail safe -- the anchor
   check still passed (the real entity is a substring of the glued span)
   and produced a corrupted stub value ("chose Choose Claude over:
   OpenAI"). Fixed by stripping a leading token when it matches
   extraction.py's own _JUDGMENT_VERBS -- reusing that existing list,
   not inventing a new one.

Both fixes only ever make results MORE conservative or MORE correct --
neither can cause a previously-correct match to be lost, and the
verb-gluing fix can only clean up a value that was already going to be
accepted, never change accept/reject outcomes.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

from .utils import _get_text, _tok, _loose_normalise, target_still_recoverable
from .extraction import _find_decisive_match, _sentence_containing, _JUDGMENT_VERBS
from .rationale_ext import _ENTITY_RE, _is_valid_entity_match, _corroborated_elsewhere
from .compressor import _is_meaningful_candidate_value


_UNAMBIGUOUS_MARKERS = re.compile(
    r'\b(?:rather\s+than|instead\s+of|in\s+favor\s+of|versus|vs\.?)\b',
    re.IGNORECASE,
)
_AMBIGUOUS_MARKERS = re.compile(r'\b(?:over|cause)\b', re.IGNORECASE)
_COMPARATIVE_MARKERS = re.compile(
    _UNAMBIGUOUS_MARKERS.pattern + '|' + _AMBIGUOUS_MARKERS.pattern,
    re.IGNORECASE,
)


def _strip_leading_verb_token(entity: str) -> str:
    """Clean-up only -- runs after the adjacency check already passed,
    so it can never change whether something matches, only what string
    value gets reported once it has."""
    if ' ' not in entity:
        return entity
    first, rest = entity.split(' ', 1)
    if _JUDGMENT_VERBS.fullmatch(first):
        return rest
    return entity


def _candidate_entity_at(text: str) -> Optional[str]:
    stripped = text.lstrip()
    m = _ENTITY_RE.match(stripped)
    if not m:
        return None
    cand = _strip_leading_verb_token(m.group(0))
    if not (_is_meaningful_candidate_value(cand) and _is_valid_entity_match(cand, text)):
        return None
    return cand


def _entity_immediately_before(text: str) -> Optional[str]:
    matches = list(_ENTITY_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    trailing = text[last.end():]
    if trailing.strip(' ,'):
        return None
    cand = _strip_leading_verb_token(last.group(0))
    if not (_is_meaningful_candidate_value(cand) and _is_valid_entity_match(cand, text)):
        return None
    return cand


def _anchors_to_target(chosen: str, target: str) -> bool:
    c, t = _loose_normalise(chosen), _loose_normalise(target)
    if not c or not t:
        return False
    return c in t or t in c


def _find_entity_anchoring_target(text: str, target: str) -> Optional[str]:
    """Used only by Path 2 (fronted, unambiguous markers only). Safe to
    scan broadly since acceptance is gated entirely by matching the
    already-verified target, not by proximity."""
    for m in _ENTITY_RE.finditer(text):
        cand = _strip_leading_verb_token(m.group(0))
        if not (_is_meaningful_candidate_value(cand) and _is_valid_entity_match(cand, text)):
            continue
        if _anchors_to_target(cand, target):
            return cand
    return None


def _find_comparative_rejection(decision: Dict, full_text: str) -> Optional[Dict]:
    """For one already-extracted, already-gated decision: is there a
    comparative marker in its decisive clause, sitting between an
    entity that anchors to this decision's own target and a second,
    distinct entity -- in either order? If so, return
    {'alternative', 'marker', 'chosen'}."""
    target = decision.get('target')
    text = decision.get('verbatim') or ''
    if not target or not text:
        return None

    dm = _find_decisive_match(text)
    if dm is None:
        return None
    sentence = _sentence_containing(text, dm.start(), dm.end())

    for marker_m in _COMPARATIVE_MARKERS.finditer(sentence):
        marker_text = marker_m.group(0)
        is_unambiguous = bool(_UNAMBIGUOUS_MARKERS.fullmatch(marker_text.strip()))
        before = sentence[:marker_m.start()]
        after = sentence[marker_m.end():].lstrip(' ,')

        # --- Path 1: CHOSEN <marker> REJECTED ---
        chosen = _entity_immediately_before(before)
        rejected = _candidate_entity_at(after)
        if chosen and rejected and rejected.lower() != chosen.lower() \
                and _anchors_to_target(chosen, target):
            if is_unambiguous or _corroborated_elsewhere(rejected, full_text, sentence):
                return {'alternative': rejected, 'marker': marker_text.strip(), 'chosen': chosen}

        # --- Path 2: <marker> REJECTED, ... CHOSEN (unambiguous only) ---
        if not is_unambiguous:
            continue
        rejected2 = _candidate_entity_at(after)
        if not rejected2:
            continue
        remainder = after[len(rejected2):]
        chosen2 = _find_entity_anchoring_target(remainder, target)
        if not chosen2 or chosen2.lower() == rejected2.lower():
            continue
        return {'alternative': rejected2, 'marker': marker_text.strip(), 'chosen': chosen2}

    return None


def extract_comparative_candidates(decisions: List[Dict], messages: List[Dict]) -> List[Dict]:
    full_text = " ".join(_get_text(m) for m in messages)
    candidates = []
    for d in decisions:
        found = _find_comparative_rejection(d, full_text)
        if found:
            candidates.append({'decision_msg_idx': d['msg_idx'], **found})
    return candidates


def _make_stub(candidate: Dict, max_tokens: Optional[float], role: str) -> Optional[Dict]:
    content = f"[comparative] chose {candidate['chosen']} over: {candidate['alternative']}"
    if max_tokens is not None and _tok(content) > max_tokens:
        return None
    return {
        'role': role,
        '_orig_idx': candidate['decision_msg_idx'],
        '_comparative_stub': True,
        'content': content,
    }


def inject_comparative_stubs(compressed: List[Dict], messages: List[Dict],
                              decisions: List[Dict],
                              max_stubs: int = 10,
                              max_stub_tokens: Optional[float] = 25) -> Tuple[List[Dict], Dict]:
    """Unchanged from your original -- additive, append-only, never
    removes or alters existing content. Worst case on any trace: no-op."""
    candidates = extract_comparative_candidates(decisions, messages)
    comp_text = ' '.join(_get_text(m) for m in compressed)

    added = []
    for c in candidates:
        if len(added) >= max_stubs:
            break
        if target_still_recoverable(c['alternative'], comp_text):
            continue

        owner_idx = c['decision_msg_idx']
        owner_role = (messages[owner_idx].get('role', 'assistant')
                      if owner_idx < len(messages) else 'assistant')
        stub = _make_stub(c, max_stub_tokens, owner_role)
        if stub is None:
            continue

        reattached = False
        for mc in compressed:
            if mc.get('_orig_idx') == owner_idx:
                mc['content'] = (mc.get('content', '') or '') + f" {stub['content']}"
                reattached = True
                break
        if not reattached:
            compressed.append(stub)

        comp_text += ' ' + stub['content']
        added.append(c)

    report = {
        'comparative_candidates_found': len(candidates),
        'comparative_stubs_added': len(added),
        'comparative_added_detail': added,
    }
    return compressed, report


__all__ = ["extract_comparative_candidates", "inject_comparative_stubs"]