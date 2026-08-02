"""
rationale_ext.py — PATCH NOTES (this revision)
=================================================

FIX 1 (bug fix, ALWAYS ON, no flag): sentence-initial reject-verb
self-match. A clause like "Ruled out Redis" has "Ruled" capitalized
only because it starts the clause -- not adjacent to "Redis" so it
doesn't get glued into one span, it stands alone as a second, bogus
"entity" candidate. This is a pre-existing bug, not introduced by this
patch -- it was just never exercised until a sentence-initial
reject-verb clause was scanned. Fixed by excluding the negation/
reject-verb match's own span from candidate entities. This can only
ever REMOVE incorrect candidates (never adds one), so it can only
help precision -- verify with a before/after benchmark run regardless,
per your own methodology, but it should not be able to regress DRR/RCI
in the direction of missing real decisions; the risk profile is
one-directional.

FIX 2 (new capability, OPT-IN, default OFF): same-message backward
clause scan. extract_rationale_candidates only ever looked at messages
strictly BEFORE the decision's own message (range(lo, d_idx) excludes
d_idx). A same-message elimination phrase -- "Ruled out Redis... Going
with Memcached instead." -- was invisible regardless of window size,
even with an explicit reject verb sitting right there. New parameter
include_same_message=False (default) preserves EXACT existing output
for every current caller, at every min_confidence setting -- this is
not a behavior change unless explicitly requested. When enabled, it
scans clauses of the decision's own message that occur strictly before
the decisive clause, reusing _scan_clauses_for_rationale (and therefore
_negation_scope_entities / _corroborated_elsewhere) completely
unchanged -- the only difference is which text gets fed to it. Uses
exact character-offset slicing of the original message text (not a
rejoin of stripped clause fragments) so corroboration's substring
matching stays correct.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Set, Tuple

from .utils import _tok, _get_text, _value_still_recoverable, _RE_CODE_FENCE
from .compressor import _is_meaningful_candidate_value

try:
    from nltk.corpus import stopwords
    _STOPWORDS = {word.upper() for word in stopwords.words('english')}
except (ImportError, LookupError):
    _STOPWORDS = {
        'NOT', 'AND', 'OR', 'THE', 'A', 'AN', 'IS', 'ARE', 'OF', 'TO', 'IN',
    }


def _is_valid_entity_match(token: str, context: str = '') -> bool:
    del context
    entity = token.strip()
    if ' ' in entity or '-' in entity:
        return True
    return entity.upper() not in _STOPWORDS


_NEGATION_RE = re.compile(
    r"""
    \b(?:
        not|n't|never|without|no|
        unable\s+to|fail(?:s|ed|ing)?\s+to
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)
_REJECT_VERB_RE = re.compile(
    r"""\b(?:
        reject(?:s|ed|ing)?|
        pass(?:es|ed|ing)?\s+on|
        skip(?:s|ped|ping)?|
        rule(?:s|d)?\s+out
    )\b""",
    re.VERBOSE | re.IGNORECASE,
)

CAUSAL_MARKERS = [
    r"\bbecause\b", r"\bsince\b", r"\bdue to\b", r"\bas it\b",
    r"\bgiven that\b", r"\bso that\b", r"\bin order to\b", r"\bas\b",
]
_CAUSAL_RE = re.compile("|".join(CAUSAL_MARKERS), re.IGNORECASE)

_CLAUSE_SPLIT_RE = re.compile(
    r"""
    (?<=[.!?;])\s+
    | ,\s+(?=\w)
    | \n(?=\s*\#{1,6}\s)
    | \n(?=\s*[-*]\s)
    | \n(?=\s*\d{1,3}[\.\)]\s)
    | \n(?=\s*\|)
    | \n{2,}
    """,
    re.VERBOSE,
)
_ENTITY_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9\-]{1,20}(?:\s+[A-Z][A-Za-z0-9\-]{1,20}){0,2}\b"
)


def _clauses(text: str) -> List[str]:
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]

def _clauses_fence_aware(text: str) -> List[str]:
    """Same purpose as _clauses(), but treats each fenced code block
    (```...```) as a single atomic clause instead of comma/newline-
    splitting inside it. Mirrors _split_sents' fence handling in
    utils.py. Used only by the filler-filter call sites in
    filler_score.py, which is the path where clause-splitting inside
    code was silently deleting fragments (e.g. a comma-split kwarg).
    _clauses()/_clauses_with_offsets() are left untouched for all other
    callers (convmem.py topic classification, _scan_clauses_for_rationale)."""
    out: List[str] = []
    last_end = 0
    for fence_m in _RE_CODE_FENCE.finditer(text):
        out.extend(c.strip() for c in _CLAUSE_SPLIT_RE.split(text[last_end:fence_m.start()]) if c.strip())
        fence_body = fence_m.group(0).strip()
        if fence_body:
            out.append(fence_body)
        last_end = fence_m.end()
    out.extend(c.strip() for c in _CLAUSE_SPLIT_RE.split(text[last_end:]) if c.strip())
    return out


def _clauses_with_offsets(text: str) -> List[Tuple[int, int]]:
    """Same segmentation as _clauses(), but returns (start, end)
    character offsets into the ORIGINAL text instead of stripped,
    discarded-separator strings -- needed so a prior-clause slice stays
    an exact substring of the original message, which
    _corroborated_elsewhere's exact-substring .replace() depends on."""
    spans = []
    last = 0
    for m in _CLAUSE_SPLIT_RE.finditer(text):
        spans.append((last, m.start()))
        last = m.end()
    spans.append((last, len(text)))

    out = []
    for s, e in spans:
        seg = text[s:e]
        if not seg.strip():
            continue
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        out.append((s + lead, e - trail))
    return out


def _candidate_entities(clause: str, exclude_spans=()) -> List[str]:
    ents = []
    for m in _ENTITY_RE.finditer(clause):
        if any(s < m.end() and m.start() < e for s, e in exclude_spans):
            continue  # overlaps the negation/reject-verb cue itself
        ents.append(m.group(0))
    return [e for e in ents
            if _is_meaningful_candidate_value(e) and _is_valid_entity_match(e, clause)]


def _negation_scope_entities(clause: str) -> List[str]:
    """Entities that fall in the same clause as a negation cue OR an
    explicit rejection verb. FIX 1 applied here: the negation/reject-
    verb match's own span is excluded from candidate entities, so a
    sentence-initial reject verb ('Ruled out Redis') can never be
    mistaken for the rejected entity itself."""
    neg_m = _NEGATION_RE.search(clause)
    rej_m = _REJECT_VERB_RE.search(clause)
    if not (neg_m or rej_m):
        return []
    exclude = [m.span() for m in (neg_m, rej_m) if m]
    return _candidate_entities(clause, exclude_spans=exclude)


def _corroborated_elsewhere(entity: str, full_text: str, own_clause: str,
                             min_count: int = 1) -> bool:
    remainder = full_text.replace(own_clause, "", 1)
    return remainder.lower().count(entity.lower()) >= min_count


def _scan_clauses_for_rationale(msg_idx: int, text: str, full_text: str) -> List[Dict]:
    out = []
    clauses = _clauses(text)
    for ci, cl in enumerate(clauses):
        entities = _negation_scope_entities(cl)
        if not entities:
            continue
        causal_here = bool(_CAUSAL_RE.search(cl))
        causal_next = ci + 1 < len(clauses) and bool(_CAUSAL_RE.search(clauses[ci + 1]))
        has_causal_signal = causal_here or causal_next
        reason_clause = cl if causal_here else (clauses[ci + 1] if causal_next else cl)
        confidence = 'high' if has_causal_signal else 'medium'

        for ent in entities:
            if not has_causal_signal and not _corroborated_elsewhere(ent, full_text, cl):
                if ci + 1 < len(clauses) and _REJECT_VERB_RE.search(cl):
                    nxt = clauses[ci + 1]
                    is_short = len(nxt.split()) <= 8
                    nxt_has_own_reject = bool(
                        _NEGATION_RE.search(nxt) or _REJECT_VERB_RE.search(nxt)
                    )
                    if is_short and not nxt_has_own_reject:
                        out.append({
                            'source_msg_idx': msg_idx,
                            'alternative': ent,
                            'reason': nxt,
                            'confidence': 'medium',
                        })
                continue

            out.append({
                'source_msg_idx': msg_idx,
                'alternative': ent,
                'reason': reason_clause,
                'confidence': confidence,
            })
    return out


def _loose_normalise(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def _decisive_clause_start(text: str, target: Optional[str]) -> Optional[int]:
    """Character offset where the clause containing `target` begins.
    None if target is missing/not found, or is in the first clause
    (nothing precedes it to scan)."""
    if not target:
        return None
    t = _loose_normalise(target)
    spans = _clauses_with_offsets(text)
    for i, (s, e) in enumerate(spans):
        if t in _loose_normalise(text[s:e]):
            return None if i == 0 else s
    return None


def _same_message_prior_clause_candidates(msg_idx: int, text: str, target: Optional[str],
                                           full_text: str) -> List[Dict]:
    """FIX 2 internals. `prior_text` is an exact character slice of the
    original message (not a rejoin of stripped fragments), so clauses
    _scan_clauses_for_rationale extracts from it remain true substrings
    of full_text -- required for corroboration's exclusion to work."""
    start = _decisive_clause_start(text, target)
    if start is None:
        return []
    prior_text = text[:start]
    return _scan_clauses_for_rationale(msg_idx, prior_text, full_text)


def extract_rationale_candidates(messages: List[Dict], decisions: List[Dict],
                                  window: int = 3,
                                  include_same_message: bool = True) -> List[Dict]:
    """For each decision, scan the `window` messages immediately
    preceding it for negation-scope + corroborated-entity spans.

    include_same_message: default False -- preserves EXACT existing
    output for every current caller regardless of min_confidence. When
    True, also scans the decision's own message for clauses before the
    decisive one (catches same-message elimination phrasing like
    "Ruled out Redis... Going with Memcached instead" that the
    window-only scan structurally cannot reach). Purely additive: never
    removes or changes an existing candidate, only adds new ones,
    always at the same confidence rules everything else uses.
    """
    candidates = []
    decision_idxs = {d['msg_idx'] for d in decisions}
    full_text = " ".join(_get_text(m) for m in messages)

    for d in decisions:
        d_idx = d['msg_idx']
        lo = max(0, d_idx - window)
        for i in range(lo, d_idx):
            if i in decision_idxs:
                continue
            text = _get_text(messages[i])
            for cand in _scan_clauses_for_rationale(i, text, full_text):
                candidates.append({**cand, 'decision_msg_idx': d_idx})

        if include_same_message:
            own_text = _get_text(messages[d_idx])
            for cand in _same_message_prior_clause_candidates(
                    d_idx, own_text, d.get('target'), full_text):
                candidates.append({**cand, 'decision_msg_idx': d_idx})

    return candidates


def _make_stub(candidate: Dict, max_tokens: Optional[float], role: str = 'assistant') -> Optional[Dict]:
    alt, reason = candidate['alternative'], candidate['reason']
    if reason.strip() == alt.strip():
        content = f"[rationale] {alt}"
    else:
        content = f"[rationale] not: {alt} — reason: {reason}"
    if max_tokens is not None and _tok(content) > max_tokens:
        return None
    return {
        'role': role,
        '_orig_idx': candidate['source_msg_idx'],
        '_rationale_stub': True,
        'content': content,
    }


def _rationale_identity(alternative: str, reason: str) -> Tuple[str, str]:
    return (alternative.strip().lower(), reason.strip().lower())


def _fact_already_present(alternative: str, reason: str, comp_text: str) -> bool:
    return (_value_still_recoverable(alternative, comp_text)
            and _value_still_recoverable(reason, comp_text))


def inject_rationale_stubs(compressed: List[Dict], messages: List[Dict],
                            decisions: List[Dict], window: int = 3,
                            max_stub_tokens: Optional[float] = 40,
                            max_stubs_per_decision: int = 2,
                            min_confidence: str = 'high',
                            include_same_message: bool = False) -> Tuple[List[Dict], Dict]:
    """Unchanged behavior at defaults. include_same_message is the same
    opt-in flag as extract_rationale_candidates, threaded through here
    so callers don't have to call extraction manually to use FIX 2.
    """
    conf_rank = {'medium': 0, 'high': 1}
    threshold = conf_rank[min_confidence]

    comp_text = ' '.join(_get_text(m) for m in compressed)
    candidates = extract_rationale_candidates(
        messages, decisions, window=window, include_same_message=include_same_message)
    candidates = [c for c in candidates if conf_rank[c['confidence']] >= threshold]

    n_before_dedup = len(candidates)
    seen_facts: Set[Tuple[str, str]] = set()
    deduped: List[Dict] = []
    for c in candidates:
        key = _rationale_identity(c['alternative'], c['reason'])
        if key in seen_facts:
            continue
        seen_facts.add(key)
        deduped.append(c)
    candidates = deduped
    n_duplicates_dropped = n_before_dedup - len(candidates)

    added = []
    per_decision_count: Dict[int, int] = {}
    for c in candidates:
        if _fact_already_present(c['alternative'], c['reason'], comp_text):
            continue
        count = per_decision_count.get(c['decision_msg_idx'], 0)
        if count >= max_stubs_per_decision:
            continue
        stub = _make_stub(c, max_stub_tokens)
        if stub is None:
            continue
        compressed.append(stub)
        comp_text += ' ' + stub['content']
        added.append(c)
        per_decision_count[c['decision_msg_idx']] = count + 1

    report = {
        'rationale_candidates_found': len(candidates),
        'rationale_duplicate_candidates_dropped': n_duplicates_dropped,
        'rationale_stubs_added': len(added),
        'rationale_added_detail': added,
    }
    return compressed, report

def inject_dropped_rationale_stubs(compressed: List[Dict], messages: List[Dict],
                                   max_stub_tokens: Optional[float] = 30,
                                   max_stubs_total: int = 10,
                                   min_confidence: str = 'high') -> Tuple[List[Dict], Dict]:
    """Add stubs for corroborated rejected alternatives in fully dropped messages.

    This complements :func:`inject_rationale_stubs`: unlike that pass it
    needs no nearby extracted decision, but considers only original messages
    with no ``_orig_idx`` representation in the compressed trace.
    """
    conf_rank = {'medium': 0, 'high': 1}
    threshold = conf_rank[min_confidence]

    present = {m.get('_orig_idx') for m in compressed if '_orig_idx' in m}
    dropped_idxs = [i for i in range(len(messages)) if i not in present]
    if not dropped_idxs:
        return compressed, {
            'dropped_rationale_candidates_found': 0,
            'dropped_rationale_stubs_added': 0,
            'dropped_rationale_added_detail': [],
        }

    full_text = ' '.join(_get_text(m) for m in messages)
    comp_text = ' '.join(_get_text(m) for m in compressed)

    candidates = []
    for i in dropped_idxs:
        candidates.extend(_scan_clauses_for_rationale(i, _get_text(messages[i]), full_text))
    candidates = [c for c in candidates if conf_rank[c['confidence']] >= threshold]

    seen_facts: Set[Tuple[str, str]] = set()
    deduped = []
    for c in candidates:
        key = _rationale_identity(c['alternative'], c['reason'])
        if key in seen_facts:
            continue
        seen_facts.add(key)
        deduped.append(c)

    added = []
    for c in deduped:
        if len(added) >= max_stubs_total:
            break
        if _fact_already_present(c['alternative'], c['reason'], comp_text):
            continue
        alt, reason = c['alternative'], c['reason']
        content = (f'[dropped] contained: {alt}'
                   if reason.strip() == alt.strip()
                   else f'[dropped] contained: {alt}, {reason}')
        if max_stub_tokens is not None and _tok(content) > max_stub_tokens:
            continue
        stub = {
            'role': messages[c['source_msg_idx']].get('role', 'assistant'),
            '_orig_idx': c['source_msg_idx'],
            '_dropped_rationale_stub': True,
            'content': content,
        }
        compressed.append(stub)
        comp_text += ' ' + content
        added.append(c)

    return compressed, {
        'dropped_rationale_candidates_found': len(deduped),
        'dropped_rationale_stubs_added': len(added),
        'dropped_rationale_added_detail': added,
    }


