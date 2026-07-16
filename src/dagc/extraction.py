"""
Decision extractor: turns a raw agent trace into a list of structured
decisions (tool calls, judgments, confirmations) with extracted targets,
rationale, and cited artifacts. Pure regex/heuristic -- no LLM calls.
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .utils import (
    CRITICAL, STOPWORDS, _artifacts, _get_text, _tok, _uw,
)
_JUDGMENT_VERBS = re.compile(
    r'\b(recommend|conclude|suggest|decide|choose|select|prefer|'
    r'best|winner|optimal|final|confirmed?|'
    r'implement|adopt|use)\b', re.IGNORECASE)

_JUDGMENT_CONNECTIVE_WORDS = ('therefore', 'thus', 'hence')

_JUDGMENT_CONNECTIVES = re.compile(
    r'\b(' + '|'.join(_JUDGMENT_CONNECTIVE_WORDS) + r')\b', re.IGNORECASE)

_JUDGMENT_SIGNALS = re.compile(
    _JUDGMENT_VERBS.pattern[2:-2] + '|' + _JUDGMENT_CONNECTIVES.pattern[2:-2],
    re.IGNORECASE)

_STRONG_JUDGMENT_SIGNALS = re.compile(
    r'\b(recommend|conclude|decide|choose|select|winner|'
    r'therefore|thus|hence|adopt)\b', re.IGNORECASE)

_STRONG_DECISIVE_VERBS_NO_CONFIRM = re.compile(
    r'\b(recommend|conclude|decide|choose|select|winner|adopt)\b',
    re.IGNORECASE)

_STRONG_JUDGMENT_VERBS = re.compile(
    r'\b(recommend|conclude|decide|choose|select|winner|adopt)\b',
    re.IGNORECASE)

_CONFIRM_SIGNALS = re.compile(
    r'\b(confirm(?:ed|ation|s)?|verified?|preserv|ensur|kept?|maintain)\b', re.IGNORECASE)
_STRONG_DECISIVE_NO_CONFIRM = re.compile(
    r'\b(recommend|conclude|decide|choose|select|winner|therefore|thus|hence|adopt)\b',
    re.IGNORECASE)
_ACTION_VERB_PRIORITY = [
    'recommend', 'confirm', 'adopt', 'implement', 'select', 'choose',
    'decide', 'prefer', 'suggest', 'conclude', 'use',
    'best', 'optimal', 'winner', 'final',
]
_ACTION_DECISION_CUE = re.compile(
    r"\b(will|shall|should|must|going to|plan(?:s|ned)? to|"
    r"we(?:'re| are| will| have)|let's|going with)\b", re.IGNORECASE)

_ENTITY_BLOCKLIST = CRITICAL | STOPWORDS | {
    'evidence', 'confirmed', 'confirm', 'confirmation', 'confirms',
    'root',
    'winner', 'best', 'optimal', 'reading', 'comparing', 'report',
    'results', 'recommendation', 'recommended', 'implementing',
    'preserved', 'clear', 'metrics', 'detailed',
    'tool_call', 'tool_calls', 'function_call', 'tool_response', 'tool_result',
}
_RE_ENTITY = re.compile(r'\b[A-Z][A-Za-z0-9]{2,}\b')

_RE_WINNER_BEFORE = re.compile(
    r'\b(?:recommendation|winner|best|optimal|recommended|selected|chosen|'
    r'preferred|adopted|confirmed|approved|finalized|final\s+decision)'
    r'\W{0,8}([a-zA-Z0-9][a-zA-Z0-9._-]+(?:\s+[a-zA-Z0-9][a-zA-Z0-9._-]+)?)',
    re.I)
_RE_WINNER_AFTER = re.compile(
    r'([a-zA-Z0-9][a-zA-Z0-9._-]+(?:\s+[a-zA-Z0-9][a-zA-Z0-9._-]+)?)'
    r'\s+(?:is\s+(?:the\s+)?(?:clear\s+)?(?:winner|best|optimal|choice)'
    r'|wins\b|was\s+(?:selected|chosen|confirmed|approved|adopted))',
    re.I)
_RE_WINNER_INLINE = re.compile(
    r'\b(?:winner|best|recommended?|optimal)\b'
    r'[ \t]{0,3}[:\-]?[ \t]{0,3}'
    r'(?!\.)([A-Za-z][\w._-]{1,30})',
    re.I)
_RE_ENTITY_SNAKE = re.compile(
    r'\b[a-z][a-z0-9]*(?:[_-][a-z][a-z0-9.]*){1,}\b')

_RE_METRIC_KV_INLINE = re.compile(
    r'([A-Za-z][\w-]*)[ \t]*[=:][ \t]*'
    r'(\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\$?\d+(?:\.\d+)?%?)'
    r'(?!\s*\.\s)'
)

_RE_PRESERVED_TAG = re.compile(r'\[preserved:\s*([^\]]+)\]')
_RE_OWNER_SUFFIX = re.compile(r'^(.*?)#d([\d,]+)$')


_RE_PRESERVED_TAG = re.compile(r'\[preserved:\s*([^\]]+)\]')
_RE_OWNER_SUFFIX = re.compile(r'^(.*?)#d([\d,]+)$')
_RE_ENTRY_SPLIT = re.compile(r',\s+')


def _preserved_tag_candidates(text, decision_idx=None):
    """
    Parses '[preserved: value#dN, value2#dM,K]' tags. Entries are
    separated by comma+whitespace; an owner suffix's own id list uses
    bare commas with no space (e.g. '#d1,2' means owned by decisions 1
    AND 2) so it must never be split as if it were a new entry.
    Entries with a suffix are only returned when decision_idx matches
    one of the listed owners; entries with NO suffix (legacy/untagged)
    are always returned, since we can't tell whose they are and
    excluding them would silently drop real preserved values.
    """
    m = _RE_PRESERVED_TAG.search(text)
    if not m:
        return []
    out = []
    for raw in _RE_ENTRY_SPLIT.split(m.group(1)):
        raw = raw.strip()
        if not raw:
            continue
        owner_m = _RE_OWNER_SUFFIX.match(raw)
        if owner_m:
            value, owners = owner_m.group(1).strip(), owner_m.group(2)
            owner_ids = {int(o) for o in owners.split(',') if o}
            if decision_idx is not None and decision_idx not in owner_ids:
                continue
            out.append(value)
        else:
            out.append(raw)
    return out

def _extract_entity_target(text, known_ids=None):
    known_ids = known_ids or []
    id_prefixes = {i.split('-')[0] for i in known_ids if '-' in i}

    for pattern in (_RE_WINNER_BEFORE, _RE_WINNER_AFTER):
        for m in pattern.finditer(text):
            val = m.group(1).strip('.,;: ')
            if (val.lower() not in _ENTITY_BLOCKLIST
                    and val not in id_prefixes
                    and len(val.split()) <= 3
                    and len(val) >= 3):
                return val

    # A following gerund clause is often a better target than a nearby ID.
    verb_m = _find_strong_judgment_match(text)
    if verb_m:
        tail = text[verb_m.end():verb_m.end() + 80].strip()
        gerund_m = re.match(r'\s*([a-z]+ing\b[^.;]{0,60})', tail, re.I)
        if gerund_m:
            phrase = gerund_m.group(1).strip()
            if len(phrase.split()) >= 2:
                return _cap_target_length(phrase, max_words=8)

    camel = [w for w in _RE_ENTITY.findall(text)
             if w.lower() not in _ENTITY_BLOCKLIST and w not in id_prefixes]
    snake = [w for w in _RE_ENTITY_SNAKE.findall(text)
             if w.lower() not in _ENTITY_BLOCKLIST and w not in id_prefixes]

    camel_top = Counter(camel).most_common(1)[0] if camel else None
    snake_top = Counter(snake).most_common(1)[0] if snake else None

    if camel_top and snake_top:
        # Prefer the more frequently mentioned identifier.
        return snake_top[0] if snake_top[1] > camel_top[1] else camel_top[0]
    if camel_top:
        return camel_top[0]
    if snake_top:
        return snake_top[0]

    words = re.findall(r'\b[a-z][a-z0-9]{1,20}\b', text)
    bigrams = [f'{words[i]} {words[i+1]}' for i in range(len(words) - 1)
               if words[i] not in STOPWORDS and words[i + 1] not in STOPWORDS
               and words[i] not in _ENTITY_BLOCKLIST]
    if bigrams:
        top_bg, top_count = Counter(bigrams).most_common(1)[0]
        if top_count >= 2:
            return top_bg

    return None


def _cap_target_length(t, max_words=6):
    if not t:
        return t
    return t if len(t.split()) <= max_words else None


def _flatten_arg_values(obj, prefix='', max_depth=3, _depth=0):
    """Yield (key, value) leaf pairs from nested dict/list tool args."""
    out = []
    if _depth > max_depth:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_flatten_arg_values(v, k, max_depth, _depth + 1))
    elif isinstance(obj, list):
        for v in obj[:5]:
            out.extend(_flatten_arg_values(v, prefix, max_depth, _depth + 1))
    elif isinstance(obj, (str, int, float)):
        out.append((prefix, obj))
    return out


_FILLER_CONNECTORS = {
    'on', 'in', 'at', 'for', 'of', 'to', 'by', 'with', 'is', 'was', 'are',
    'a', 'an', 'the',
}
_FILLER_ACK_WORDS = {
    'great', 'sure', 'okay', 'ok', 'yes', 'no', 'thanks', 'certainly',
    'absolutely', 'perfect', 'awesome', 'alright', 'understood', 'done',
    'welcome', 'hello', 'hi', 'sorry', 'yep', 'nope',
}

# Stable target-key priority shared by extraction and evaluation.
_CANONICAL_TARGET_KEY_TIERS: Tuple[Tuple[str, ...], ...] = (
    ('new_', 'updated_', 'target_'),
    ('email',),
    ('account_id', 'client_id', 'customer_id', 'user_id', 'id'),
    ('confirmation', 'reference', 'booking', 'reservation', 'claim',
     'policy', 'invoice', 'ticket', 'tracking', 'pnr', 'mrn', 'npi',
     'record', 'case'),
    ('query', 'search_term'),
    ('transaction_id', 'target'),
    ('zip', 'zip_code', 'postal_code'),
    ('name',),
    ('options',),
    ('file', 'path'),
    ('experiment', 'metric'),
)


def _tokenize_key(key: str) -> List[str]:
    s = re.sub(r'[^A-Za-z0-9]+', '_', str(key))
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', s)
    return [t for t in s.lower().split('_') if t]


def _key_tier_rank(key: Optional[str]) -> Optional[int]:
    """Format-agnostic key-priority lookup: matches by SUBSTRING against
    each tier's tokens instead of requiring exact key equality."""
    if key is None:
        return None
    key_norm = re.sub(r'[^a-z0-9_]', '', str(key).lower())
    if not key_norm:
        return None
    for tier_idx, tokens in enumerate(_CANONICAL_TARGET_KEY_TIERS):
        for tok in tokens:
            if tok in key_norm:
                return tier_idx
    return None


def _stringify_arg_value(v):
    if isinstance(v, (dict, list)):
        s = json.dumps(v, separators=(',', ':'))
    else:
        s = str(v).strip()
    return s if len(s) >= 2 else None
# Do not split decimal values as sentence boundaries.
_RE_SENTENCE_END = re.compile(r'(?<!\d)([.!?])(?!\d)(?:\s+|$)|\n+')


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    start = 0
    for m in _RE_SENTENCE_END.finditer(text):
        end = m.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _sentence_containing(text: str, pos_start: int, pos_end: int) -> str:
    for s, e in _sentence_spans(text):
        if s <= pos_start < e:
            return text[s:e]
    # Use a small local window if no sentence span is found.
    return text[max(0, pos_start - 80):pos_end + 80]


def _connective_is_corroborated(text: str, m: 're.Match', verb_pattern) -> bool:
    """A connective match counts only if its own sentence also contains
    a real judgment verb or an explicit action cue."""
    sentence = _sentence_containing(text, m.start(), m.end())
    return bool(verb_pattern.search(sentence) or _ACTION_DECISION_CUE.search(sentence))


def _has_judgment_signal(text: str) -> bool:
    if _JUDGMENT_VERBS.search(text):
        return True
    return any(_connective_is_corroborated(text, m, _JUDGMENT_VERBS)
               for m in _JUDGMENT_CONNECTIVES.finditer(text))


def _has_strong_judgment_signal(text: str) -> bool:
    if _STRONG_JUDGMENT_VERBS.search(text):
        return True
    return any(_connective_is_corroborated(text, m, _STRONG_JUDGMENT_VERBS)
               for m in _JUDGMENT_CONNECTIVES.finditer(text))


def _has_strong_decisive_no_confirm(text: str) -> bool:
    if _STRONG_DECISIVE_VERBS_NO_CONFIRM.search(text):
        return True
    return any(_connective_is_corroborated(text, m, _STRONG_DECISIVE_VERBS_NO_CONFIRM)
               for m in _JUDGMENT_CONNECTIVES.finditer(text))


def _find_strong_judgment_match(text: str):
    """Returns the Match anchoring the strongest decisive signal, for
    callers (like _extract_entity_target) that need match.end() to slice
    the clause after it. Prefers a real verb; falls back to a
    corroborated connective; returns None if neither qualifies."""
    verb_m = _STRONG_JUDGMENT_VERBS.search(text)
    if verb_m:
        return verb_m
    for m in _JUDGMENT_CONNECTIVES.finditer(text):
        if _connective_is_corroborated(text, m, _STRONG_JUDGMENT_VERBS):
            return m
    return None

_RE_IDLIKE_VALUE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{2,29}$')


def _looks_identifier_shaped(cand: str) -> bool:
    """Value-shape prior, independent of key name or domain: short,
    no-whitespace, alphanumeric token with a digit reads as an
    identifier/code regardless of field name or business domain."""
    c = cand.strip()
    if not _RE_IDLIKE_VALUE.match(c):
        return False
    return bool(re.search(r'\d', c))


def _looks_structural_numeric(cand):
    """Bare small integers read as pagination/limit/offset-style params
    regardless of what key they're stored under."""
    return bool(re.fullmatch(r'-?\d{1,4}', cand.strip()))


def _is_sane_candidate(s: str) -> bool:
    return '\n' not in s and sum(c.isalnum() for c in s) >= len(s) * 0.4


def _score_target_candidate(cand, source, tool_name, text, key=None):
    low = cand.strip().lower()
    if len(low) < 2:
        return -1e9
    if tool_name and low == str(tool_name).strip().lower():
        return -1e9

    score = {'arg': 3.0, 'text_artifact': 2.0, 'text_entity': 1.0}.get(source, 0.0)

    from_stopwords = low in STOPWORDS or low in _FILLER_ACK_WORDS
    if from_stopwords:
        score -= 2.5
    if _looks_identifier_shaped(cand):
        score += 1.2
    if _looks_structural_numeric(cand):
        score -= 1.0
    a = _artifacts(cand)
    if a['ids'] or a['paths'] or a['urls'] or '@' in cand:
        score += 2.0

    score += min(1.5, len(cand) / 20.0)
    if ' ' in cand.strip():
        score += 0.3

    if key is not None:
        rank = _key_tier_rank(key)
        if rank is not None:
            n_tiers = len(_CANONICAL_TARGET_KEY_TIERS)
            score += 3.0 - (rank / n_tiers) * 2.0

    return score


def _extract_target(args, text, tool_name=None):
    candidates = []

    for k, v in _flatten_arg_values(args):
        cand = _stringify_arg_value(v)
        if cand:
            candidates.append((cand, 'arg', k))

    arts = _artifacts(text)
    ent = _extract_entity_target(text, known_ids=arts['ids'])
    if ent:
        candidates.append((ent, 'text_entity', None))
    for kind in ('ids', 'paths'):
        for a in arts[kind]:
            candidates.append((a, 'text_artifact', None))

    # Include plausible numeric identifiers such as ZIP and account codes.
    for a in arts['numbers']:
        a_clean = a.rstrip('%')
        if a_clean.isdigit() and 4 <= len(a_clean) <= 12:
            candidates.append((a_clean, 'text_artifact', None))

    # Include values retained by the compressor's preservation tags.
    for val in _preserved_tag_candidates(text):
        candidates.append((val, 'text_artifact', None))

    candidates = [c for c in candidates if _is_sane_candidate(c[0])]

    if not candidates:
        return None

    scored = [(_score_target_candidate(c, src, tool_name, text, key=k), c)
              for c, src, k in candidates]
    scored.sort(key=lambda x: -x[0])
    best_score, best_cand = scored[0]
    if best_score <= 0:
        return None
    return _cap_target_length(best_cand, max_words=12)

def _extract_verb(text):
    verb_matches = [g.lower() for g in _JUDGMENT_VERBS.findall(text)]
    if verb_matches:
        for v in _ACTION_VERB_PRIORITY:
            if v in verb_matches:
                return v
        return verb_matches[0]

    # Preservation tags can retain a decision verb after surrounding text is trimmed.
    for val in _preserved_tag_candidates(text):
        low = val.strip().lower()
        if _JUDGMENT_VERBS.fullmatch(low) or _JUDGMENT_CONNECTIVES.fullmatch(low):
            return low

    # Connectives count only when their sentence also signals a decision.
    for m in _JUDGMENT_CONNECTIVES.finditer(text):
        if _connective_is_corroborated(text, m, _JUDGMENT_VERBS):
            return m.group(1).lower()

    return 'decide'


def _is_meaningful_rationale_key(key: str) -> bool:
    """
    A rationale key/value is only worth keeping if it names something --
    not a function word that happened to sit next to a number or a
    'winner'-style signal. Centralizing this means every extraction loop
    in _extract_rationale (present or future) gets the same standard,
    instead of each loop carrying its own ad-hoc word list.
    """
    low = key.strip().lower()
    if len(low) < 2:
        return False
    if low in STOPWORDS or low in _FILLER_CONNECTORS or low in _FILLER_ACK_WORDS:
        return False
    return True


def _is_meaningful_candidate_value(val: str) -> bool:
    """
    Stricter gate for values entering critical-artifact sets directly
    (action verbs, resolved targets, rationale k=v values) rather than
    being scored/ranked candidates first. Reuses the stopword/filler
    check everything else already relies on, and additionally rejects
    bare short numerics that carry no identifying information on their
    own (a disk-size fragment like "14" or "08" is not reproducible
    evidence; "Mirror_1=14" is).
    """
    if not _is_meaningful_rationale_key(val):
        return False
    stripped = val.strip()
    if re.fullmatch(r'\d{1,3}%?', stripped):
        return False
    return True

def _bare_rationale_value(rat: str) -> str:
    """
    Strip whichever label prefix produced this rationale token (id:,
    winner:, preserved:, beats:, or key=value) down to just the
    substantive value -- so the SAME underlying fact isn't kept twice
    under two different extraction paths (e.g. 'id:X' and 'preserved:X'
    both surviving when X is genuinely a single fact, not two).
    """
    if ':' in rat:
        return rat.split(':', 1)[1].strip().lower()
    if '=' in rat:
        return rat.split('=', 1)[1].strip().lower()
    return rat.strip().lower()


def _format_rationale_entry(val: str, default_prefix: str = 'preserved') -> str:
    """Preserve an existing rationale tag prefix when present; otherwise
    fall back to the generic preserved marker used for compressor tags."""
    stripped = str(val).strip()
    if not stripped:
        return stripped
    if re.match(r'^[a-z][a-z0-9_-]*:', stripped, re.I):
        return stripped
    return f'{default_prefix}:{stripped}'

def _extract_rationale(text, arts, decision_values=None, decision_idx=None):
    """
    decision_values: optional set of this decision's own critical values,
        used as a fallback filter/recovery mechanism (see below).
    decision_idx: this decision's msg_idx, used to filter the
        compressor's [preserved: value#dN] tag by explicit ownership
        suffix when present -- the precise mechanism. decision_values
        remains as a secondary check for untagged/legacy preserved
        entries and for fact-recovery from surviving text.
    """
    rat = []

    for val in _preserved_tag_candidates(text, decision_idx=decision_idx):
        if decision_values is not None and val not in decision_values:
            continue
        rat.append(_format_rationale_entry(val, default_prefix='preserved'))

    for m in _RE_METRIC_KV_INLINE.finditer(text):
        if _is_meaningful_rationale_key(m.group(1)) and _is_meaningful_candidate_value(m.group(2)):
            rat.append(f'{m.group(1)}={m.group(2)}')
    for e in arts['errors'][:2]:
        rat.append(f'error:{e[:60]}')
    for i in arts['ids'][:3]:
        rat.append(f'id:{i}')
    for m in _RE_WINNER_INLINE.finditer(text):
        val = m.group(1).strip('.,;:')
        if _is_meaningful_rationale_key(val) and val.lower() not in _ENTITY_BLOCKLIST:
            rat.append(f'winner:{val}')
    for m in re.finditer(
            r'([A-Za-z][\w._-]{1,25})\s+(?:outperforms?|beats?|surpasses?)',
            text, re.I):
        if _is_meaningful_rationale_key(m.group(1)):
            rat.append(f'beats:{m.group(1)}')

    if decision_values:
        already_bare = {_bare_rationale_value(r) for r in rat}
        for v in decision_values:
            if not v or len(v) < 2:
                continue
            looks_like_fact = any(c.isdigit() for c in v) or _looks_identifier_shaped(v)
            if not looks_like_fact:
                continue
            bare_v = v.strip().lower()
            if bare_v in already_bare or v not in text:
                continue
            rat.append(f'value:{v}')
            already_bare.add(bare_v)

    seen, out = set(), []
    for r in rat:
        bare = _bare_rationale_value(r)
        if bare in seen:
            continue
        seen.add(bare)
        out.append(r)
    return out[:8]

# Tool-call accessors shared by the compressor and evaluator.

def _get_tool_call_args(tc: Optional[dict]) -> dict:
    """
    Handles both common tool_call shapes:
      Format 1: {"name": "tool_name", "args": {...}}
      Format 2: {"function": {"name": "tool_name", "arguments": "..."}}
    """
    if not isinstance(tc, dict):
        return {}
    if 'function' in tc and isinstance(tc['function'], dict):
        raw = tc['function'].get('arguments', '{}')
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw) if isinstance(raw, str) else {}
        except Exception:
            return {}
    return tc.get('args', {}) or {}


def _get_tool_call_name(tc: Optional[dict], default: str = 'unknown_tool') -> str:
    if not isinstance(tc, dict):
        return default
    if 'function' in tc and isinstance(tc['function'], dict):
        return tc['function'].get('name', default)
    return tc.get('name', default)


def extract_decisions(messages: List[Dict],
                       decision_roles: Tuple[str, ...] = ('assistant',)) -> List[Dict]:
    """
    Turn a raw agent trace into a list of structured decisions.
    decision_roles: which message roles are scanned for decisions.
    Defaults to ('assistant',) for chat-style traces. Multi-agent or
    orchestrator traces can pass e.g. ('assistant', 'agent', 'worker')
    or whatever role labels their adapter emits.
    """
    decisions = []
    for idx, msg in enumerate(messages):
        if msg.get('role') not in decision_roles:
            continue
        text = _get_text(msg)
        arts = _artifacts(text)
        if isinstance(msg.get('tool_call'), dict):
            tc = msg['tool_call']
            tool_name = _get_tool_call_name(tc, 'unknown_tool')
            tool_args = _get_tool_call_args(tc)
            target = _extract_target(tool_args, text, tool_name)

            this_decision_values = {v for v in (target, tool_name) if v}
            decisions.append({'type': 'action', 'action': tool_name,
                'target': target,
                'rationale': _extract_rationale(text, arts, decision_values=this_decision_values,
                                               decision_idx=idx),
                'artifacts': {k: arts[k] for k in ('paths', 'ids', 'errors')},
                'verbatim': text, 'msg_idx': idx})
        elif _has_strong_judgment_signal(text):
            action = _extract_verb(text)
            target = _extract_target({}, text)

            this_decision_values = {v for v in (target, action) if v}
            decisions.append({'type': 'judgment', 'action': action,
                'target': target,
                'rationale': _extract_rationale(text, arts, decision_values=this_decision_values,
                                               decision_idx=idx),
                'artifacts': {k: arts[k] for k in ('paths', 'ids', 'errors')},
                'verbatim': text, 'msg_idx': idx})
        elif (_CONFIRM_SIGNALS.search(text) and (arts['paths'] or arts['ids'])
              and not _has_strong_decisive_no_confirm(text)):
            target = _extract_target({}, text)
            if target is None:
                fallback = [a for a in (arts['paths'] + arts['ids']) if _is_sane_candidate(a)]
                target = fallback[0] if fallback else None
                fallback = [f for f in fallback if _is_sane_candidate(f)]
                target = fallback[0] if fallback else None

            this_decision_values = {v for v in (target,) if v}
            decisions.append({'type': 'confirmation', 'action': 'confirm', 'target': target,
                'rationale': _extract_rationale(text, arts, decision_values=this_decision_values,
                                               decision_idx=idx),
                'artifacts': {k: arts[k] for k in ('paths', 'ids', 'errors')},
                'verbatim': text, 'msg_idx': idx})
        elif _has_judgment_signal(text):
            action = _extract_verb(text)
            target = _extract_target({}, text)

            this_decision_values = {v for v in (target, action) if v}
            decisions.append({'type': 'judgment', 'action': action,
                'target': target,
                'rationale': _extract_rationale(text, arts, decision_values=this_decision_values,
                                               decision_idx=idx),
                'artifacts': {k: arts[k] for k in ('paths', 'ids', 'errors')},
                'verbatim': text, 'msg_idx': idx})

        elif _CONFIRM_SIGNALS.search(text) and (arts['paths'] or arts['ids']):
            target = _extract_target({}, text)
            if target is None:
                fallback = [a for a in (arts['paths'] + arts['ids']) if _is_sane_candidate(a)]
                target = fallback[0] if fallback else None
                fallback = [f for f in fallback if _is_sane_candidate(f)]
                target = fallback[0] if fallback else None

            this_decision_values = {v for v in (target,) if v}
            decisions.append({'type': 'confirmation', 'action': 'confirm', 'target': target,
                'rationale': _extract_rationale(text, arts, decision_values=this_decision_values,
                                               decision_idx=idx),
                'artifacts': {k: arts[k] for k in ('paths', 'ids', 'errors')},
                'verbatim': text, 'msg_idx': idx})
    return decisions
