"""
Decision extractor: turns a raw agent trace into a list of structured
decisions (tool calls, judgments, confirmations) with extracted targets,
rationale, and cited artifacts. Pure regex/heuristic -- no LLM calls.
# extraction.py
EXTRACTION_LOGIC_VERSION = "2026-08-02.1"  # bumped: _mask_code_fences now masks a
# dangling (unpaired) ``` to end-of-text instead of leaving it as visible prose.

# wherever ground truth is loaded for scoring, before comparing
if decision.get("_extractor_version") != EXTRACTION_LOGIC_VERSION:
    raise RuntimeError(
        f"Stale ground truth (built with {decision.get('_extractor_version')!r}, "
        f"current extractor is {EXTRACTION_LOGIC_VERSION!r}). Regenerate fixtures."
    )
"""

from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from .utils import (
    CRITICAL, STOPWORDS, _artifacts, _get_text, _tok, _uw,
)
class StaleGroundTruthError(RuntimeError):
    """Raised when a stored ground-truth decision was stamped with an
    EXTRACTION_LOGIC_VERSION that no longer matches the live extractor.
    This turns a previously-silent failure mode into a loud one: without
    this check, compress_dagc's fresh re-extraction (which drives the
    hard-guarantee set) can silently diverge from a stale fixture's
    action/target -- the fixture keeps expecting an old label (e.g.
    'request') that the current extractor no longer produces for that
    text, so nothing protects it, and reproduction (correctly, per the
    CURRENT extractor) comes back with the current label instead. That
    reads as an action mismatch during scoring, but it's actually a
    fixture/version skew bug, not a reproduction failure."""
    pass


def assert_ground_truth_current(decisions: List[Dict], source: str = "") -> None:
    """Call this immediately after loading any stored/frozen ground-truth
    decision list, before using it for compression guarantees or scoring.
    Raises StaleGroundTruthError listing every decision whose stamped
    _extractor_version doesn't match the live EXTRACTION_LOGIC_VERSION,
    so drift is caught at load time instead of surfacing as mysterious
    per-trace action mismatches downstream."""
    stale = [
        (d.get('msg_idx'), d.get('_extractor_version'))
        for d in decisions
        if d.get('_extractor_version') != EXTRACTION_LOGIC_VERSION
    ]
    if stale:
        detail = ', '.join(f"msg_idx={i} (built with {v!r})" for i, v in stale[:10])
        more = f" (+{len(stale) - 10} more)" if len(stale) > 10 else ""
        raise StaleGroundTruthError(
            f"{source or 'ground truth'}: {len(stale)} decision(s) stamped with a "
            f"stale extractor version, current is {EXTRACTION_LOGIC_VERSION!r}: "
            f"{detail}{more}. Regenerate fixtures before scoring/compressing."
        )  # bumped: closes the remaining
# false-negative sample gaps that had NO decisive-verb match at all (so no
# amount of target/rationale tuning could have caught them) --
# (1) "need to" joins the intent-modal family, and intent-modal matches
#     ('d like to / want to / need to / wish to) now resolve to the real
#     verb that follows them instead of leaking the bare modal phrase as
#     the action label;
# (2) bare standalone-imperative instructions with an open-class leading
#     verb ("Obtain...", "Get directions...") now anchor target extraction
#     on their own leading verb instead of relying on the bigram-
#     corroboration fallback, which requires the same bigram twice and so
#     silently drops any single-mention instruction;
# (3) a "Could/Can/Would/Will you (please) VERB...?" polite request now
#     fires even when VERB sits outside the closed _JUDGMENT_VERBS
#     vocabulary, not only when the existing interrogative-mood override
#     already found a decisive match to override mood for;
# (4) a first-person future-intent statement ("I will/I'll/we'll VERB...")
#     is now recognized as a decision even though it is, by design,
#     excluded from the standalone-imperative check;
# (5) a first-person certainty assertion re-stating/correcting a concrete
#     number or id ("I'm absolutely certain there were 3 of us") is now
#     captured as a confirmation, mirroring what _CONFIRM_SIGNALS already
#     does for assistant turns.
# Fixtures built with 2026-07-25.2 or earlier must be regenerated.

_JUDGMENT_VERBS = re.compile(
    r'\b(recommend|conclude|suggest|decide|choos(?:e|es|ing)|chose|select(?:s|ed|ing)?|prefer|'
    r'best|winner|optimal|final|confirm(?:ed|ation|s)?|'
    r'implement|adopt|us(?:e|es|ed|ing)|switch(?:ed|ing)?|migrat(?:e|ed|ing)|'
    r'deploy(?:s|ed|ing)?|provision(?:s|ed|ing)?|'
    r'(?:go(?:es|ing)?|went)\s+with|'
    r'keep(?:s|ing)?|remov(?:e|es|ed|ing)|push(?:es|ed|ing)?|merg(?:e|es|ed|ing)|'
    r'mov(?:e|es|ed|ing)|target(?:s|ed|ing)?|delet(?:e|es|ed|ing)|renam(?:e|es|ed|ing)|'
    r'revert(?:s|ed|ing)?|'
    r'cancel\w*|exchang\w*|refund\w*|rebook\w*|reschedul\w*|'
    r'return(?:s|ed|ing)?|book(?:s|ed|ing)?|add(?:s|ed|ing)?|'
    # NEW 2026-08-01: "need to" added to the intent-modal family below.
    # Proven miss: "I need this to be resolved" (trace54_tau) carries the
    # same decision-intent weight as "I want this resolved" but had zero
    # coverage. Same tight "must be followed by to" shape as its siblings
    # so it doesn't swallow plain uses of "need" with no following verb
    # ("I need a refund" stays uncovered by this branch, as intended).
    r"(?:'d|would)\s+like\s+to|want(?:s|ed|ing)?\s+to|wish(?:es)?\s+to|need(?:s|ed)?\s+to)\b",
    re.IGNORECASE)

# NEW 2026-08-01: intent-modal phrases ("'d like to", "want to", "need to",
# "wish to") match _JUDGMENT_VERBS above so the SIGNAL fires correctly, but
# they are not themselves actions -- the real action verb always follows
# them ("I'd like to CANCEL the order"). Before this fix, _extract_verb
# could return the bare modal phrase itself as the action label whenever
# it won priority or was the first match (proven: trace0/trace57/trace58/
# trace33_rwt3 in the false-negative sample review all surfaced
# action="'d like to" / "would like to" instead of the real verb -- not
# wrong exactly, just uninformative and inconsistent with every other verb
# this function returns). _resolve_intent_modal_action looks immediately
# past the modal for the next word and uses THAT as the action when it
# reads as a plausible verb, falling back to a generic 'request' label
# rather than leaking the modal phrase.
_RE_INTENT_MODAL_HEAD = re.compile(
    r"^(?:'d|would|want(?:s|ed|ing)?|wish(?:es)?|need(?:s|ed)?)\s+(?:like\s+)?to$",
    re.IGNORECASE)

_BARE_EVALUATIVE_ADJ_WORDS = {'final', 'best', 'optimal'}
_DECISION_NOUN_NEARBY = re.compile(
    r'\b(decision|answer|choice|option|pick|call|verdict|recommendation|'
    r'winner|result|conclusion)\b', re.IGNORECASE)

def _bare_adjective_is_decisive(text: str, m: 're.Match') -> bool:
    if m.group(0).lower() not in _BARE_EVALUATIVE_ADJ_WORDS:
        return True
    window = text[max(0, m.start() - 30):min(len(text), m.end() + 30)]
    return bool(_DECISION_NOUN_NEARBY.search(window))

_RE_CODE_FENCE = re.compile(r'```.*?```', re.S)

def _mask_code_fences(text: str) -> str:
    return _RE_CODE_FENCE.sub(lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text)

_JUDGMENT_CONNECTIVE_WORDS = ('therefore', 'thus', 'hence')

_JUDGMENT_CONNECTIVES = re.compile(
    r'\b(' + '|'.join(_JUDGMENT_CONNECTIVE_WORDS) + r')\b', re.IGNORECASE)



_STRONG_JUDGMENT_SIGNALS = re.compile(
    r'\b(recommend|conclude|decide|choose|select(?:s|ed|ing)?|'
    r'therefore|thus|hence|adopt)\b', re.IGNORECASE)

_STRONG_DECISIVE_VERBS_NO_CONFIRM = re.compile(
    r'\b(recommend|conclude|decide|choos(?:e|es|ing)|chose|select(?:s|ed|ing)?|winner|adopt)\b',
    re.IGNORECASE)

_STRONG_JUDGMENT_VERBS = re.compile(
    r'\b(recommend|conclude|decide|choos(?:e|es|ing)|chose|select(?:s|ed|ing)?|winner|adopt)\b',
    re.IGNORECASE)

_CONFIRM_SIGNALS = re.compile(
    r'\b(confirm(?:ed|ation|s)?|verified?|preserv|ensur|kept?|maintain)\b', re.IGNORECASE)

_OUTCOME_CONFIRM_SIGNALS = re.compile(
    r'\b(successfully|(?:has|have) been (?:applied|processed|completed|updated|modified|'
    r'refunded|charged|cancelled|confirmed|saved)|(?:was|were) (?:applied|processed|completed)|'
    r"(?:i(?:'ve| have)|we(?:'ve| have)) (?:applied|processed|completed|updated|modified|"
    r'refunded|charged|cancelled|confirmed|finished|done|saved)|'
    r'(?:has|have) (?:already )?saved\b)\b',
    re.IGNORECASE
)

_ACTION_VERB_PRIORITY = [
    'recommend', 'confirm', 'adopt', 'implement', 'select', 'choose',
    'decide', 'prefer', 'suggest', 'conclude', 'use',
    'best', 'optimal', 'winner', 'final',
    'cancel', 'exchange', 'refund', 'rebook', 'reschedule',
    'keep', 'remove', 'push', 'merge', 'move', 'target', 'delete', 'rename', 'revert',
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
_RE_ENTITY = re.compile(r'\b[A-Za-z][a-z0-9]+[A-Z][A-Za-z0-9]*\b')

_RE_WINNER_BEFORE = re.compile(
    r'\b(?:recommendation|winner|best|optimal|recommended|selected|chosen|'
    r'preferred|adopted|confirmed|approved|finalized|final\s+decision)\b'
    r'\W{0,8}([a-zA-Z0-9][a-zA-Z0-9._-]+(?:\s+[a-zA-Z0-9][a-zA-Z0-9._-]+)?)',)
_RE_WINNER_AFTER = re.compile(
    r'([a-zA-Z0-9][a-zA-Z0-9._-]+(?:\s+[a-zA-Z0-9][a-zA-Z0-9._-]+)?)'
    r'\s+(?:is\s+(?:the\s+)?(?:clear\s+|better\s+|best\s+|right\s+|correct\s+|'
    r'safer\s+|simpler\s+|preferred\s+|obvious\s+)?(?:winner|best|optimal|choice)'
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
    r'(?<![A-Za-z0-9_])([A-Za-z][\w-]*)[ \t]*[=:][ \t]*'
    r'(\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\$?\d+(?:\.\d+)?%?)'
    r'(?!\s*\.\s)'
)
_RE_EQUATION_OPERATOR = re.compile(r'[=≈]')

def _is_chained_equation(sentence: str) -> bool:
    return len(_RE_EQUATION_OPERATOR.findall(sentence)) > 1



_RE_PRESERVED_TAG = re.compile(r'\[preserved:\s*([^\]]+)\]')
_RE_OWNER_SUFFIX = re.compile(r'^(.*?)#d([\d,]+)$')
_RE_WORD_CHAR = re.compile(r'[^\W\d_]', re.UNICODE)

def _snap_to_word_boundaries(text: str, s: int, e: int) -> Tuple[int, int]:
    while s < e and s > 0 and _RE_WORD_CHAR.match(text[s - 1]) and _RE_WORD_CHAR.match(text[s]):
        s += 1
    while e > s and e < len(text) and _RE_WORD_CHAR.match(text[e - 1]) and _RE_WORD_CHAR.match(text[e]):
        e -= 1
    return s, e

_RE_ENTRY_SPLIT = re.compile(r',\s+')
_RE_INTERROGATIVE_END = re.compile(r'\?\s*$')
_RE_WH_FRONT = re.compile(r'^\s*(?:so\s+)?(what|which|who|whom|whose|why|how)\b', re.I)
_RE_AUX_INVERSION_FRONT = re.compile(
    r"^\s*(is|are|was|were|do|does|did|can|could|would|should|will|shall|"
    r"have|has|had)\b.*\?\s*$", re.I)
_KNOWN_FILE_EXTENSIONS = {
    'csv', 'tsv', 'json', 'jsonl', 'txt', 'md', 'yml', 'yaml', 'xml',
    'xlsx', 'xls', 'parquet', 'pkl', 'pickle', 'npy', 'npz', 'h5', 'hdf5',
    'png', 'jpg', 'jpeg', 'gif', 'svg', 'bmp', 'tiff', 'webp',
    'pdf', 'doc', 'docx', 'ppt', 'pptx',
    'py', 'ipynb', 'js', 'ts', 'sh', 'sql', 'r', 'cfg', 'ini', 'toml',
    'log', 'db', 'sqlite', 'zip', 'tar', 'gz', 'env', 'ckpt', 'pt', 'onnx',
}

_RE_PATH = re.compile(
    r'(?<![A-Za-z0-9<])'
    r'(?:/[\w.-]+(?:/[\w.-]+)+(?<![.,;:!?])'
    r'|/[\w.-]*\d[\w.-]*(?<![.,;:!?])'
    r'|[A-Za-z]:[\\/][\w.\\/-]+(?<![.,;:!?])'
    r'|\b[\w][\w\-]{2,60}\.(?:' + '|'.join(_KNOWN_FILE_EXTENSIONS) + r')\b(?<![.,;:!?]))',
    re.IGNORECASE)

_DISCOURSE_CONNECTIVES = set(_JUDGMENT_CONNECTIVE_WORDS) | {
    'however', 'since', 'after', 'before', 'because', 'although', 'though',
    'while', 'whereas', 'despite', 'meanwhile', 'moreover', 'furthermore',
    'additionally', 'otherwise', 'instead', 'consequently', 'nevertheless',
    'nonetheless', 'overall', 'once', 'given', 'based', 'then', 'now',
    'next', 'finally', 'still', 'yet', 'so', 'accordingly', 'likewise',
    'similarly', 'conversely',
}

_ENTITY_BLOCKLIST = CRITICAL | STOPWORDS | _DISCOURSE_CONNECTIVES | {
    'evidence', 'confirmed', 'confirm', 'confirmation', 'confirms',
    'root',
    'winner', 'best', 'optimal', 'reading', 'comparing', 'report',
    'results', 'recommendation', 'recommended', 'implementing',
    'preserved', 'clear', 'metrics', 'detailed',
    'tool_call', 'tool_calls', 'function_call', 'tool_response', 'tool_result',
}
_RE_OR_SPLIT = re.compile(r'\bor\b', re.IGNORECASE)
_HEDGE_MARKERS = re.compile(
    r"\b(probably|maybe|perhaps|possibly|might|could|likely|presumably|"
    r"i think|i guess|sort of|kind of|leaning (?:towards|toward)|"
    r"not (?:fully |totally |completely )?sure|not certain|"
    r"back and forth|torn between|can'?t decide|haven'?t decided|"
    r"not (?:yet )?decided|still deciding|still evaluating|still weighing|"
    r"undecided)\b",
    re.IGNORECASE)

_RE_LOCAL_NEGATION = re.compile(r"\b(?:not|never|no longer|without)\s*$|n't\s*$",
                                 re.IGNORECASE)
_RE_SENTENCE_END = re.compile(r'(?<!\d)([.!?])(?!\d)(?:\s+|$)|\n+|\s+--\s+'r'|(?<![A-Za-z0-9_])</?[A-Za-z_][A-Za-z0-9_]{0,29}\s*/?>\s*')

_RE_PRESERVED_TAG_SPAN = re.compile(r'\[preserved:[^\]]*\]')
_RE_TOOL_FOOTPRINT_SPAN = re.compile(r'→TOOL:[A-Za-z_][\w.]*\([^)]*\)')

_RE_DANGLING_FENCE = re.compile(r'```.*\Z', re.S)

def _mask_code_fences(text: str) -> str:
    text = _RE_CODE_FENCE.sub(lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text)
    # Compression can truncate a code block so only its OPENING ``` survives
    # (the matching close got cut by sentence-selection/budget trimming).
    # Left unhandled, everything after that dangling marker -- including
    # code tokens like 'return'/'push' that _JUDGMENT_VERBS matches -- reads
    # as ordinary prose and can surface as a spurious decisive match that
    # was never visible (and never decisive) in the original message. An
    # odd ``` count after the paired substitution above means exactly one
    # is left dangling; mask from there to the end of the text the same way
    # a genuine closed fence is masked.
    if text.count('```') % 2 == 1:
        text = _RE_DANGLING_FENCE.sub(lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text)
    text = _RE_PRESERVED_TAG_SPAN.sub(lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text)
    text = _RE_TOOL_FOOTPRINT_SPAN.sub(lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text)
    return text

def _extract_question_options(prior_text: str) -> List[str]:
    opts = set()
    for m in _RE_ENTITY.finditer(prior_text):
        w = m.group(0)
        if w.lower() not in _ENTITY_BLOCKLIST:
            opts.add(w.lower())
    for m in _RE_ENTITY_SNAKE.finditer(prior_text):
        w = m.group(0)
        if w.lower() not in _ENTITY_BLOCKLIST:
            opts.add(w.lower())
    for part in _RE_OR_SPLIT.split(prior_text):
        tok = part.strip().strip('?.,!').split()
        if tok and 1 <= len(tok[-1]) and tok[-1].lower() not in _ENTITY_BLOCKLIST | STOPWORDS:
            opts.add(tok[-1].lower())
    return list(opts)


def _is_bare_option_answer(text: str, prior_text: Optional[str]) -> Optional[str]:
    if not prior_text or '?' not in prior_text:
        return None
    options = _extract_question_options(prior_text)
    if len(options) < 2:
        return None

    first_clause = re.split(r'[.!\n]', text.strip(), maxsplit=1)[0].strip()
    if not first_clause or len(first_clause.split()) > 4:
        return None
    if _clause_is_interrogative(first_clause):
        return None

    fc_low = first_clause.lower().strip('.,;:')
    for opt in options:
        if opt == fc_low or fc_low.startswith(opt + ' ') or fc_low == opt:
            return first_clause.strip('.,;:')
    return None

def _looks_grounded_np(tokens: List[str], allow_bare_single: bool = False) -> bool:
    if len(tokens) >= 2:
        return True
    if len(tokens) == 1 and any(ch.isdigit() for ch in tokens[0]):
        return True
    if len(tokens) == 1 and tokens[0][:1].isupper():
        return True
    if allow_bare_single and len(tokens) == 1 and tokens[0].isalpha():
        return True
    return False


_FOCUS_ADVERBS = {'only', 'just', 'also', 'even', 'solely', 'merely', 'simply'}
_COORD_CONJUNCTIONS = {'and', 'or'}


def _extract_decisive_object_phrase(text: str, decisive_span: Optional[Tuple[int, int]]) -> Optional[str]:
    if decisive_span is None:
        return None

    tail = text[decisive_span[1]:decisive_span[1] + 100]
    tail = re.split(r'[,;.!?\n]|--', tail, maxsplit=1)[0]
    words = _RE_WORD_TOKEN.findall(tail)
    if not words:
        return None

    i = 0
    if words[i].lower() in _PREPOSITIONS_OBJECT:
        i += 1
    while i < len(words) and (words[i].lower() in _DETERMINERS
                               or words[i].lower() in _DEGREE_MODIFIERS
                               or words[i].lower() in _FOCUS_ADVERBS
                               or words[i].lower() in _VAGUE_QUALIFIERS):
        i += 1

    phrase_tokens = []
    content_count = 0
    for w in words[i:i + 8]:
        wl = w.lower()
        if wl in _COORD_CONJUNCTIONS:
            if not phrase_tokens or content_count >= 4:
                break
            phrase_tokens.append(wl)
            continue
        if wl in _NP_STOP_WORDS:
            break
        phrase_tokens.append(w)
        content_count += 1
        if content_count >= 4:
            break

    if not _looks_grounded_np(phrase_tokens, allow_bare_single=True):
        return None
    return ' '.join(t.lower() for t in phrase_tokens)

def _is_sentence_initial(text: str, pos: int) -> bool:
    for s, e in _sentence_spans(text):
        if s <= pos < e:
            i = s
            while i < e and text[i].isspace():
                i += 1
            return pos == i
    return False


def _best_corroborated_match(matches, text, id_prefixes):
    groups = defaultdict(list)
    for m in matches:
        w = m.group(0)
        if w.lower() in _ENTITY_BLOCKLIST or w in id_prefixes:
            continue
        groups[w].append(m.start())

    best = None
    for w, positions in groups.items():
        corroborated = len(positions) >= 2 or not _is_sentence_initial(text, positions[0])
        if not corroborated:
            continue
        if best is None or len(positions) > best[1]:
            best = (w, len(positions))
    return best

def _is_meaningful_target_candidate(val: str) -> bool:
    words = val.strip().split()
    if not words:
        return False
    first = words[0].lower().strip('.,;:')
    return (first not in STOPWORDS
            and first not in _FILLER_CONNECTORS
            and first not in _FILLER_ACK_WORDS)

def _extract_entity_target(text, known_ids=None, decisive_span=None):
    known_ids = known_ids or []
    id_prefixes = {i.split('-')[0] for i in known_ids if '-' in i}

    search_text = text
    if decisive_span is not None:
        search_text = _sentence_containing(text, decisive_span[0], decisive_span[1])

    for pattern in (_RE_WINNER_BEFORE, _RE_WINNER_AFTER):
        for m in pattern.finditer(search_text):
            val = m.group(1).strip('.,;: ')
            if (val.lower() not in _ENTITY_BLOCKLIST
                    and val not in id_prefixes
                    and len(val.split()) <= 3
                    and len(val) >= 3
                    and _is_meaningful_target_candidate(val)):
                return val

    verb_m = _find_strong_judgment_match(search_text)
    if verb_m:
        tail = search_text[verb_m.end():verb_m.end() + 80].strip()
        gerund_m = re.match(r'\s*([a-z]+ing\b[^.;]{0,60})', tail, re.I)
        if gerund_m:
            phrase = gerund_m.group(1).strip()
            if len(phrase.split()) >= 2:
                return _cap_target_length(phrase, max_words=8)

    camel_top = _best_corroborated_match(_RE_ENTITY.finditer(search_text), search_text, id_prefixes)
    snake_top = _best_corroborated_match(_RE_ENTITY_SNAKE.finditer(search_text), search_text, id_prefixes)
    if camel_top and snake_top:
        return snake_top[0] if snake_top[1] > camel_top[1] else camel_top[0]
    if camel_top:
        return camel_top[0]
    if snake_top:
        return snake_top[0]

    all_words = _RE_WORD_TOKEN_UNICODE.findall(search_text)

    if any(not w.isascii() for w in all_words):
        bigrams = []
    else:
        def _is_bigram_content_word(w: str) -> bool:
            wl = w.lower()
            return (wl not in STOPWORDS
                    and wl not in _ENTITY_BLOCKLIST
                    and bool(re.fullmatch(r'[a-z][a-z0-9]{1,20}', wl)))

        bigrams = []
        for chunk in re.split(r'[,;:]|--|—', search_text):
            cw = _RE_WORD_TOKEN_UNICODE.findall(chunk)
            bigrams.extend(
                f'{cw[i].lower()} {cw[i + 1].lower()}'
                for i in range(len(cw) - 1)
                if _is_bigram_content_word(cw[i]) and _is_bigram_content_word(cw[i + 1])
            )

    if bigrams:
        top_bg, top_count = Counter(bigrams).most_common(1)[0]
        if top_count >= 2:
            return top_bg

def _clause_is_interrogative(clause: str) -> bool:
    c = clause.strip()
    return bool(_RE_INTERROGATIVE_END.search(c)
                or _RE_WH_FRONT.match(c)
                or _RE_AUX_INVERSION_FRONT.match(c))

def _clause_is_hedged(clause: str) -> bool:
    check_text = clause.strip()
    front_m = _RE_POLITE_REQUEST_FRONT.match(check_text)
    if front_m:
        check_text = check_text[front_m.end() - 1:]
    return bool(_HEDGE_MARKERS.search(check_text))


def _verb_locally_negated(text: str, m: 're.Match') -> bool:
    window = text[max(0, m.start() - 20):m.start()]
    return bool(_RE_LOCAL_NEGATION.search(window))

_RE_POLITE_REQUEST_FRONT = re.compile(
    r"^\s*(?:[A-Za-z ]{0,20}?,\s*)?"
    r"(?:so\s+)?(?:could|can|would|will)\s+you\s+(?:please\s+|kindly\s+)?\w",
    re.IGNORECASE)

def _is_polite_request_question(sentence: str) -> bool:
    return bool(_RE_POLITE_REQUEST_FRONT.match(sentence.strip()))

EXTRACTION_LOGIC_VERSION = "2026-08-02.2"  # bumped: a JUDGMENT_VERBS match inside
# a fence-less, stranded code-statement fragment (compression can cut a code
# block mid-fence, leaving no ``` markers to mask) is no longer treated as
# decisive -- see _sentence_looks_like_code.
_RE_CODE_SYMBOL = re.compile(r'[{}();<>=\[\]]')

def _sentence_looks_like_code(sentence: str) -> bool:
    """True if `sentence` reads like a bare source-code statement rather
    than natural-language prose.

    Needed because compressed output can leave a code line stranded
    WITHOUT its original ``` fence markers -- compression's sentence
    splitter can cut a code block mid-fence, so _mask_code_fences (which
    only recognizes fence markers) has nothing left to mask. An ordinary
    English word that also doubles as a common code token (return/add/
    keep/push/target/use) inside such a stranded fragment is syntax, not
    a decisive prose verb, and must not be treated as one.

    Two independent, cheap signals, BOTH required (either alone is too
    easy to trip on legitimate prose that happens to mention one symbol
    or one technical term):
      1. Code-punctuation density -- prose rarely has more than a symbol
         or two per clause; a code statement is dense with them.
      2. Low natural-language function-word density -- prose is glued
         together with stopwords (the/a/to/is/with/...); a bare code
         statement mostly isn't.
    """
    s = sentence.strip()
    if len(s) < 3:
        return False
    words = _RE_WORD_TOKEN.findall(s)
    if not words:
        return False
    symbol_density = len(_RE_CODE_SYMBOL.findall(s)) / max(len(s), 1)
    stopword_ratio = sum(1 for w in words if w.lower() in STOPWORDS) / len(words)
    return symbol_density >= 0.05 and stopword_ratio <= 0.15


def _verb_match_is_decisive(text: str, m: 're.Match') -> bool:
    sentence = _sentence_containing(text, m.start(), m.end())
    is_question = (_clause_is_interrogative(sentence)
                   and not _is_polite_request_question(sentence))
    return (not is_question
            and not _clause_is_hedged(sentence)
            and not _verb_locally_negated(text, m)
            and not _sentence_looks_like_code(sentence)
            and _bare_adjective_is_decisive(text, m))
def _prior_message_is_interrogative(messages: List[Dict], idx: int) -> bool:
    if idx <= 0:
        return False
    prior_text = _get_text(messages[idx - 1]).strip()
    if not prior_text:
        return False
    clauses = [c for c in re.split(r'(?<=[.!?])\s+', prior_text) if c.strip()]
    tail = clauses[-1] if clauses else prior_text
    return _clause_is_interrogative(prior_text) or _clause_is_interrogative(tail)


_SUBJECT_LEADING_WORDS = frozenset({
    "i", "we", "you", "he", "she", "it", "they", "this", "that", "there",
    "is", "are", "was", "were", "will", "would", "can", "could", "should",
    "shall", "must", "might", "may", "the", "a", "an", "so", "and", "but",
    "ok", "okay", "yes", "no", "sure",
})


def _prior_message_is_imperative_directive(messages: List[Dict], idx: int) -> bool:
    if idx <= 0:
        return False
    prior_text = _get_text(messages[idx - 1]).strip()
    if not prior_text:
        return False
    clauses = [c for c in re.split(r'(?<=[.!?])\s+', prior_text) if c.strip()]
    tail = clauses[-1] if clauses else prior_text
    if _clause_is_interrogative(tail) or _clause_is_hedged(tail):
        return False
    first_word = re.match(r"^[A-Za-z']+", tail)
    if not first_word:
        return False
    return first_word.group(0).lower() not in _SUBJECT_LEADING_WORDS


def _has_imperative_response_signal(text: str, messages: List[Dict], idx: int) -> bool:
    if not _prior_message_is_imperative_directive(messages, idx):
        return False
    first_clause = re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)[0]
    return not _clause_is_interrogative(first_clause) and not _clause_is_hedged(first_clause)


def _has_directive_response_signal(text: str, messages: List[Dict], idx: int) -> bool:
    if not _prior_message_is_interrogative(messages, idx):
        return False
    first_clause = re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)[0]
    return not _clause_is_interrogative(first_clause) and not _clause_is_hedged(first_clause)


_POLITENESS_LEAD_IN = re.compile(
    r'^(?:please|kindly|could you please|just|now)\s+', re.IGNORECASE)


def _is_standalone_imperative(text: str) -> bool:
    first_clause = re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)[0]
    if _clause_is_interrogative(first_clause) or _clause_is_hedged(first_clause):
        return False
    stripped = _POLITENESS_LEAD_IN.sub('', first_clause.strip())
    first_word = re.match(r"^[A-Za-z']+", stripped)
    if not first_word:
        return False
    return first_word.group(0).lower() not in _SUBJECT_LEADING_WORDS


def _has_standalone_imperative_signal(text: str) -> bool:
    return _is_standalone_imperative(text)


def _first_sentence(text: str) -> str:
    return re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)[0]


def _leading_offset(text: str) -> int:
    return len(text) - len(text.lstrip())


def _leading_verb_span(text: str) -> Optional[Tuple[int, int]]:
    """A standalone imperative ('Obtain comprehensive details about a
    movie...', 'Get directions from X to Y...') that doesn't happen to use
    one of the closed-class _JUDGMENT_VERBS has no decisive_span to anchor
    object-phrase extraction on. _extract_target then falls through to the
    bigram-corroboration fallback in _extract_entity_target -- which only
    returns a bigram that repeats TWICE in the message, so a single bare
    instruction (the overwhelmingly common case for an opening task
    request) silently produces no target and the whole decision gets
    dropped by _has_concrete_referent. Proven gap: 'Obtain comprehensive
    details about a movie, excluding cast information and images.'
    Anchor directly on the clause's own leading verb instead, so the
    already-reliable _extract_decisive_object_phrase /
    _extract_all_object_phrases machinery has a local span to work from --
    exactly as it already does for a genuine _JUDGMENT_VERBS hit."""
    first_clause = _first_sentence(text)
    stripped = _POLITENESS_LEAD_IN.sub('', first_clause.strip())
    m = re.match(r"[A-Za-z']+", stripped)
    if not m:
        return None
    verb = m.group(0)
    start = text.find(verb, _leading_offset(text))
    if start == -1:
        return None
    return (start, start + len(verb))


# NEW 2026-08-01: broadened, last-resort sibling of _is_polite_request_question.
# 'Could you help me cancel...' already works today because 'cancel' is
# itself a _JUDGMENT_VERBS hit and the polite-request check only overrides
# the interrogative-mood veto for an ALREADY-found decisive match. 'Can you
# make the exoskeleton walk?' / 'Can you try to be more imaginative?' carry
# the exact same second-person-modal request shape but front a verb
# ('make', 'try') outside the closed _JUDGMENT_VERBS vocabulary, so no
# decisive match is ever found and the polite-request exception never gets
# a chance to fire. This signal doesn't require a _JUDGMENT_VERBS hit at
# all -- it only requires the closed second-person-modal-request SHAPE
# (mirroring _RE_POLITE_REQUEST_FRONT) at the front of a genuine question.
# Deliberately checked only after every more specific branch in
# extract_decisions, so it can only fill a gap, never shadow a better
# classification.
_RE_POLITE_REQUEST_VERB = re.compile(
    r"^\s*(?:[A-Za-z ]{0,20}?,\s*)?(?:so\s+)?(?:could|can|would|will)\s+you\s+"
    r"(?:please\s+|kindly\s+)?(?:help me\s+)?([a-z]+)\b", re.IGNORECASE)


def _polite_request_verb_match(text: str) -> Optional['re.Match']:
    first = _first_sentence(text)
    if not first.rstrip().endswith('?'):
        return None
    return _RE_POLITE_REQUEST_VERB.match(first.strip())


def _has_polite_request_signal(text: str) -> bool:
    return _polite_request_verb_match(text) is not None


# NEW 2026-08-01: a first-person future-intent statement ("I will check
# X...", "I'll cancel Y...", "We'll proceed with Z...") carries the same
# decision content as an imperative, but _is_standalone_imperative
# structurally cannot catch it -- it explicitly excludes clauses that open
# with a pronoun/auxiliary (_SUBJECT_LEADING_WORDS includes "i"/"we"/
# "will"/"'ll" territory) precisely so it doesn't mistake an ordinary
# declarative sentence for an instruction. That exclusion is correct in
# general but throws out this one real pattern along with it. Proven
# miss: "I will first check the details of your reservations to ensure
# they meet the criteria for cancellation." (trace69, assistant turn) --
# no _JUDGMENT_VERBS hit ('check' isn't one), no directive/imperative
# response context, no standalone-imperative match. Narrow and
# closed-class on purpose: only the "I will / I'll / we will / we'll"
# fronting, optionally through one throwaway adverb ("first", "now",
# "then", "also"), immediately followed by the real verb.
_RE_FIRST_PERSON_INTENT = re.compile(
    r"^\s*(?:so\s+)?(?:I(?:'ll|\s+will)|we(?:'ll|\s+will))\s+"
    r"(?:first\s+|now\s+|then\s+|also\s+|just\s+)?([a-z]+)\b",
    re.IGNORECASE)


def _first_person_intent_match(text: str) -> Optional['re.Match']:
    first = _first_sentence(text)
    if _clause_is_interrogative(first) or _clause_is_hedged(first):
        return None
    return _RE_FIRST_PERSON_INTENT.match(first)


def _has_first_person_intent_signal(text: str) -> bool:
    return _first_person_intent_match(text) is not None


# NEW 2026-08-01: a user firmly restating/correcting a concrete fact
# ("I'm absolutely certain there were 3 of us", "I distinctly remember
# paying for the insurance") functions the same way _CONFIRM_SIGNALS
# does for an assistant's "confirmed"/"verified" -- it's the person
# asserting a value should be taken as settled -- but none of that
# vocabulary ("confirm/verify/preserve/ensure/keep/maintain") appears.
# Proven miss: "No, I'm absolutely certain there were 3 of us on that
# delayed flight!" (trace76) carries a concrete numeric correction with
# zero coverage under the existing confirm-signal vocabulary. Gated on
# the SAME sentence also containing a digit, so a bare confident opinion
# with no concrete fact attached ("I'm sure that's a bad idea") is not
# swept in.
_RE_CERTAINTY_ASSERTION = re.compile(
    r"\b(?:i(?:'m|\s+am)\s+(?:absolutely\s+|completely\s+|100%\s+)?"
    r"(?:certain|sure|positive)|i know (?:for (?:a )?fact )?that|"
    r"i distinctly remember)\b",
    re.IGNORECASE)


def _has_certainty_assertion_signal(text: str) -> bool:
    m = _RE_CERTAINTY_ASSERTION.search(text)
    if not m:
        return False
    sentence = _sentence_containing(text, m.start(), m.end())
    if _clause_is_interrogative(sentence):
        return False
    return bool(re.search(r'\d', sentence))


def _preserved_tag_candidates(text, decision_idx=None):
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
            out.append((value, True))
        else:
            out.append((raw, False))
    return out

def _cap_target_length(t, max_words=6):
    if not t:
        return t
    return t if len(t.split()) <= max_words else None


def _flatten_arg_values(obj, prefix='', max_depth=3, _depth=0):
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
_PREPOSITIONS_OBJECT = {'to', 'as', 'with', 'for', 'into', 'onto', 'on'}
_DETERMINERS = {'the', 'a', 'an', 'this', 'that', 'these', 'those'}

_VAGUE_QUALIFIERS = {
    'different', 'various', 'certain', 'other', 'similar', 'particular',
    'some', 'any', 'another',
}

_TRAILING_ADVERBIALS = {
    'real', 'quick', 'quickly', 'now', 'soon', 'immediately', 'right',
    'away', 'today', 'briefly', 'fast', 'shortly', 'already', 'just',
}
_DEGREE_MODIFIERS = {
    'more', 'less', 'very', 'much', 'quite', 'somewhat', 'rather', 'fairly',
}

_NP_STOP_WORDS = (STOPWORDS | _DISCOURSE_CONNECTIVES | _FILLER_CONNECTORS
                   | _FILLER_ACK_WORDS | _ENTITY_BLOCKLIST | _TRAILING_ADVERBIALS)

_RE_WORD_TOKEN = re.compile(r"[A-Za-z0-9]+")
_RE_WORD_TOKEN_UNICODE = re.compile(r"[^\W\d_]+", re.UNICODE)
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
            if e - s <= 240:
                return text[s:e]
            lo, hi = _snap_to_word_boundaries(text, max(s, pos_start - 120), min(e, pos_end + 120))
            return text[lo:hi]
    lo, hi = _snap_to_word_boundaries(text, max(0, pos_start - 80), min(len(text), pos_end + 80))
    return text[lo:hi]


def _connective_is_corroborated(text: str, m: 're.Match', verb_pattern) -> bool:
    sentence = _sentence_containing(text, m.start(), m.end())
    return bool(verb_pattern.search(sentence) or _ACTION_DECISION_CUE.search(sentence))


def _has_judgment_signal(text: str) -> bool:
    text = _mask_code_fences(text)
    if any(_verb_match_is_decisive(text, m) for m in _JUDGMENT_VERBS.finditer(text)):
        return True
    return any(_connective_is_corroborated(text, m, _JUDGMENT_VERBS)
               and _verb_match_is_decisive(text, m)
               for m in _JUDGMENT_CONNECTIVES.finditer(text))


def _has_strong_judgment_signal(text: str) -> bool:
    text = _mask_code_fences(text)
    if any(_verb_match_is_decisive(text, m) for m in _STRONG_JUDGMENT_VERBS.finditer(text)):
        return True
    return any(_connective_is_corroborated(text, m, _STRONG_JUDGMENT_VERBS)
               and _verb_match_is_decisive(text, m)
               for m in _JUDGMENT_CONNECTIVES.finditer(text))

def _has_strong_decisive_no_confirm(text: str) -> bool:
    text = _mask_code_fences(text)
    if any(_verb_match_is_decisive(text, m) for m in _STRONG_DECISIVE_VERBS_NO_CONFIRM.finditer(text)):
        return True
    return any(_connective_is_corroborated(text, m, _STRONG_DECISIVE_VERBS_NO_CONFIRM)
               and _verb_match_is_decisive(text, m)
               for m in _JUDGMENT_CONNECTIVES.finditer(text))

def _find_decisive_match(text):
    text = _mask_code_fences(text)
    m = _find_strong_judgment_match(text)
    if m is not None:
        return m
    for m in _JUDGMENT_VERBS.finditer(text):
        if _verb_match_is_decisive(text, m):
            return m
    for m in _JUDGMENT_CONNECTIVES.finditer(text):
        if (_connective_is_corroborated(text, m, _JUDGMENT_VERBS)
                and _verb_match_is_decisive(text, m)):
            return m
    return None

def _find_strong_judgment_match(text: str):
    text = _mask_code_fences(text)
    for m in _STRONG_JUDGMENT_VERBS.finditer(text):
        if _verb_match_is_decisive(text, m):
            return m
    for m in _JUDGMENT_CONNECTIVES.finditer(text):
        if (_connective_is_corroborated(text, m, _STRONG_JUDGMENT_VERBS)
                and _verb_match_is_decisive(text, m)):
            return m
    return None

_RE_IDLIKE_VALUE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{2,29}$')


def _looks_identifier_shaped(cand: str) -> bool:
    c = cand.strip()
    if not _RE_IDLIKE_VALUE.match(c):
        return False
    return bool(re.search(r'\d', c))


def _looks_structural_numeric(cand):
    return bool(re.fullmatch(r'-?\d{1,4}', cand.strip()))


def _is_sane_candidate(s: str) -> bool:
    return '\n' not in s and sum(c.isalnum() for c in s) >= len(s) * 0.4


def _number_is_threshold_shaped(number: str, text: str) -> bool:
    positions = [m.start() for m in re.finditer(re.escape(number), text)]
    if not positions:
        return False
    return all(_preceded_by_threshold_cue(text, p) for p in positions)


def _score_target_candidate(cand, source, tool_name, text, key=None, is_owned=False, local=False):
    low = cand.strip().lower()
    if len(low) < 2 and not _is_numeric_literal(low):
        return -1e9
    if tool_name and low == str(tool_name).strip().lower():
        return -1e9

    score = {'arg': 3.0, 'text_artifact': 2.0, 'text_entity': 1.0}.get(source, 0.0)

    if is_owned:
        score += 2.5

    from_stopwords = low in STOPWORDS or low in _FILLER_ACK_WORDS
    if from_stopwords:
        score -= 2.5
    if _looks_identifier_shaped(cand):
        score += 1.2
    if _looks_structural_numeric(cand) and not _looks_like_reportable_number(cand):
        score -= 1.0
    if _is_numeric_literal(low) and _number_is_threshold_shaped(cand.strip(), text):
        score -= 4.0

    a = _artifacts(cand)
    looks_like_url_or_email = bool(a.get('urls')) or '@' in cand
    looks_like_real_id_or_path = (a.get('ids') or a.get('paths')) and _looks_identifier_shaped(cand)
    if looks_like_url_or_email or looks_like_real_id_or_path:
        score += 2.0

    score += min(1.5, len(cand) / 20.0)
    if ' ' in cand.strip():
        score += 0.3
    if key is not None:
        rank = _key_tier_rank(key)
        if rank is not None:
            n_tiers = len(_CANONICAL_TARGET_KEY_TIERS)
            score += 3.0 - (rank / n_tiers) * 2.0

    if local:
        score += 1.5

    return score


_NUMERIC_LITERAL_RE = re.compile(
    r'^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?%?$'
)

def _is_numeric_literal(s: str) -> bool:
    return bool(_NUMERIC_LITERAL_RE.fullmatch(s.strip()))


def _looks_like_reportable_number(s: str) -> bool:
    if not _is_numeric_literal(s):
        return False
    core = s.rstrip('%').lstrip('-')
    if '.' in core or 'e' in core.lower():
        return True
    return len(core) >= 4

def _extract_all_object_phrases(sentence: str) -> List[str]:
    candidates = []
    for chunk in re.split(r'[,;:]|--|—', sentence):
        words = _RE_WORD_TOKEN.findall(chunk)
        for i, w in enumerate(words):
            wl = w.lower()
            if wl not in _PREPOSITIONS_OBJECT and wl not in _DETERMINERS:
                continue
            j = i + 1
            if wl in _PREPOSITIONS_OBJECT and j < len(words) and words[j].lower() in _DETERMINERS:
                j += 1
            while j < len(words) and words[j].lower() in _DEGREE_MODIFIERS:
                j += 1
            phrase = []
            for w2 in words[j:j + 4]:
                if w2.lower() in _NP_STOP_WORDS:
                    break
                phrase.append(w2)
            if _looks_grounded_np(phrase):
                candidates.append(' '.join(t.lower() for t in phrase))
    seen, out = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def _extract_target(args, text, tool_name=None, decision_idx=None, decisive_span=None):
    candidates = []
    for k, v in _flatten_arg_values(args):
        cand = _stringify_arg_value(v)
        if cand:
            candidates.append((cand, 'arg', k, False))

    arts = _artifacts(text)
    ent = _extract_entity_target(text, known_ids=arts['ids'], decisive_span=decisive_span)
    if ent:
        candidates.append((ent, 'text_entity', None, False))

    if decisive_span is not None:
        obj_phrase = _extract_decisive_object_phrase(text, decisive_span)
        if obj_phrase:
            candidates.append((obj_phrase, 'text_entity', None, False))

        decisive_sentence = _sentence_containing(text, decisive_span[0], decisive_span[1])
        for phrase in _extract_all_object_phrases(decisive_sentence):
            candidates.append((phrase, 'text_entity', None, False))
            
    for kind in ('ids', 'paths'):
        for a in arts[kind]:
            candidates.append((a, 'text_artifact', None, False))

    for a in arts['numbers']:
            candidates.append((a, 'text_artifact', None, False))

    for val, is_owned in _preserved_tag_candidates(text, decision_idx=decision_idx):
        candidates.append((val, 'text_artifact', None, is_owned))

    candidates = [c for c in candidates if _is_sane_candidate(c[0])]
    if not candidates:
        return None

    decisive_sentence = (_sentence_containing(text, decisive_span[0], decisive_span[1])
                          if decisive_span is not None else None)

    scored = [(_score_target_candidate(c, src, tool_name, text, key=k, is_owned=owned,
                                        local=bool(decisive_sentence) and c in decisive_sentence), c)
              for c, src, k, owned in candidates]
    scored.sort(key=lambda x: -x[0])
    best_score, best_cand = scored[0]
    if best_score <= 0:
        return None
    return _cap_target_length(best_cand, max_words=12)

def _try_last_resort(text, role, idx, arts, llm_fallback_fn=None):
    from .reclassify_decisions import find_decision_evidence

    m = find_decision_evidence(text, role)
    if m is not None:
        span = (m.start(), m.end())
        target = _extract_target({}, text, decision_idx=idx, decisive_span=span)
        if _has_concrete_referent(target, None, arts):
            return {
                "type": "judgment", "action": m.group(0).lower(), "target": target,
                "rationale": _extract_rationale(text, arts, decision_values={target} if target else set(), decision_idx=idx),
                "artifacts": {k: arts[k] for k in ("paths", "ids", "errors")},
                "verbatim": text, "msg_idx": idx,
            }

    from .multilingual_decision_detector import MULTILINGUAL_PATTERNS
    for _, pattern in MULTILINGUAL_PATTERNS:
        mm = pattern.search(text)
        if mm:
            span = (mm.start(), mm.end())
            target = _extract_target({}, text, decision_idx=idx, decisive_span=span)
            if _has_concrete_referent(target, None, arts):
                return {
                    "type": "judgment", "action": "request", "target": target,
                    "rationale": _extract_rationale(text, arts, decision_values={target} if target else set(), decision_idx=idx),
                    "artifacts": {k: arts[k] for k in ("paths", "ids", "errors")},
                    "verbatim": text, "msg_idx": idx,
                }

    if llm_fallback_fn is not None:
        return llm_fallback_fn(text, role)

    return None

def _resolve_intent_modal_action(scope_text: str, verb_text: str, match_end: int) -> str:
    """See _RE_INTENT_MODAL_HEAD above: a bare intent-modal match ('d like
    to / want to / need to / wish to) is never itself the action -- walk
    past it to the real verb ('...to CANCEL the order') and return that
    instead. Non-modal matches pass through untouched."""
    if not _RE_INTENT_MODAL_HEAD.match(verb_text.strip()):
        return verb_text
    tail = scope_text[match_end:match_end + 40].strip()
    nxt = re.match(r"[A-Za-z']+", tail)
    if nxt and nxt.group(0).lower() not in _NP_STOP_WORDS:
        return nxt.group(0).lower()
    return 'request'

EXTRACTION_LOGIC_VERSION = "2026-08-02.4"
def _extract_verb(text, decision_idx=None, decisive_span=None):
    text = _mask_code_fences(text)

    def _pick(scope_text):
        matches = [
            m for m in _JUDGMENT_VERBS.finditer(scope_text)
            if not _sentence_looks_like_code(
                _sentence_containing(scope_text, m.start(), m.end()))
        ]
        if not matches:
            return None
        by_text = defaultdict(list)
        for m in matches:
            by_text[m.group(0).lower()].append(m)
        for v in _ACTION_VERB_PRIORITY:
            if v in by_text:
                m = by_text[v][0]
                return _resolve_intent_modal_action(scope_text, v, m.end())
        m = matches[0]
        return _resolve_intent_modal_action(scope_text, m.group(0).lower(), m.end())

    if decisive_span is not None:
        local = _sentence_containing(text, decisive_span[0], decisive_span[1])
        picked = _pick(local)
        if picked:
            return picked

    picked = _pick(text)
    if picked:
        return picked

    for val, _owned in _preserved_tag_candidates(text, decision_idx=decision_idx):
        low = val.strip().lower()
        if _JUDGMENT_VERBS.fullmatch(low) or _JUDGMENT_CONNECTIVES.fullmatch(low):
            return low

    for m in _JUDGMENT_CONNECTIVES.finditer(text):
        if _connective_is_corroborated(text, m, _JUDGMENT_VERBS):
            return m.group(1).lower()

    return 'decide'



def _is_meaningful_rationale_key(key: str) -> bool:
    low = key.strip().lower()
    if len(low) < 2 and not _is_numeric_literal(low):
        return False
    if low in STOPWORDS or low in _FILLER_CONNECTORS or low in _FILLER_ACK_WORDS:
        return False
    return True


def _is_meaningful_candidate_value(val: str) -> bool:
    return _is_meaningful_rationale_key(val)

def _bare_rationale_value(rat: str) -> str:
    if ':' in rat:
        return rat.split(':', 1)[1].strip().lower()
    if '=' in rat:
        return rat.split('=', 1)[1].strip().lower()
    return rat.strip().lower()


def _format_rationale_entry(val: str, default_prefix: str = 'preserved') -> str:
    stripped = str(val).strip()
    if not stripped:
        return stripped
    if re.match(r'^[a-z][a-z0-9_-]*:', stripped, re.I):
        return stripped
    return f'{default_prefix}:{stripped}'

def _extract_rationale(text, arts, decision_values=None, decision_idx=None):
    rat = []
    for val, is_owned in _preserved_tag_candidates(text, decision_idx=decision_idx):
        if not is_owned and decision_values is not None and val not in decision_values:
            continue
        rat.append(_format_rationale_entry(val, default_prefix='preserved'))

    for m in _RE_METRIC_KV_INLINE.finditer(text):
        sentence = _sentence_containing(text, m.start(), m.end())
        if _is_chained_equation(sentence):
            continue
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

def _get_tool_call_args(tc: Optional[dict]) -> dict:
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


_RE_INLINE_TOOL_INVOKE = re.compile(
    r'<(?:\w+:)?invoke\s+name=["\']([A-Za-z_][\w.]*)["\']', re.IGNORECASE)
_RE_INLINE_TOOL_PARAM = re.compile(
    r'<parameter\s+name=["\']([\w.]+)["\']>(.*?)</parameter>',
    re.IGNORECASE | re.DOTALL)
_RE_BRACKET_CALL = re.compile(
    r'\[calls?\s+([A-Za-z_][\w.]*)\s*\(', re.IGNORECASE)


def _extract_inline_tool_call(text):
    m = _RE_INLINE_TOOL_INVOKE.search(text)
    if m:
        name = m.group(1)
        block_end = text.find('</invoke>', m.end())
        block = text[m.end(): block_end if block_end != -1 else len(text)]
        args = {pm.group(1): pm.group(2).strip()
                for pm in _RE_INLINE_TOOL_PARAM.finditer(block)}
        return {"name": name, "args": args}
    m2 = _RE_BRACKET_CALL.search(text)
    if m2:
        return {"name": m2.group(1), "args": {}}
    return None


def _has_concrete_referent(target: Optional[str], tool_name: Optional[str], arts: Dict) -> bool:
    if target:
        return True
    if tool_name:
        return True
    if arts.get('paths') or arts.get('ids'):
        return True
    if arts.get('numbers'):
        return True
    return False


def _confirming_match(text):
    m = _OUTCOME_CONFIRM_SIGNALS.search(text)
    if m:
        return m
    return _CONFIRM_SIGNALS.search(text)


def _anchor_span_for_numbers(text, decisive_span):
    if decisive_span is not None:
        return decisive_span
    m = _confirming_match(text)
    return (m.start(), m.end()) if m is not None else None


_THRESHOLD_CUE = re.compile(
    r'\b(?:under|below|over|above|at least|at most|less than|more than|'
    r'up to|no more than|not (?:to )?exceed(?:ing)?|exceeding|'
    r'threshold|limit|cap(?:ped)?)\s*$',
    re.IGNORECASE)

def _preceded_by_threshold_cue(sentence: str, match_start: int, window: int = 20) -> bool:
    start = max(0, match_start - window)
    return bool(_THRESHOLD_CUE.search(sentence[start:match_start]))


def _full_sentence_containing(text: str, pos_start: int, pos_end: int) -> str:
    for s, e in _sentence_spans(text):
        if s <= pos_start < e:
            return text[s:e]
    lo, hi = _snap_to_word_boundaries(text, max(0, pos_start - 80), min(len(text), pos_end + 80))
    return text[lo:hi]

# extraction.py — new shared helper
def _resolve_multilingual_action(text: str, span: Tuple[int, int]) -> str:
    """Single source of truth for non-English judgment action labels.
    Used identically by ground-truth extraction (_try_multilingual) and
    by the reproducer's deterministic fallback, so the two can never
    disagree on a translated action."""
    from .multilingual_decision_detector import lightweight_translate
    translated = lightweight_translate(text)
    dm = _find_decisive_match(translated)
    if dm is not None:
        return _extract_verb(translated, decisive_span=(dm.start(), dm.end()))
    return 'request'   # only used when translation truly yields no verb signal

def _confirmed_numbers_in_message(text, arts, decisive_span=None, target=None):
    numbers = arts.get('numbers', [])
    if not numbers:
        return set()

    spans = []
    if decisive_span is not None:
        spans.append(decisive_span)
    m = _confirming_match(text)
    if m is not None:
        spans.append((m.start(), m.end()))
    if not spans and target:
        tm = re.search(r'(?<![\d.])' + re.escape(target) + r'(?![\d])', text)
        if tm:
            spans.append((tm.start(), tm.end()))
    if not spans:
        return set()

    found = set()
    for s, e in spans:
        sentence = _full_sentence_containing(text, s, e)
        sentence_hits = []
        for n in numbers:
            if n in found:
                continue
            for num_m in re.finditer(r'(?<![\d.])' + re.escape(n) + r'(?![\d])', sentence):
                sentence_hits.append((n, num_m.start()))
                break

        non_threshold = [n for n, pos in sentence_hits
                          if not _preceded_by_threshold_cue(sentence, pos)]
        keep = non_threshold if non_threshold else [n for n, _ in sentence_hits]
        found.update(keep)

    return found


def _build_decision_for_message(
    messages: List[Dict],
    idx: int,
    llm_fallback_fn=None,
) -> Optional[Dict]:
    """
    Build the decision (or None) for messages[idx] alone, using the exact
    priority-ordered builder chain extract_decisions() applies per-message.

    Pulled out into its own function so it is the SINGLE SOURCE OF TRUTH for
    "what decision, and what action, does this message represent." Both
    ground-truth extraction (extract_decisions, which now just loops and
    calls this) and deterministic reproduction (dagc.reproduce.
    _deterministic_extract, via a reconstructed positional message list)
    call this SAME function.

    ROOT CAUSE this closes: reproduce.py used to re-implement its own
    partial/reordered subset of this priority chain. Whenever an EARLIER,
    non-judgment-verb builder here (imperative directive, confirm-signal,
    certainty assertion, bare-option-answer, polite-request, first-person-
    intent, etc.) claimed a message and its own _extract_verb call landed
    on the generic 'decide' fallback, ground truth recorded action='decide'
    -- but the old reproduction shortcut, seeing decisive_span=None, would
    independently re-check MULTILINGUAL_PATTERNS / the last-resort evidence
    table against the SAME text and often find a translated or
    table-matched verb ground truth never had a chance to consider, since
    ground truth's loop stops at the FIRST successful builder and never
    reaches those later stages once an earlier one wins. Calling this exact
    function on recovered/compressed text instead closes that gap
    structurally: the two paths can no longer take divergent routes through
    the same priority list.
    """
    msg = messages[idx]
    text = _get_text(msg)
    arts = _artifacts(text)

    dm = _find_decisive_match(text)
    decisive_span = (dm.start(), dm.end()) if dm is not None else None

    tool_call = msg.get("tool_call")
    if not isinstance(tool_call, dict):
        tool_call = _extract_inline_tool_call(text)

    def _finalize(dtype, action, target, span_for_numbers, extra_values=None):
        if not _has_concrete_referent(target, None, arts):
            return None
        values = {v for v in (target,) if v}
        if extra_values:
            values |= extra_values
        values |= _confirmed_numbers_in_message(text, arts, span_for_numbers, target=target)
        return {
            "type": dtype,
            "action": action,
            "target": target,
            "rationale": _extract_rationale(text, arts, decision_values=values, decision_idx=idx),
            "artifacts": {k: arts[k] for k in ("paths", "ids", "errors")},
            "verbatim": text,
            "msg_idx": idx,
        }

    def _try_tool_call():
        if not isinstance(tool_call, dict):
            return None
        tool_name = _get_tool_call_name(tool_call, "unknown_tool")
        tool_args = _get_tool_call_args(tool_call)
        target = _extract_target(tool_args, text, tool_name, decision_idx=idx, decisive_span=decisive_span)
        if not _has_concrete_referent(target, tool_name, arts):
            return None
        values = {v for v in (target,) if v} | _confirmed_numbers_in_message(
            text, arts, decisive_span, target=target)
        return {
            "type": "action", "action": tool_name, "target": target,
            "rationale": _extract_rationale(text, arts, decision_values=values, decision_idx=idx),
            "artifacts": {k: arts[k] for k in ("paths", "ids", "errors")},
            "verbatim": text, "msg_idx": idx,
        }

    def _try_strong_judgment():
        if not _has_strong_judgment_signal(text):
            return None
        action = _extract_verb(text, decision_idx=idx, decisive_span=decisive_span)
        target = _extract_target({}, text, decision_idx=idx, decisive_span=decisive_span)
        return _finalize("judgment", action, target, decisive_span)

    def _try_confirm_paths_ids_no_strong():
        if not (_CONFIRM_SIGNALS.search(text) and (arts["paths"] or arts["ids"])
                and not _has_strong_decisive_no_confirm(text)):
            return None
        target = _extract_target({}, text, decision_idx=idx, decisive_span=decisive_span)
        if target is None:
            fallback = [a for a in (arts["paths"] + arts["ids"]) if _is_sane_candidate(a)]
            target = fallback[0] if fallback else None
        return _finalize("confirmation", "confirm", target, decisive_span)

    def _try_outcome_confirm():
        if not (_OUTCOME_CONFIRM_SIGNALS.search(text)
                and (arts["numbers"] or arts["ids"] or arts["paths"])
                and not _has_strong_decisive_no_confirm(text)):
            return None
        target = _extract_target({}, text, decision_idx=idx, decisive_span=decisive_span)
        if target is None:
            fallback = [a for a in (arts["paths"] + arts["ids"] + arts["numbers"]) if _is_sane_candidate(a)]
            target = fallback[0] if fallback else None
        return _finalize("confirmation", "confirm", target, decisive_span)

    def _try_judgment():
        if not _has_judgment_signal(text):
            return None
        action = _extract_verb(text, decision_idx=idx, decisive_span=decisive_span)
        target = _extract_target({}, text, decision_idx=idx, decisive_span=decisive_span)
        return _finalize("judgment", action, target, decisive_span)

    def _try_confirm_paths_ids():
        if not (_CONFIRM_SIGNALS.search(text) and (arts["paths"] or arts["ids"])):
            return None
        target = _extract_target({}, text, decision_idx=idx, decisive_span=decisive_span)
        if target is None:
            fallback = [a for a in (arts["paths"] + arts["ids"]) if _is_sane_candidate(a)]
            target = fallback[0] if fallback else None
        return _finalize("confirmation", "confirm", target, decisive_span)

    def _try_directive_response():
        if not _has_directive_response_signal(text, messages, idx):
            return None
        action = _extract_verb(text, decision_idx=idx, decisive_span=decisive_span)
        target = _extract_target({}, text, decision_idx=idx, decisive_span=decisive_span)
        return _finalize("judgment", action, target, decisive_span)

    def _try_imperative_response():
        if not _has_imperative_response_signal(text, messages, idx):
            return None
        action = _extract_verb(text, decision_idx=idx, decisive_span=decisive_span)
        target = _extract_target({}, text, decision_idx=idx, decisive_span=decisive_span)
        return _finalize("judgment", action, target, decisive_span)

    def _try_standalone_imperative():
        if not _has_standalone_imperative_signal(text):
            return None
        local_span = decisive_span or _leading_verb_span(text)
        action = _extract_verb(text, decision_idx=idx, decisive_span=local_span)
        target = _extract_target({}, text, decision_idx=idx, decisive_span=local_span)
        return _finalize("judgment", action, target, local_span)

    def _try_polite_request():
        m = _polite_request_verb_match(text)
        if m is None:
            return None
        verb_span = (_leading_offset(text) + m.start(1), _leading_offset(text) + m.end(1))
        action = m.group(1).lower()
        target = _extract_target({}, text, decision_idx=idx, decisive_span=verb_span)
        return _finalize("judgment", action, target, verb_span)

    def _try_first_person_intent():
        m = _first_person_intent_match(text)
        if m is None:
            return None
        verb_span = (_leading_offset(text) + m.start(1), _leading_offset(text) + m.end(1))
        action = m.group(1).lower()
        target = _extract_target({}, text, decision_idx=idx, decisive_span=verb_span)
        return _finalize("judgment", action, target, verb_span)

    def _try_certainty_assertion():
        if not _has_certainty_assertion_signal(text):
            return None
        m = _RE_CERTAINTY_ASSERTION.search(text)
        cert_span = (m.start(), m.end())
        target = _extract_target({}, text, decision_idx=idx, decisive_span=cert_span)
        if target is None:
            fallback = [a for a in (arts["ids"] + arts["numbers"]) if _is_sane_candidate(a)]
            target = fallback[0] if fallback else None
        return _finalize("confirmation", "confirm", target, cert_span)

    def _try_bare_option_answer():
        prior_text = _get_text(messages[idx - 1]) if idx > 0 else None
        bare_target = _is_bare_option_answer(text, prior_text)
        if not bare_target:
            return None
        return _finalize("judgment", "decide", bare_target, decisive_span)

    def _try_multilingual():
        from .multilingual_decision_detector import MULTILINGUAL_PATTERNS
        for _, pattern in MULTILINGUAL_PATTERNS:
            m = pattern.search(text)
            if m:
                span = (m.start(), m.end())
                action = _resolve_multilingual_action(text, span)
                target = _extract_target({}, text, decision_idx=idx, decisive_span=span)
                return _finalize("judgment", action, target, span)
        return None

    builders = (
        _try_tool_call,
        _try_strong_judgment,
        _try_confirm_paths_ids_no_strong,
        _try_outcome_confirm,
        _try_judgment,
        _try_confirm_paths_ids,
        _try_directive_response,
        _try_imperative_response,
        _try_standalone_imperative,
        _try_polite_request,
        _try_first_person_intent,
        _try_certainty_assertion,
        _try_bare_option_answer,
        _try_multilingual,
    )

    for build in builders:
        decision = build()
        if decision is not None:
            decision["_extractor_version"] = EXTRACTION_LOGIC_VERSION
            return decision

    fallback = _try_last_resort(text, msg.get("role", ""), idx, arts, llm_fallback_fn=llm_fallback_fn)
    if fallback is not None:
        fallback["_extractor_version"] = EXTRACTION_LOGIC_VERSION
        fallback["_source"] = "last_resort_fallback"
    return fallback


def extract_decisions(
    messages: List[Dict],
    decision_roles: Tuple[str, ...] = ("assistant", "user"),
    llm_fallback_fn=None,
) -> List[Dict]:
    """
    Turn a raw agent trace into a list of structured decisions.

    Thin loop over _build_decision_for_message() -- see that function's
    docstring for the priority-chain rationale that makes it safe to reuse
    from dagc.reproduce for deterministic re-derivation.
    """
    decisions = []
    for idx, msg in enumerate(messages):
        if msg.get("role") not in decision_roles:
            continue
        decision = _build_decision_for_message(messages, idx, llm_fallback_fn=llm_fallback_fn)
        if decision is not None:
            decisions.append(decision)
    return decisions