"""
Decision extractor: turns a raw agent trace into a list of structured
decisions (tool calls, judgments, confirmations) with extracted targets,
rationale, and cited artifacts. Pure regex/heuristic -- no LLM calls.
# extraction.py
EXTRACTION_LOGIC_VERSION = "2026-07-22.1"

# wherever a decision is built and frozen into a fixture
decision_record["_extractor_version"] = EXTRACTION_LOGIC_VERSION

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
EXTRACTION_LOGIC_VERSION = "2026-07-25.2"  # bumped: verb extraction scoped to the
# decisive sentence + preserved-tag/tool-footprint spans masked (fixes verbs
# leaking from other decisions' rescue tags); object-phrase/bigram target
# fallbacks now respect clause boundaries instead of a punctuation-stripped
# flat word list (fixes multi-clause-spanning nonsense targets); decisive
# sentence hard-guaranteed during compression. Fixtures built with
# 2026-07-25.1 or earlier must be regenerated.

_JUDGMENT_VERBS = re.compile(
    r'\b(recommend|conclude|suggest|decide|choos(?:e|es|ing)|chose|select(?:s|ed|ing)?|prefer|'
    r'best|winner|optimal|final|confirm(?:ed|ation|s)?|'
    r'implement|adopt|us(?:e|es|ed|ing)|switch(?:ed|ing)?|migrat(?:e|ed|ing)|'
    r'deploy(?:s|ed|ing)?|provision(?:s|ed|ing)?|'
    r'(?:go(?:es|ing)?|went)\s+with|'
    # NEW: directive/imperative action verbs. Same open-class tradeoff
    # _JUDGMENT_VERBS already accepts (see rationale_ext.py's own
    # docstring on this exact issue) -- these are the specific gaps
    # T001/T004/T005/T006/T009/T011 exposed. Extend further as new
    # gaps turn up; this list is not, and can never be, complete.
    r'keep(?:s|ing)?|remov(?:e|es|ed|ing)|push(?:es|ed|ing)?|merg(?:e|es|ed|ing)|'
    r'mov(?:e|es|ed|ing)|target(?:s|ed|ing)?|delet(?:e|es|ed|ing)|renam(?:e|es|ed|ing)|'
    r'revert(?:s|ed|ing)?)\b',
    re.IGNORECASE)

_RE_CODE_FENCE = re.compile(r'```.*?```', re.S)

def _mask_code_fences(text: str) -> str:
    """Blank out fenced code blocks before judgment-signal/verb detection,
    preserving length and structure so match offsets (used by
    _sentence_containing, decisive_span, etc.) stay valid against the
    original text. Code syntax (import/using/keep/target/...) must never
    be mistaken for a decision verb."""
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

_ACTION_VERB_PRIORITY = [
    'recommend', 'confirm', 'adopt', 'implement', 'select', 'choose',
    'decide', 'prefer', 'suggest', 'conclude', 'use',
    'best', 'optimal', 'winner', 'final',
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
    """True if the sentence contains more than one '='/'≈'-like operator --
    the shape of a multi-step derivation ('A = B = C ≈ D'), not a single
    declarative key-value statement. In this shape, the word immediately
    preceding any one operator is often a fragment of an intermediate
    restatement (e.g. '...number of payments = 7088.34 * 300 ≈ 2,126,502'
    -- '7088.34' is the monthly payment, not 'number of payments'), not
    the true label of the number that follows it -- naive nearest-token
    attribution silently mislabels the value. Requiring a single operator
    per sentence keeps the simple, reliable 'key = value' case and drops
    the ambiguous chained case rather than emitting a confidently wrong
    label. ':'-separated labels (e.g. 'Mirror_1: 18TB_disk') are
    unaffected -- ':' isn't counted as an equation operator here, since
    a paragraph can legitimately contain many unrelated ':' labels
    without any chained-derivation ambiguity."""
    return len(_RE_EQUATION_OPERATOR.findall(sentence)) > 1



_RE_PRESERVED_TAG = re.compile(r'\[preserved:\s*([^\]]+)\]')
_RE_OWNER_SUFFIX = re.compile(r'^(.*?)#d([\d,]+)$')
_RE_WORD_CHAR = re.compile(r'[^\W\d_]', re.UNICODE)

def _snap_to_word_boundaries(text: str, s: int, e: int) -> Tuple[int, int]:
    """Nudge a raw character slice so it never starts or ends mid-word.
    A word-tokenizing regex run on a slice that begins/ends inside a
    word treats the truncated remainder as a real token (e.g.
    'strictly' cut to 'ly') -- it has no way to detect the cut
    happened. Trim the partial word off each edge instead of guessing
    where the original word 'should' have started."""
    while s < e and s > 0 and _RE_WORD_CHAR.match(text[s - 1]) and _RE_WORD_CHAR.match(text[s]):
        s += 1
    while e > s and e < len(text) and _RE_WORD_CHAR.match(text[e - 1]) and _RE_WORD_CHAR.match(text[e]):
        e -= 1
    return s, e

_RE_PRESERVED_TAG = re.compile(r'\[preserved:\s*([^\]]+)\]')
_RE_OWNER_SUFFIX = re.compile(r'^(.*?)#d([\d,]+)$')
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
# ── Bug C: entity fallback needs corroboration, not just capitalization ──

# Closed-class discourse connectives/adverbials. Unlike verbs or entity
# names this genuinely IS a small, finite class in English grammar — 
# enumerating it once covers the category permanently, it isn't the
# same kind of whack-a-mole as enumerating open-class content words.
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

_RE_LOCAL_NEGATION = re.compile(r"\b(?:not|n't|never|no longer|without)\s*$",
                                 re.IGNORECASE)
_RE_SENTENCE_END = re.compile(r'(?<!\d)([.!?])(?!\d)(?:\s+|$)|\n+|\s+--\s+'r'|(?<![A-Za-z0-9_])</?[A-Za-z_][A-Za-z0-9_]{0,29}\s*/?>\s*')

# extraction.py

_RE_PRESERVED_TAG_SPAN = re.compile(r'\[preserved:[^\]]*\]')
_RE_TOOL_FOOTPRINT_SPAN = re.compile(r'→TOOL:[A-Za-z_][\w.]*\([^)]*\)')

def _mask_code_fences(text: str) -> str:
    """Blank out fenced code blocks AND compressor-injected bookkeeping
    spans ([preserved: ...] rescue tags, →TOOL:(...) footprints) before
    judgment-signal/verb detection, preserving length/offsets. These
    spans are DAGC's own accounting artifacts, not language the model
    produced -- a verb-shaped token inside one (e.g. a rescued value
    that belongs to a DIFFERENT decision) must never be mistaken for
    this message's own decisive clause. Same discipline as code-fence
    masking, extended to the other non-semantic span types."""
    text = _RE_CODE_FENCE.sub(lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text)
    text = _RE_PRESERVED_TAG_SPAN.sub(lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text)
    text = _RE_TOOL_FOOTPRINT_SPAN.sub(lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text)
    return text
# Starting points, not final values -- calibrate against your real
# corpus and report what you land on + its effect, same as everything
# else in this pass.
def _extract_question_options(prior_text: str) -> List[str]:
    """Candidate option names from a preceding either/or or open
    question -- reuses the SAME entity/snake_case shapes the rest of
    this file already trusts as identifier-like, plus a plain 'X or Y'
    split for lowercase alternatives ('postgres or mysql?')."""
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
    """
    Detects an elliptical answer-then-justify decision: no judgment verb
    anywhere, but the message's first clause names an option that the
    IMMEDIATELY PRIOR message posed as an alternative. Returns the
    matched option as the target, or None.

    Gated deliberately narrow to avoid false positives:
    - prior message must actually look like a question (ends in '?')
      AND contain >=2 extractable options -- a single-option or
      non-question prior can't corroborate anything.
    - this message's first clause (up to first '.', '!', or newline)
      must be short (<=4 words) -- rules out normal-length statements
      that happen to mention an option word in passing.
    - first clause must NOT itself be interrogative (reuses
      _clause_is_interrogative -- same mood check used elsewhere).
    """
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

def _looks_grounded_np(tokens: List[str]) -> bool:
    """Structural gate, not a word list: modifier+head ('staging
    bucket', 'v2 endpoint') vs. a bare single noun ('meeting', 'flow').
    >=2 content tokens, one token carrying a digit (version/resource
    tag), or a single capitalized token (proper-noun shaped -- product/
    tool/company names like 'Redis', 'Claude', 'Kubernetes') separates
    the grounded shapes from a bare common noun, without naming either
    category. Callers must pass ORIGINAL-CASE tokens for the
    capitalization check to mean anything."""
    if len(tokens) >= 2:
        return True
    if len(tokens) == 1 and any(ch.isdigit() for ch in tokens[0]):
        return True
    if len(tokens) == 1 and tokens[0][:1].isupper():
        return True
    return False


# Closed, tiny grammatical classes -- safe to special-case the same way
# _DEGREE_MODIFIERS already is, unlike open-class content verbs.
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
    # Was two single-shot checks in a fixed order (determiner, then
    # degree modifier) -- "only the X" never got past "only" (a focus
    # adverb, matching neither check) to reach the determiner skip.
    # Loop until none apply, in whatever order they actually occur.
    while i < len(words) and (words[i].lower() in _DETERMINERS
                               or words[i].lower() in _DEGREE_MODIFIERS
                               or words[i].lower() in _FOCUS_ADVERBS):
        i += 1

    # Coordinated lists ("accuracy and latency plots") shouldn't be cut
    # at the conjunction.
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
        # Keep original casing so _looks_grounded_np can recognize
        # single-token proper nouns (product/tool/company names like
        # 'Claude', 'Redis') by capitalization shape, not just digits.
        phrase_tokens.append(w)
        content_count += 1
        if content_count >= 4:
            break

    if not _looks_grounded_np(phrase_tokens):
        return None
    return ' '.join(t.lower() for t in phrase_tokens)

def _is_sentence_initial(text: str, pos: int) -> bool:
    """True if the character at `pos` is the first non-whitespace char
    of its sentence -- meaning any capitalization there is a pure
    orthographic artifact of starting a sentence, not evidence the word
    is a proper noun/identifier. General, position-based -- catches any
    connective, listed or not."""
    for s, e in _sentence_spans(text):
        if s <= pos < e:
            i = s
            while i < e and text[i].isspace():
                i += 1
            return pos == i
    return False


def _best_corroborated_match(matches, text, id_prefixes):
    """
    Among regex matches for one candidate pool (camel or snake), picks
    the most frequent candidate -- but a candidate only counts if it's
    corroborated: it recurs somewhere in the text, OR its one occurrence
    isn't sentence-initial. A single sentence-initial hit is precisely
    the shape of a false positive ("However", "Since", "Based" opening a
    clause) -- this filters that shape out regardless of which specific
    word it is, so it also catches connectives never enumerated above.
    """
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
    """
    Same meaningful-content bar _is_meaningful_rationale_key already
    applies to rationale keys, extended to winner-pattern target
    candidates. A signal word ('best', 'winner', 'recommended', ...)
    matching is necessary but not sufficient -- the captured text after
    it must not be pure function-word filler, or the match latched onto
    incidental phrasing near the signal word rather than a real target.
    Rejects 'to provide' out of 'do my best to provide...' by checking
    its first token is a function word, not by hand-listing the phrase --
    so it also catches the next occurrence of this shape with different
    words.
    """
    words = val.strip().split()
    if not words:
        return False
    first = words[0].lower().strip('.,;:')
    return (first not in STOPWORDS
            and first not in _FILLER_CONNECTORS
            and first not in _FILLER_ACK_WORDS)

def _extract_entity_target(text, known_ids=None, decisive_span=None):
    """
    decisive_span: (start, end) of the verb/connective match that
    qualified this message as a decision, if known. When given, the
    camelCase/snake_case fallback searches only the sentence containing
    that span -- not the whole message -- so it can't return an entity
    that's merely frequent/corroborated elsewhere in unrelated text
    (e.g. code blocks, prior context) but unconnected to the actual
    decisive clause. Falls back to whole-text search only when no span
    is available (matches current behavior for callers that can't
    supply one yet).
    """
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

        # Same clause-boundary discipline as _extract_all_object_phrases:
        # build bigrams PER CHUNK, never across a comma/semicolon/colon/
        # dash, so "prevent data loss, makes use" can't yield "loss makes"
        # and a comma-separated list can't yield a false compound phrase.
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
    """Deterministic mood check, no parser needed. A judgment-shaped word
    inside a QUESTION means someone is being asked to judge, not that a
    judgment was asserted. This gates on sentence STRUCTURE, not
    vocabulary -- transfers across every domain without naming a single
    new word, unlike adding/removing entries from _JUDGMENT_VERBS."""
    c = clause.strip()
    return bool(_RE_INTERROGATIVE_END.search(c)
                or _RE_WH_FRONT.match(c)
                or _RE_AUX_INVERSION_FRONT.match(c))

def _clause_is_hedged(clause: str) -> bool:
    """Epistemic-modality check, same closed-class logic as the
    connectives above: hedge markers are a small, finite grammatical
    class, not open-class vocabulary -- enumerate once, covers the
    category permanently."""
    return bool(_HEDGE_MARKERS.search(clause))


def _verb_locally_negated(text: str, m: 're.Match') -> bool:
    """True only if negation sits immediately before THIS verb match --
    'NOT using', "isn't adopting". Deliberately a short window ending
    at the match, not the whole sentence: negation elsewhere in the
    sentence for a different purpose ('using X, not Y') must not
    suppress a real decision -- only negation attached to the verb
    itself should."""
    window = text[max(0, m.start() - 20):m.start()]
    return bool(_RE_LOCAL_NEGATION.search(window))

def _verb_match_is_decisive(text: str, m: 're.Match') -> bool:
    """Clause-scoped mood check on the sentence containing this match:
    not a question, not hedged, not locally negated."""
    sentence = _sentence_containing(text, m.start(), m.end())
    return (not _clause_is_interrogative(sentence)
            and not _clause_is_hedged(sentence)
            and not _verb_locally_negated(text, m))
def _prior_message_is_interrogative(messages: List[Dict], idx: int) -> bool:
    """True if the message immediately before `idx` reads as a question --
    the same mood check already used to REJECT verb matches inside
    questions, reused here as a POSITIVE signal that idx's message is
    likely answering something."""
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
    """Sibling check to _prior_message_is_interrogative, same idea for
    the other closed mood category: is the immediately preceding
    message a COMMAND rather than a question or a declarative
    statement? Imperative sentences in English open on a bare verb
    with no explicit subject ('Upload X', 'Set Y to Z', 'Delete W') --
    a small, structurally closed distinction, not a verb whitelist:
    any leading pronoun/auxiliary/article instead marks a declarative
    or interrogative sentence, not a command."""
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
    """
    Closed STRUCTURAL signal, sibling to _has_directive_response_signal:
    an assistant turn immediately following a directive/command is a
    decision regardless of which word it uses ('Sent report.pdf to
    Priya', 'Saved config.yaml with port: 8443') -- imperative
    commands ('Upload X', 'Save it', 'Delete Y') are the gap an
    enumerated verb list can never fully close, same as Q&A pairs were
    before this.

    Two conditions required, same discipline as the interrogative case:
      1. The immediately preceding message reads as a command.
      2. This message is not itself a question and not hedged.
    Concreteness is left to the existing _has_concrete_referent gate
    downstream, not duplicated here.
    """
    if not _prior_message_is_imperative_directive(messages, idx):
        return False
    first_clause = re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)[0]
    return not _clause_is_interrogative(first_clause) and not _clause_is_hedged(first_clause)


def _has_directive_response_signal(text: str, messages: List[Dict], idx: int) -> bool:
    """
    Closed STRUCTURAL signal, not another verb for the pile: an assistant
    turn immediately answering a preceding question is a decision
    regardless of which word it uses ("moved to Friday", "cosine
    similarity is the better choice", "goes under cache_v2") -- the gap
    an enumerated verb list can never fully close.

    Two conditions required, mirroring rationale_ext.py's own
    negation+corroboration discipline (one cue alone isn't enough):
      1. The immediately preceding message reads as a question.
      2. This message is not itself a question and not hedged.
    Concreteness is left to the existing _has_concrete_referent gate
    downstream, not duplicated here.
    """
    if not _prior_message_is_interrogative(messages, idx):
        return False
    first_clause = re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)[0]
    return not _clause_is_interrogative(first_clause) and not _clause_is_hedged(first_clause)

def _preserved_tag_candidates(text, decision_idx=None):
    """
    Parses '[preserved: value#dN, value2#dM,K]' tags. Returns a list of
    (value, is_owned) pairs:
      is_owned=True  -> carried an explicit '#dN' suffix that matched
                         decision_idx (or decision_idx was not given).
      is_owned=False -> untagged/legacy entry -- no ownership info, so
                         downstream callers should corroborate it against
                         something else before trusting it for THIS decision.
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
            out.append((value, True))
        else:
            out.append((raw, False))
    return out

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
_DEGREE_MODIFIERS = {                                    # <-- new
    'more', 'less', 'very', 'much', 'quite', 'somewhat', 'rather', 'fairly',
}

_NP_STOP_WORDS = (STOPWORDS | _DISCOURSE_CONNECTIVES | _FILLER_CONNECTORS
                   | _FILLER_ACK_WORDS | _ENTITY_BLOCKLIST
                   | _VAGUE_QUALIFIERS | _TRAILING_ADVERBIALS)

_RE_WORD_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")
_RE_WORD_TOKEN_UNICODE = re.compile(r"[^\W\d_]+", re.UNICODE)
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
    """
    Returns the sentence containing the match. Long sentences are
    windowed to the text immediately around the match rather than
    returned in full, and -- whether windowed or falling back to a raw
    local slice because no sentence span was found -- the result is
    always snapped to word boundaries first (_snap_to_word_boundaries),
    so a downstream word-tokenizing regex never sees a truncated
    fragment (e.g. 'strictly' cut to 'ly') as if it were a real token.
    """
    for s, e in _sentence_spans(text):
        if s <= pos_start < e:
            if e - s <= 240:
                return text[s:e]
            lo, hi = _snap_to_word_boundaries(text, max(s, pos_start - 120), min(e, pos_end + 120))
            return text[lo:hi]
    # No sentence span found at all -- small local window.
    lo, hi = _snap_to_word_boundaries(text, max(0, pos_start - 80), min(len(text), pos_end + 80))
    return text[lo:hi]


def _connective_is_corroborated(text: str, m: 're.Match', verb_pattern) -> bool:
    """A connective match counts only if its own sentence also contains
    a real judgment verb or an explicit action cue."""
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
    """
    Locate the single match — verb or connective — that qualifies this
    message as a decision, using the SAME priority extract_decisions()
    already applies (strong signal first, then general signal). Returns
    a re.Match or None.

    This is the one place "what counts as the decisive clause" is
    decided. Both extract_decisions (ground truth) and reproduce.py's
    deterministic fallback (reproduced side) call this instead of each
    re-implementing their own notion of "the decisive part of the
    text" — so both sides scope the entity fallback to the identical
    sentence, and can no longer silently drift apart the way the
    unscoped whole-message fallback did.
    """
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


def _score_target_candidate(cand, source, tool_name, text, key=None, is_owned=False, local=False):
    low = cand.strip().lower()
    if len(low) < 2:
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
    if _looks_structural_numeric(cand):
        score -= 1.0

    # Self-referential "does this candidate look like an id/path/url"
    # bonus -- gated on the SAME digit-bearing definition
    # _looks_identifier_shaped already uses everywhere else in this file,
    # not a separate, looser check. A dotted token with no digit at all
    # (a library/module path like "scipy.stats.kendalltau") is exactly as
    # "id-shaped" under a loose dots-and-letters pattern as a real
    # business identifier ("order-4471") is -- but only the latter is
    # ever actually the ANSWER to a question. URLs/emails keep their own
    # unambiguous signal (scheme, @) and aren't put through this gate,
    # since that signal doesn't depend on digits at all.
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

    # Locality: a candidate that actually occurs in the same sentence as
    # the verb/connective that MADE this a decision is far more likely to
    # be what the decision is about than a number or phrase sitting in an
    # earlier, unrelated sentence of the same message (a supporting
    # statistic mentioned before the actual verdict, e.g.). This is a
    # BONUS, not a filter -- an out-of-sentence candidate can still win
    # if nothing local scores higher, so a genuinely correct target
    # phrased apart from the decisive clause is never made unreachable,
    # only deprioritized relative to a same-sentence candidate.
    if local:
        score += 1.5

    return score


# A standalone numeric literal, in any form a model would actually report
# one: signed/unsigned integers, decimals, scientific notation, and
# optional percent sign. This exists as its own recognizer -- distinct
# from the generic identifier/entity candidates -- because a numeric
# answer's validity has nothing to do with digit-count length; a
# 17-digit p-value and a 1-digit class label are both legitimate targets,
# just of different shapes. Length-based filtering was conflating "is
# this numeric" with "is this the right numeric", which are different
# questions that need different answers.
_NUMERIC_LITERAL_RE = re.compile(
    r'^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?%?$'
)

def _is_numeric_literal(s: str) -> bool:
    return bool(_NUMERIC_LITERAL_RE.fullmatch(s.strip()))


# Bare short integers (page numbers, list indices, small counts like "14",
# "08") are genuinely likely to be noise on their own -- that part of the
# old heuristic was doing real work and shouldn't be thrown out. The bug
# was applying that same suspicion to EVERY numeric shape, including
# decimals and long digit runs, which are essentially never noise: nobody
# accidentally writes "0.05595605111076938" as filler.
def _looks_like_reportable_number(s: str) -> bool:
    if not _is_numeric_literal(s):
        return False
    core = s.rstrip('%').lstrip('-')
    if '.' in core or 'e' in core.lower():
        return True                      # any decimal/scientific-notation value: always reportable
    return len(core) >= 4                 # bare integers: keep the existing noise filter

def _extract_all_object_phrases(sentence: str) -> List[str]:
    candidates = []
    # A candidate phrase must never straddle a comma/semicolon/colon/
    # dash -- those mark clause boundaries or coordinate list items
    # ("consumption, investment, government spending"). Word-tokenizing
    # discards that punctuation, so without this cut the builder can't
    # tell "the end of one clause" from "an ordinary adjacent word" --
    # it silently glues "loss" to the next clause's "makes".
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
                # Keep original casing so _looks_grounded_np can recognize
                # single-token proper nouns (product/tool/company names
                # like 'Claude', 'Redis') by capitalization shape, not
                # just digits.
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

    # Same sentence _find_decisive_match/entity_target already anchor to --
    # computed once here so every candidate source is judged by the same
    # locality standard, not just the entity fallback.
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

def _extract_verb(text, decision_idx=None, decisive_span=None):
    text = _mask_code_fences(text)

    def _pick(matches):
        for v in _ACTION_VERB_PRIORITY:
            if v in matches:
                return v
        return matches[0] if matches else None

    # Prefer the sentence that actually qualified this message as a
    # decision. A verb-shaped token anywhere ELSE in the message is not
    # evidence about THIS decision -- it may be a different clause, a
    # rescued value's neighbor, or leftover phrasing from earlier
    # compression. Falls back to whole-message scanning only if the
    # decisive sentence itself has no match, so nothing findable today
    # becomes unreachable.
    if decisive_span is not None:
        local = _sentence_containing(text, decisive_span[0], decisive_span[1])
        picked = _pick([g.lower() for g in _JUDGMENT_VERBS.findall(local)])
        if picked:
            return picked

    picked = _pick([g.lower() for g in _JUDGMENT_VERBS.findall(text)])
    if picked:
        return picked

    # Ownership-scoped: only THIS decision's own tagged rescue values
    # count as candidate verbs, not another decision's leaked tag.
    for val, _owned in _preserved_tag_candidates(text, decision_idx=decision_idx):
        low = val.strip().lower()
        if _JUDGMENT_VERBS.fullmatch(low) or _JUDGMENT_CONNECTIVES.fullmatch(low):
            return low

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
    check everything else already relies on.

    NOTE: this previously also rejected bare short numerics (e.g. "14",
    "08", "150", "100%") on the theory they're usually incidental noise
    (a disk-size fragment). In practice every value reaching this gate
    has ALREADY been vetted upstream -- by _extract_rationale's own
    digit-or-identifier-shaped check for rationale entries, or by
    _score_target_candidate's ranking for targets -- so re-rejecting
    short numbers here doesn't filter noise that got through; it drops
    real decision-critical numbers (percentages, day counts, dollar
    amounts, class counts) that upstream extraction had already
    correctly identified as meaningful. That gap is what leaves them
    unprotected during compression.
    """
    return _is_meaningful_rationale_key(val)

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
    rat = []
    for val, is_owned in _preserved_tag_candidates(text, decision_idx=decision_idx):
        # Only gate UNTAGGED/legacy entries against decision_values.
        # Owner-tagged entries are already correctly scoped -- don't
        # additionally require them to equal the target/action.
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

def extract_decisions(
    messages: List[Dict],
    decision_roles: Tuple[str, ...] = ("assistant",),) -> List[Dict]:
    decisions = []

    for idx, msg in enumerate(messages):
        if msg.get("role") not in decision_roles:
            continue

        text = _get_text(msg)
        arts = _artifacts(text)

        dm = _find_decisive_match(text)
        decisive_span = (dm.start(), dm.end()) if dm is not None else None

        if isinstance(msg.get("tool_call"), dict):
            tc = msg["tool_call"]
            tool_name = _get_tool_call_name(tc, "unknown_tool")
            tool_args = _get_tool_call_args(tc)

            target = _extract_target(
                tool_args,
                text,
                tool_name,
                decision_idx=idx,
                decisive_span=decisive_span,
            )

            if not _has_concrete_referent(target, tool_name, arts):
                continue

            this_decision_values = {v for v in (target, tool_name) if v}

            decisions.append(
                {
                    "type": "action",
                    "action": tool_name,
                    "target": target,
                    "rationale": _extract_rationale(
                        text,
                        arts,
                        decision_values=this_decision_values,
                        decision_idx=idx,
                    ),
                    "artifacts": {
                        k: arts[k] for k in ("paths", "ids", "errors")
                    },
                    "verbatim": text,
                    "msg_idx": idx,
                }
            )

        elif _has_strong_judgment_signal(text):
            action = _extract_verb(text, decision_idx=idx, decisive_span=decisive_span)

            target = _extract_target(
                {},
                text,
                decision_idx=idx,
                decisive_span=decisive_span,
            )

            if not _has_concrete_referent(target, None, arts):
                continue

            this_decision_values = {v for v in (target, action) if v}

            decisions.append(
                {
                    "type": "judgment",
                    "action": action,
                    "target": target,
                    "rationale": _extract_rationale(
                        text,
                        arts,
                        decision_values=this_decision_values,
                        decision_idx=idx,
                    ),
                    "artifacts": {
                        k: arts[k] for k in ("paths", "ids", "errors")
                    },
                    "verbatim": text,
                    "msg_idx": idx,
                }
            )

        elif (
            _CONFIRM_SIGNALS.search(text)
            and (arts["paths"] or arts["ids"])
            and not _has_strong_decisive_no_confirm(text)
        ):
            target = _extract_target(
                {},
                text,
                decision_idx=idx,
                decisive_span=decisive_span,
            )

            if target is None:
                fallback = [
                    a
                    for a in (arts["paths"] + arts["ids"])
                    if _is_sane_candidate(a)
                ]
                target = fallback[0] if fallback else None

            if not _has_concrete_referent(target, None, arts):
                continue

            this_decision_values = {v for v in (target,) if v}

            decisions.append(
                {
                    "type": "confirmation",
                    "action": "confirm",
                    "target": target,
                    "rationale": _extract_rationale(
                        text,
                        arts,
                        decision_values=this_decision_values,
                        decision_idx=idx,
                    ),
                    "artifacts": {
                        k: arts[k] for k in ("paths", "ids", "errors")
                    },
                    "verbatim": text,
                    "msg_idx": idx,
                }
            )

        elif _has_judgment_signal(text):
            action = _extract_verb(text, decision_idx=idx, decisive_span=decisive_span)

            target = _extract_target(
                {},
                text,
                decision_idx=idx,
                decisive_span=decisive_span,
            )

            if not _has_concrete_referent(target, None, arts):
                continue

            this_decision_values = {v for v in (target, action) if v}

            decisions.append(
                {
                    "type": "judgment",
                    "action": action,
                    "target": target,
                    "rationale": _extract_rationale(
                        text,
                        arts,
                        decision_values=this_decision_values,
                        decision_idx=idx,
                    ),
                    "artifacts": {
                        k: arts[k] for k in ("paths", "ids", "errors")
                    },
                    "verbatim": text,
                    "msg_idx": idx,
                }
            )

        elif _CONFIRM_SIGNALS.search(text) and (
            arts["paths"] or arts["ids"]
        ):
            target = _extract_target(
                {},
                text,
                decision_idx=idx,
                decisive_span=decisive_span,
            )

            if target is None:
                fallback = [
                    a
                    for a in (arts["paths"] + arts["ids"])
                    if _is_sane_candidate(a)
                ]
                target = fallback[0] if fallback else None

            if not _has_concrete_referent(target, None, arts):
                continue

            this_decision_values = {v for v in (target,) if v}

            decisions.append(
                {
                    "type": "confirmation",
                    "action": "confirm",
                    "target": target,
                    "rationale": _extract_rationale(
                        text,
                        arts,
                        decision_values=this_decision_values,
                        decision_idx=idx,
                    ),
                    "artifacts": {
                        k: arts[k] for k in ("paths", "ids", "errors")
                    },
                    "verbatim": text,
                    "msg_idx": idx,
                }
            )

        elif _has_directive_response_signal(text, messages, idx):
            action = _extract_verb(text, decision_idx=idx, decisive_span=decisive_span)
            
            target = _extract_target(
                {}, 
                text, 
                decision_idx=idx, 
                decisive_span=decisive_span
            )
            
            if not _has_concrete_referent(target, None, arts):
                continue
                
            this_decision_values = {v for v in (target, action) if v}
            
            decisions.append(
                {
                    "type": "judgment",
                    "action": action,
                    "target": target,
                    "rationale": _extract_rationale(
                        text, 
                        arts, 
                        decision_values=this_decision_values, 
                        decision_idx=idx
                    ),
                    "artifacts": {k: arts[k] for k in ("paths", "ids", "errors")},
                    "verbatim": text,
                    "msg_idx": idx,
                }
            )

        elif _has_imperative_response_signal(text, messages, idx):
            action = _extract_verb(text, decision_idx=idx, decisive_span=decisive_span)

            target = _extract_target(
                {},
                text,
                decision_idx=idx,
                decisive_span=decisive_span
            )

            if not _has_concrete_referent(target, None, arts):
                continue

            this_decision_values = {v for v in (target, action) if v}

            decisions.append(
                {
                    "type": "judgment",
                    "action": action,
                    "target": target,
                    "rationale": _extract_rationale(
                        text,
                        arts,
                        decision_values=this_decision_values,
                        decision_idx=idx
                    ),
                    "artifacts": {k: arts[k] for k in ("paths", "ids", "errors")},
                    "verbatim": text,
                    "msg_idx": idx,
                }
            )
        elif _has_imperative_response_signal(text, messages, idx):
            action = _extract_verb(text, decision_idx=idx, decisive_span=decisive_span)

            target = _extract_target(
                {},
                text,
                decision_idx=idx,
                decisive_span=decisive_span
            )

            if not _has_concrete_referent(target, None, arts):
                continue

            this_decision_values = {v for v in (target, action) if v}

            decisions.append(
                {
                    "type": "judgment",
                    "action": action,
                    "target": target,
                    "rationale": _extract_rationale(
                        text,
                        arts,
                        decision_values=this_decision_values,
                        decision_idx=idx
                    ),
                    "artifacts": {k: arts[k] for k in ("paths", "ids", "errors")},
                    "verbatim": text,
                    "msg_idx": idx,
                }
            )

        else:
            prior_text = (
                _get_text(messages[idx - 1]) if idx > 0 else None
            )
            bare_target = _is_bare_option_answer(text, prior_text)

            if bare_target and _has_concrete_referent(bare_target, None, arts):
                decisions.append(
                    {
                        "type": "judgment",
                        "action": "decide",
                        "target": bare_target,
                        "rationale": _extract_rationale(
                            text, arts, decision_values={bare_target}, decision_idx=idx,
                        ),
                        "artifacts": {k: arts[k] for k in ("paths", "ids", "errors")},
                        "verbatim": text,
                        "msg_idx": idx,
                        "_extractor_version": EXTRACTION_LOGIC_VERSION,
                    }
                )

    return decisions
