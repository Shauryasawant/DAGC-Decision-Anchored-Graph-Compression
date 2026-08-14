"""
Shared utilities: artifact/ID extraction, tokenization, embedding cache,
sentence splitting. No LLM dependency anywhere in this module.
"""
from __future__ import annotations
import re
from functools import lru_cache
from typing import Dict, List, Tuple
from enum import Enum, auto
import numpy as np
import re as _re
from .config import runtime

CRITICAL = {
    'error', 'exception', 'warning', 'fail', 'success', 'result', 'metric', 'score',
    'recommend', 'decision', 'conclusion', 'path', 'file', 'experiment', 'key', 'final',
    'found', 'fix', 'next', 'required', 'critical', 'important', 'output', 'summary'
}
STOPWORDS = {
    'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'then', 'than',
    'what', 'when', 'where', 'read', 'give', 'show', 'tell', 'make', 'does', 'will',
    'would', 'could', 'should', 'have', 'your', 'task', 'same', 'rate', 'also',
    'about', 'while', 'want', 'need', 'please', 'provide', 'compare', 'answer',
    'original', 'estimate', 'meaning', 'just', 'very', 'like', 'more', 'you', 'your', 'yours', 'using', 'use', 'used',
    'to', 'for', 'of', 'in', 'on', 'at', 'by', 'with',
    'i', 'me', 'my', 'we', 'us', 'our', 'he', 'she', 'they', 'them', 'their',
    'it', 'its', 'this', 'that', 'these', 'those',
}

_RE_PATH = re.compile(
    r'(?<![A-Za-z0-9<])'
    r'(?:/[\w.-]+(?:/[\w.-]+)+(?<![.,;:!?])'
    r'|/[\w.-]*\d[\w.-]*(?<![.,;:!?])'
    r'|[A-Za-z]:[\\/][\w.\\/-]+(?<![.,;:!?])'
    r'|\b[A-Za-z][\w.-]*(?:/[\w.-]+)+(?<![.,;:!?])'        # relative path / branch name
    r'|(?<![\w.])\.[A-Za-z_][\w.-]{1,30}\b(?<![.,;:!?])'   # dotfile
    r'|(?<!,)\b[\w][\w\-]{2,60}\.(?=[a-z0-9]*[a-z])[a-z0-9]{1,5}\b(?<![.,;:!?]))')

_RE_ID_ALNUM = re.compile(r'\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{8,15}\b')
_RE_URL = re.compile(r'https?://\S+')
_RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
_RE_ID_AZ = re.compile(r'\b[A-Z][A-Z0-9]{1,}-[A-Z0-9]*\d[A-Z0-9]*\b')
_RE_ID_HX = re.compile(r'\b[a-f0-9]{8,32}\b')
_RE_COORD = re.compile(r'-?\d{1,3}\.\d{3,8},-?\d{1,3}\.\d{3,8}')
_RE_NUM = re.compile(r'\$?\b\d{1,3}(?:,\d{3})+(?:\.\d+)?%?\b|\$?\b\d+(?:\.\d+)?%?\b')
_RE_ERR = re.compile(r'(?i)(?:error|exception|warning):\s.{0,120}')

_INFRA_NOISE_PATTERNS = (
    re.compile(r'message serialization failed', re.I),
    re.compile(r'is deprecated in jupyter-client', re.I),
    re.compile(r'^\s*content\s*=\s*self\.pack', re.I),
    re.compile(r'out of range float values are not json compliant', re.I),
)
_RE_PATH_IDIOM_FALSE_POSITIVE = re.compile(
    r'^(?:and/or|his/her|he/she|him/her|her/him|yes/no|on/off|'
    r'true/false|either/or|before/after|then/now|input/output|'
    r'read/write)$', re.I)
_RE_SLUG_ID = re.compile(
    r'\b[A-Za-z][A-Za-z0-9]*[_-][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b'
)
_STATE_WORDS = r'enabled|disabled|active|inactive|true|false|on|off|open|closed|locked|unlocked|running|stopped|paused'

_RE_STATE_KV = re.compile(
    r'\b([A-Za-z_][A-Za-z0-9_]{1,40})\s*(?:=|:|->|set to|switched to|turned)\s*'
    r'(' + _STATE_WORDS + r')\b',
    re.IGNORECASE
)

def _is_infra_noise(err: str) -> bool:
    """True for known Jupyter/runtime boilerplate warnings that are
    structurally errors (RE_ERR matches them correctly) but carry no
    decision-relevant meaning -- e.g. NaN JSON-serialization notices
    from the notebook kernel, not anything the model reasoned about."""
    return any(p.search(err) for p in _INFRA_NOISE_PATTERNS)


_RE_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=\S)')
DOMAIN_ID_PATTERNS: Dict[str, re.Pattern] = {
    'banking_finance': re.compile(r'\b(?:IBAN[:\s]?[A-Z]{2}\d{2}[A-Z0-9]{10,30}|ACCT-\d{6,12}|SWIFT-[A-Z0-9]{8,11}|TXN-[A-Z0-9]{8,16})\b', re.I),
    'healthcare': re.compile(r'\b(?:MRN-\d{6,10}|NPI-\d{10}|ICD-10-[A-Z]\d{2}(?:\.\d{1,2})?|CPT-\d{5})\b', re.I),
    'insurance': re.compile(r'\b(?:CLAIM-\d{6,10}|POLICY-[A-Z0-9]{6,12})\b', re.I),
    'legal': re.compile(r'\b(?:DOCKET-[A-Z0-9-]{4,15}|CASE-\d{2,4}-[A-Z0-9]{4,10})\b', re.I),
    'hr': re.compile(r'\bEMP-\d{4,8}\b', re.I),
    'manufacturing': re.compile(r'\b(?:PART-[A-Z0-9]{4,12}|LOT-[A-Z0-9]{4,12}|BATCH-[A-Z0-9]{4,12})\b', re.I),
    'supply_chain': re.compile(r'\b(?:SHIP-[A-Z0-9]{6,14}|TRACK-[A-Z0-9]{6,20}|PO-\d{5,10})\b', re.I),
    'retail_ecommerce': re.compile(r'\b(?:SKU-[A-Z0-9]{4,12}|ORDER-\d{6,12})\b', re.I),
    'telecom': re.compile(r'\b(?:IMEI-\d{15}|MSISDN-\d{10,15})\b', re.I),
    'government': re.compile(r'\b(?:CASE-GOV-\d{4,10}|FILE-\d{4,10})\b', re.I),
    'education': re.compile(r'\bSTU-\d{4,10}\b', re.I),
    'travel_hospitality': re.compile(r'\b(?:PNR-[A-Z0-9]{6}|BOOKING-[A-Z0-9]{6,12})\b', re.I),
    'cybersecurity_soc': re.compile(r'\b(?:CVE-\d{4}-\d{4,7}|INCIDENT-\d{4,10})\b', re.I),
    'pharmaceutical': re.compile(r'\b(?:NDC-\d{4,5}-\d{3,4}-\d{1,2}|BATCH-RX-[A-Z0-9]{4,12})\b', re.I),
}

def _loose_normalise(s):
    if s is None:
        return ''
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

# Shape test for "does this value get word-boundary matching, or the loose
# concatenated-substring match below". Deliberately kept identical in scope
# to compressor.py's _art_in_text condition (`re.fullmatch(r'[a-z0-9_]+', ...)`),
# not just pure alphabetic strings -- see _value_still_recoverable docstring
# for why. Name kept as _RE_PLAIN_WORD (not renamed) to avoid touching any
# other call sites in the package that may reference it.
_RE_PLAIN_WORD = re.compile(r'^[A-Za-z0-9_]+$')

def _value_still_recoverable(value, text):
    """Core check: is this string still literally present (loosely
    normalised) in the surviving text? Shared primitive underlying
    target_still_recoverable and action_still_recoverable -- one
    recoverability rule, applied uniformly, rather than a heuristic
    re-implemented per field.

    Matching modes, chosen by shape:
      - Purely numeric tokens (e.g. '12', '14', '08') use a DIGIT-ONLY
        boundary: (?<![0-9])12(?![0-9]). These values are commonly
        glued to a unit suffix in source text ("12" -> "12TB_disk"),
        so a following/preceding letter is not evidence of a different
        number and must not block the match. A following/preceding
        DIGIT still blocks it, so '12' correctly still fails to match
        inside '120' or '112' -- the original false-positive case this
        function guards against is unaffected.
      - Other alphanumeric-plus-underscore tokens with no other
        punctuation (action verbs like 'use', 'confirm', id-shaped
        values like 'step_2' or 'v2') keep the full alnum-boundary
        match. Loose-normalise strips spaces along with punctuation,
        so a bare substring check on the result would silently match
        'use' inside 'used'/'user'/'because'/'cause', or 'v2' inside
        'review2' -- exactly the false-positive shape _art_in_text
        already guards against for compressor-side coverage tracking.
        Recoverability must use the same standard here it uses there,
        for the same token shape, or DME-Verify's "is this still here"
        check can disagree with the compressor's "did I keep this"
        check for the same value.
      - Everything else (ids, paths, multi-word phrases, values with
        punctuation such as hyphens) keeps the loose concatenated-
        substring check -- tolerating reformatting (e.g. "TXN-123"
        surviving as "TXN123") is correct, intended behavior for that
        shape, and is unaffected by either boundary rule above since
        any value containing a hyphen or space still falls through to
        this branch.
    """
    if not value:
        return False
    value_str = str(value).strip()
    v_norm = _loose_normalise(value)
    if not v_norm:
        return False
    if _RE_PLAIN_WORD.match(value_str):
        text_l = str(text).lower()
        if re.fullmatch(r'\d+', value_str):
            pattern = re.compile(r'(?<![0-9])' + re.escape(value_str) + r'(?![0-9])')
        else:
            pattern = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(value_str.lower()) + r'(?![A-Za-z0-9_])')
        return bool(pattern.search(text_l))
    return v_norm in _loose_normalise(text)


def target_still_recoverable(target, text, arts=None):
    """True if the ground-truth target is still findable in this message's
    surviving text/artifacts -- i.e. verify, don't re-derive."""
    if _value_still_recoverable(target, text):
        return True
    if arts:
        t_norm = _loose_normalise(target) if target else None
        if t_norm:
            for kind in ('ids', 'paths', 'numbers'):
                for a in arts.get(kind, []):
                    if _loose_normalise(a) == t_norm:
                        return True
    return False


def action_still_recoverable(action, text):
    """Same recoverability check as target_still_recoverable, applied to
    the decision's action verb. No artifact-list fallback -- actions are
    verbs, not ids/paths/numbers, so there's nothing to check there."""
    return _value_still_recoverable(action, text)

def register_domain_pattern(name: str, pattern: str, flags=re.IGNORECASE) -> None:
    """Register a domain identifier pattern that feeds the standard IDs bucket."""
    DOMAIN_ID_PATTERNS[name] = re.compile(pattern, flags)


def _domain_ids(text: str) -> List[str]:
    out: List[str] = []
    for pat in DOMAIN_ID_PATTERNS.values():
        out.extend(m.group(0) for m in pat.finditer(text))
    return out


def _tok(text) -> int:
    return len(runtime.tokenizer.encode(str(text)))


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _cos_norm(a, b, norm_a: float, norm_b: float) -> float:
    """Same formula as _cos(), but takes pre-computed norms for a and b
    instead of recomputing np.linalg.norm on every call. Use when the same
    vectors get compared repeatedly (e.g. a greedy selection loop) and
    their norms are already known -- identical result, less redundant work.
    """
    return float(np.dot(a, b) / (norm_a * norm_b + 1e-12))


def _uw(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


_EMBED_CACHE: Dict[str, np.ndarray] = {}


def clear_embed_cache() -> None:
    _EMBED_CACHE.clear()


def _encode(texts, max_chunk=200):
    uncached_idx = [i for i, t in enumerate(texts) if str(t) not in _EMBED_CACHE]
    if uncached_idx:
        fresh = _encode_uncached([texts[i] for i in uncached_idx], max_chunk)
        for i, emb in zip(uncached_idx, fresh):
            _EMBED_CACHE[str(texts[i])] = emb
    return np.vstack([_EMBED_CACHE[str(t)] for t in texts])


def _encode_uncached(texts, max_chunk=200):
    out = []
    for text in texts:
        toks = runtime.tokenizer.encode(str(text))
        if len(toks) <= max_chunk:
            emb = runtime.embedder.encode([text], normalize_embeddings=False)[0]
        else:
            step = max_chunk - 20
            chunks = [runtime.tokenizer.decode(toks[i:i + max_chunk])
                      for i in range(0, len(toks), step)]
            emb = runtime.embedder.encode(chunks, normalize_embeddings=False).mean(axis=0)
        out.append(_norm(np.array(emb, dtype=np.float32)))
    return out


import ast

_RE_STRINGIFIED_CONTENT_BLOCKS = re.compile(r"^\s*\[\s*\{.*\}\s*\]\s*$", re.S)

def _extract_text_from_content_blocks(blocks) -> str:
    """blocks: list of content-block dicts (OpenAI Responses-API shape,
    e.g. [{'type': 'output_text', 'text': '...'}]). Concatenates the
    'text' fields in order; blocks with no 'text' are skipped."""
    out = []
    for b in blocks:
        if isinstance(b, dict):
            t = b.get('text')
            if isinstance(t, str):
                out.append(t)
    return ' '.join(out)


def _normalize_content_value(v):
    """Return the real prose string behind msg['content'].

    Three shapes:
      - plain str -> returned as-is (the common case, unaffected).
      - a native list of content-block dicts (not yet stringified) ->
        text fields extracted and joined. Previously silently dropped.
      - a string that is itself a Python repr of that same list shape
        (content stringified upstream, e.g. str(response.content)
        before the trace was serialized to JSON) -> parsed back via
        ast.literal_eval and handled like the native-list case.
        Falls back to the raw string unchanged if parsing fails or
        the parsed value doesn't look like content blocks, so an
        unrelated string that happens to start with '[{' is never
        mangled.
    """
    if isinstance(v, list):
        return _extract_text_from_content_blocks(v)
    if isinstance(v, str) and _RE_STRINGIFIED_CONTENT_BLOCKS.match(v):
        try:
            parsed = ast.literal_eval(v)
        except Exception:
            return v
        if isinstance(parsed, list) and any(
                isinstance(b, dict) and 'text' in b for b in parsed):
            return _extract_text_from_content_blocks(parsed)
        return v
    if isinstance(v, str):
        return v
    return None


def _get_text(msg):
    parts = []
    for k in ('content', 'name'):
        v = msg.get(k)
        normalized = _normalize_content_value(v)
        if normalized:
            parts.append(normalized)
    tc = msg.get('tool_call')
    if isinstance(tc, dict):
        import json
        parts.append(json.dumps(tc, sort_keys=True))
    return ' '.join(parts)


@lru_cache(maxsize=4096)
def _artifacts_cached(text: str) -> Dict[str, Tuple[str, ...]]:
    paths = [p for p in _RE_PATH.findall(text)
             if not _RE_PATH_IDIOM_FALSE_POSITIVE.match(p)]

    # Character spans occupied by list-item enumeration numerals
    # ("10." at the start of a line) -- excluded below so they never
    # enter the numeric-artifact pool as if they were real values.
    list_marker_spans = [m.span(1) for m in _RE_LIST_MARKER_NUM.finditer(text)]

    def _is_list_marker(m):
        return any(s <= m.start() and m.end() <= e for s, e in list_marker_spans)

    numbers = [m.group(0) for m in _RE_NUM.finditer(text) if not _is_list_marker(m)]
    states = [f"{m.group(1)}={m.group(2).lower()}" for m in _RE_STATE_KV.finditer(text)]

    return {
        'paths': tuple(_uw(paths)),
        'urls': tuple(_uw(_RE_URL.findall(text))),
        'emails': tuple(_uw(_RE_EMAIL.findall(text))),
        'ids': tuple(_uw(_RE_ID_AZ.findall(text) + _RE_ID_HX.findall(text)
                          + _RE_ID_ALNUM.findall(text) + _RE_COORD.findall(text)
                          + _RE_SLUG_ID.findall(text)
                          + _domain_ids(text))),
        'numbers': tuple(_uw(numbers)),
        'states': tuple(_uw(states)),
        'errors': tuple(e for e in _uw(_RE_ERR.findall(text)) if not _is_infra_noise(e)),
    }

def _artifacts(text):
    return {k: list(v) for k, v in _artifacts_cached(text).items()}


def _art_count(text):
    a = _artifacts(text)
    return sum(len(a[k]) for k in ('paths', 'ids', 'urls', 'errors'))


def _art_bonus(s):
    a = _artifacts(s)
    return min(1.0,
        0.40 * min(1.0, len(a['paths']) / 1) + 0.30 * min(1.0, len(a['ids']) / 1) +
        0.25 * min(1.0, len(a['errors']) / 1) + 0.10 * min(1.0, len(a['numbers']) / 3))


def _art_density(text):
    return _art_count(text) / max(1, _tok(text))


_RE_LIST_ITEM = re.compile(
    r'(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+'   # "- ", "* ", "• ", "1. ", "2) "
)
_RE_LIST_MARKER_NUM = re.compile(r'(?:^|\n)\s*(\d+)[.)]\s+')


_RE_CODE_FENCE = re.compile(r'```.*?```', re.S)


def _split_sents_plain(text, min_t=4):
    """Original line/sentence splitter for text OUTSIDE fenced code
    blocks. Byte-identical to the pre-fence-fix _split_sents body --
    factored out so the fence-aware wrapper below can apply it only to
    fence-free segments, without touching non-fence behavior at all."""
    out = []
    lines = _RE_LIST_ITEM.split(text) if _RE_LIST_ITEM.search(text) else [text]
    for line in lines:
        for raw_line in line.split('\n'):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            for s in _RE_SENT.split(raw_line):
                s = s.strip()
                if s and (_art_count(s) > 0 or _tok(s) >= min_t):
                    out.append(s)
    return out


def _needs_fence_protection(line: str) -> bool:
    """
    True only if this line contains something that could be
    misinterpreted as a decision verb/connective once its surrounding
``` fence is gone -- i.e. after the compressor selects and rejoins a
    subset of lines pulled from inside a code block, dropping the
    original fence delimiters in the process.

    Lazy-imported from extraction.py to avoid a circular import:
    extraction.py already imports from utils.py at module load time, so
    utils.py importing extraction.py at ITS top level would fail. By the
    time _split_sents actually runs (well after both modules finish
    loading), the import below is a cheap already-cached lookup, not a
    real re-import.

    Deliberately narrow: the large majority of code lines (assignments,
    bare function calls, data literals) contain no English judgment
    verb and never match either pattern, so they pay zero fence
    overhead. Only the rare line that could actually flip an extracted
    action verb once unfenced gets the extra backtick tokens -- so
    compression is only sacrificed exactly where correctness requires
    it, never as a blanket cost across every code line.
    """
    from .extraction import _JUDGMENT_VERBS, _JUDGMENT_CONNECTIVES
    return bool(_JUDGMENT_VERBS.search(line) or _JUDGMENT_CONNECTIVES.search(line))


def _split_sents(text, min_t=4):
    """
    Sentence/line splitter used to build the compressor's selection pool.

    Fence-aware, cost-minimized: a fenced code block (```...```) is
    walked separately from surrounding prose. Lines extracted from
    INSIDE a fence are re-wrapped in their own ``` markers ONLY when
    _needs_fence_protection says the line could be misread as a
    decision verb/connective once unfenced -- lines with no such
    content are emitted exactly as before, with zero added tokens, so
    compression on code-heavy traces is unaffected except where the
    correctness fix actually requires it.

    Why re-fencing is needed at all: the compressor selects a SUBSET of
    these lines and rejoins them with spaces when building compressed
    message content. Without per-line re-fencing, that reassembled text
    can lose its fence delimiters entirely -- even though the original
    message had them -- letting code syntax/comments be read as plain
    prose by extraction._mask_code_fences on the compressed side while
    the original side (still fenced) correctly masks the same content.
    That asymmetry could make a decision's action verb extract
    differently on the two sides purely because of code content neither
    side was meant to treat as English text.

    Prose outside any fence is completely unaffected -- routed through
    _split_sents_plain unchanged.
    """
    out = []
    last_end = 0
    for fence_m in _RE_CODE_FENCE.finditer(text):
        out.extend(_split_sents_plain(text[last_end:fence_m.start()], min_t))

        fence_body = fence_m.group(0)
        inner = fence_body.strip('`')
        inner_lines = inner.split('\n')
        # Drop an optional language tag on the fence's opening line
        # (```R, ```python, ```json, ...) -- it's not code content and
        # would otherwise be re-emitted as its own bogus "sentence".
        if inner_lines and re.fullmatch(r'[A-Za-z0-9_+-]{0,12}', inner_lines[0].strip()):
            inner_lines = inner_lines[1:]

        # FIX (code-block atomicity): previously each line inside a fence
        # was scored and selected INDEPENDENTLY, so the selector could keep
        # e.g. `labels = data.keys()` while dropping `data = {...}` --
        # syntactically broken code kept while looking "selected" fine.
        # A fenced block has hard line-to-line dependencies a sentence-
        # level scorer has no way to see, so it's now treated as ONE
        # atomic candidate: the whole block survives selection or none of
        # it does. This only changes granularity going INTO the selection
        # pool -- target_arts, by_decision, phase1/phase2 budgeting, and
        # the hard-guarantee rescue passes are all unaffected; decision-
        # critical values inside code are still separately protected via
        # target_arts_all regardless of this block being selected or not.
        has_content = any(
            _art_count(ln.strip()) > 0 or _tok(ln.strip()) >= min_t
            for ln in inner_lines if ln.strip()
        )
        if has_content:
            out.append(fence_body)
        last_end = fence_m.end()

    out.extend(_split_sents_plain(text[last_end:], min_t))
    return out


def _head_tail_cap(content: str, max_toks: int, head_frac: float = 0.55) -> str:
    toks = runtime.tokenizer.encode(content)
    if len(toks) <= max_toks:
        return content
    head_n = int(max_toks * head_frac)
    tail_n = max_toks - head_n

    head = runtime.tokenizer.decode(toks[:head_n])
    tail = runtime.tokenizer.decode(toks[-tail_n:]) if tail_n > 5 else ''
    result = (head + ' […] ' + tail).strip()

    # Guard: widen the cut (bounded) if it fragmented an artifact
    # instead of dropping it cleanly. Uses only primitives already in
    # this module -- _artifacts, _value_still_recoverable.
    arts = _artifacts(content)
    all_values = [v for k in ('paths', 'ids', 'numbers', 'errors') for v in arts[k]]
    fragmented = [v for v in all_values
                  if v in content and not _value_still_recoverable(v, result)]
    if fragmented:
        max_extra = max(4, int(max_toks * 0.15))
        for extra in range(1, max_extra + 1):
            tail = runtime.tokenizer.decode(toks[-(tail_n + extra):]) if (tail_n + extra) > 5 else tail
            head = runtime.tokenizer.decode(toks[:head_n + extra])
            result = (head + ' […] ' + tail).strip()
            fragmented = [v for v in all_values
                          if v in content and not _value_still_recoverable(v, result)]
            if not fragmented:
                break
    return result
