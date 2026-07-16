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
    r'|\b[\w][\w\-]{2,60}\.[a-z0-9]{1,5}\b(?<![.,;:!?]))')

_RE_ID_ALNUM = re.compile(r'\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{8,15}\b')
_RE_URL = re.compile(r'https?://\S+')
_RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
_RE_ID_AZ = re.compile(r'\b[A-Z][A-Z0-9]{1,}-[A-Z0-9]*\d[A-Z0-9]*\b')
_RE_ID_HX = re.compile(r'\b[a-f0-9]{8,32}\b')
_RE_COORD = re.compile(r'-?\d{1,3}\.\d{3,8},-?\d{1,3}\.\d{3,8}')
_RE_NUM = re.compile(r'\b\d+(?:\.\d+)?%?\b')
_RE_ERR = re.compile(r'(?i)(?:error|exception|warning):\s.{0,120}')

_INFRA_NOISE_PATTERNS = (
    re.compile(r'message serialization failed', re.I),
    re.compile(r'is deprecated in jupyter-client', re.I),
    re.compile(r'^\s*content\s*=\s*self\.pack', re.I),
    re.compile(r'out of range float values are not json compliant', re.I),
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

def _value_still_recoverable(value, text):
    """Core check: is this string still literally present (loosely
    normalised) in the surviving text? Shared primitive underlying
    target_still_recoverable and action_still_recoverable -- one
    recoverability rule, applied uniformly, rather than a heuristic
    re-implemented per field.
    """
    if not value:
        return False
    v_norm = _loose_normalise(value)
    if not v_norm:
        return False
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


def _get_text(msg):
    parts = []
    for k in ('content', 'name'):
        v = msg.get(k)
        if isinstance(v, str):
            parts.append(v)
    tc = msg.get('tool_call')
    if isinstance(tc, dict):
        import json
        parts.append(json.dumps(tc, sort_keys=True))
    return ' '.join(parts)


@lru_cache(maxsize=4096)
def _artifacts_cached(text: str) -> Dict[str, Tuple[str, ...]]:
    return {
        'paths': tuple(_uw(_RE_PATH.findall(text))),
        'urls': tuple(_uw(_RE_URL.findall(text))),
        'emails': tuple(_uw(_RE_EMAIL.findall(text))),
        'ids': tuple(_uw(_RE_ID_AZ.findall(text) + _RE_ID_HX.findall(text)
                          + _RE_ID_ALNUM.findall(text) + _RE_COORD.findall(text)
                          + _domain_ids(text))),
        'numbers': tuple(_uw(_RE_NUM.findall(text))),
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


def _split_sents(text, min_t=4):
    out = []
    for s in _RE_SENT.split(text):
        s = s.strip()
        if s and (_art_count(s) > 0 or _tok(s) >= min_t):
            out.append(s)
    return out


def _head_tail_cap(content: str, max_toks: int, head_frac: float = 0.55) -> str:
    toks = runtime.tokenizer.encode(content)
    if len(toks) <= max_toks:
        return content
    head_n = int(max_toks * head_frac)
    tail_n = max_toks - head_n
    head = runtime.tokenizer.decode(toks[:head_n])
    tail = runtime.tokenizer.decode(toks[-tail_n:]) if tail_n > 5 else ''
    return (head + ' […] ' + tail).strip()
