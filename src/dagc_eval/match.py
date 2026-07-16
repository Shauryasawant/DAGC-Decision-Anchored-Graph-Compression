"""
Decision matcher: scores a reproduced decision against the original
ground-truth decision (action/target/rationale F1). No LLM, no network --
pure string comparison.
"""
from __future__ import annotations
import json
import re
from dagc.extraction import (
    _bare_rationale_value,
    _looks_identifier_shaped,
    _JUDGMENT_CONNECTIVE_WORDS,
)

W_ACTION = 0.40
W_TARGET = 0.40
W_RATIONALE = 0.20
DRR_THRESHOLD = 0.75


def _normalise(text):
    if text is None:
        return ''
    return re.sub(r'[^a-z0-9]', ' ', str(text).lower()).strip()


def _token_f1(pred, gold):
    p = set(_normalise(pred).split())
    g = set(_normalise(gold).split())
    if not p or not g:
        return 0.0
    ov = len(p & g)
    pr = ov / len(p)
    rc = ov / len(g)
    return 2 * pr * rc / (pr + rc + 1e-9)


def _stem(word):
    for s in ('ing', 'ed', 's'):
        if word.endswith(s) and len(word) - len(s) >= 3:
            return word[:-len(s)]
    return word

# Keep synonym groups separate so close matches score higher than broad ones.
_VERB_SYNONYM_FAMILIES = [
    {'recommend', 'suggest', 'advise', 'propose'},
    {'confirm', 'approve', 'verify', 'finalize', 'finalise'},
    {'select', 'choose', 'pick'},
    {'decide', 'determine', 'conclude'},
    {'adopt', 'implement', 'apply'},
    {'use', 'utilize', 'utilise', 'employ'},
    {'prefer'},
    {'best', 'optimal', 'winner', 'final'},
]

_CONNECTIVE_STEMS = {_stem(w) for w in _JUDGMENT_CONNECTIVE_WORDS}


def _build_family_index(families):
    """Stem every family member once, so lookups are inflection-agnostic
    (confirmed/confirming/confirms all resolve to the 'confirm' family)
    without needing a separate hardcoded list of tenses per verb."""
    index = {}
    for fam_idx, family in enumerate(families):
        for word in family:
            index[_stem(word)] = fam_idx
    return index


_VERB_FAMILY_INDEX = _build_family_index(_VERB_SYNONYM_FAMILIES)


_DECIDE_FALLBACK_STEM = _stem('decide')

def _action_match(original, reproduced):
    orig = _normalise(original.get('action', ''))
    repr_ = _normalise(reproduced.get('action', ''))
    if not orig or not repr_:
        return 0.0
    if orig == repr_:
        return 1.0

    orig_stem = _stem(orig)
    repr_stem = _stem(repr_)

    if orig_stem == repr_stem:
        return 0.97

    orig_fam = _VERB_FAMILY_INDEX.get(orig_stem)
    repr_fam = _VERB_FAMILY_INDEX.get(repr_stem)

    score = 0.0
    if orig_fam is not None and orig_fam == repr_fam:
        score = 0.95

    orig_is_connective = orig_stem in _CONNECTIVE_STEMS
    repr_is_connective = repr_stem in _CONNECTIVE_STEMS
    if orig_is_connective and repr_is_connective:
        score = max(score, 0.85)
    elif orig_is_connective and repr_fam is not None:
        score = max(score, 0.70)
    elif repr_is_connective and orig_fam is not None:
        score = max(score, 0.70)

    if orig_fam is not None and repr_fam is not None:
        score = max(score, 0.70)

    # Treat the extractor's fallback as a partial match for any decision verb.
    if (orig_stem == _DECIDE_FALLBACK_STEM and repr_fam is not None) or \
       (repr_stem == _DECIDE_FALLBACK_STEM and orig_fam is not None):
        score = max(score, 0.85)

    if score > 0.0:
        return score

    os_ = {_stem(w) for w in orig.split()}
    rs_ = {_stem(w) for w in repr_.split()}
    if os_ and os_ <= rs_:
        return 0.9
    if rs_ and rs_ <= os_:
        return 0.8
    if os_ & rs_:
        return 0.6
    return _token_f1(repr_, orig)


def _target_match(original, reproduced):
    orig = original.get('target', '')
    repr_ = reproduced.get('target', '')
    if isinstance(orig, list):
        orig = ' '.join(str(x) for x in orig)
    if isinstance(repr_, list):
        repr_ = ' '.join(str(x) for x in repr_)
    if not orig or not repr_:
        return 0.0

    orig_str = str(orig).strip()
    if orig_str.startswith('['):
        try:
            items = json.loads(orig_str)
            if isinstance(items, list) and items:
                rn = _normalise(str(repr_))
                hits = sum(1 for it in items
                           if _normalise(str(it)) in rn or _token_f1(rn, _normalise(str(it))) > 0.5)
                return hits / len(items)
        except Exception:
            pass

    orig_n = _normalise(orig)
    repr_n = _normalise(repr_)
    if orig_n == repr_n:
        return 1.0
    if _looks_identifier_shaped(str(orig).strip()):
        orig_compact = orig_n.replace(' ', '')
        repr_compact = repr_n.replace(' ', '')
        if orig_compact and orig_compact in repr_compact:
            return 0.9
        return 0.0

    base_f1 = _token_f1(repr_n, orig_n)
    if base_f1 >= 0.5:
        return base_f1

    # Evidence can strengthen a partial target match, not create one.
    if base_f1 > 0.0:
        if original.get('type') == 'confirmation':
            arts = original.get('artifacts', {})
            for kind in ('paths', 'ids'):
                for a in arts.get(kind, []):
                    if _token_f1(repr_n, _normalise(str(a))) > 0.5:
                        return max(base_f1, 0.80)

        if original.get('type') == 'action':
            for rat_item in original.get('rationale', []):
                if repr_n and repr_n in _normalise(rat_item):
                    return max(base_f1, 0.65)

    return base_f1


def _rationale_f1(original, reproduced):
    og_raw = [r for r in original.get('rationale', []) if r]
    rp_raw = [r for r in reproduced.get('rationale', []) if r]
    if not og_raw:
        return 1.0
    if not rp_raw:
        return 0.0

    scores = []
    for o_raw in og_raw:
        o_bare = _bare_rationale_value(o_raw)
        o_full = _normalise(o_raw)

        best = 0.0
        for r_raw in rp_raw:
            r_bare = _bare_rationale_value(r_raw)
            r_full = _normalise(r_raw)

            full = _token_f1(r_full, o_full)
            bare = _token_f1(r_bare, o_bare)
            best = max(best, full, bare)
        scores.append(best)

    return sum(scores) / len(scores)


def match_decision(original, reproduced):
    if not reproduced.get('_success', True):
        return {
            'action_score': 0.0, 'target_score': 0.0, 'rationale_score': 0.0,
            'decision_score': 0.0, 'reproduced': False, 'failure': 'llm_error',
            'target_scoreable': original.get('target') is not None,
        }
    a_score = _action_match(original, reproduced)
    r_score = _rationale_f1(original, reproduced)

    # Avoid over-penalizing tool calls with only a thin rationale.
    if (original.get('type') == 'action' and a_score >= 0.90
            and len(original.get('rationale', [])) <= 1):
        r_score = max(r_score, 0.50)

    ts = original.get('target') is not None
    if ts:
        t_score = _target_match(original, reproduced)
        overall = W_ACTION * a_score + W_TARGET * t_score + W_RATIONALE * r_score
    else:
        t_score = None
        overall = ((W_ACTION / (W_ACTION + W_RATIONALE)) * a_score
                   + (W_RATIONALE / (W_ACTION + W_RATIONALE)) * r_score)
    return {
        'action_score': round(a_score, 4),
        'target_score': None if t_score is None else round(t_score, 4),
        'rationale_score': round(r_score, 4),
        'decision_score': round(overall, 4),
        'reproduced': overall >= DRR_THRESHOLD,
        'failure': None,
        'target_scoreable': ts,
    }
