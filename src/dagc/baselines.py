"""
Optional baseline compressors for comparison against compress_dagc().
Not imported by dagc/__init__.py automatically -- pull in what you need:

    from dagc.baselines import compress_tail_truncation, compress_random_drop
"""
from __future__ import annotations
import random
from typing import Dict, List

from .utils import _get_text, _split_sents, _tok
from .compressor import DAGCConfig, DAGC_CFG, compress_dagc


def compress_tail_truncation(messages: List[Dict], cfg: DAGCConfig = DAGC_CFG, **kw) -> List[Dict]:
    """Naive baseline: keep system prompt + as many trailing messages as fit."""
    orig_toks = sum(_tok(_get_text(m)) for m in messages)
    budget = max(1, int(orig_toks * (1 - cfg.TARGET_REDUCTION)))
    sys_msgs = [(i, m) for i, m in enumerate(messages) if m.get('role') == 'system']
    rest = [(i, m) for i, m in enumerate(messages) if m.get('role') != 'system']
    kept, used = {}, 0
    for i, m in sys_msgs:
        mc = dict(m)
        toks = _tok(m.get('content', ''))
        if toks > cfg.SYSTEM_MAX_TOKENS:
            mc['content'] = m.get('content', '')[: cfg.SYSTEM_MAX_TOKENS * 4] + '[…]'
        mc['_orig_idx'] = i
        kept[i] = mc
        used += _tok(mc.get('content', ''))
    for i, m in reversed(rest):
        t = _tok(_get_text(m))
        if used + t > budget and kept:
            continue
        kept[i] = dict(m, _orig_idx=i)
        used += t
    return [kept[i] for i in sorted(kept)]


def compress_random_drop(messages: List[Dict], cfg: DAGCConfig = DAGC_CFG, seed: int = 0, **kw) -> List[Dict]:
    """Naive baseline: randomly keep sentences until the budget is spent."""
    rng = random.Random(seed)
    orig_toks = sum(_tok(_get_text(m)) for m in messages)
    budget = max(1, int(orig_toks * (1 - cfg.TARGET_REDUCTION)))
    n = len(messages)
    protected = {i for i, m in enumerate(messages)
                 if m.get('role') == 'system' or i >= n - cfg.KEEP_LAST_K}
    protected_toks = sum(_tok(_get_text(messages[i])) for i in protected)
    remaining = max(0, budget - protected_toks)
    pool = [(s, i) for i, m in enumerate(messages) if i not in protected
            for s in _split_sents(_get_text(m), cfg.MIN_SENT_TOKENS)]
    rng.shuffle(pool)
    chosen, used = [], 0
    for s, i in pool:
        t = _tok(s)
        if used + t <= remaining:
            chosen.append((s, i))
            used += t
    msg_sents: Dict[int, List[str]] = {}
    for s, i in chosen:
        msg_sents.setdefault(i, []).append(s)
    out = []
    for i, m in enumerate(messages):
        if i in protected:
            out.append(dict(m, _orig_idx=i))
        elif i in msg_sents:
            mc = dict(m)
            mc['content'] = ' '.join(msg_sents[i])
            mc['_orig_idx'] = i
            out.append(mc)
    return out


def compress_identity(messages: List[Dict], **kw) -> List[Dict]:
    """No-op baseline: returns the trace unchanged (for measuring the ceiling)."""
    return [dict(m, _orig_idx=i) for i, m in enumerate(messages)]


BASELINES = {
    'tail_truncation': compress_tail_truncation,
    'random_drop': compress_random_drop,
    'identity': compress_identity,
}


def compress_dagc_baseline(messages: List[Dict], cfg: DAGCConfig = DAGC_CFG, seed: int = 0, **kw) -> List[Dict]:
    """
    Adapter so compress_dagc() can be called through the same
    `fn(trace, seed=...)` interface as the other BASELINES entries.

    compress_dagc() itself takes no `seed` -- it is deterministic given a
    trace and cfg (no internal RNG), so seed is accepted here purely for
    interface compatibility and otherwise ignored. Does not call or modify
    compress_dagc()'s own logic in any way; this is a passthrough.
    """
    return compress_dagc(messages, cfg=cfg)


# Same as BASELINES, plus DAGC itself under the 'DAGC' key -- for callers
# that want a head-to-head comparison set (e.g. dagc_eval.run_method_comparison,
# the `dagc compare` CLI command) rather than baselines alone. Added instead
# of changing BASELINES in place so existing importers of BASELINES keep
# getting exactly the baseline methods, unchanged.
METHODS_WITH_DAGC = {
    'DAGC': compress_dagc_baseline,
    **BASELINES,
}
