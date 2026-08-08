"""
anchor_lifecycle.py -- lifecycle resolution and compression-safe repair
for State/Goal anchors produced by state_goal_extract.py.

This is the retrospective/prospective fork from the anchor taxonomy
made concrete: Decision/Evidence anchors (rationale_ext.py) describe
something that already happened and are corroboration-gated. State and
Goal anchors govern what happens NEXT, so "was this only mentioned
once?" is not evidence against them -- what matters is whether they are
still in force. That's a lifecycle question, not a corroboration
question, and it's resolved here with two independent, content-overlap-
gated checks:

  STATE  -> active | superseded
    Superseded only on an EXPLICIT update/contradiction signal in a
    later same-topic State candidate. Mere repetition or topical
    similarity is not enough (that would just be corroboration wearing
    a different hat) -- see _is_state_supersession.

  GOAL   -> active | completed | abandoned
    Resolved by scanning later messages for a completion or abandonment
    cue that ALSO shares a real content word with the goal's own
    clause. The cue alone is not sufic ient: "Done with lunch, that was
    good." contains a completion word ("Done") but shares no topic with
    "I want to migrate the backend to Postgres" and must NOT resolve it
    -- content-overlap gating, not keyword presence (see
    tests/test_state_goal_gold.py::test_lifecycle_gold_set,
    case "unrelated_completion_does_not_resolve_goal").

Both checks reuse convmem.py's existing shared-content-token gate
(_content_tokens / MIN_SHARED_CONTENT_TOKENS) -- the same anti-false-
positive discipline convmem.Memory.add() already uses for its own
supersession detection -- rather than inventing a second, differently-
tuned topic-overlap heuristic.

COMPRESSION-SAFE REPAIR: inject_anchor_stubs
-----------------------------------------------------------------------
Mirrors rationale_ext.py's inject_rationale_stubs / sv_dagc.py's own
repair pass: additive only (never removes or alters existing content),
checks recoverability with the same primitive
(utils._value_still_recoverable) the rest of the package already trusts
for "is this still here" checks, and is bounded by an explicit stub
count AND per-stub token cap so the repair cost can never scale with
how many candidates Stage 1 generated -- protecting the compression
ratio is a property of this budget, not of Stage 1 precision.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from .convmem import _content_tokens, MIN_SHARED_CONTENT_TOKENS
from .utils import _get_text, _tok, _value_still_recoverable

# --- STATE supersession ------------------------------------------------
#
# Checked against the NEW candidate's full source-message text (not just
# its extracted clause) -- clause splitting on commas can separate the
# update cue ("Actually, ...", "..., not Pune.") from the fact clause
# itself ("I live in Mumbai now"), so the cue check must look at the
# whole message, while the topic-overlap check stays clause-to-clause.
_RE_STATE_UPDATE_CUE = re.compile(
    r"\b(actually|instead|correction|updated?|no longer|now)\b",
    re.IGNORECASE,
)

# --- GOAL lifecycle ------------------------------------------------------

_RE_GOAL_COMPLETE_CUE = re.compile(
    r"\b(finished|finish(?:ed)?|done|complete[d]?|wrapped up|shipped|resolved)\b",
    re.IGNORECASE,
)
_RE_GOAL_ABANDON_CUE = re.compile(
    r"\bnever mind\b|\bdecided not to\b|\bno longer want\b|\bchanged my mind\b|"
    r"\bnot going to\b|\bcancel(?:led|ing)?\b|\bscrap(?:ped)?\b",
    re.IGNORECASE,
)


def _is_state_supersession(old: Dict, new: Dict) -> bool:
    """True if `new` (a later State candidate) explicitly supersedes
    `old` (an earlier one of the same category): same-topic (shared
    content word) AND an explicit update/contradiction cue present
    somewhere in new's source message.

    Takes resolved text (old['clause'], new['clause'], new['msg_text'])
    rather than a message index to dereference. Callers must resolve
    msg_text against their own messages list BEFORE constructing these
    dicts, at the point where the index is known to be valid. This
    function never looks anything up itself, so a record whose index
    later goes stale (e.g. after ShadowBuffer._trim() reindexes) can't
    silently produce a wrong answer -- there is no index left to go
    stale."""
    shared = _content_tokens(old['clause']) & _content_tokens(new['clause'])
    if len(shared) < MIN_SHARED_CONTENT_TOKENS:
        return False
    return bool(_RE_STATE_UPDATE_CUE.search(new['msg_text']))


def _goal_resolution_status(goal: Dict, messages: List[Dict]) -> str:
    """Scan messages after the goal's own message for a content-
    overlapping completion or abandonment signal. Returns 'active' if
    none is found (the default -- a goal stays open until something
    explicitly closes it, not until it's re-mentioned)."""
    goal_tokens = _content_tokens(goal['clause'])
    for i in range(goal['msg_idx'] + 1, len(messages)):
        text = _get_text(messages[i])
        shared = goal_tokens & _content_tokens(text)
        if len(shared) < MIN_SHARED_CONTENT_TOKENS:
            continue
        if _RE_GOAL_ABANDON_CUE.search(text):
            return 'abandoned'
        if _RE_GOAL_COMPLETE_CUE.search(text):
            return 'completed'
    return 'active'


def resolve_lifecycle(messages: List[Dict], candidates: List[Dict]) -> List[Dict]:
    """Attach a lifecycle `status` to every State/Goal candidate.

    Returns a new list of dicts -- each input candidate plus 'status':
      state -> 'active' | 'superseded'
      goal  -> 'active' | 'completed' | 'abandoned'

    State candidates are compared only against other State candidates
    (never against Goal candidates or vice versa); order is msg_idx
    ascending, matching state_goal_extract's own candidate order.
    """
    ordered = sorted(candidates, key=lambda c: c['msg_idx'])
    state_cands = [c for c in ordered if c['anchor_type'] == 'state']
    goal_cands = [c for c in ordered if c['anchor_type'] == 'goal']

    out: List[Dict] = []

    for i, c in enumerate(state_cands):
        status = 'active'
        for later in state_cands[i + 1:]:
            later_resolved = {**later, 'msg_text': _get_text(messages[later['msg_idx']])}
            if _is_state_supersession(c, later_resolved):
                status = 'superseded'
                break
        out.append({**c, 'status': status})

    for c in goal_cands:
        out.append({**c, 'status': _goal_resolution_status(c, messages)})

    return out


# 'active' and 'in_progress' both mean "not yet resolved" -- an anchor
# still governing future behavior. 'in_progress' is part of the goal
# lifecycle contract (active -> in_progress -> completed/abandoned) even
# though nothing in this module currently emits it, so a future caller
# supplying it is treated correctly rather than silently excluded.
_OPEN_STATUSES = {'active', 'in_progress'}


def active_anchors(anchors: Sequence[Dict],
                    categories: Optional[Tuple[str, ...]] = None) -> List[Dict]:
    """Filter to still-open anchors (status in _OPEN_STATUSES), optionally
    restricted to one or more anchor_type categories, e.g.
    active_anchors(anchors, categories=('state',))."""
    out = [a for a in anchors if a.get('status') in _OPEN_STATUSES]
    if categories is not None:
        out = [a for a in out if a['anchor_type'] in categories]
    return out


def inject_anchor_stubs(compressed: List[Dict], messages: List[Dict],
                         anchors: List[Dict], max_stubs: int = 10,
                         max_stub_tokens: Optional[float] = 30
                         ) -> Tuple[List[Dict], Dict]:
    """Additive pass: for each still-open anchor, if its clause isn't
    already recoverable in the compressed text, append a small stub.
    Never removes or alters existing content.

    Bounded cost, independent of how many anchors were passed in:
    at most `max_stubs` stubs added, each capped at `max_stub_tokens`
    (an anchor whose clause alone exceeds the per-stub cap is skipped,
    not truncated -- silently truncating a constraint could change its
    meaning, e.g. dropping a word out of "under 200 words").
    """
    comp_text = ' '.join(_get_text(m) for m in compressed)
    added: List[Dict] = []
    total_tokens = 0

    for a in anchors:
        if len(added) >= max_stubs:
            break
        if _value_still_recoverable(a['clause'], comp_text):
            continue
        content = f"[{a['anchor_type']}] {a['clause']}"
        cost = _tok(content)
        if max_stub_tokens is not None and cost > max_stub_tokens:
            continue
        stub = {
            'role': messages[a['msg_idx']].get('role', 'user'),
            '_orig_idx': a['msg_idx'],
            '_anchor_stub': True,
            'anchor_type': a['anchor_type'],
            'content': content,
        }
        compressed.append(stub)
        comp_text += ' ' + content
        added.append(a)
        total_tokens += cost

    return compressed, {
        'anchor_stubs_added': len(added),
        'anchor_stub_tokens': total_tokens,
        'anchor_added_detail': added,
    }


__all__ = ["resolve_lifecycle", "active_anchors", "inject_anchor_stubs"]
