"""
state_goal_extract.py -- Stage 1 candidate tagging for State and Goal
anchors, extending dagc's existing Decision/Evidence extraction with the
two prospective anchor categories from the anchor taxonomy (see
ANCHOR_TAXONOMY.md):

  State anchor -- a persistent fact future decisions depend on (config,
                  preference, standing constraint). No natural expiry;
                  relevant until explicitly superseded.
  Goal anchor  -- an active objective or commitment. Drives what counts
                  as relevant going forward; has a lifecycle (active ->
                  completed/abandoned), resolved separately in
                  anchor_lifecycle.py.

WHY THIS IS A SEPARATE MODULE, NOT AN EDIT TO rationale_ext.py OR
extraction.py
-----------------------------------------------------------------------
Per the design discussion this package was built from: Decision/Evidence
detection is already tuned and passing (25/0 regression suite,
DRR_soft >= 0.95). State/Goal candidate tagging runs as a fully separate
code path that reads the same message stream but never touches how
Decision anchors get flagged, scored, or corroborated. A false-positive-
prone State/Goal classifier can therefore only ever affect its own
category's precision/recall numbers -- it cannot regress the existing
Decision pipeline, because it shares no mutable state or control flow
with it. It CAN and does reuse existing detection *code* it doesn't
compete with -- see below.

WHAT THIS REUSES INSTEAD OF REINVENTING
-----------------------------------------------------------------------
convmem.classify() is dagc's existing precise-but-narrow regex extractor
for exactly this state/goal distinction (see convmem.py's own module
docstring). It is Stage 1's primary candidate source. It is deliberately
NOT replaced -- only *extended*, with one additional cue class it
structurally cannot see (see below), and *gated*, with one Stage 2 check
it doesn't do (interrogative rejection).

STAGE 1: TWO CANDIDATE SOURCES
-----------------------------------------------------------------------
1. convmem.classify() -- possessive/first-person declarative cues
   ("I live in...", "my deadline is...", "I want to...", "I'm trying
   to..."). Covers self-descriptive state and intent-verb goals.

2. _stage1_directive_candidates() (new, this module) -- standing
   INSTRUCTIONS directed at the assistant, not self-descriptions: "please
   keep responses under 200 words", "from now on always cite sources",
   "as a rule never use profanity". These are State anchors too (they
   are persistent facts/constraints future turns depend on) but they
   don't have a first-person subject at all, so convmem's cue set --
   built entirely around "I"/"my"/"our" -- structurally cannot match
   them. This is not a bug in convmem.classify to fix; it is a different
   grammatical shape needing its own cue class, kept separate exactly so
   a change here cannot alter convmem.classify's own tested behavior.

Both sources are deliberately generous (a Stage 1 candidate tagger should
favor recall) -- precision is clawed back in Stage 2, not by narrowing
Stage 1's cues.

STAGE 2: VERIFICATION GATE
-----------------------------------------------------------------------
A candidate is rejected if it fails a defensibility check given local
context, regardless of which cue matched it. Currently one check:
  - Interrogative rejection: a clause that is itself a question ("What
    time zone should I use for the logs?") can share cue vocabulary with
    a real anchor ("... I use ...") without being one. Rejected by
    trailing "?".
Ack-only filler ("Sure, sounds good, thanks!") and decision-shaped text
("Recommend the best index... therefore B-tree is the winner.") are not
separately filtered here because neither convmem's cues nor the new
directive cue ever fire on them in the first place (no first-person
subject, no directive-instruction phrase) -- verified against the gold
set in tests/test_state_goal_gold.py. If a future cue addition starts
catching that shape, add a Stage 2 check for it then, rather than
guessing at defenses for a false-positive that doesn't exist yet.

CORROBORATION POLICY, EXPLICITLY NOT APPLIED HERE
-----------------------------------------------------------------------
Unlike rationale_ext.py's _corroborated_elsewhere gate (which requires a
second independent mention before trusting a rejected-alternative
candidate), Stage 1 here does NOT require repetition. A State anchor's
lack of repetition is the normal case, not weak evidence -- a preference
or constraint is usually stated once. Gating single-mention State/Goal
candidates on corroboration is exactly the failure mode this taxonomy
exists to fix (see ANCHOR_TAXONOMY.md and
tests/test_state_goal_gold.py::test_single_mention_state_anchor_dropped_by_baseline_but_kept_by_sv).
Lifecycle resolution (supersession, completion, abandonment) is a
separate, explicit-signal-gated concern handled in anchor_lifecycle.py,
not a recall-limiting gate on candidate generation.
"""
from __future__ import annotations

import re
from typing import Dict, List

from .convmem import classify as _classify_cues
from .convmem import _clauses as _cm_clauses
from .utils import _get_text

# --- New Stage 1 cue class: directive/standing-instruction State -----------
#
# Distinct grammatical shape from convmem's first-person cues: an
# imperative or standing rule directed AT the assistant, not a
# self-description. Every phrase here is a closed, specific anchor
# phrase (not a bare "always"/"never", which would be far too generic
# and would risk colliding with rationale_ext.py's own negation-scope
# scanning for rejected alternatives -- a different concern entirely).
_RE_DIRECTIVE_STATE_CUE = re.compile(
    r"\b("
    r"from now on|as a rule|for (?:this|the) (?:whole|entire) session|"
    r"please (?:always|never|keep|avoid|don'?t|do not|use|include|cite|respond|remember)|"
    r"always (?:cite|use|keep|respond|include|remember)|"
    r"never (?:use|include|respond|forget)"
    r")\b",
    re.IGNORECASE,
)

# Stage 2 gate: a clause ending in "?" is a question, not an assertion --
# reject regardless of which cue matched it.
_RE_INTERROGATIVE = re.compile(r"\?\s*$")


def _stage1_directive_candidates(messages: List[Dict]) -> List[Dict]:
    """Standing-instruction State candidates -- see module docstring.
    Same {'msg_idx', 'clause'} shape convmem.classify produces, so
    downstream code (dedup, Stage 2 gate, lifecycle resolution) treats
    every candidate identically regardless of source."""
    out = []
    for i, m in enumerate(messages):
        text = _get_text(m)
        for cl in _cm_clauses(text):
            if _RE_DIRECTIVE_STATE_CUE.search(cl):
                out.append({'msg_idx': i, 'clause': cl})
    return out


def _passes_stage2_gate(candidate: Dict) -> bool:
    return not _RE_INTERROGATIVE.search(candidate['clause'])


def extract_state_goal_candidates(messages: List[Dict]) -> List[Dict]:
    """Two-stage State/Goal candidate extraction.

    Returns a flat list of candidates, each:
        {'msg_idx': int, 'clause': str, 'anchor_type': 'state'|'goal',
         'cue': str}   -- cue is 'convmem' or 'directive', for audit/debug only.

    Deduplicated by (msg_idx, clause, anchor_type) -- the same clause can
    in principle match both cue sources (e.g. convmem's "please" is not
    among its cues today, but this keeps the contract safe against future
    cue additions on either side without silently double-counting).
    """
    raw: List[Dict] = []

    cues = _classify_cues(messages)
    for cat in ('state', 'goal'):
        for a in cues.get(cat, []):
            raw.append({'msg_idx': a['msg_idx'], 'clause': a['clause'],
                        'anchor_type': cat, 'cue': 'convmem'})

    for a in _stage1_directive_candidates(messages):
        raw.append({'msg_idx': a['msg_idx'], 'clause': a['clause'],
                     'anchor_type': 'state', 'cue': 'directive'})

    seen = set()
    deduped = []
    for c in raw:
        key = (c['msg_idx'], c['clause'], c['anchor_type'])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    return [c for c in deduped if _passes_stage2_gate(c)]


__all__ = ["extract_state_goal_candidates"]
