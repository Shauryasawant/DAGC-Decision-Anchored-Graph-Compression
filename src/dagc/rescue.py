"""
rescue.py — cross-turn decision rescue for DAGC.

Extends decision-awareness from *compression time* to *every subsequent
turn*, without reimplementing anything compress_dagc already does better.

Wiring (this is the entire integration surface):

    from dagc import compress
    from dagc.rescue import ShadowBuffer, RescueEngine

    shadow = ShadowBuffer(max_turns=200)
    engine = RescueEngine()

    history: list[dict] = []          # raw, uncompressed messages so far
    last_compressed: list[dict] = []  # whatever compress() returned last turn

    def on_new_message(msg, budget_tokens):
        history.append(msg)
        force_preserve, events, unrescuable = engine.process_turn(
            new_message=msg,
            shadow=shadow,
            last_compressed_messages=last_compressed,
            compression_budget_tokens=budget_tokens,
        )
        compressed = compress(history, force_preserve=force_preserve,
                               ABSOLUTE_BUDGET_TOKENS=budget_tokens)
        return compressed

Why this is simpler than the first draft, not just different: compress_dagc
already guarantees zero loss of anything in target_arts_all via
_phase1_hard_guarantee + the unconditional final verify-and-repair pass,
with no budget ceiling on that guarantee by design (see the
MSTAR_HARD_DROP fix comment in compressor.py — losing a decision costs
more than blowing the budget). Duplicating that as a second eviction
system risked the two disagreeing. This version doesn't duplicate it: it
only decides *which strings* need that guarantee this turn, and lets
force_preserve carry them in. Budget-neutrality is therefore whatever
compress_dagc's own accounting produces — bounded growth, not a promise
this module can independently keep.

What "critical values" means for a decision is also no longer reinvented
here — _decision_critical_values (dagc.compressor) is the exact function
compress_dagc uses to build target_arts/target_arts_all, so reusing it
means rescue's notion of "this decision's protectable values" can never
drift from what force_preserve will actually hard-guarantee.

Two deliberately-justified pieces of real math (a third — budget-neutral
eviction — is gone; see above for why):

  1. Exponentially-weighted recurrence (DecayedRecurrenceTracker) — O(1)-
     update decay, same trick as an EMA/decayed frequency counter.
     Distinguishes a burst of near-term re-references (likely load-
     bearing) from sparse hits scattered across a long session (likely
     noise), without storing full rescue history.
  2. Capacity-bounded promotion (GuaranteedSet) — caps the permanently-
     protected set structurally, so a long session can't let force_preserve
     grow without bound just because things keep getting referenced once.
     Weakest-incumbent eviction is a straight O(K) argmin at the small K
     this runs at.

v1 (default): single-hop — rescue the one decision that owns the
referenced value. Validate this against real traces before touching v2.

v2 (opt-in, multi_hop=True): depth-capped backward walk over a decision
DAG (edge A -> B iff A's target/artifacts feed B's rationale, A before B).
Helps root-cause-style references ("why did latency spike" needs the
upstream log mention, not just the spike sentence) but costs more force-
preserve budget and carries more false-positive risk — validate
separately, and only after confirming v1's trigger tiers look sane (see
validation note at the bottom).
"""
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple,Callable
import re
from .utils import _artifacts, _get_text, _tok
from .anchor_lifecycle import _is_state_supersession
from .extraction import (
    _build_decision_for_message,
    _find_decisive_match,
    _bare_rationale_value,
    _RE_ENTITY,
    _RE_ENTITY_SNAKE,
    _ENTITY_BLOCKLIST,
)
from .compressor import _decision_critical_values, _art_in_text, _footprint_text


# --------------------------------------------------------------------------
# 1. Critical-value extraction — thin wrapper, not a reimplementation.
#    Delegates to the SAME function compress_dagc uses to build
#    target_arts/target_arts_all, so rescue can never protect a different
#    notion of "critical" than the compressor actually enforces.
# --------------------------------------------------------------------------

def _critical_values(decision: Dict, force_action: bool = False) -> Set[str]:
    """force_action=True mirrors compress_dagc's own force_action_msg_idxs
    reasoning: a decision currently being rescued has, by definition, no
    other protection route this turn (that's WHY it needed rescuing), so
    its action verb is force-included even if it's one of the generic
    verbs (recommend/decide/confirm/...) normally excluded as
    noncritical. Ancestor decisions pulled in via multi-hop use the
    default (force_action=False) since they aren't the ones that failed
    to survive -- they're supporting context."""
    force_idxs = {decision['msg_idx']} if force_action else None
    return _decision_critical_values([decision], force_action_msg_idxs=force_idxs)


def _message_reference_values(text: str) -> Set[str]:
    """What does this NEW message actually mention? Deliberately broader
    than _critical_values (raw entities, not just decision-derived
    values) since we don't yet know if what's referenced was ever a
    decision at all -- that's precisely the "evidence that never became a
    decision" gap noted for the SRE case."""
    arts = _artifacts(text)
    vals: Set[str] = set()
    for kind in ("paths", "ids", "numbers"):
        for a in arts.get(kind, []):
            vals.add(str(a).strip().lower())
    for m in _RE_ENTITY.finditer(text):
        w = m.group(0).lower()
        if w not in _ENTITY_BLOCKLIST:
            vals.add(w)
    for m in _RE_ENTITY_SNAKE.finditer(text):
        w = m.group(0).lower()
        if w not in _ENTITY_BLOCKLIST:
            vals.add(w)
    return vals


# --------------------------------------------------------------------------
# 2. Decision DAG — edge A -> B iff A's target/artifacts feed B's
#    rationale and A.msg_idx < B.msg_idx. Acyclic by construction.
# --------------------------------------------------------------------------

@dataclass
class DecisionDAG:
    nodes: List[Dict]
    edges: Dict[int, List[int]]
    value_owner: Dict[str, List[int]]

    def upstream(self, node_id: int) -> List[int]:
        return [src for src, dsts in self.edges.items() if node_id in dsts]


def build_decision_dag(decisions: List[Dict]) -> DecisionDAG:
    owner: Dict[str, List[int]] = defaultdict(list)
    for i, d in enumerate(decisions):
        for v in _critical_values(d):
            owner[v.lower()].append(i)

    edges: Dict[int, List[int]] = defaultdict(list)
    for j, b in enumerate(decisions):
        rationale_vals = {_bare_rationale_value(r) for r in (b.get("rationale") or [])}
        for v in rationale_vals:
            for i in owner.get(v, []):
                if i != j and decisions[i]["msg_idx"] < b["msg_idx"]:
                    edges[i].append(j)

    return DecisionDAG(nodes=decisions, edges=dict(edges), value_owner=dict(owner))


# --------------------------------------------------------------------------
# 3. Shadow buffer — windowed original messages + their extracted
#    decisions, so a later turn can be back-traced against what was
#    actually said, independent of what survived any given compression.
# --------------------------------------------------------------------------

@dataclass
class ShadowBuffer:
    max_turns: int = 200
    decision_roles: Tuple[str, ...] = ("user", "assistant")
    messages: List[Dict] = field(default_factory=list)
    decisions: List[Dict] = field(default_factory=list)
    _dag: Optional[DecisionDAG] = None

    def ingest(self, new_messages: List[Dict],
               on_evict: Optional[Callable[[List[Dict], List[Dict]], None]] = None) -> None:
        start_idx = len(self.messages)
        self.messages.extend(new_messages)
        for offset, msg in enumerate(new_messages):
            idx = start_idx + offset
            if msg.get("role") not in self.decision_roles:
                continue
            d = _build_decision_for_message(self.messages, idx)
            if d is not None:
                self.decisions.append(d)
        self._dag = None
        self._trim(on_evict=on_evict)

    def _trim(self, on_evict: Optional[Callable[[List[Dict], List[Dict]], None]] = None) -> None:
        if len(self.messages) <= self.max_turns:
            return
        overflow = len(self.messages) - self.max_turns

        # Snapshot BEFORE any reindexing -- msg_idx values here are still
        # valid against self.messages exactly as it stands right now.
        about_to_evict = [d for d in self.decisions if d["msg_idx"] < overflow]
        if on_evict is not None and about_to_evict:
            on_evict(about_to_evict, self.messages)

        self.messages = self.messages[overflow:]
        kept = []
        for d in self.decisions:
            if d["msg_idx"] >= overflow:
                d = dict(d)
                d["msg_idx"] -= overflow
                kept.append(d)
        self.decisions = kept
        self._dag = None

    @property
    def dag(self) -> DecisionDAG:
        if self._dag is None:
            self._dag = build_decision_dag(self.decisions)
        return self._dag
@dataclass
class UnrescuableEviction:
    """A decision that fell out of ShadowBuffer's window and could not
    be promoted -- either GuaranteedSet was full and nothing in it was
    superseded by this decision, or only some of its critical values
    found room. This is the literal, observable instance of Theorem 1's
    capacity boundary: real information was just permanently lost.
    Surfaced explicitly, never swallowed -- silent loss is the one
    failure mode this design exists to prevent."""
    msg_idx: int
    clause: str
    unpromoted_values: Set[str]

# --------------------------------------------------------------------------
# 4. Tiered reference matcher.
#    Tier 1 — already recoverable from what was actually shipped last
#             turn: not missing, no-op.
#    Tier 2 — exact match against the shadow buffer's decisions: it
#             existed, wasn't shipped, is now referenced -> rescue.
#             No false-positive risk beyond genuine value collisions,
#             since this is exact-string equality, not fuzzy matching.
#    Tier 3 — corroborated fallback for values with no exact prior
#             decision-value match. This is the ONLY tier with real
#             false-positive risk (a coincidental substring hit costs
#             compression ratio for zero real benefit), so it carries
#             three independent tightenings on top of the original
#             decisive-sentence gate:
#               (a) word-boundary matching, not raw substring — "id"
#                   must not match inside "rapid";
#               (b) a minimum length + non-numeric-shape filter — a
#                   bare number or 1-3 char token has essentially no
#                   specificity and should only ever rescue via tier-2's
#                   exact match, never a fuzzy substring;
#               (c) at most ONE tier-3 candidate per value — if a value
#                   fuzzy-matches several decisions, only the most
#                   recent is used, so one ambiguous reference can't
#                   trigger several redundant rescues at once.
# --------------------------------------------------------------------------

@dataclass
class RescueCandidate:
    value: str
    tier: int
    owning_node_id: int
    confidence: float


_word_boundary_cache: Dict[str, "re.Pattern"] = {}


def _word_boundary_contains(value: str, text: str) -> bool:
    """Word-boundary-aware containment, so a short value can't
    spuriously match as a substring of an unrelated longer word.
    Multi-word values are matched punctuation-tolerant between words,
    mirroring how compressor._art_in_text handles multi-word targets."""
    import re
    pattern = _word_boundary_cache.get(value)
    if pattern is None:
        escaped = re.escape(value)
        if ' ' in value:
            escaped = escaped.replace(r'\ ', r'[^A-Za-z0-9]*')
        pattern = re.compile(r'(?<![A-Za-z0-9_])' + escaped + r'(?![A-Za-z0-9_])', re.IGNORECASE)
        _word_boundary_cache[value] = pattern
    return bool(pattern.search(text))


_MIN_TIER3_LEN = 4


def _tier3_eligible(value: str) -> bool:
    """Fuzzy rescue is restricted to values with enough shape to be a
    real identifier: minimum length, and not a bare number. A bare
    number has no specificity for substring matching — it either
    exactly equals a known decision value (tier-2 already covers that)
    or it's a coincidence."""
    if len(value) < _MIN_TIER3_LEN:
        return False
    stripped = value.replace('.', '').replace('-', '')
    if stripped.isdigit():
        return False
    return True


def find_missing_references(
    new_message_text: str,
    last_compressed_text: str,
    shadow: ShadowBuffer,
) -> List[RescueCandidate]:
    referenced = _message_reference_values(new_message_text)

    shadow_index: Dict[str, int] = {}
    for i, d in enumerate(shadow.decisions):
        for v in _critical_values(d):
            shadow_index.setdefault(v.lower(), i)

    candidates: List[RescueCandidate] = []
    unresolved: List[str] = []
    for v in referenced:
        if _art_in_text(v, last_compressed_text):
            continue  # tier 1 -- genuinely still there, nothing to do
        node_id = shadow_index.get(v)
        if node_id is not None:
            candidates.append(RescueCandidate(v, tier=2, owning_node_id=node_id, confidence=1.0))
        else:
            unresolved.append(v)

    eligible = [v for v in unresolved if _tier3_eligible(v)]
    if eligible:
        best_match: Dict[str, int] = {}  # value -> most recent owning decision
        for node_id, d in enumerate(shadow.decisions):
            sentence_text = d.get("verbatim", "")
            if _find_decisive_match(sentence_text) is None:
                continue
            for v in eligible:
                if _word_boundary_contains(v, sentence_text):
                    if v not in best_match or d["msg_idx"] > shadow.decisions[best_match[v]]["msg_idx"]:
                        best_match[v] = node_id
        for v, node_id in best_match.items():
            candidates.append(RescueCandidate(v, tier=3, owning_node_id=node_id, confidence=0.6))

    return candidates

# --------------------------------------------------------------------------
# 5. Decayed recurrence tracker.
# --------------------------------------------------------------------------

@dataclass
class _RecurrenceEntry:
    score: float = 0.0
    last_turn: int = -1


class DecayedRecurrenceTracker:
    """R(v, t) = R(v, t_prev) * decay^(t - t_prev) + 1 on each touch.
    decay in (0, 1); half_life_turns() reports ln(0.5)/ln(decay)."""

    def __init__(self, decay: float = 0.85):
        assert 0.0 < decay < 1.0
        self.decay = decay
        self._entries: Dict[str, _RecurrenceEntry] = {}

    def touch(self, value: str, turn: int) -> float:
        e = self._entries.setdefault(value, _RecurrenceEntry())
        if e.last_turn >= 0:
            e.score *= self.decay ** max(0, turn - e.last_turn)
        e.score += 1.0
        e.last_turn = turn
        return e.score

    def score(self, value: str, turn: int) -> float:
        e = self._entries.get(value)
        if e is None:
            return 0.0
        return e.score * (self.decay ** max(0, turn - e.last_turn))

    def half_life_turns(self) -> float:
        return math.log(0.5) / math.log(self.decay)


# --------------------------------------------------------------------------
# 6. Capacity-bounded guaranteed set — the permanently-force-preserved
#    tier. Structurally capped so a long session can't let repeated
#    single references silently grow force_preserve without bound.
# --------------------------------------------------------------------------

class GuaranteedSet:
    """Capacity-bounded, permanently-force-preserved tier.

    Eviction is supersession-aware, not score-based: a member is only
    ever displaced when a newcomer explicitly supersedes it (same
    topic + an update/contradiction cue -- see
    anchor_lifecycle._is_state_supersession). If full and nothing
    present is superseded by the newcomer, promotion is refused -- a
    live fact is never evicted just because something else currently
    scores higher on recency."""

    def __init__(self, max_size: int):
        self.max_size = max(1, max_size)
        self._members: Dict[str, Dict] = {}

    @property
    def members(self) -> Set[str]:
        return set(self._members.keys())

    def resize(self, new_max_size: int) -> None:
        self.max_size = max(1, new_max_size)

    def try_promote(self, value: str, clause: str, msg_idx: int,
                     messages: List[Dict]) -> Optional[str]:
        """Returns the displaced value's name if a superseded incumbent
        had to be dropped to make room, else None. Refuses promotion
        (returns None, value NOT added) if full and nothing present is
        superseded by this newcomer."""
        if value in self._members:
            return None
        # Resolve msg_text EAGERLY here, while msg_idx is guaranteed
        # valid against `messages` (this call's fresh argument) -- never
        # store the raw index for later dereference. ShadowBuffer._trim()
        # can reindex messages between this promotion and any future
        # call that uses this record as the incumbent `old` side of a
        # supersession check; storing resolved text instead of an index
        # means this record's value can never go stale, by construction.
        msg_text = _get_text(messages[msg_idx]) if 0 <= msg_idx < len(messages) else clause
        new_rec = {'clause': clause, 'msg_text': msg_text}
        if len(self._members) < self.max_size:
            self._members[value] = new_rec
            return None
        for incumbent_val, incumbent_rec in self._members.items():
            if _is_state_supersession(incumbent_rec, new_rec):
                del self._members[incumbent_val]
                self._members[value] = new_rec
                return incumbent_val
        return None

    def __contains__(self, value: str) -> bool:
        return value in self._members


# --------------------------------------------------------------------------
# 7. Orchestration.
# --------------------------------------------------------------------------

@dataclass
class RescueEvent:
    value: str
    tier: int
    owning_msg_idx: int
    contributed_values: Set[str]
    promoted: bool
    displaced_from_guaranteed: Optional[str]


class RescueEngine:
    def __init__(
        self,
        decay: float = 0.85,
        promote_threshold: float = 2.2,
        guaranteed_alpha: float = 0.15,
        guaranteed_min: int = 10,  # Calibrated via calibrate_guaranteed_min.py against 25
        # independently-seeded synthetic long-session ITBench corpora (seeds
        # 42-66, ~35 real sessions chained per group, ~270-350 turns after
        # adjacent-dedup). Pooled: 3452 trims, p50=1.0, mean=2.67, p95=6.0,
        # observed max=8 -- bimodal (only ever 6 or 8 across all 25 seeds,
        # never in between). Running max flattened after seed 5 (only 1 new
        # record in 24 subsequent seeds), supporting max=8 as a real ceiling
        # for this corpus rather than an artifact of small sample size.
        # guaranteed_min = ceil(8 * 1.25) = 10, applying a safety margin on
        # top of the observed max since max-of-N-seeds is a lower bound on
        # the true worst case, not the true worst case itself. Supersedes
        # the earlier 4-seed calibration (which had produced 8 with weaker
        # convergence evidence).
        multi_hop: bool = False,
        multi_hop_depth: int = 2,
        multi_hop_token_budget: int = 400,
    ):
        """decay/promote_threshold/guaranteed_alpha remain uncalibrated
        defaults and need a real-trace sweep before being trusted at scale.
        guaranteed_min is the exception: calibrated against 25 seeded ITBench
        runs (see comment on the parameter above)."""
        self.tracker = DecayedRecurrenceTracker(decay=decay)
        self.promote_threshold = promote_threshold
        self.guaranteed_alpha = guaranteed_alpha
        self.guaranteed_min = guaranteed_min
        self.guaranteed = GuaranteedSet(max_size=guaranteed_min)
        self.multi_hop = multi_hop
        self.multi_hop_depth = multi_hop_depth
        self.multi_hop_token_budget = multi_hop_token_budget
        self.turn = 0
        self._pending_unrescuable: List[UnrescuableEviction] = []
        self.unrescuable_log: List[UnrescuableEviction] = []  # cumulative, whole-session audit trail

    def _resize_guaranteed(self, compression_budget_tokens: int) -> None:
        # Same placeholder-normalizer caveat as before: this constant needs
        # calibration against real per-value token costs from actual
        # traces, not a made-up divisor. Flagged, not fixed.
        cap = max(self.guaranteed_min,
                  int(self.guaranteed_alpha * max(1, compression_budget_tokens) / 10))
        self.guaranteed.resize(cap)

    def _on_shadow_evict(self, evicted: List[Dict], messages_before_trim: List[Dict]) -> None:
        # Chronological order (ShadowBuffer.decisions is append-ordered):
        # earliest-evicted decisions get first shot at any free/superseded
        # slot. Named explicitly -- this is a real tie-break policy, not
        # an accident of iteration order.
        for d in evicted:
            unpromoted: Set[str] = set()
            for v in _critical_values(d):
                if v in self.guaranteed:
                    continue
                self.guaranteed.try_promote(
                    v, d.get('verbatim', ''), d['msg_idx'], messages_before_trim)
                if v not in self.guaranteed:
                    unpromoted.add(v)
            if unpromoted:
                rec = UnrescuableEviction(
                    msg_idx=d['msg_idx'], clause=d.get('verbatim', ''),
                    unpromoted_values=unpromoted)
                self._pending_unrescuable.append(rec)
                self.unrescuable_log.append(rec)

    def process_turn(
        self,
        new_message: Dict,
        shadow: ShadowBuffer,
        last_compressed_messages: List[Dict],
        compression_budget_tokens: int,
    ) -> Tuple[Set[str], List[RescueEvent]]:
        """Run one turn of rescue. Returns (force_preserve, events) --
        pass force_preserve straight into compress()/compress_dagc() for
        this turn; caller owns assembling `history` and calling compress."""
        self.turn += 1
        self._resize_guaranteed(compression_budget_tokens)
        self._pending_unrescuable = []  
        shadow.ingest([new_message], on_evict=self._on_shadow_evict)
        last_text = "\n".join(_footprint_text(m) for m in last_compressed_messages)
        text = _get_text(new_message)
        candidates = find_missing_references(text, last_text, shadow)

        turn_force_preserve: Set[str] = set()
        events: List[RescueEvent] = []

        for cand in candidates:
            owning = shadow.decisions[cand.owning_node_id]
            values = _critical_values(owning, force_action=True)
            if self.multi_hop:
                for ancestor in self._backward_walk(shadow, cand.owning_node_id, last_text):
                    values |= _critical_values(ancestor)
            turn_force_preserve |= values

            r_score = self.tracker.touch(cand.value, self.turn)
            promoted, displaced = False, None
            if r_score >= self.promote_threshold:
                displaced = self.guaranteed.try_promote(
                    cand.value, owning.get('verbatim', ''), owning['msg_idx'],
                    shadow.messages)
                promoted = True

            events.append(RescueEvent(
                value=cand.value,
                tier=cand.tier,
                owning_msg_idx=owning["msg_idx"],
                contributed_values=values,
                promoted=promoted,
                displaced_from_guaranteed=displaced,
            ))

        # Guaranteed members ride along every turn regardless of whether
        # they were re-referenced THIS turn -- that's the point of
        # promotion: stop depending on continued re-reference to survive.
        turn_force_preserve |= self.guaranteed.members

        return turn_force_preserve, events, self._pending_unrescuable

    def _backward_walk(self, shadow: ShadowBuffer, node_id: int, last_text: str) -> List[Dict]:
        """Depth-capped upstream walk, off by default. Pulls ancestor
        decisions whose critical values aren't already recoverable in
        what was actually shipped last turn, within multi_hop_depth hops
        and a fixed token budget -- so one rescue can't cascade into
        re-hydrating a whole trace."""
        out: List[Dict] = []
        frontier: List[Tuple[int, int]] = [(node_id, 0)]
        seen = {node_id}
        spent = 0
        while frontier:
            nid, depth = frontier.pop(0)
            if depth >= self.multi_hop_depth or spent >= self.multi_hop_token_budget:
                continue
            for up in shadow.dag.upstream(nid):
                if up in seen:
                    continue
                seen.add(up)
                card = shadow.decisions[up]
                vals = _critical_values(card)
                if not vals or all(_art_in_text(v, last_text) for v in vals):
                    continue
                cost = sum(_tok(v) for v in vals)
                if spent + cost > self.multi_hop_token_budget:
                    continue
                out.append(card)
                spent += cost
                frontier.append((up, depth + 1))
        return out
    # --------------------------------------------------------------------------
# 8. Automatic session management for compress(enable_rescue=True).
#
# compress() is called with the FULL growing message list each turn, not
# just the newest message -- so "what's new since last call" is inferred
# by remembering how many messages were seen last time, per session_id.
# This means `messages` must be a growing prefix across calls sharing a
# session_id (append-only). A new conversation needs a new session_id, or
# reset_rescue_session() first -- there is no way to detect an unrelated
# trace reusing the same session_id except the length-shrank case handled
# below.
# --------------------------------------------------------------------------

import threading as _threading
from typing import Any as _Any

_rescue_sessions_lock = _threading.Lock()
_rescue_sessions: Dict[str, Dict[str, _Any]] = {}


def reset_rescue_session(session_id: str = "default") -> None:
    """Drop all rescue state for this session_id (ShadowBuffer, engine,
    last-compressed cache, seen-count). Call this when starting a new,
    unrelated conversation that happens to reuse a session_id."""
    with _rescue_sessions_lock:
        _rescue_sessions.pop(session_id, None)


def _get_or_create_session(session_id: str, engine_kwargs: Optional[Dict] = None) -> Dict[str, _Any]:
    with _rescue_sessions_lock:
        sess = _rescue_sessions.get(session_id)
        if sess is None:
            sess = {
                "shadow": ShadowBuffer(),
                "engine": RescueEngine(**(engine_kwargs or {})),
                "last_compressed": [],
                "n_seen": 0,
            }
            _rescue_sessions[session_id] = sess
        return sess


def _run_rescue_for_call(
    messages: List[Dict],
    session_id: str,
    budget_tokens: int,
    engine_kwargs: Optional[Dict] = None,
) -> Tuple[Set[str], List["RescueEvent"], List[UnrescuableEviction], Dict[str, _Any]]:
    """Diffs `messages` against what this session has already seen, feeds
    only the NEW messages through process_turn (so decisions aren't
    re-extracted for old turns every call), and returns the accumulated
    force_preserve set for THIS compress() call plus the live session
    dict so the caller can update last_compressed after compression."""
    sess = _get_or_create_session(session_id, engine_kwargs)

    if len(messages) < sess["n_seen"]:
        # Shorter than what we've seen -- almost certainly a different,
        # unrelated trace reusing this session_id. Reset rather than
        # silently diffing against the wrong prefix.
        sess = _get_or_create_session(session_id + "__reset_marker", engine_kwargs)
        with _rescue_sessions_lock:
            _rescue_sessions[session_id] = sess
        sess["n_seen"] = 0
        sess["last_compressed"] = []

    new_msgs = messages[sess["n_seen"]:]
    force_preserve_total: Set[str] = set()
    events_total: List["RescueEvent"] = []
    unrescuable_total: List[UnrescuableEviction] = []

    for msg in new_msgs:
        fp, events, unrescuable = sess["engine"].process_turn(
            new_message=msg,
            shadow=sess["shadow"],
            last_compressed_messages=sess["last_compressed"],
            compression_budget_tokens=budget_tokens,
        )
        force_preserve_total |= fp
        events_total.extend(events)
        unrescuable_total.extend(unrescuable)

    sess["n_seen"] = len(messages)
    return force_preserve_total, events_total, unrescuable_total, sess