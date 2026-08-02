"""
filler_score.py — probabilistic filler-vs-decision clause scoring.

WHY NOT ANOTHER REGEX BLOCKLIST
--------------------------------
rationale_ext.py's own docstring already names the trap: an open
semantic class ("words that mean X") turns into permanent whack-a-mole,
where every gap becomes another word added to the list. Courtesy
phrasing has the same shape ("happy to help", "I understand", "feel
free to", ...), AND it has a second failure mode a blocklist can't
see at all: the exact same opener ("I understand...") is sometimes
pure filler and sometimes the matrix clause of a sentence whose
COMPLEMENT carries the real decision ("I understand your account is
suspended because of a missed payment."). Phrase-level matching can't
tell these apart because the danger lives in what follows the phrase,
not the phrase itself.

This module replaces "is this phrase on a list" with two independent,
combinable, low-dependency mathematical signals, evaluated per CLAUSE
(not per sentence, not per phrase):

  1. NAIVE BAYES LOG-ODDS over closed-class structural features --
     does this clause carry a judgment verb, a causal connective, an
     enumeration, an artifact/ID, a directive cue? Each is a feature
     this codebase already detects elsewhere (extraction.py,
     rationale_ext.py); this module just combines their presence/
     absence into a calibrated log-likelihood-ratio sum instead of an
     ad hoc point score. Ships with hand-set priors, but `fit()`
     re-estimates every weight from plain frequency counts over
     labeled (clause, is_filler) examples -- no external ML library,
     just Counter + math.log. Feed it DME-Verify's own restoration
     events as labels and it calibrates itself against your real
     corpus instead of staying fixed at hand-tuned priors forever.

  2. SHANNON SURPRISAL over candidate entities/artifacts inside the
     clause: -log2(count_in_trace / total_mentions). A fact mentioned
     once anywhere in the trace has high surprisal -- deleting this
     clause may destroy the ONLY copy. A fact mentioned five times has
     low surprisal -- redundant, safe to drop this one instance. This
     is rationale_ext.py's own `_corroborated_elsewhere` check, made
     continuous instead of a binary >=1-mention threshold, so a
     "mentioned twice, both times in isolated single-clause asides"
     case is scored differently from "mentioned twice in solid,
     well-attested context" -- both currently just read as
     "corroborated=True".

DECISION RULE
-------------
delete(clause)  iff  P_filler(clause) >= T_prob  AND  max_surprisal(clause) <= T_bits

Both gates required, matching this codebase's existing two-signal
discipline (negation+corroboration in rationale_ext.py, courtesy+
decisiveness implicit in extraction.py's mood checks): reading like an
acknowledgment is necessary but not sufficient. A clause phrased exactly
like a courtesy closer that happens to contain the ONLY mention of a
dollar figure is still blocked by the surprisal gate, regardless of how
high its filler probability scores.

INTEGRATION
-----------
Treat this as a CANDIDATE GENERATOR, not a final authority. Run its
output through your existing DME-Verify (RCI-gated verify & repair) --
that subsystem already exists specifically to restore accidentally-
dropped critical artifacts, so it is the correct ground-truth safety
net for this exact problem, not a new one to build:

    candidates = filler_deletion_candidates(text, full_text, scorer)
    compressed  = apply_deletions(text, [c for c, p, s in candidates])
    restored    = DME_Verify.verify_and_repair(original, compressed)

Any clause DME-Verify restores is a labeled false positive -- append it
to your labeled-example set as (clause, is_filler=False) and periodically
re-run `scorer.fit()`. Confirmed-safe deletions (nothing restored)
become (clause, is_filler=True) labels. This turns filler detection into
an online-calibrated classifier instead of a fixed heuristic, using
exactly the corroboration/verification machinery you already built.

DEPENDENCIES: stdlib only (re, math, collections, typing). Everything
that needs real linguistic detection (artifacts, judgment verbs, causal
connectives, entity shapes) is imported from your existing extraction.py
/ rationale_ext.py / utils.py rather than reimplemented here, so a
detection improvement made there benefits this module automatically.
"""
from __future__ import annotations
import math
import json   # <-- add this
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple
from .extraction import _STRONG_DECISIVE_VERBS_NO_CONFIRM

# in _clause_features:

# Reuse existing detectors rather than re-implementing them -- keeps this
# module's notion of "artifact" / "judgment verb" / "causal connective"
# identical to the rest of the package, and means improvements made in
# those modules propagate here for free.
from .utils import _artifacts, STOPWORDS
from .extraction import _ACTION_DECISION_CUE, _ENTITY_BLOCKLIST, _RE_ENTITY, _RE_ENTITY_SNAKE
from .rationale_ext import _CAUSAL_RE, _clauses, _clauses_fence_aware


# --- Closed-class courtesy openers -----------------------------------------
# Same finite-grammatical-class argument _DISCOURSE_CONNECTIVES already
# makes in extraction.py: this is a small, closed set of conventional
# opener TEMPLATES, not an open-class attempt to name every way someone
# could be polite. It is deliberately a WEAK signal (see weights below) --
# matching it never deletes anything by itself.
_COURTESY_OPENER_RE = re.compile(
    r"""^\s*(?:
        i'?m\s+happy\s+to\s+help |
        i'?m\s+here\s+to\s+help |
        i\s+understand\b |
        feel\s+free\s+to |
        please\s+let\s+me\s+know |
        i\s+want\s+to\s+ensure |
        thank(?:s|\s+you)\s+for\s+your\s+patience |
        i\s+appreciate\s+your\s+patience |
        happy\s+to\s+assist
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Numbered/lettered option menus -- "Would you like me to: 1) ... 2) ...",
# "option A / option B". Deliberately protective: an enumerated menu is
# the customer's actual available-actions list, not a pleasantry, even
# though it often sits inside a "Would you like me to..." courtesy frame.
_ENUMERATION_RE = re.compile(
    r"\boption\s+[A-Za-z0-9]\b|(?:^|\n|\s)\(?\d{1,2}[\.\)]\s+\S",
    re.IGNORECASE,
)

# --- add to filler_score.py, near the bottom, before "Combined candidate generator" ---

def apply_deletions(text: str, clauses_to_delete: List[str]) -> str:
    """
    Remove the given clause strings from `text`, preserving the order of
    surviving clauses. Matches by exact string against _clauses(text) --
    if a clause isn't found verbatim (e.g. a whitespace/normalization
    difference upstream), it is left in place rather than risking a wrong
    partial removal.
    """
    to_delete = set(clauses_to_delete)
    kept = [c for c in _clauses_fence_aware(text) if c not in to_delete]
    return ' '.join(c.strip() for c in kept if c.strip())

def _clause_features(clause: str) -> Dict[str, bool]:
    """
    Binary structural feature vector for one clause. Every feature here
    is either imported from existing detectors or a small closed-class
    regex defined above -- nothing open-class, nothing that becomes a
    maintenance burden as new phrasings appear.
    """
    arts = _artifacts(clause)
    has_entity = any(
        m.group(0).lower() not in _ENTITY_BLOCKLIST
        for pattern in (_RE_ENTITY, _RE_ENTITY_SNAKE)
        for m in pattern.finditer(clause)
    )
    tokens = clause.strip().split()
    return {
    'courtesy_opener':   bool(_COURTESY_OPENER_RE.match(clause)),
    'has_directive_cue': bool(_ACTION_DECISION_CUE.search(clause)),
    'has_artifact':      bool(arts.get('ids') or arts.get('paths')
                               or arts.get('numbers') or arts.get('errors')),
    'has_causal':        bool(_CAUSAL_RE.search(clause)),
    'has_enumeration':   bool(_ENUMERATION_RE.search(clause)),
    'has_judgment_verb': bool(_STRONG_DECISIVE_VERBS_NO_CONFIRM.search(clause)),
    'has_entity':        has_entity,
    'is_short':          len(tokens) <= 6,
}


class FillerScorer:
    """
    Naive Bayes log-odds classifier over the features above.

    Weights are natural-log likelihood ratios: positive = evidence
    FOR filler, negative = evidence for "protected / decision-bearing".
    Score = prior_log_odds + sum(weight_f for each feature present);
    P(filler) = sigmoid(score). This is textbook Naive Bayes text
    classification -- the same math spam filters used before neural
    methods existed -- chosen deliberately because it is (a) fully
    interpretable per-feature, (b) trivially re-fittable from small
    labeled sets with no external library, and (c) numerically stable
    with nothing but `math.log`/`math.exp`.
    """
    FEATURES: Tuple[str, ...] = (
        'courtesy_opener', 'has_judgment_verb', 'has_directive_cue',
        'has_artifact', 'has_causal', 'has_enumeration', 'has_entity',
        'is_short',
    )

    # Hand-set starting priors. Treat these as a cold-start, not a final
    # answer -- call `fit()` on labeled (clause, is_filler) pairs from
    # your own DME-Verify restoration log as soon as you have any, and
    # these get replaced by corpus-estimated values.
    _DEFAULT_WEIGHTS: Dict[str, float] = {
        'courtesy_opener':    2.0,
        'has_judgment_verb': -2.5,
        'has_directive_cue': -2.0,
        'has_artifact':      -3.0,
        'has_causal':        -2.5,
        'has_enumeration':   -3.0,
        'has_entity':        -1.5,
        'is_short':           0.5,
    }
    # Slight prior AGAINST deletion: in a typical trace most clauses are
    # not pure filler, so an all-zero feature vector should still lean
    # "protected" rather than 50/50.
    _DEFAULT_PRIOR_LOG_ODDS: float = -1.0

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 prior_log_odds: Optional[float] = None):
        self.weights = dict(weights) if weights else dict(self._DEFAULT_WEIGHTS)
        self.prior_log_odds = (self._DEFAULT_PRIOR_LOG_ODDS
                                if prior_log_odds is None else prior_log_odds)

    def score(self, clause: str) -> float:
        """Returns P(filler | clause) in [0, 1]."""
        feats = _clause_features(clause)
        log_odds = self.prior_log_odds + sum(
            self.weights.get(f, 0.0) for f, present in feats.items() if present
        )
        # Numerically stable sigmoid.
        if log_odds >= 0:
            z = math.exp(-log_odds)
            return 1.0 / (1.0 + z)
        z = math.exp(log_odds)
        return z / (1.0 + z)

    def fit(self, labeled_examples: Iterable[Tuple[str, bool]], alpha: float = 1.0,
            min_support: int = 5) -> None:
        """
        Re-estimate every weight via Laplace-smoothed Naive Bayes:

            weight_f = log( P(f=1 | filler) / P(f=1 | protected) )

        labeled_examples: iterable of (clause_text, is_filler). Build
        this incrementally from your own pipeline: label=True for
        clauses you deleted and DME-Verify did NOT restore (confirmed
        safe), label=False for clauses DME-Verify DID restore (confirmed
        you were about to lose something). This is the calibration loop
        described in the module docstring -- run it periodically as the
        log accumulates, not just once.

        alpha is the Laplace smoothing constant: guards against zero
        counts producing a divide-by-zero or an infinite log-ratio for
        features rarely seen in one class yet.

        min_support: minimum number of observations a feature must have
        in BOTH classes before its corpus-estimated weight is trusted.
        Below that, Laplace smoothing's pseudo-count can dominate the
        true signal -- a feature with 0-4 real observations in the
        minority class can flip sign purely from alpha, not from
        evidence -- so the feature keeps its CURRENT weight (whatever
        self.weights already holds, i.e. the prior/default or a previous
        fit()'s value) instead of being overwritten by a noisy estimate
        built on a near-empty cell.
        """
        n_filler = n_protected = 0
        f_filler: Counter = Counter()
        f_protected: Counter = Counter()

        for clause, is_filler in labeled_examples:
            feats = _clause_features(clause)
            bucket = f_filler if is_filler else f_protected
            if is_filler:
                n_filler += 1
            else:
                n_protected += 1
            for f, present in feats.items():
                if present:
                    bucket[f] += 1

        if n_filler == 0 or n_protected == 0:
            # Not enough of both classes yet to estimate a ratio --
            # keep current weights rather than producing degenerate
            # (all +inf or all -inf) values.
            return

        new_weights = {}
        skipped = []
        for f in self.FEATURES:
            if f_filler[f] < min_support or f_protected[f] < min_support:
                # Thin support in at least one class -- don't trust a
                # corpus estimate built mostly from the smoothing
                # constant. Keep whatever weight this feature currently
                # holds (default prior, or a prior fit()'s value).
                new_weights[f] = self.weights.get(f, self._DEFAULT_WEIGHTS.get(f, 0.0))
                skipped.append((f, f_filler[f], f_protected[f]))
                continue
            p_f_given_filler = (f_filler[f] + alpha) / (n_filler + 2 * alpha)
            p_f_given_protected = (f_protected[f] + alpha) / (n_protected + 2 * alpha)
            new_weights[f] = math.log(p_f_given_filler / p_f_given_protected)
        self.weights = new_weights
        if skipped:
            print(f"[fit] kept prior weight for {len(skipped)} thin-support feature(s) "
                  f"(need >={min_support} obs/class): "
                  f"{[(f, ff, fp) for f, ff, fp in skipped]}")

        self.prior_log_odds = math.log((n_filler + alpha) / (n_protected + alpha))
        self.weights = new_weights

    def to_json(self) -> str:
            """Serialize calibrated weights + prior so they survive process
            restarts. Call this once after fit() and save the result."""
            return json.dumps({'weights': self.weights, 'prior_log_odds': self.prior_log_odds})

    @classmethod
    def from_json(cls, s: str) -> 'FillerScorer':
        """Reconstruct a calibrated scorer from to_json() output."""
        d = json.loads(s)
        return cls(weights=d['weights'], prior_log_odds=d['prior_log_odds'])


# --- Shannon surprisal over candidate spans ---------------------------------

def _candidate_spans(text: str) -> List[str]:
    """Every artifact/entity shape this codebase already trusts as
    potentially decision-critical, lower-cased for counting."""
    arts = _artifacts(text)
    spans = [a.lower() for a in arts.get('ids', [])]
    spans += [a.lower() for a in arts.get('paths', [])]
    spans += [a.lower() for a in arts.get('numbers', [])]
    for pattern in (_RE_ENTITY, _RE_ENTITY_SNAKE):
        for m in pattern.finditer(text):
            w = m.group(0)
            if w.lower() not in _ENTITY_BLOCKLIST:
                spans.append(w.lower())
    return spans


def _mention_counts(full_text: str) -> Tuple[Counter, int]:
    """
    Frequency table of every candidate span across the WHOLE trace, plus
    the total mention count used to normalize surprisal. Computed once
    per trace and reused for every clause's lookup -- O(n) over the
    trace instead of re-scanning per clause.
    """
    counts = Counter(_candidate_spans(full_text))
    total = sum(counts.values()) or 1
    return counts, total


def _clause_max_surprisal(clause: str, mention_counts: Counter, total: int) -> float:
    """
    Shannon self-information (bits) of the RAREST candidate span in this
    clause: -log2(count(span) / total). Returns 0.0 for a clause with no
    candidate spans -- there is nothing informational to protect, so it
    cannot block deletion on this gate (the Naive Bayes gate still can).

    Interpretation: 0 bits = perfectly redundant/ubiquitous. Each +1 bit
    roughly halves how often that fact recurs elsewhere in the trace.
    A span appearing in only 1 of 64 equally-likely "slots" carries 6
    bits -- treat anything above your chosen threshold as irreplaceable.
    """
    spans = _candidate_spans(clause)
    if not spans:
        return 0.0
    return max(-math.log2(mention_counts.get(s, 1) / total) for s in spans)


# --- Combined candidate generator -------------------------------------------

def filler_deletion_candidates(
    text: str,
    full_text: str,
    scorer: FillerScorer,
    prob_threshold: float = 0.75,
    surprisal_threshold_bits: float = 4.0,
) -> List[Tuple[str, float, float]]:
    """
    Returns [(clause, p_filler, max_surprisal_bits), ...] for every
    clause in `text` that clears BOTH gates:

        p_filler            >= prob_threshold
        max_surprisal_bits  <= surprisal_threshold_bits

    Treat the return value as CANDIDATES to delete, then run the result
    through DME-Verify before trusting it (see module docstring). Do not
    wire this directly into a destructive compression step without that
    check.

    prob_threshold=0.75 and surprisal_threshold_bits=4.0 are starting
    points, not derived constants -- sweep both against check_rationale.py
    and your wildchat traces, and report precision/recall at whatever you
    land on, the same way every other threshold in this codebase already
    is documented as calibrated-not-assumed.
    """
    mention_counts, total = _mention_counts(full_text)
    out = []
    for clause in _clauses_fence_aware(text):
        p = scorer.score(clause)
        if p < prob_threshold:
            continue
        s = _clause_max_surprisal(clause, mention_counts, total)
        if s <= surprisal_threshold_bits:
            out.append((clause, p, s))
    return out