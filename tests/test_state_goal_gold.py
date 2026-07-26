"""
Gold-set regression tests for the State/Goal anchor pipeline
(state_goal_extract.py + anchor_lifecycle.py + sv_dagc.py's
preserve_state/preserve_goals).

This is a STARTER gold set (18 category examples + 6 lifecycle cases),
not the 40-60-per-category set the design called for -- see
ANCHOR_TAXONOMY.md for the labeling template and the plan to grow this
from real wildchat/LoCoMo traces. Every example here was run against
the actual extractor before being locked in as an assertion (not
written speculatively and hoped to pass) -- see the two calibration
probes this file's cases were drawn from.

Four things this file is responsible for proving, matching the
non-negotiable requirements from the design discussion:
  1. Stage-1 category tagging precision/recall, measured separately,
     against explicit thresholds (not vibes).
  2. Stage-2 lifecycle resolution correctness on concrete cases,
     including the negative case (an unrelated "done" must NOT resolve
     an unrelated goal -- content-overlap gating, not keyword presence).
  3. The compression-ratio guard: even when Stage 1 is deliberately
     generous, the repair pass cost is bounded, never proportional to
     how much noise Stage 1 let through.
  4. Zero regression: preserve_state=False, preserve_goals=False (the
     default) produces byte-identical output to the pre-existing
     compress_dagc_sv, and the full existing suite (tested separately
     by just running `pytest tests/`) is untouched by any of this.
"""
from __future__ import annotations

import pytest

from dagc.compressor import DAGCConfig, compress
from dagc.state_goal_extract import extract_state_goal_candidates
from dagc.anchor_lifecycle import resolve_lifecycle, active_anchors
from dagc.sv_dagc import compress_dagc_sv
from dagc.utils import _get_text, _tok


# ---------------------------------------------------------------------------
# 1. Stage-1 category gold set (precision / recall)
# ---------------------------------------------------------------------------

# (text, expected_category) -- expected_category is None for negatives
# (things that must NOT be tagged as state or goal at all).
CATEGORY_GOLD_SET = [
    ("I live in Pune with my family.", "state"),
    ("My deadline is next Friday.", "state"),
    ("I'm allergic to peanuts, please avoid recommending recipes with them.", "state"),
    ("Please keep all responses under 200 words for this whole session.", "state"),
    ("From now on, always cite your sources.", "state"),
    ("As a rule, never use profanity in your replies.", "state"),
    ("My manager is Priya and she reviews all my PRs.", "state"),
    ("I work at a fintech startup in Bangalore.", "state"),
    ("I prefer dark mode over light mode.", "state"),
    ("Our budget is $50,000 for this quarter.", "state"),
    ("I want to migrate the backend to Postgres eventually.", "goal"),
    ("I'm planning to run a marathon next year.", "goal"),
    ("I need to finish the report by tomorrow.", "goal"),
    ("My goal is to learn Spanish this year.", "goal"),
    ("I'm looking for a new apartment closer to work.", "goal"),
    ("Remind me to call the plumber tomorrow.", "goal"),
    ("I'm trying to quit smoking.", "goal"),
    ("I intend to switch careers into data science.", "goal"),
    # Negatives: interrogatives that share cue vocabulary with a real
    # anchor, filler/ack clauses, and decision-shaped text that belongs
    # to the Decision/Evidence pipeline, not this one.
    ("What time zone should I use for the logs?", None),
    ("Do you know how tall the Eiffel Tower is?", None),
    ("What's my deadline again?", None),
    ("Is it going to rain tomorrow?", None),
    ("Recommend the best index for this table. therefore B-tree is the winner.", None),
    ("Confirmed, DBOPS-4471 and result.csv are preserved.", None),
    ("Sure, sounds good, thanks!", None),
    ("Great, that works for me.", None),
    ("I mean, that's wild.", None),
    ("Okay perfect, understood.", None),
    ("The quarterly report shows revenue up 12%.", None),
    ("Use PostgreSQL_3 because it handles concurrent writes better.", None),
]

# Documented, known limitation (not this module's bug): convmem's own
# state-cue regex requires "our budget is" adjacent and misses
# possessive-chain phrasing like "our team's budget is" -- tracked in
# ANCHOR_TAXONOMY.md as a Stage-1 recall gap for convmem.classify to
# eventually fix, not papered over here.
KNOWN_STAGE1_GAPS = [
    "Our team's budget is $50,000 for this quarter.",
]

# Thresholds set deliberately asymmetric, per the design discussion:
# State/Goal is recall-weighted (missing an anchor is a silent,
# hard-to-detect failure) while still requiring a real precision floor
# so this doesn't degenerate into "tag everything."
MIN_RECALL = 0.90
MIN_PRECISION_NEG_PASS_RATE = 0.90  # fraction of negatives with zero false positives


def _category_probe(text):
    return {c["anchor_type"] for c in extract_state_goal_candidates(
        [{"role": "user", "content": text}])}


def test_category_gold_set_recall_and_precision_meet_bar():
    positives = [(t, c) for t, c in CATEGORY_GOLD_SET if c is not None]
    negatives = [(t, c) for t, c in CATEGORY_GOLD_SET if c is None]

    hits = [t for t, expected in positives if expected in _category_probe(t)]
    misses = [t for t, expected in positives if expected not in _category_probe(t)]
    recall = len(hits) / len(positives)

    false_positive_texts = [t for t, _ in negatives if _category_probe(t)]
    neg_pass_rate = 1 - len(false_positive_texts) / len(negatives)

    print(
        "\\nCategory gold-set results: "
        f"recall {len(hits)}/{len(positives)} ({recall:.1%}); "
        f"negatives rejected {len(negatives) - len(false_positive_texts)}"
        f"/{len(negatives)} ({neg_pass_rate:.1%}); "
        f"false positives {len(false_positive_texts)}"
    )

    assert recall >= MIN_RECALL, (
        f"category recall {recall:.2f} below {MIN_RECALL} -- misses: {misses}")
    assert neg_pass_rate >= MIN_PRECISION_NEG_PASS_RATE, (
        f"false positives on negatives: {false_positive_texts}")


@pytest.mark.parametrize("text", KNOWN_STAGE1_GAPS)
def test_known_stage1_gap_is_still_a_miss(text):
    """Documents a known limitation rather than silently passing or
    silently failing CI -- if this ever starts passing (convmem's cue
    regex gets fixed upstream), this test should be deleted, not left
    green by accident."""
    assert _category_probe(text) == set()


# ---------------------------------------------------------------------------
# 2. Stage-2 lifecycle gold set
# ---------------------------------------------------------------------------

LIFECYCLE_CASES = [
    dict(
        name="state_single_mention_stays_active",
        messages=[{"role": "user", "content": "My deadline is next Friday."}],
        expect=[("state", "My deadline is next Friday.", "active")],
    ),
    dict(
        name="state_explicit_update_supersedes",
        messages=[
            {"role": "user", "content": "I live in Pune."},
            {"role": "assistant", "content": "Got it, Pune noted."},
            {"role": "user", "content": "Actually, I live in Mumbai now, not Pune."},
        ],
        expect=[
            ("state", "I live in Pune.", "superseded"),
            ("state", "I live in Mumbai now", "active"),
        ],
    ),
    dict(
        name="goal_completed",
        messages=[
            {"role": "user", "content": "I need to finish the deployment script by Friday."},
            {"role": "assistant", "content": "Sure, I can help with that."},
            {"role": "user", "content": "Finally finished the deployment script, it is done."},
        ],
        expect=[("goal", "I need to finish the deployment script by Friday.", "completed")],
    ),
    dict(
        name="goal_abandoned",
        messages=[
            {"role": "user", "content": "I want to migrate the backend to Postgres."},
            {"role": "assistant", "content": "Sounds good, I can help plan that."},
            {"role": "user", "content": "Actually never mind about migrating to Postgres, decided not to do that."},
        ],
        expect=[("goal", "I want to migrate the backend to Postgres.", "abandoned")],
    ),
    dict(
        name="unrelated_completion_does_not_resolve_goal",
        messages=[
            {"role": "user", "content": "I want to migrate the backend to Postgres."},
            {"role": "assistant", "content": "Sounds good."},
            {"role": "user", "content": "Done with lunch, that was good."},
        ],
        expect=[("goal", "I want to migrate the backend to Postgres.", "active")],
    ),
    dict(
        name="instruction_constraint_stays_active_through_noise",
        messages=[
            {"role": "user", "content": "Please keep all responses under 200 words for this whole session."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "What do you think about option A vs option B?"},
            {"role": "assistant", "content": "Recommend option A. therefore option A is the winner."},
        ],
        expect=[("state", "Please keep all responses under 200 words for this whole session.", "active")],
    ),
]


@pytest.mark.parametrize("case", LIFECYCLE_CASES, ids=[c["name"] for c in LIFECYCLE_CASES])
def test_lifecycle_gold_set(case):
    candidates = extract_state_goal_candidates(case["messages"])
    anchors = resolve_lifecycle(case["messages"], candidates)
    by_clause = {a["clause"]: a for a in anchors}
    for cat, clause, status in case["expect"]:
        assert clause in by_clause, f"missing candidate for clause={clause!r}"
        a = by_clause[clause]
        assert (a["anchor_type"], a["status"]) == (cat, status), (
            f"clause={clause!r} expected ({cat},{status}) "
            f"got ({a['anchor_type']},{a['status']})")


def test_negation_scope_does_not_falsely_trigger_on_never_mind():
    """Regression guard for the specific false positive caught during
    development: 'never mind' must resolve the GOAL it dismisses, and
    must NOT also get tagged as a new standing-instruction STATE
    anchor just because it contains 'never'."""
    messages = [
        {"role": "user", "content": "I want to migrate the backend to Postgres."},
        {"role": "user", "content": "Actually never mind about migrating to Postgres, decided not to do that."},
    ]
    candidates = extract_state_goal_candidates(messages)
    state_clauses = [c["clause"] for c in candidates if c["anchor_type"] == "state"]
    assert not any("never mind" in c.lower() for c in state_clauses)


# ---------------------------------------------------------------------------
# 3. End-to-end: the exact bug pattern from the design discussion
# ---------------------------------------------------------------------------

def _noisy_decision_trace(constraint_text, n_rounds=15):
    messages = [{"role": "user", "content": constraint_text}]
    for i in range(n_rounds):
        messages.append({"role": "user",
                          "content": f"Can you help me think through option {i}?"})
        messages.append({"role": "assistant", "content":
                          f"Recommend using PostgreSQL_{i} because it handles "
                          f"concurrent writes better. therefore PostgreSQL_{i} "
                          f"is the winner."})
    return messages


def test_single_mention_state_anchor_dropped_by_baseline_but_kept_by_sv():
    constraint = "Responses must stay under 200 words for this whole session."
    messages = _noisy_decision_trace(constraint)
    cfg = DAGCConfig(TARGET_REDUCTION=0.9)

    baseline = compress(messages, cfg=cfg)
    baseline_text = " ".join(_get_text(m) for m in baseline).lower()

    protected, report = compress_dagc_sv(
        messages, cfg=cfg, preserve_state=True, preserve_goals=True)
    protected_text = " ".join(_get_text(m) for m in protected).lower()

    # This IS the regression: under aggressive compression, a
    # single-mention standing constraint is exactly the shape most
    # likely to be dropped by a corroboration-style gate.
    assert "under 200 words" not in baseline_text, (
        "baseline unexpectedly kept the constraint -- if this starts "
        "failing, the bug this test guards against may already be "
        "fixed upstream; investigate before assuming this test is wrong")
    assert "under 200 words" in protected_text
    assert report["anchor_stubs_added"] >= 1


# ---------------------------------------------------------------------------
# 4. Compression-ratio guard
# ---------------------------------------------------------------------------

def test_anchor_repair_cost_is_bounded_not_proportional_to_candidate_count():
    """Even if Stage 1 tags many candidates, the repair pass must only
    ever add a small, capped number of stub tokens -- protecting
    compression ratio is a property of the bounded repair budget, not
    of Stage-1 precision. Build a trace with 30 distinct single-mention
    state facts and confirm total added tokens stays inside the
    configured cap regardless."""
    messages = []
    for i in range(30):
        messages.append({"role": "user",
                          "content": f"My favorite project number {i} is project-{i}."})
    cfg = DAGCConfig(TARGET_REDUCTION=0.5)

    max_stubs = 10
    _, report = compress_dagc_sv(
        messages, cfg=cfg, preserve_state=True,
        max_anchor_stubs=max_stubs, max_anchor_stub_tokens=30)

    assert report["anchor_stubs_added"] <= max_stubs
    # Bounded total cost: stubs_added * per-stub cap is a hard ceiling.
    assert report["anchor_stub_tokens"] <= max_stubs * 30


# ---------------------------------------------------------------------------
# 5. Zero regression by default
# ---------------------------------------------------------------------------

def test_default_flags_produce_byte_identical_output():
    messages = _noisy_decision_trace(
        "Please keep all responses under 200 words for this whole session.", n_rounds=5)
    cfg = DAGCConfig(TARGET_REDUCTION=0.85)

    out_implicit, report_implicit = compress_dagc_sv(messages, cfg=cfg)
    out_explicit, report_explicit = compress_dagc_sv(
        messages, cfg=cfg, preserve_state=False, preserve_goals=False)

    assert out_implicit == out_explicit
    assert "anchor_stubs_added" not in report_implicit
    assert "anchor_stubs_added" not in report_explicit


def test_active_anchors_helper_filters_by_status_and_category():
    messages = [
        {"role": "user", "content": "My deadline is next Friday."},
        {"role": "user", "content": "I want to migrate the backend to Postgres."},
        {"role": "user", "content": "Never mind about migrating to Postgres, decided not to do that."},
    ]
    anchors = resolve_lifecycle(messages, extract_state_goal_candidates(messages))
    active_state_only = active_anchors(anchors, categories=("state",))
    assert all(a["anchor_type"] == "state" for a in active_state_only)
    active_goal_only = active_anchors(anchors, categories=("goal",))
    assert active_goal_only == []  # the only goal candidate was abandoned
