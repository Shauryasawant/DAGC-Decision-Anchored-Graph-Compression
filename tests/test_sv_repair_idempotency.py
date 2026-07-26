"""
tests/test_sv_repair_idempotency.py

Regression tests for the double-invocation bug in compress_dagc_sv, where
_verify_and_repair was called twice in a row with identical arguments --
the first call actually repaired `compressed` in place, but its report
was discarded; the second call then measured the already-repaired state,
found RCI already at/above floor, and returned repaired=False,
artifacts_added=0, RCI_pre_repair=<the post-repair value>.

These tests fail loudly if that pattern reappears, in two complementary
ways:
  1. test_verify_and_repair_called_exactly_once -- structural guard.
     Doesn't depend on any specific trace needing repair; just proves
     compress_dagc_sv invokes the repair pass a single time per call.
  2. test_repair_report_reflects_actual_repair -- behavioral guard,
     directly unit-testing _verify_and_repair (not the wrapper), so it
     can't be fooled by anything upstream silently calling it twice.
"""
from unittest.mock import patch

import pytest

from dagc.sv_dagc import compress_dagc_sv, _verify_and_repair


# ---------------------------------------------------------------------
# 1. Structural guard: compress_dagc_sv must call _verify_and_repair
#    exactly once. This is the direct regression test for the bug that
#    was found -- it will fail immediately if the duplicated call block
#    is ever reintroduced, regardless of what trace is passed in.
# ---------------------------------------------------------------------
def test_verify_and_repair_called_exactly_once():
    messages = [
        {"role": "user", "content": "Use mirrored vdevs instead of RAIDZ because rebuilds are faster."},
        {"role": "assistant", "content": "Understood, I'll design the pool using only mirrored vdevs."},
    ]

    with patch("dagc.sv_dagc._verify_and_repair", wraps=_verify_and_repair) as spy:
        compress_dagc_sv(messages, rci_floor=1.0)

    assert spy.call_count == 1, (
        f"_verify_and_repair was called {spy.call_count} times; expected exactly 1. "
        "If this fails, check compress_dagc_sv for a duplicated call block -- "
        "the second call silently discards the first call's real repair report."
    )


# ---------------------------------------------------------------------
# 2. Behavioral guard: unit-test _verify_and_repair directly (bypassing
#    compress_dagc_sv entirely) on a fixture engineered so RCI_pre_repair
#    is guaranteed to start below rci_floor. Asserts on side effects that
#    cannot lie under double-invocation: artifacts_added, the presence of
#    "[repaired: ...]" text in the output, and RCI actually improving.
#
# NOTE: adjust the `decisions` dict shape below if your actual decision
# schema differs (this mirrors the action/target/rationale shape seen in
# _decision_critical_values usage elsewhere in the package).
# ---------------------------------------------------------------------
def test_repair_report_reflects_actual_repair():
    # Original trace: msg 0 contains the artifact "RAIDZ" that the
    # decision depends on.
    messages = [
        {"role": "user", "content": "Don't use RAIDZ, mirrored arrays rebuild faster."},
        {"role": "assistant", "content": "OK, using mirrored vdevs."},
    ]

    # Simulate compression having dropped msg 0 entirely -- only msg 1
    # survived, carrying its original _orig_idx.
    compressed = [
        {"role": "assistant", "content": "OK, using mirrored vdevs.", "_orig_idx": 1},
    ]

    # A decision whose critical value ("RAIDZ") is no longer present
    # anywhere in `compressed`, forcing RCI_pre_repair below floor.
    # Schema matches extract_decisions()'s actual return shape exactly
    # (type/action/target/rationale/artifacts/verbatim/msg_idx) -- confirmed
    # against src/dagc/extraction.py. `artifacts` needs all three of
    # paths/ids/errors since build_dependency_graph's `cited` set only
    # reads paths+ids, but other code paths may expect all three keys
    # to exist.
    decisions = [
        {
            "type": "action",
            "action": "use",
            "target": "RAIDZ",
            "rationale": [],
            "artifacts": {"paths": [], "ids": [], "errors": []},
            "verbatim": "OK, using mirrored vdevs.",
            # msg_idx=1 (not 0): build_dependency_graph only creates a
            # dependency edge when origin < msg_idx -- RAIDZ's origin is
            # msg 0, so the decision must be attributed to msg 1 (the
            # assistant's reply that depends on it) for the edge to exist.
            "msg_idx": 1,
        },
    ]

    compressed_out, report = _verify_and_repair(
        messages, compressed, decisions,
        rci_floor=1.0, max_repair_tokens=None, use_extended_scope=False,
    )

    assert report["repaired"] is True, (
        f"Expected repair to trigger, got report={report}"
    )
    assert report["artifacts_added"] > 0, (
        f"Expected at least one artifact repaired, got report={report}"
    )
    assert report["RCI_pre_repair"] < 1.0, (
        "RCI_pre_repair should reflect the ORIGINAL (pre-repair) deficient "
        f"state, not an already-repaired one. Got {report['RCI_pre_repair']}."
    )
    assert report["RCI_post_repair"] >= report["RCI_pre_repair"], (
        "Repair should not make RCI worse."
    )

    repaired_text = " ".join(m.get("content", "") for m in compressed_out)
    assert "[repaired:" in repaired_text, (
        "Expected a '[repaired: ...]' tag or stub in the repaired output, "
        f"got: {repaired_text!r}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))