"""
Preservation round-trip invariant.

This is deliberately NOT a regression test for one specific bug. It
encodes the actual guarantee DAGC is supposed to make: any value the
extractor judged decision-critical in the ORIGINAL trace must still be
recoverable -- verbatim, or via an owned [preserved: ...] tag entry --
after compress_dagc() + a fresh extract_decisions() pass on the
compressed output.

It is parametrized over every trace JSON you already have lying around
for eval (the same files compete_benchmark_v2.py / the DRR harness use),
so it runs the same invariant check across your whole corpus instead of
one hand-picked example.
"""
from __future__ import annotations

import glob
import json
import os

import pytest

from dagc.extraction import extract_decisions
from dagc.compressor import compress_dagc, DAGCConfig, _decision_critical_values# Point this at wherever your benchmark trace JSON files live -- adjust
# to match your repo layout (this mirrors the shape used in
# compete_benchmark_v2.py: a list of {"name", "trace": [...]}).
TRACE_GLOB = os.environ.get(
    "DAGC_TEST_TRACE_GLOB",
    os.path.join(os.path.dirname(__file__), "fixtures", "*.json"),
)


def _load_traces():
    traces = []
    for path in sorted(glob.glob(TRACE_GLOB)):
        with open(path) as f:
            data = json.load(f)
        # Support both "one trace per file" and "list of traces per file".
        items = data if isinstance(data, list) else [data]
        for item in items:
            name = item.get("name", os.path.basename(path))
            trace = item.get("trace", item if isinstance(item, list) else None)
            if trace:
                traces.append(pytest.param(trace, id=name))
    return traces


@pytest.mark.parametrize("trace", _load_traces())
@pytest.mark.parametrize("target_reduction", [0.70, 0.85, 0.90])
def test_critical_facts_survive_compression(trace, target_reduction):
    """
    For every decision extracted from the ORIGINAL trace, every
    decision-critical value (action / target / rationale fact) must
    still be findable in the corresponding reproduced decision after
    compress_dagc() + extract_decisions() on the compressed output.

    Matching is done by the STABLE original msg_idx (compressed messages
    carry `_orig_idx`; extract_decisions must resolve msg_idx/decision_idx
    from it, not from position in the shorter compressed list -- if that
    resolution is broken, everything below fails, which is the point).
    """
    original_decisions = extract_decisions(trace)
    if not original_decisions:
        pytest.skip("no decisions extracted from this trace at all")

    cfg = DAGCConfig(TARGET_REDUCTION=target_reduction)
    compressed = compress_dagc(trace, cfg=cfg)
    reproduced = extract_decisions(compressed)
    reproduced_by_idx = {d["msg_idx"]: d for d in reproduced}

    failures = []
    for od in original_decisions:
        rd = reproduced_by_idx.get(od["msg_idx"])
        if rd is None:
            failures.append(f"decision at msg_idx={od['msg_idx']} vanished entirely "
                             f"(original type={od['type']!r} action={od['action']!r})")
            continue

        critical_facts = _decision_critical_values([od])
        blob = " ".join(rd.get("rationale", []) or []) + " " + \
            str(rd.get("target") or "") + " " + str(rd.get("action") or "")
        blob_low = blob.lower()

        for fact in critical_facts:
            if fact.lower() not in blob_low:
                failures.append(
                    f"msg_idx={od['msg_idx']}: critical fact {fact!r} "
                    f"(from original {od['type']} action={od['action']!r} "
                    f"target={od['target']!r}) did not survive compression "
                    f"at target_reduction={target_reduction}"
                )

    assert not failures, (
        f"{len(failures)} preservation failure(s) at target_reduction={target_reduction}:\n"
        + "\n".join(failures)
    )