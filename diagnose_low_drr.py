"""
diagnose_low_drr.py
=====================================================================
Finds every trace where DAGC's DRR_soft < DRR_THRESHOLD (default 0.95)
and dumps a full diagnostic for each one:

  1. explain_drr()      -- which decision(s) failed, and why
                            (action mismatch / target lost / rationale
                            not recoverable / outright reproduction failure)
  2. diff_trace_report() -- which artifacts (paths/ids/emails/errors)
                            were dropped by compression, and which whole
                            messages got fully elided

Also finds every trace where RCI is None/NaN/unusable (or where
compute_drr() raises outright) -- independent of DRR_CUTOFF -- and dumps
a granular readout of what's actually in the result dict so you can see
*why* RCI didn't compute, without touching dagc_eval internals.

Nothing new is computed here -- this is pure read-off of data
compute_drr() already produces, using dagc_eval.diagnostics, which is
already in your package and was previously unused.

Edit the CONFIG block below to match your existing paths (same values
as compete_benchmark_v2.py), then:

    python diagnose_low_drr.py

Output prints to console AND is written to text file(s) you can
paste back wholesale.
=====================================================================
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict

# Configuration
ROOT_DIR = Path(__file__).resolve().parent

TRACES_PATH = str(ROOT_DIR / "/Users/shaurya/python/DDR/real_world_nemotron.json")
LIB_SOURCE_PATH = "/Users/shaurya/python/dagc_pkg_v2/src"
DRR_CUTOFF = 0.95
MAX_TRACES = None
OUTPUT_FILE = str(ROOT_DIR / "low_drr_diagnostics.txt")
ONLY_NAMES = None

RUN_RCI_DIAGNOSTICS = True
RCI_DEBUG_OUTPUT_FILE = str(ROOT_DIR / "rci_failure_diagnostics.txt")

def _bootstrap_dagc():
    priority_order = [
        LIB_SOURCE_PATH,
        "/Users/shaurya/python/dagc_pkg_v2/src",
        "/Users/shaurya/python/dagc_pkg_v2",
        str(ROOT_DIR),
    ]

    candidate_paths = []
    for candidate in priority_order:
        if os.path.isdir(candidate) and candidate not in candidate_paths:
            candidate_paths.append(candidate)

    # Reverse insertion preserves the declared search order.
    for candidate in reversed(candidate_paths):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    print(f"[info] using library search paths: {candidate_paths}")
    return candidate_paths


candidate_paths = _bootstrap_dagc()

try:
    if importlib.util.find_spec("dagc") is None or importlib.util.find_spec("dagc_eval") is None:
        raise ModuleNotFoundError("dagc / dagc_eval not found")

    import dagc
    import dagc_eval
    from dagc import compress_dagc, extract_decisions
    from dagc.adapters import TiktokenTokenizer, SentenceTransformerEmbedder
    from dagc.graph import build_dependency_graph
    from dagc_eval.benchmark import compute_drr
    from dagc_eval.diagnostics import explain_drr, diff_trace_report

    print(f"[info] dagc loaded from: {dagc.__file__}")
    print(f"[info] dagc_eval loaded from: {dagc_eval.__file__}")

    try:
        from dagc_eval.diagnostics import explain_drr_full
    except ImportError:
        def explain_drr_full(result, show_all=False):
            decisions = result.get("decisions", [])
            if not decisions:
                return "No decisions were found in this trace -- DRR is undefined."

            lines = [f"DRR_soft: {result.get('DRR_soft')}  (mean of per-decision scores)"]
            lines.append(f"{'idx':>4}  {'type':<12}  {'action':>7}  {'target':>7}  {'rationale':>9}  {'overall':>8}  {'pass?':>6}")

            ranked = sorted(decisions, key=lambda x: x.get("match", {}).get("decision_score", 0.0))
            for d in ranked:
                orig = d.get("original", {})
                match = d.get("match", {})
                ts = "n/a" if match.get("target_score") is None else f"{match.get('target_score', 0.0):.3f}"
                lines.append(
                    f"{orig.get('msg_idx', '?'):>4}  {orig.get('type', ''):<12}  "
                    f"{match.get('action_score', 0.0):>7.3f}  {ts:>7}  "
                    f"{match.get('rationale_score', 0.0):>9.3f}  "
                    f"{match.get('decision_score', 0.0):>8.3f}  "
                    f"{'✓' if match.get('reproduced', False) else '✗':>6}"
                )
                if not show_all and match.get('decision_score', 0.0) >= 0.99:
                    continue

            deficits = {'action': 0.0, 'target': 0.0, 'rationale': 0.0}
            for d in decisions:
                m = d.get("match", {})
                deficits['action'] += (1.0 - m.get('action_score', 0.0))
                if m.get("target_score") is not None:
                    deficits['target'] += (1.0 - m['target_score'])
                deficits['rationale'] += (1.0 - m.get('rationale_score', 0.0))

            lines.append("\nTotal deficit by component (higher = bigger drag on DRR_soft):")
            for k, v in sorted(deficits.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {k:<10}: {v:.3f}")

            return "\n".join(lines)

    try:
        from dagc_eval.diagnostics import dump_rationale
    except ImportError:
        def dump_rationale(result: Dict) -> str:
            """Raw original vs reproduced rationale text for every decision."""
            lines = []
            for d in result.get("decisions", []):
                orig = d.get('original', {})
                repro = d.get('reproduced', {})
                match = d.get('match', {})
                lines.append(
                    f"--- idx={orig.get('msg_idx')} type={orig.get('type')} "
                    f"rationale_score={match.get('rationale_score')}"
                )
                lines.append(f"  ORIGINAL rationale: {orig.get('rationale')}")
                lines.append(f"  REPRODUCED rationale: {repro.get('rationale')}")
            return "\n".join(lines)

    try:
        from dagc_eval.diagnostics import dump_actions
    except ImportError:
        def dump_actions(result: Dict) -> str:
            """Raw original vs reproduced action verb for every judgment-type
            decision -- this is where the new tiered _action_match scoring
            (Tier 0/1/2 + connective wildcard) lives, so it's the fastest way
            to tell whether a 0.700 score is a genuine near-synonym match
            (case a: fix is correct, extractor is just noisy) or a structural
            extraction bug (case b: same pair repeating, or a fixed/default
            action string on one side)."""
            lines = []
            for d in result.get("decisions", []):
                orig = d.get('original', {})
                repro = d.get('reproduced', {})
                match = d.get('match', {})
                if orig.get('type') != 'judgment':
                    continue
                lines.append(
                    f"--- idx={orig.get('msg_idx')} action_score={match.get('action_score')}"
                )
                lines.append(f"  ORIGINAL action:    {orig.get('action')!r}")
                lines.append(f"  REPRODUCED action:  {repro.get('action')!r}")
            return "\n".join(lines) if lines else "(no judgment-type decisions in this trace)"

except ModuleNotFoundError as e:
    print("[error] Could not import dagc or dagc_eval.")
    print(f"Checked paths: {candidate_paths}")
    raise

dagc.configure(
    tokenizer=TiktokenTokenizer("cl100k_base"),
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
)
print("[ok] dagc configured with TiktokenTokenizer + SentenceTransformerEmbedder.\n")


def load_traces(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    traces = []
    for item in data:
        tr = item.get("trace")
        if not tr:
            continue
        traces.append({
            "name": item.get("name", f"trace_{len(traces)}"),
            "domain": item.get("domain", "unknown"),
            "trace": tr,
        })

    if MAX_TRACES:
        traces = traces[:MAX_TRACES]
    return traces


def _is_bad_rci(value):
    """True if RCI is None, NaN, or otherwise unusable."""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def diagnose_rci_failures(traces):
    """
    Scans every trace (independent of DRR_CUTOFF) and flags any trace where
    RCI is None / NaN / missing, or where compute_drr() raises outright.
    Dumps everything readable off the result dict plus basic trace-shape
    stats and a few heuristic "likely cause" checks, so you can see *why*
    RCI didn't compute without having to instrument dagc_eval internals.
    """
    chunks = []
    n_checked = 0
    n_bad = 0

    for item in traces:
        name, domain, trace = item["name"], item["domain"], item["trace"]

        decisions = extract_decisions(trace)
        if not decisions:
            continue

        try:
            result = compute_drr(trace, compressor=compress_dagc, verbose=False)
        except Exception as e:
            n_bad += 1
            chunk = [
                "=" * 100,
                f"TRACE: {name}  (domain={domain})  -- compute_drr() RAISED",
                "-" * 100,
                f"exception type : {type(e).__name__}",
                f"exception msg  : {e}",
                f"messages       : {len(trace)}",
                f"decisions      : {len(decisions)}",
                "=" * 100,
                "",
            ]
            report = "\n".join(chunk)
            print(report)
            chunks.append(report)
            continue

        n_checked += 1
        rci = result.get("RCI")

        if not _is_bad_rci(rci):
            continue

        n_bad += 1
        compressed = result.get("compressed")

        chunk = []
        chunk.append("=" * 100)
        chunk.append(f"TRACE: {name}  (domain={domain})  -- RCI is {rci!r}")
        chunk.append("-" * 100)
        chunk.append(f"messages (original)    : {len(trace)}")
        chunk.append(f"messages (compressed)  : {len(compressed) if compressed is not None else 'N/A'}")
        chunk.append(f"decisions extracted     : {len(decisions)}")
        chunk.append(f"reduction (%)           : {result.get('reduction')}")
        chunk.append(f"DRR_soft                : {result.get('DRR_soft')}")
        chunk.append(f"decision_art_ret        : {result.get('decision_art_ret')}")

        edges = build_dependency_graph(trace, decisions)
        chunk.append(f"dependency edges found : {len(edges)}")
        if len(edges) == 0:
            chunk.append("  -> RCI is None because this trace has zero cross-message artifact "
                         "dependencies (no artifact introduced in one message is referenced "
                         "again in a later one) -- this is expected/correct, not a failure.")

        # Include scalar result fields for troubleshooting.
        chunk.append("-" * 100)
        chunk.append("Full result dict (scalars only; lists/dicts shown as type+len):")
        for k, v in result.items():
            if k in ("compressed", "decisions"):
                continue
            if isinstance(v, (list, dict)):
                chunk.append(f"  {k}: <{type(v).__name__} len={len(v)}>")
            else:
                chunk.append(f"  {k}: {v!r}")
            
        # Add common causes of an unusable RCI.
        chunk.append("-" * 100)
        chunk.append("Likely-cause checks:")
        hinted = False
        if len(decisions) == 0:
            chunk.append("  -> zero decisions extracted (RCI denominator likely 0/undefined)")
            hinted = True
        if result.get("reduction") in (0, 0.0, None):
            chunk.append("  -> reduction is 0/None -- no compression happened on this trace "
                          "(check target_reduction / token cap logic for this trace's length)")
            hinted = True
        if compressed is not None and len(compressed) == len(trace):
            chunk.append("  -> compressed length == original length (compressor was a no-op here)")
            hinted = True
        if compressed is not None and len(compressed) == 0:
            chunk.append("  -> compressed trace is EMPTY (over-compression / everything elided)")
            hinted = True
        if not hinted:
            chunk.append("  -> none of the standard heuristics fired; inspect the full result dict above")

        chunk.append("=" * 100)
        chunk.append("")

        block = "\n".join(chunk)
        print(block)
        chunks.append(block)

    print(f"\n[RCI diagnostics] checked {n_checked} traces with decisions; "
          f"{n_bad} had unusable RCI (None/NaN) or raised an exception.")

    with open(RCI_DEBUG_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks))

    print(f"[RCI diagnostics] report written -> {RCI_DEBUG_OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose low-DRR traces from a JSON file")
    parser.add_argument("traces_path", nargs="?", default=TRACES_PATH,
                        help="Path to a traces JSON file (defaults to the built-in trace file)")
    args = parser.parse_args()

    traces_path = args.traces_path
    traces = load_traces(traces_path)
    print(f"Loaded {len(traces)} traces from {traces_path}\n")

    if ONLY_NAMES:
        traces = [t for t in traces if t["name"] in ONLY_NAMES]
        print(f"Filtered to {len(traces)} named trace(s).\n")

    if RUN_RCI_DIAGNOSTICS:
        print("=" * 100)
        print("RUNNING RCI FAILURE SCAN (independent of DRR_CUTOFF)")
        print("=" * 100)
        diagnose_rci_failures(traces)
        print()

    report_chunks = []
    n_scored = 0
    n_low = 0

    for item in traces:
        name, domain, trace = item["name"], item["domain"], item["trace"]

        decisions = extract_decisions(trace)
        if not decisions:
            continue

        try:
            result = compute_drr(trace, compressor=compress_dagc, verbose=False)
        except Exception as e:
            print(f"[warn] compute_drr raised for trace '{name}': {type(e).__name__}: {e} "
                  f"(see rci_failure_diagnostics.txt for detail)")
            continue

        drr = result.get("DRR_soft")
        if drr is None:
            continue
        n_scored += 1

        if not ONLY_NAMES and drr >= DRR_CUTOFF:
            continue
        n_low += 1

        compressed = result["compressed"]

        chunk = []
        chunk.append("=" * 100)
        chunk.append(f"TRACE: {name}   (domain={domain})")
        chunk.append(
            f"messages={len(trace)}  decisions={len(decisions)}  "
            f"reduction={result.get('reduction', float('nan')):.1f}%  "
            f"RCI={result.get('RCI')}  "
            f"decision_art_ret={result.get('decision_art_ret', float('nan')):.2f}"
        )
        chunk.append("-" * 100)
        chunk.append("[explain_drr]")
        chunk.append(explain_drr(result))
        chunk.append("")
        chunk.append("[explain_drr_full]")
        chunk.append(explain_drr_full(result, show_all=True))
        chunk.append("")
        chunk.append("[dump_rationale]")
        chunk.append(dump_rationale(result))
        chunk.append("")
        chunk.append("[dump_actions]")
        chunk.append(dump_actions(result))
        chunk.append("")
        chunk.append("[diff_trace_report]")
        chunk.append(diff_trace_report(trace, compressed))
        chunk.append("=" * 100)
        chunk.append("")

        block = "\n".join(chunk)
        print(block)
        report_chunks.append(block)

    print(f"\nScored {n_scored} traces with decisions; "
          f"{n_low} below DRR_soft={DRR_CUTOFF} (or matched ONLY_NAMES).")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_chunks))

    print(f"\nFull report written -> {OUTPUT_FILE}")
    print("Paste that file's contents back for review.")


if __name__ == "__main__":
    main()
