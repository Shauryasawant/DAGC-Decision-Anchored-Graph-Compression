"""
calibrate_filler_scorer.py — turn sv_dagc's own verify+repair signal
into labeled examples for FillerScorer.fit(), and re-fit.

sv_dagc.py IS NOT MODIFIED BY THIS FILE. compress_dagc_sv is imported
and called as a black box, exactly the way its own docstring describes
using it -- this harness adds a calibration loop ON TOP, it doesn't
touch compression.py's or sv_dagc.py's code paths at all.

THE LOOP
--------
For each filler-deletion candidate clause:
  1. Deep-copy the original messages.
  2. Strip that ONE clause out of the ONE message it came from.
  3. Run compress_dagc_sv(trial_messages, rci_floor=...) -- this runs
     compression + RCI check + repair on the trial trace, using
     sv_dagc.py completely unchanged.
  4. If report['repaired'] is False and report['shortfall'] is empty:
     nothing needed fixing -> this deletion was SAFE -> label
     (clause, is_filler=True).
     If repair was needed OR something couldn't be repaired
     (shortfall non-empty): this deletion broke something -> label
     (clause, is_filler=False), i.e. PROTECTED.

One clause tested per compress_dagc_sv call, in isolation, so a single
run's causality is never ambiguous -- consistent with sv_dagc.py's own
"repair every failing edge at most once" discipline. This is more
compress_dagc_sv calls than testing a whole trace's deletions at once,
but it's a calibration/offline step, not a runtime hot path, and it
means every label you accumulate is unambiguous.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import dagc
from dagc.adapters import TiktokenTokenizer, SentenceTransformerEmbedder
from .filler_score import FillerScorer, filler_deletion_candidates
from .sv_dagc import compress_dagc_sv
from .utils import _get_text
from .rationale_ext import _clauses
from collections import Counter

# Same configuration check_progress.py uses -- compress_dagc_sv relies on
# this being set (at minimum for _tok()); without it, token-budget logic
# inside compression/repair may silently use wrong defaults rather than
# raising, so this is set unconditionally at import time here, matching
# the known-working setup rather than assuming it's a no-op if skipped.
dagc.configure(
    tokenizer=TiktokenTokenizer("cl100k_base"),
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
)

LABEL_STORE = Path(__file__).parent / "filler_labels.jsonl"


import random

def generate_candidates_per_message(
    messages: List[Dict], scorer: FillerScorer,
    max_per_feature_bucket: int = 15,
    seed: int = 0,
) -> List[Tuple[int, str, float, float]]:
    """
    UNGATED, STRATIFIED sampling for calibration.

    filler_deletion_candidates() is threshold-gated (p_filler >= 0.75)
    -- correct for production, wrong here: it only returns clauses the
    CURRENT weights already like, so fit() never sees a labeled example
    with has_causal/has_artifact/has_directive_cue/has_enumeration/
    has_entity == True, and those weights can't be estimated. That's
    what produced the six identical -1.2237... weights.

    Fix: enumerate every clause via _clauses(), bucket by its exact
    feature signature, and sample up to max_per_feature_bucket clauses
    per bucket. This guarantees every feature is observed both True and
    False without exhaustively compress_dagc_sv-testing every clause in
    every trace.
    """
    from .filler_score import _clause_features, _mention_counts, _clause_max_surprisal

    full_text = " ".join(_get_text(m) for m in messages)
    mention_counts, total = _mention_counts(full_text)

    buckets: Dict[Tuple[bool, ...], List[Tuple[int, str]]] = {}
    for idx, m in enumerate(messages):
        if m.get('role') == 'system':
            continue
        text = _get_text(m)
        if not text:
            continue
        for clause in _clauses(text):
            feats = _clause_features(clause)
            key = tuple(feats[f] for f in FillerScorer.FEATURES)
            buckets.setdefault(key, []).append((idx, clause))

    rng = random.Random(seed)
    out = []
    for items in buckets.values():
        sample = items if len(items) <= max_per_feature_bucket else rng.sample(items, max_per_feature_bucket)
        for idx, clause in sample:
            p = scorer.score(clause)
            bits = _clause_max_surprisal(clause, mention_counts, total)
            out.append((idx, clause, p, bits))
    return out


def test_single_deletion(
    messages: List[Dict], msg_idx: int, clause: str,
    rci_floor: float = 1.0, **sv_kwargs,
) -> Tuple[bool, Dict]:
    """
    Ground truth for "was this deletion safe" must come from the ORIGINAL
    trace's decisions -- not decisions re-extracted from the already-
    stripped trial trace, which can silently lose the decision itself if
    the deleted clause WAS the decision's own text. Re-extracting after
    deletion answers a circular question ("does the mutilated trace still
    need what's left of it") instead of the real one ("did compressing the
    mutilated trace still preserve everything the ORIGINAL trace needed").
    """
    from .extraction import extract_decisions
    from .graph import compute_rci
    from .graph_ext import compute_rci_extended

    original_decisions = extract_decisions(messages)

    trial_messages = copy.deepcopy(messages)
    original_content = trial_messages[msg_idx].get('content', '') or ''
    stripped = original_content.replace(clause, '').strip()

    if stripped == original_content:
        return False, {'error': 'clause_not_found_in_message', 'msg_idx': msg_idx}

    trial_messages[msg_idx]['content'] = stripped

    compressed, _inner_report = compress_dagc_sv(trial_messages, rci_floor=rci_floor, **sv_kwargs)

    # Score the trial's compressed output against the ORIGINAL trace's
    # decisions -- the real question calibration needs answered.
    rci_result = compute_rci_extended(messages, compressed, original_decisions,
                                       include_critical_values=True)
    safe = rci_result['RCI'] is not None and rci_result['RCI'] >= rci_floor
    return safe, rci_result


def build_labels_for_trace(
    messages: List[Dict], scorer: FillerScorer,
    rci_floor: float = 1.0, **sv_kwargs,
) -> List[Tuple[str, bool]]:
    """Full loop for one trace: generate candidates, test each in
    isolation, return (clause, is_filler) labels."""
    candidates = generate_candidates_per_message(messages, scorer)
    labels = []
    for msg_idx, clause, _p, _bits in candidates:
        safe, _report = test_single_deletion(messages, msg_idx, clause, rci_floor, **sv_kwargs)
        labels.append((clause, safe))
    return labels


def append_labels(labels: List[Tuple[str, bool]]) -> None:
    """Persist labels across runs so fit() accumulates evidence over
    many traces instead of re-learning from scratch each time."""
    with LABEL_STORE.open("a", encoding="utf-8") as f:
        for clause, is_filler in labels:
            f.write(json.dumps({"clause": clause, "is_filler": is_filler}) + "\n")


def load_all_labels() -> List[Tuple[str, bool]]:
    if not LABEL_STORE.exists():
        return []
    out = []
    with LABEL_STORE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append((rec["clause"], rec["is_filler"]))
    return out

# in the __main__ block of calibrate_filler_scorer.py, before calling calibrate():
from .extraction import extract_decisions

def _filter_traces_with_decisions(traces: List[List[Dict]]) -> List[List[Dict]]:
    kept = []
    dropped = 0
    for t in traces:
        try:
            decs = extract_decisions(t)
        except Exception:
            decs = []
        if decs:
            kept.append(t)
        else:
            dropped += 1
    print(f"filtered out {dropped} zero-decision traces, {len(kept)} remain")
    return kept
def _filter_english_traces(raw_items: List[Dict]) -> List[List[Dict]]:
    """
    Filter out known-non-English traces. 'domain' means different things
    across source files: in real_world_traces.json it's a language label
    (English/Chinese/Spanish/...); in real_world_traces3.json it's a task
    type (tool_calling). Only exclude an item when its domain value is a
    recognized NON-English language label -- anything else (a task-type
    label like 'tool_calling', or a missing field) is kept, since we have
    no positive evidence it's non-English.
    """
    NON_ENGLISH_LABELS = {
        'chinese', 'spanish', 'russian', 'turkish', 'french', 'japanese', 'italian'
    }
    kept, dropped = [], 0
    for item in raw_items:
        domain = str(item.get('domain', '')).strip().lower()
        if domain in NON_ENGLISH_LABELS:
            dropped += 1
            continue
        kept.append(item['trace'])
    print(f"filtered out {dropped} non-English traces (by domain field), {len(kept)} remain")
    return kept
def calibrate(traces: List[List[Dict]], rci_floor: float = 1.0, **sv_kwargs) -> FillerScorer:
    scorer = FillerScorer()
    all_new_labels = []
    for messages in traces:
        labels = build_labels_for_trace(messages, scorer, rci_floor, **sv_kwargs)
        all_new_labels.extend(labels)

    append_labels(all_new_labels)
    full_history = load_all_labels()
    _print_calibration_diagnostics(full_history, traces)
    scorer.fit(full_history)
    return scorer


def _print_calibration_diagnostics(all_labels, trace_list):
    """Print class balance and per-feature support so a skewed or
    thin calibration set doesn't get trusted silently."""
    from .filler_score import _clause_features

    n_filler = sum(1 for _, y in all_labels if y)
    n_protected = sum(1 for _, y in all_labels if not y)
    total = n_filler + n_protected

    print(f"\n=== Calibration label diagnostics ===")
    print(f"total labeled clauses: {total}  (filler={n_filler}, protected={n_protected})")
    if total == 0:
        print("WARNING: zero labeled examples -- fit() will no-op and keep prior weights.")
        return
    ratio = n_filler / total
    print(f"filler ratio: {ratio:.1%}")
    if ratio > 0.90 or ratio < 0.10:
        print("WARNING: severe class imbalance -- prior_log_odds will dominate over "
              "feature weights. Treat resulting weights as provisional.")
    if total < 200:
        print(f"WARNING: only {total} labeled examples -- per-feature weights below "
              f"~30-50 examples per class are likely noisy (Laplace smoothing masks "
              f"this by pulling weak-count features toward 0, e.g. watch has_artifact).")

    # Per-feature support: how many filler vs protected examples actually
    # HAD each feature present -- tells you whether has_artifact's weak
    # weight is "genuinely uninformative" or "starved of examples."
    feat_filler = Counter()
    feat_protected = Counter()
    for clause, is_filler in all_labels:
        feats = _clause_features(clause)
        bucket = feat_filler if is_filler else feat_protected
        for f, present in feats.items():
            if present:
                bucket[f] += 1

    print(f"\n{'feature':<20}{'filler_n':<10}{'protected_n':<12}")
    for f in FillerScorer.FEATURES:
        print(f"{f:<20}{feat_filler[f]:<10}{feat_protected[f]:<12}")

    # Per-trace decision count -- flags traces where extract_decisions
    # returned zero decisions (non-English or otherwise), which means
    # that trace contributed NO protected (is_filler=False) labels and
    # skews calibration toward filler.
    from .extraction import extract_decisions
    print(f"\nzero-decision traces (contribute 0 protected labels):")
    zero_dec_count = 0
    for i, trace in enumerate(trace_list):
        try:
            decs = extract_decisions(trace)
        except Exception:
            decs = []
        if not decs:
            zero_dec_count += 1
    print(f"{zero_dec_count} / {len(trace_list)} traces had zero extracted decisions")
    if zero_dec_count > 0:
        print("WARNING: these traces contribute filler-only labels, biasing the "
              "prior toward deletion. Check for non-English content overlap.")
from pathlib import Path
from typing import Dict, List, Union

TRACES_PATH_DEFAULT = [
    Path("/Users/shaurya/python/DDR/real_world_traces.json"),
    Path("/Users/shaurya/python/DDR/real_world_traces3.json"),
]


def load_traces(
    traces_path: Union[Path, List[Path]] = TRACES_PATH_DEFAULT
) -> Dict[str, List[Dict]]:
    """
    Loads JSON file(s) containing lists of {"name": ..., "trace": [...]} entries.
    Accepts either a single Path or a list of Path objects.
    Returns {name: messages_list}, skipping any entry with no trace.
    """
    paths = [traces_path] if isinstance(traces_path, Path) else traces_path
    traces_dict = {}

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if item.get("trace"):
                    traces_dict[item["name"]] = item["trace"]
        except FileNotFoundError:
            print(f"[warn] File not found, skipping: {path}")

    return traces_dict



if __name__ == "__main__":
    import sys

    # Usage: python -m dagc.calibrate_filler_scorer [traces_json_path] [trace_name ...]
    #   no args              -> load ALL traces from the default path(s), English-filtered
    #   one path arg         -> load ALL traces from that path, English-filtered
    #   path + trace names   -> load only those named traces (no language filter
    #                           applied to an explicit by-name selection)
    args = sys.argv[1:]
    if args and args[0].endswith(".json"):
        traces_paths = [Path(args[0])]
        selected_names = args[1:]
    else:
        traces_paths = TRACES_PATH_DEFAULT if isinstance(TRACES_PATH_DEFAULT, list) else [TRACES_PATH_DEFAULT]
        selected_names = args

    # Load raw JSON directly (not via load_traces()) so 'domain' survives --
    # load_traces() discards everything except name/trace.
    raw_items = []
    for p in traces_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw_items.extend(json.load(f))
        except FileNotFoundError:
            print(f"[warn] File not found, skipping: {p}")

    if selected_names:
        by_name = {item["name"]: item for item in raw_items if item.get("trace")}
        missing = [n for n in selected_names if n not in by_name]
        if missing:
            print(f"WARNING: not found: {missing}")
        traces = [by_name[n]["trace"] for n in selected_names if n in by_name]
    else:
        traces = _filter_english_traces(raw_items)

    print(f"Calibrating on {len(traces)} trace(s) from {traces_paths}")
    calibrated = calibrate(traces)
    print("Calibrated weights:", calibrated.weights)
    print("Calibrated prior log-odds:", calibrated.prior_log_odds)
    with open(Path(__file__).parent / "filler_scorer_calibrated.json", "w") as f:
        f.write(calibrated.to_json())
    print("saved calibrated weights to filler_scorer_calibrated.json")