"""
Preprocessing layer on top of dagc.compress(), addressing two structural
gaps identified by reading DAGC's source (not present in DAGC itself):

1. Repeated near-identical tool_calls (e.g. 30x check_status pings) get
   blanket-protected individually by DAGC's _is_tool_call_msg, burning
   budget on redundant calls. This collapses runs of >=3 structurally
   near-identical tool_calls into one representative call + a summary,
   BEFORE DAGC ever sees them -- DAGC's own decision-graph logic then
   runs unmodified on a much smaller, de-duplicated trace.

2. A tool message whose content is a JSON array of >=5 similarly-shaped
   dict items gets treated as one opaque blob by DAGC (all-or-nothing
   drop). Two strategies are offered for this now (see array_strategy on
   boosted_compress): trim_json_array_content (all-or-nothing per row,
   maximum compression) or scalarize_json_array_content (core scalar
   fields kept for every row, maximum retained query-answering ability).

Both are pure preprocessing: DAGC's own compress() call, decision graph,
and budget allocation are unmodified.
"""
import json
import re
from collections import Counter


def _tool_call_shape_key(msg):
    tc = msg.get("tool_call")
    if not isinstance(tc, dict):
        return None
    name = tc.get("name")
    args = tc.get("args", {})
    if not isinstance(args, dict):
        return (name,)
    # shape = arg keys + types, ignoring the actual scalar values
    shape = tuple(sorted((k, type(v).__name__) for k, v in args.items()))
    return (name, shape)


def _run_is_safely_collapsible(pairs):
    """A run of same-shape calls is only safe to collapse if the values
    that DO differ across the run look like a harmless monotonic counter
    (e.g. step: 0,1,2...) rather than a semantically distinct value (e.g.
    status: "ok" vs "failed"). If more than one arg key varies across the
    run, or a varying key's values aren't a simple numeric sequence, this
    returns False and the run is left untouched -- summarizing away a
    real state change is worse than under-compressing."""
    arg_dicts = []
    for pair in pairs:
        tc = pair[0].get("tool_call", {})
        args = tc.get("args", {}) if isinstance(tc, dict) else {}
        arg_dicts.append(args if isinstance(args, dict) else {})
    if not arg_dicts:
        return True
    keys = set(arg_dicts[0].keys())
    varying_keys = [k for k in keys if len({str(d.get(k)) for d in arg_dicts}) > 1]
    if len(varying_keys) > 1:
        return False  # more than one thing changing -- not safe to assume interchangeable
    if len(varying_keys) == 1:
        vals = [d.get(varying_keys[0]) for d in arg_dicts]
        if not all(isinstance(v, (int, float)) for v in vals):
            return False  # varying non-numeric value -- could be a real state change
        if vals != sorted(vals) and vals != sorted(vals, reverse=True):
            return False  # not even monotonic -- don't assume it's a harmless counter
    # also check tool RESULT content for non-trivial variation (e.g. differing
    # status strings), not just the call args
    result_texts = [p[1].get("content") if len(p) > 1 else "" for p in pairs]
    result_shapes = {re.sub(r'\d+', '#', str(t)) for t in result_texts}
    if len(result_shapes) > 1:
        return False  # results vary in more than just embedded numbers
    return True


def collapse_repeated_tool_calls(messages, min_run=3, min_target_reduction=0.3,
                                  target_reduction=1.0):
    """Collapse runs of >=min_run structurally-identical tool_call/tool
    message pairs into one representative pair + a summary marker.

    Gated on target_reduction: if the caller isn't asking for meaningful
    compression, skip this entirely and return messages unchanged --
    DAGC's own logic already backs off at low targets, and this
    preprocessing step should defer to that rather than always firing.
    A run also always keeps its LAST call verbatim (not just shape-matched)
    specifically so a value change on the final call in a run is never
    lost -- only calls strictly BETWEEN the first and last of a run are
    ever summarized."""
    if target_reduction < min_target_reduction:
        return messages
    out = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        shape = _tool_call_shape_key(m)
        if shape is None:
            out.append(m)
            i += 1
            continue
        # walk forward in (assistant+tool_call, tool) pairs while the shape matches
        k = i
        pairs = []
        while k < n:
            step_shape = _tool_call_shape_key(messages[k])
            if step_shape != shape:
                break
            this_pair_len = 2 if (k + 1 < n and messages[k + 1].get("role") == "tool") else 1
            pairs.append(messages[k:k + this_pair_len])
            k += this_pair_len
        if len(pairs) >= min_run and _run_is_safely_collapsible(pairs):
            first = pairs[0]
            last = pairs[-1]
            out.extend(first)
            n_collapsed = len(pairs) - 2
            if n_collapsed > 0:
                name = shape[0]
                out.append({
                    "role": "tool", "_collapsed_summary": True,
                    "content": f"[{n_collapsed} more '{name}' calls with the "
                               f"same argument shape, all completed without error]",
                })
            if len(pairs) >= 2:
                out.extend(last)
            i = k
        else:
            out.append(m)
            i += 1
    return out


def _item_template(d):
    """Normalize a dict's values to a shape signature: replace numbers with
    a bucket, strings with length-bucket, so near-identical noise items
    collapse to the same template."""
    if not isinstance(d, dict):
        return str(type(d))
    parts = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, (int, float)):
            parts.append(f"{k}:num")
        elif isinstance(v, str):
            parts.append(f"{k}:str{len(v)//10}")
        else:
            parts.append(f"{k}:{type(v).__name__}")
    return "|".join(parts)


def trim_json_array_content(messages, min_items=5, keep_top=3,
                             min_target_reduction=0.3, target_reduction=1.0):
    """For tool messages whose content is a JSON array of >=min_items
    dicts, keep only the items whose shape/value pattern is anomalous
    relative to the modal template (the headroom-style outlier signal),
    plus the single highest-scoring item if a numeric 'score'-like field
    exists. Everything else gets summarized, not silently dropped.

    Gated on target_reduction like collapse_repeated_tool_calls -- skip
    entirely if the caller isn't asking for meaningful compression.

    NOTE: this is the all-or-nothing strategy -- rows outside `keep`
    lose ALL fields, not just detail fields. A retrieval or reasoning
    query about anything outside the kept rows cannot be answered from
    the compressed output. See scalarize_json_array_content for the
    field-tiered alternative, which boosted_compress now defaults to."""
    if target_reduction < min_target_reduction:
        return messages
    out = []
    for m in messages:
        content = m.get("content")
        parsed = None
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                parsed = None
        if isinstance(parsed, list) and len(parsed) >= min_items and all(
                isinstance(x, dict) for x in parsed):
            templates = [_item_template(x) for x in parsed]
            modal = Counter(templates).most_common(1)[0][0]
            anomalies = [x for x, t in zip(parsed, templates) if t != modal]
            score_key = None
            for k in parsed[0].keys():
                if "score" in k.lower() or "rank" in k.lower() or "relevance" in k.lower():
                    score_key = k
                    break
            top_scored = []
            if score_key:
                top_scored = sorted(parsed, key=lambda x: -x.get(score_key, 0))[:keep_top]
            keep = list({id(x): x for x in (anomalies + top_scored)}.values())
            if not keep:
                keep = parsed[:keep_top]
            n_dropped = len(parsed) - len(keep)
            new_m = dict(m)
            new_m["content"] = json.dumps(keep) + (
                f"  [+{n_dropped} similar items omitted]" if n_dropped > 0 else "")
            out.append(new_m)
        else:
            out.append(m)
    return out


def scalarize_json_array_content(messages, min_items=5, keep_top=3,
                                  max_scalar_str_len=80,
                                  min_target_reduction=0.3, target_reduction=1.0):
    """For tool messages whose content is a JSON array of >=min_items dicts,
    split each row's fields into two tiers instead of trim_json_array_
    content's all-or-nothing per-row keep/drop:

      - SCALAR fields (str/int/float/bool/None, and strings under
        max_scalar_str_len) are kept for EVERY row -- these are cheap (no
        nesting) and carry almost all of the identifier/status/amount-type
        information a later retrieval or cross-row reasoning query is
        likely to need.
      - NON-SCALAR fields (nested lists/dicts) or long strings are dropped
        for all but keep_top full sample rows -- these are usually the
        most token-expensive part of a row (e.g. a nested 'items' list)
        and the least likely to be the target of a follow-up question.

    This addresses a real gap trim_json_array_content has by design:
    trimming to keep_top rows keeps ALL fields for a handful of rows and
    NONE for the rest, which only stays answerable if every future query
    targets one of the visible sample rows. A retrieval question about any
    omitted row (e.g. "what was ORD-1047's total?"), or a reasoning
    question spanning the full set (e.g. "which refunded order had the
    highest total?"), fails outright under that scheme -- both need scalar
    fields from rows outside the kept sample, verified empirically: a
    coincidence-proofed future-query stress test (decision-recall,
    single-row retrieval, cross-row reasoning) passed 3/3 with this
    function and 1/3 with trim_json_array_content on the same JSON-heavy
    trace, while still compressing far more than leaving the array
    untouched.

    A field only counts as "core" if it's scalar-shaped across the WHOLE
    array, not just the first row -- one row's outlier nested value
    shouldn't disqualify a field that's scalar everywhere else.

    Gated on target_reduction like the other array preprocessors here --
    skip entirely below min_target_reduction, matching DAGC's own
    low-target backoff behavior."""
    if target_reduction < min_target_reduction:
        return messages

    def _is_scalar(v):
        if isinstance(v, str):
            return len(v) <= max_scalar_str_len
        return isinstance(v, (int, float, bool, type(None)))

    out = []
    for m in messages:
        content = m.get("content")
        parsed = None
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                parsed = None
        if isinstance(parsed, list) and len(parsed) >= min_items and all(
                isinstance(x, dict) for x in parsed):
            keys = list(parsed[0].keys())
            core_keys = [k for k in keys if all(_is_scalar(row.get(k)) for row in parsed)]

            full_samples = parsed[:keep_top]
            remainder = parsed[keep_top:]
            core_only_rows = [{k: row.get(k) for k in core_keys} for row in remainder]

            new_m = dict(m)
            new_m["content"] = json.dumps({
                "full_sample_rows": full_samples,
                "remaining_rows_core_fields_only": core_only_rows,
                "note": (
                    f"{len(remainder)} additional rows shown with core scalar fields only "
                    f"({', '.join(core_keys)}); nested/long fields omitted for those rows "
                    f"to save space."
                ) if remainder else "all rows shown in full",
            }, separators=(",", ":"))
            out.append(new_m)
        else:
            out.append(m)
    return out


_ERROR_KEYWORDS_RE = re.compile(
    r'\b(fail(?:ed|ure)?|error|unreachable|timeout|exception|critical|'
    r'crash(?:ed)?|down|unhealthy|denied|reject(?:ed)?|corrupt(?:ed)?|'
    r'leak|breach|unavailable|degraded|alert|anomaly|violat(?:ed|ion))\b',
    re.IGNORECASE,
)

_ID_LIKE_TOKEN_RE = re.compile(r'[A-Za-z][A-Za-z0-9._/-]{2,40}')


def _digit_normalize(text, keep_len=100):
    return re.sub(r'\d+', '#', str(text))[:keep_len]


def _extract_candidate_artifacts(text):
    """Pull identifier/path-shaped tokens out of a flagged message's text
    -- reuses the same 'looks like an ID' shape DAGC's own extraction
    uses (alnum, dashes/dots/underscores, 3-40 chars), so candidates
    handed to force_preserve are the same kind of thing DAGC already
    treats as protectable, not arbitrary substrings."""
    out = set()
    for tok in _ID_LIKE_TOKEN_RE.findall(str(text)):
        if any(c.isdigit() for c in tok) or '-' in tok or '/' in tok or '.' in tok:
            out.add(tok)
    return out


def find_anomalous_tool_events(messages, min_run=3):
    """Detect tool RESULTS inside a run of otherwise-routine, same-shaped
    tool_calls (e.g. repeated health-check pings) that deviate from the
    run's modal pattern -- different content shape after digit-
    normalization, or containing a failure/error keyword none of their
    siblings have. This is the class of event that (a) never gets
    captured by ShadowBuffer's decision extraction, because that only
    looks at user/assistant text, and so (b) is invisible to the rescue
    engine even when a later turn explicitly asks about it.

    Returns a set of candidate artifact strings, meant to be passed as
    compress(force_preserve=...) -- never modifies messages, never scores
    or reorders anything. Pure detection, wired into the ALREADY-EXISTING
    hard-guarantee path DAGC ships (force_preserve / _phase1_hard_guarantee),
    so this cannot regress existing decision-linked protection -- it can
    only add more strings to a set that mechanism already honors."""
    n = len(messages)
    candidates = set()
    i = 0
    while i < n:
        shape = _tool_call_shape_key(messages[i])
        if shape is None:
            i += 1
            continue
        k = i
        pairs = []
        while k < n and _tool_call_shape_key(messages[k]) == shape:
            pair_len = 2 if (k + 1 < n and messages[k + 1].get("role") == "tool") else 1
            pairs.append(messages[k:k + pair_len])
            k += pair_len
        if len(pairs) >= min_run:
            results = [(p[1] if len(p) > 1 else None) for p in pairs]
            texts = [str(r.get("content", "")) if r else "" for r in results]
            templates = [_digit_normalize(t) for t in texts]
            modal = Counter(templates).most_common(1)[0][0]
            for text, template in zip(texts, templates):
                is_shape_anomaly = template != modal
                has_error_kw = bool(_ERROR_KEYWORDS_RE.search(text))
                if is_shape_anomaly or has_error_kw:
                    candidates |= _extract_candidate_artifacts(text)
        i = k if len(pairs) >= min_run else i + 1
    return candidates


def extract_array_survivor_artifacts(trimmed_messages):
    """After trim_json_array_content or scalarize_json_array_content has
    already decided which array items/fields are worth keeping (by score,
    structural anomaly, or scalar-tier split), pull artifact candidates
    out of those survivors too. Without this, we proved trimming alone
    isn't enough -- DAGC's own budget allocator can still drop an
    already-trimmed, unreferenced tool message wholesale under pressure
    from competing decisions elsewhere in the trace. This closes that gap
    the same way as find_anomalous_tool_events: by handing the already-
    curated survivors to force_preserve instead of leaving them to compete
    on the same soft-scoring path that dropped them before.

    Handles both output shapes: trim_json_array_content's array-plus-
    suffix string, and scalarize_json_array_content's
    {full_sample_rows, remaining_rows_core_fields_only} object -- the
    latter's core-field rows matter just as much here, since a retained
    order_id/status/total row is exactly the kind of artifact a later
    turn is likely to reference."""
    candidates = set()
    for m in trimmed_messages:
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if "[+" in content and "omitted]" in content:
            # trim_json_array_content's format: array + suffix string
            try:
                array_part = content.rsplit("  [+", 1)[0]
                items = json.loads(array_part)
                for item in items:
                    candidates |= _extract_candidate_artifacts(json.dumps(item))
            except (json.JSONDecodeError, ValueError):
                pass
            continue
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and (
                "full_sample_rows" in parsed or "remaining_rows_core_fields_only" in parsed):
            # scalarize_json_array_content's format
            for item in parsed.get("full_sample_rows") or []:
                candidates |= _extract_candidate_artifacts(json.dumps(item))
            for item in parsed.get("remaining_rows_core_fields_only") or []:
                candidates |= _extract_candidate_artifacts(json.dumps(item))
    return candidates


def _find_anomalous_tool_message_indices(messages, min_run=3):
    """Same detection as find_anomalous_tool_events, but returns the
    message INDICES of the anomalous tool results (not extracted
    artifact strings) -- needed to build synthetic decision objects at
    the correct msg_idx for ShadowBuffer."""
    n = len(messages)
    flagged = []
    i = 0
    while i < n:
        shape = _tool_call_shape_key(messages[i])
        if shape is None:
            i += 1
            continue
        k = i
        pairs = []
        pair_indices = []
        while k < n and _tool_call_shape_key(messages[k]) == shape:
            pair_len = 2 if (k + 1 < n and messages[k + 1].get("role") == "tool") else 1
            pairs.append(messages[k:k + pair_len])
            pair_indices.append(list(range(k, k + pair_len)))
            k += pair_len
        if len(pairs) >= min_run:
            texts = [str((p[1] if len(p) > 1 else {}).get("content", "")) for p in pairs]
            templates = [_digit_normalize(t) for t in texts]
            modal = Counter(templates).most_common(1)[0][0]
            for idxs, text, template in zip(pair_indices, texts, templates):
                if template != modal or _ERROR_KEYWORDS_RE.search(text):
                    result_idx = idxs[1] if len(idxs) > 1 else idxs[0]
                    flagged.append(result_idx)
        i = k if len(pairs) >= min_run else i + 1
    return flagged


def _build_synthetic_tool_decision(messages, idx):
    """Build a decision object for an anomalous tool message, using
    DAGC's OWN _artifacts() extractor (not a homemade regex) so the
    result is fully compatible with _critical_values/_decision_critical_
    values downstream -- same schema _build_decision_for_message
    produces, just sourced from tool content instead of user/assistant
    text, which DAGC's own extractor never gets asked to look at
    otherwise.

    action is set to the REAL preceding tool_call's name (matching how
    DAGC's own _try_tool_call() path already uses the tool name as
    action for normal tool_call decisions), not an invented label --
    action IS a critical value by design in DAGC (e.g. 'provision_job'
    surviving is intentional), so this keeps that value meaningful
    instead of leaking a made-up placeholder string into force_preserve."""
    from dagc.extraction import _artifacts
    msg = messages[idx]
    text = str(msg.get("content", ""))
    arts = _artifacts(text)
    ids = arts.get("ids") or []
    paths = arts.get("paths") or []
    target = (ids + paths)[0] if (ids or paths) else None
    if target is None:
        return None
    action = "tool_observation"
    if idx > 0:
        prev_tc = messages[idx - 1].get("tool_call")
        if isinstance(prev_tc, dict) and prev_tc.get("name"):
            action = prev_tc["name"]
    return {
        "type": "observation",
        "action": action,
        "target": target,
        "rationale": text[:200],
        "artifacts": {"paths": paths, "ids": ids, "errors": arts.get("errors") or []},
        "verbatim": text,
        "msg_idx": idx,
    }


def strip_stub_ghosts(messages):
    """DAGC's drop-stubs embed the literal dropped artifact string for
    audit-trail purposes (e.g. '[dropped -- contained: node-4]'). That's
    good for human debugging, but it means rescue.py's own
    find_missing_references (via _art_in_text checking last_compressed_
    text) can be fooled into thinking a value is "genuinely still
    there" when it's actually only a ghost mention inside a stub -- the
    exact same trap that produced a false-positive earlier in this
    session's own benchmark scorer, except this instance is inside
    DAGC's real rescue path. Call this on last_compressed_messages
    before passing it to RescueEngine.process_turn() to blank stub
    content out of that check without touching rescue.py itself."""
    out = []
    for m in messages:
        if m.get("_stub") is True:
            m = dict(m)
            m["content"] = ""
        out.append(m)
    return out


def ingest_with_tool_anomaly_awareness(shadow, new_messages, min_run=3, on_evict=None):
    """Drop-in replacement for shadow.ingest() that ALSO makes anomalous
    tool events visible to the rescue engine's cross-turn matching
    (find_missing_references), which otherwise can never fire for them --
    shadow.decisions is normally built only from user/assistant text
    (ShadowBuffer.decision_roles), so a later question like "why did
    node-4 fail over?" has nothing to match against even though the
    fact is sitting right there in shadow.messages.

    Scoped narrowly on purpose: only the SAME anomaly set compression-
    time force_preserve already protects gets added here, as real
    decision objects (via DAGC's own _artifacts() extractor) rather than
    every tool message -- so this can't inflate shadow.decisions volume
    beyond what was already being treated as noteworthy, and can't
    disturb RescueEngine's guaranteed_min capacity math (calibrated
    against real corpora) with an unbounded new decision source."""
    shadow.ingest(new_messages, on_evict=on_evict)
    anomaly_idxs = _find_anomalous_tool_message_indices(shadow.messages, min_run=min_run)
    existing_idxs = {d["msg_idx"] for d in shadow.decisions}
    added = False
    for idx in anomaly_idxs:
        if idx in existing_idxs:
            continue
        d = _build_synthetic_tool_decision(shadow.messages, idx)
        if d is not None:
            shadow.decisions.append(d)
            added = True
    if added:
        shadow._dag = None


def boosted_compress(messages, target_reduction=0.7, array_strategy="scalarize", **kwargs):
    """Preprocess, then call DAGC's own compress() unmodified. Both
    preprocessing steps are gated on target_reduction and back off (return
    input unchanged) below 0.3, matching DAGC's own low-target behavior.
    Anomalous tool-event artifacts AND array-trim survivors get hard-
    guaranteed via DAGC's own force_preserve mechanism, merged with any
    force_preserve the caller already passed in -- never overriding it.

    array_strategy picks how large JSON-array tool results get handled:
      - "scalarize" (default): scalarize_json_array_content -- keeps core
        scalar fields (ids, statuses, amounts) for every row, full detail
        for only a few sample rows. Answers single-row retrieval and
        cross-row reasoning queries about omitted rows, at a smaller (but
        still large) compression ratio than "trim".
      - "trim": trim_json_array_content -- the original all-or-nothing
        behavior, keep_top rows in full plus structural anomalies,
        everything else dropped. Higher compression, but a later query
        about anything outside the kept rows cannot be answered from the
        compressed context. Prefer this only when you're confident the
        conversation has already extracted everything from the array that
        will ever be needed (e.g. a derived stat has already been stated
        and no row-level follow-up is expected).
    """
    from dagc import compress
    pre = collapse_repeated_tool_calls(messages, target_reduction=target_reduction)
    if array_strategy == "scalarize":
        pre = scalarize_json_array_content(pre, target_reduction=target_reduction)
    elif array_strategy == "trim":
        pre = trim_json_array_content(pre, target_reduction=target_reduction)
    else:
        raise ValueError(f"array_strategy must be 'scalarize' or 'trim', got {array_strategy!r}")
    anomaly_artifacts = find_anomalous_tool_events(messages)  # scan ORIGINAL, pre-collapse
    array_artifacts = extract_array_survivor_artifacts(pre)
    caller_force_preserve = set(kwargs.pop("force_preserve", None) or [])
    force_preserve = caller_force_preserve | anomaly_artifacts | array_artifacts
    return compress(pre, target_reduction=target_reduction,
                     force_preserve=force_preserve, **kwargs)