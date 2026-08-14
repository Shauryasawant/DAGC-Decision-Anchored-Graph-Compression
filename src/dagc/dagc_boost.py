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
   drop). This pre-trims such arrays to the items that are structurally
   anomalous relative to the rest (different key-value pattern, e.g. a
   uniquely high 'score' or a differently-shaped snippet) -- the same
   kind of outlier signal headroom's SmartCrusher uses -- so DAGC's
   downstream logic only ever has to decide whether to keep a small,
   pre-curated candidate set rather than judge a 20-item undifferentiated
   blob as a single unit.

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
    entirely if the caller isn't asking for meaningful compression."""
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


def boosted_compress(messages, target_reduction=0.7, **kwargs):
    """Preprocess, then call DAGC's own compress() unmodified. Both
    preprocessing steps are gated on target_reduction and back off (return
    input unchanged) below 0.3, matching DAGC's own low-target behavior."""
    from dagc import compress
    pre = collapse_repeated_tool_calls(messages, target_reduction=target_reduction)
    pre = trim_json_array_content(pre, target_reduction=target_reduction)
    return compress(pre, target_reduction=target_reduction, **kwargs)
