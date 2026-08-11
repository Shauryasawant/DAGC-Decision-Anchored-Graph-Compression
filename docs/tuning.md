# Tuning

`DAGCConfig` controls the tradeoff between how much a trace shrinks and how much
gets hard-protected regardless of budget. This page goes field-by-field.

```python
from dagc import DAGCConfig, compress

cfg = DAGCConfig(
    TARGET_REDUCTION=0.90,
    KEEP_LAST_K=3,
)

compressed = compress(messages, cfg=cfg)

# Equivalent inline overrides, no separate config object needed:
compressed = compress(
    messages,
    TARGET_REDUCTION=0.90,
    KEEP_LAST_K=3,
)
```

## `TARGET_REDUCTION` (default `0.87`)

The fraction of tokens you're asking DAGC to remove. Treat this as a request, not
a guarantee — decision-critical evidence (from the extraction step in
[How Compression Works](./how-it-works.md)) is protected first, budget cuts come
second.

- **Push higher (0.90+)** when your traces have a lot of redundant back-and-forth
  relative to how much decision content they carry — long debugging sessions,
  chatty confirmations, repeated status checks.
- **Pull lower (0.5–0.7)** when the trace is already dense with decisions relative
  to its length — you'll get less shrinkage, but you're not fighting the protector
  for every token.
- At very high settings on decision-dense traces, don't be surprised if actual
  reduction plateaus below the target. That's the protector doing its job, not a
  bug — verify with [Evaluation](./evaluation.md) rather than assuming the number
  you requested is the number you got.

## `KEEP_LAST_K` (default `1`)

Hard-protects the most recent `K` messages regardless of what the dependency graph
says about them. Useful because the most recent turn is often the one a downstream
step will react to next, even if it doesn't yet contain a "decision" by DAGC's
extraction criteria.

- Raise this for traces where the tail end matters for continuity (an agent about
  to take its next action) even before that action produces an extractable decision.
- Keep it low (or at default) when you want the protector logic, not recency, to
  drive what survives.

## `PROTECT_TOOL_CALLS` (default `True`)

Preserves messages carrying `tool_call` / `tool_calls` payloads during selection.
Turning this off means tool calls compete for budget on the same terms as regular
conversational messages — only worth doing if you've verified (via
[Evaluation](./evaluation.md)) that your downstream task doesn't actually need
tool-call fidelity.

## `PROTECT_JUDGMENTS` (default `True`)

Preserves assistant judgments and confirmations — the "we'll go with X" style
messages that close out a decision. This is what keeps a *conclusion* in the
trace even if the *discussion* leading to it gets compressed away.

## `USE_CAUSAL_SKELETON` (default `True`)

Uses the causal dependency graph (step 2 in [How Compression Works](./how-it-works.md))
during selection rather than a flatter relevance-only heuristic. Disabling this
falls back to a simpler selection strategy — mainly useful for isolating whether
the causal graph is what's driving a particular result, when debugging DAGC itself
rather than tuning it for production use.

## `COMPRESS_PROTECTED` (default `True`)

Protected messages (from the artifact extraction and `PROTECT_*` flags above) can
still have their *content* shortened even though the message itself isn't dropped.
Set this to `False` if you need protected messages to survive byte-for-byte —
useful when downstream code parses specific formatting out of a message and
shortening would break that parse, at the cost of lower overall reduction.

## Recommended workflow

Don't tune blind. The loop that actually works:

1. Run `compress()` with defaults on a representative sample of your real traces.
2. Run `dagc_eval.compute_drr()` on the result (see [Evaluation](./evaluation.md)).
3. Adjust one field at a time — `TARGET_REDUCTION` first, since it has the largest
   effect — and re-measure DRR/RCI, not just the reduction percentage.
4. Only reach for the `PROTECT_*` / `USE_CAUSAL_SKELETON` flags if DRR shows a
   specific artifact class getting lost that the defaults should have caught.
