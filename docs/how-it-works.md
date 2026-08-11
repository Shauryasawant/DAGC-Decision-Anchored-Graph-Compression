# How Compression Works

The README summarizes this as four steps. This page walks through what each step
actually does and why it's ordered this way.

```
Trace
  |
  v
1. Extraction        --  find decision-critical artifacts
  |
  v
2. Dependency Graph   --  link each decision to its supporting evidence
  |
  v
3. Budget Allocation  --  protect critical regions, cut redundant context
  |
  v
4. Validate & Restore --  catch and repair any accidentally-dropped evidence
  |
Compressed trace
```

## 1. Extraction

DAGC scans the trace for **decision-critical artifacts** — the things a downstream
agent or reader would need to resume the task correctly:

- Tool call names and arguments (`tool_call` / `tool_calls` payloads)
- Identifiers (job IDs, PR numbers, hashes)
- File paths
- Configuration values
- Metrics and numeric targets
- Confirmed choices ("we'll go with X")

This is artifact-level extraction, not topic-level summarization — it doesn't ask
"what is this message about," it asks "what specific values does a later step
depend on."

## 2. Dependency Graph

Each extracted artifact is linked back to the message(s) that provide the evidence
for it. This produces a directed graph over the trace: decisions as nodes, supporting
context as edges. A message that only restates something already captured elsewhere
in the graph has low marginal value and becomes a strong candidate for removal — a
message that's the *sole* source of a decision artifact becomes protected.

## 3. Budget Allocation

Given `target_reduction` (or `TARGET_REDUCTION` in `DAGCConfig`), DAGC allocates the
token budget by:

- Hard-protecting decision-critical regions found in step 2
- Hard-protecting the most recent `KEEP_LAST_K` messages
- Removing redundant/low-marginal-value context first
- Optionally shortening (not deleting) protected content if `COMPRESS_PROTECTED=True`

This is why `target_reduction` is a *soft* target: decision-critical evidence is
protected regardless of how aggressive the requested reduction is, so at very high
reduction settings the actual result can fall short of the request. See
[Tuning](./tuning.md) for the full config reference.

## 4. Validate & Restore

Before returning, DAGC checks the compressed trace against the artifacts found in
step 1. Anything that should have survived but didn't gets restored — this is the
step that keeps the "decision-critical" guarantee from being just a best-effort
heuristic with no check at the end.

```python
from dagc import compress

compressed = compress(messages, target_reduction=0.85)
# Each returned message carries `_orig_idx`, its position in the original trace —
# useful for tracing a compressed message back to source, or for building your
# own "what survived" diagnostics.
for m in compressed:
    print(m["_orig_idx"], m["role"])
```

## A note on determinism

DAGC's selection strategy is heuristic, not learned — same input and config
produce the same output every time, with no model in the loop unless you've
explicitly configured an embedding or evaluation adapter (see
[Architecture](./architecture.md)). This makes it straightforward to unit-test
against a fixed trace, but it also means quality is bounded by how well the
heuristics generalize to your traces — the README's advice to evaluate on your
own data before deploying is worth taking seriously. See [Evaluation](./evaluation.md).
