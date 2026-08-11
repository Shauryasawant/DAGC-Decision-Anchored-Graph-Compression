# Decision Rationale

Extraction (see [How Compression Works](./how-it-works.md)) captures *what* was
decided. Rationale extraction goes one step further and captures *why* — including
alternatives that were considered and explicitly ruled out. This matters whenever
a downstream agent might otherwise re-propose an option that was already tried
and rejected.

## Extracting rationale

```python
from dagc import extract_decisions, extract_rationale_candidates

decisions = extract_decisions(messages)
candidates = extract_rationale_candidates(
    messages, decisions, include_same_message=True
)

for c in candidates:
    print(c["alternative"])       # the option that was ruled out
    print(c["reason"])            # the clause explaining why
    print(c["confidence"])        # 'high' (explicit causal marker) or 'medium'
    print(c["decision_msg_idx"])  # which final decision this explains
```

`confidence` distinguishes two extraction styles:

- **`high`** — an explicit causal marker in the text ("X failed because...",
  "switched to Y after Z broke")
- **`medium`** — a weaker inferential link between a candidate alternative and
  the reason given

Filter on `confidence` if you only want the rationale you're most sure about
surfaced to a downstream reader.

## Making rationale survive compression

Extraction alone doesn't protect the rationale from being cut — the message it
lives in is still subject to the same budget allocation as everything else (see
[Tuning](./tuning.md)). To make sure it survives, inject it as rationale stubs
*after* compression:

```python
from dagc import inject_rationale_stubs, inject_dropped_rationale_stubs

compressed, report = inject_rationale_stubs(compressed, messages, decisions)
print(report["rationale_stubs_added"])
```

This adds short stubs carrying the rationale into the already-compressed trace,
rather than trying to protect the full original message (which would fight the
budget allocator for space it may not need).

### Recovering rationale from dropped messages

Some rationale lives entirely in a message that gets dropped, not just shortened.
`inject_dropped_rationale_stubs` recovers rationale specifically from messages
that didn't survive compression at all:

```python
compressed, dropped_report = inject_dropped_rationale_stubs(compressed, messages)
print(dropped_report["dropped_rationale_stubs_added"])
```

## Inspecting what got added

Both injection functions return a report alongside the compressed trace. Beyond
the summary counts above, each report includes a detail list:

```python
for detail in report["rationale_added_detail"]:
    print(detail)

for detail in dropped_report["dropped_rationale_added_detail"]:
    print(detail)
```

Use this when you need to audit exactly which stubs were added and why — useful
during [Evaluation](./evaluation.md) if DRR looks lower than expected and you
suspect a rationale-heavy trace is the cause.

## When to use this

Rationale extraction has a cost — it's an extra pass over the trace, and stub
injection adds tokens back after compression already ran. It's worth it when your
downstream task involves an agent that might otherwise retry a ruled-out option;
skip it for traces where only the final decision matters and the path there doesn't.
