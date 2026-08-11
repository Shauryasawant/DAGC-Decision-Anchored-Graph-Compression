# Evaluation

`dagc_eval` measures whether compression actually preserved what it was supposed
to. Don't trust `TARGET_REDUCTION` numbers alone — measure the result.

## Metrics

| Metric | What it answers |
|---|---|
| **DRR** (Decision Reproducibility Rate) | Can the original operational decisions still be reproduced from the compressed trace? |
| **RCI** (Reasoning Chain Integrity) | Does the reasoning chain leading to each decision still hold together? |
| **`art_ret`** | Artifact retention — fraction of all recoverable artifacts that survived |
| **`decision_art_ret`** | Artifact retention specifically for artifacts tied to a decision — the strongest single signal that decision-critical info survived intact |

If you only look at one number when tuning, make it `decision_art_ret` — it's the
metric most directly tied to the thing DAGC is designed to protect (see
[How Compression Works](./how-it-works.md)).

## Running it

```python
from dagc_eval import TASKS, compute_drr, generate_trace

trace = generate_trace(TASKS[0])
result = compute_drr(trace)

print(result["DRR_soft"])
print(result["RCI"])
print(result["art_ret"])
print(result["decision_art_ret"])
```

`TASKS` ships a set of synthetic evaluation tasks so you can sanity-check the
library without bringing your own data first. Once you trust the workflow, swap
`generate_trace(TASKS[0])` for your own real trace — synthetic tasks tell you
DAGC works in general, not that it works on *your* traces.

## Deterministic by default

The default evaluation pipeline is offline and deterministic — no LLM calls, same
input always produces the same score. This is what makes it safe to run in CI
against a fixed set of traces without worrying about model drift or rate limits.

## Reconstruction-based evaluation (optional)

For a stronger signal, supply your own LLM client and let it try to actually
answer a downstream question using only the compressed trace, rather than relying
on the deterministic heuristic alone:

```python
import openai
from dagc_eval import compute_drr
from dagc_eval.interfaces import OpenAIChatClient

llm = OpenAIChatClient(openai.OpenAI())

result = compute_drr(trace, llm=llm)
```

This is opt-in and the only place in the evaluation path that makes a network
call — see [Architecture](./architecture.md) for how this fits the package's
offline-by-default design.

## CLI

The same workflows are available without writing Python:

```bash
dagc compress trace.json --target-reduction 0.85 -o compressed.json
dagc evaluate trace.json -o report.md
dagc benchmark --n-traces 3 -o benchmark.json
```

`dagc evaluate` runs the same `compute_drr` pipeline and writes a readable report.
`dagc benchmark` runs it across multiple generated traces at once — a fast way to
get a first read on default-config behavior before you've assembled your own
evaluation set.

## Reading a result

A high `DRR_soft` with a low `decision_art_ret` is worth investigating — it can
mean decisions are *technically* reconstructible but only barely, with supporting
artifacts thinned out more than you'd want. Cross-reference against
[Tuning](./tuning.md) — this pattern often means `TARGET_REDUCTION` is set higher
than the trace's decision density can comfortably support.
