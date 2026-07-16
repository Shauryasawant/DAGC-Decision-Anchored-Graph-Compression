# DAGC — Decision-Anchored Graph Compression

`dagc` shortens agent and chat histories while prioritising the values that
later decisions depend on: tool arguments, IDs, paths, metrics, and confirmed
choices. It is designed for applications that need to reduce context size
without losing the evidence behind an earlier decision.

The core package is local-only. It does not make LLM or network calls unless
you configure an optional embedding or evaluation adapter that does.

## What it includes

- Decision-aware message compression with configurable token reduction.
- Zero-dependency fallback tokenizer and embedder for local use.
- Optional adapters for `tiktoken`, Sentence Transformers, OpenAI, and Cohere.
- `compress_any()` for normalising and restoring common message formats.
- `dagc_eval`, an optional reproducibility and benchmark toolkit.
- `dagc-server`, an optional HTTP proxy for compressing message arrays before
  forwarding a request to an upstream LLM API.

## Install

Install the package from this repository while developing:

```bash
pip install -e .
```

After publishing, install it from PyPI:

```bash
pip install dagc
```

The core package only requires NumPy and SciPy. For model-aligned token counts
and semantic embeddings, install the relevant extras:

```bash
pip install "dagc[tiktoken,sentence-transformers]"
```

## Quick start

```python
from dagc import compress

compressed = compress(messages, target_reduction=0.85)
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=compressed,
)
```

`compress()` accepts a list of message dictionaries and returns a new list.
The returned messages include `_orig_idx`, which records their position in the
original trace.

For arbitrary message shapes, use `compress_any()` instead. It normalises
common message and envelope formats, compresses them, then restores the
original structure by default:

```python
from dagc import compress_any

compressed_payload = compress_any(request_payload, target_reduction=0.85)
```

## Configure production adapters

The default adapters are useful for getting started, but a tokenizer and
embedding model matched to your application usually produce better results.

```python
import dagc
from dagc.adapters import SentenceTransformerEmbedder, TiktokenTokenizer

dagc.configure(
    tokenizer=TiktokenTokenizer("cl100k_base"),
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
)
```

See [examples/byok_openai_embeddings.py](examples/byok_openai_embeddings.py)
for a Mistral embedding example and [examples/langgraph_style_node.py](examples/langgraph_style_node.py)
for a workflow-node example.

## Tuning

```python
from dagc import DAGCConfig, compress

cfg = DAGCConfig(TARGET_REDUCTION=0.90, KEEP_LAST_K=3)
compressed = compress(messages, cfg=cfg)

# Equivalent inline overrides:
compressed = compress(messages, TARGET_REDUCTION=0.90, KEEP_LAST_K=3)
```

| Field | Default | Purpose |
| --- | ---: | --- |
| `TARGET_REDUCTION` | `0.87` | Requested fraction of tokens to remove. |
| `KEEP_LAST_K` | `1` | Retains the latest messages as protected context. |
| `PROTECT_TOOL_CALLS` | `True` | Protects tool-call messages during selection. |
| `PROTECT_JUDGMENTS` | `True` | Protects assistant judgments and confirmations. |
| `USE_CAUSAL_SKELETON` | `True` | Uses the causal message graph during selection. |

Protected messages may still have their content shortened when
`COMPRESS_PROTECTED=True` (the default). Use the settings in `DAGCConfig` to
adjust those caps for your application.

## How compression works

1. The extractor identifies tool calls, decisions, and confirmations, then
   collects the IDs, paths, values, and rationale associated with them.
2. The first selection phase reserves budget for decision-critical evidence.
3. The remaining budget is filled with causally relevant, task-relevant
   sentences while reducing redundant context.
4. A final pass checks for missing critical values and adds available evidence
   back when needed.

This is heuristic software, so evaluate it with your own traces before relying
on a specific reduction target in production.

## Evaluate a trace

`dagc_eval` measures whether a compressed trace still has enough information
to reproduce the decisions in the original trace.

```python
from dagc_eval import TASKS, compute_drr, generate_trace

trace = generate_trace(TASKS[0])
result = compute_drr(trace)
print(result["DRR_soft"], result["RCI"])
```

The default evaluation path is deterministic and runs offline. You can also
pass a BYOK client for reconstruction attempts that need an LLM:

```python
import openai
from dagc_eval import compute_drr
from dagc_eval.interfaces import OpenAIChatClient

llm = OpenAIChatClient(openai.OpenAI())
result = compute_drr(trace, llm=llm)
```

The command-line interface exposes the same workflows:

```bash
dagc compress trace.json --target-reduction 0.85 -o compressed.json
dagc evaluate trace.json -o report.md
dagc benchmark --n-traces 3 -o benchmark.json
```

## Optional proxy

Install the server extra to run the wire-compatible compression proxy:

```bash
pip install "dagc[server]"
export UPSTREAM_BASE_URL="https://your-llm-provider.example"
dagc-server
```

The proxy looks for a `messages`, `trace`, `conversation`, or `turns` array in
the request body. If compression fails, it forwards the original request
unchanged.

## Project layout

```text
src/dagc/       Core compression, formats, adapters, and graph utilities
src/dagc_eval/  Evaluation, benchmarks, diagnostics, exports, and proxy server
examples/       Runnable integration examples
tests/          Unit and format-robustness tests
```
