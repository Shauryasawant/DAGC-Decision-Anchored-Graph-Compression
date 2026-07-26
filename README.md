# DAGC — Decision-Anchored Graph Compression

**DAGC (Decision-Anchored Graph Compression)** compresses conversational and agent traces while preserving the decision-critical information required for reliable downstream reasoning. Unlike generic summarization methods that primarily optimize semantic similarity, DAGC prioritizes operational artifacts such as tool arguments, identifiers, file paths, configuration values, metrics, and confirmed choices that downstream decisions depend on.

It supports both conversational messages and messages carrying `tool_call`/`tool_calls` payloads (common in agent and tool-integrated workflows). Use `compress()` for canonical chat traces and `compress_any()` to normalize, compress, and restore arbitrary message formats.

The core package runs entirely locally. It does not make LLM or network calls unless you explicitly configure an optional embedding or evaluation adapter that does.

## Why DAGC?

Traditional conversation compression and summarization methods preserve general meaning but may discard the operational details required to continue a task correctly. Information such as identifiers, file paths, configuration values, tool arguments, and confirmed decisions often matters more than semantic similarity when an agent must resume work later.

DAGC is designed to preserve these decision-critical artifacts while aggressively reducing context size. Its accompanying evaluation toolkit measures this objective using **Decision Reproducibility Rate (DRR)**, which evaluates whether the original operational decisions remain reproducible after compression.

## Features

* Decision-aware conversational and agent-trace compression.
* Preservation of tool arguments, identifiers, configuration values, file paths, metrics, and confirmed decisions.
* High compression while maintaining decision fidelity.
* `compress_any()` for normalizing and restoring common message formats.
* Lightweight fallback tokenizer and embedder for fully local execution.
* Optional adapters for `tiktoken`, Sentence Transformers, OpenAI, and Cohere.
* Decision rationale extraction, including alternatives that were considered and ruled out.
* Deterministic offline evaluation with DRR and supporting diagnostics.
* Optional HTTP proxy for production deployments.

## Install

Install the package from this repository while developing:

```bash
pip install -e .
```

After publishing, install it from PyPI:

```bash
pip install dagc
```

The core package only requires NumPy and SciPy. For model-aligned token counts and semantic embeddings, install the relevant extras:

```bash
pip install "dagc[tiktoken,sentence-transformers]"
```

## Quick start

Compress a conversation before storing it as long-term memory or forwarding it to an LLM.

```python
from dagc import compress

# Conversational messages (OpenAI / chat-style)
compressed = compress(messages, target_reduction=0.85)

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=compressed,
)

# Agent/tool-call example
agent_msg = {
    "role": "assistant",
    "content": "Called system to fetch metrics.",
    "tool_call": {
        "name": "get_metrics",
        "args": {
            "job_id": "run-42"
        }
    }
}

compressed_agent = compress([agent_msg], target_reduction=0.5)
```

`compress()` accepts a list of message dictionaries and returns a new list. Each returned message includes `_orig_idx`, recording its position in the original trace.

For arbitrary message formats, use `compress_any()`. It normalizes common message and envelope formats, compresses them, and restores the original structure automatically.

```python
from dagc import compress_any

compressed_payload = compress_any(request_payload, target_reduction=0.85)
```

## Configure production adapters

The built-in adapters are suitable for getting started, but configuring a tokenizer and embedding model that match your production environment generally improves compression quality.

```python
import dagc
from dagc.adapters import SentenceTransformerEmbedder, TiktokenTokenizer

dagc.configure(
    tokenizer=TiktokenTokenizer("cl100k_base"),
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
)
```

See `examples/byok_openai_embeddings.py` for a BYOK embedding example and `examples/langgraph_style_node.py` for a workflow integration example.

## Tuning

```python
from dagc import DAGCConfig, compress

cfg = DAGCConfig(
    TARGET_REDUCTION=0.90,
    KEEP_LAST_K=3,
)

compressed = compress(messages, cfg=cfg)

# Equivalent inline overrides
compressed = compress(
    messages,
    TARGET_REDUCTION=0.90,
    KEEP_LAST_K=3,
)
```

| Configuration         | Default | Description                                        |
| --------------------- | ------: | -------------------------------------------------- |
| `TARGET_REDUCTION`    |  `0.87` | Requested fraction of tokens to remove.            |
| `KEEP_LAST_K`         |     `1` | Protects the most recent messages.                 |
| `PROTECT_TOOL_CALLS`  |  `True` | Preserves tool-call messages during selection.     |
| `PROTECT_JUDGMENTS`   |  `True` | Preserves assistant judgments and confirmations.   |
| `USE_CAUSAL_SKELETON` |  `True` | Uses the causal dependency graph during selection. |

Protected messages may still have their content shortened when `COMPRESS_PROTECTED=True` (default). Additional limits can be adjusted through `DAGCConfig`.

`TARGET_REDUCTION` is a soft target, not a guarantee. Decision-critical evidence is hard-protected regardless of how aggressive a reduction is requested, so actual reduction can fall short of the target at very high settings. At very conservative settings, actual reduction can also plateau below what spare budget would otherwise allow, once all decision-relevant content has already been retained. Evaluate on your own traces rather than assuming the requested percentage will be met exactly.

## How compression works

1. Extract decision-critical artifacts including tool calls, identifiers, configuration values, paths, metrics, and confirmed choices.
2. Construct a dependency graph linking each decision to its supporting conversational evidence.
3. Allocate the compression budget while protecting decision-critical regions and removing redundant context.
4. Validate the compressed trace and restore any missing critical evidence before producing the final output.

DAGC uses heuristic selection strategies. Evaluate compression quality on representative production traces before deployment.

## Decision rationale

Beyond preserving what was decided, DAGC can extract *why* — including alternatives that were considered and ruled out. This is useful when an agent needs to know not just the final choice, but the reasoning that eliminated other options.

```python
from dagc import extract_decisions, extract_rationale_candidates

decisions = extract_decisions(messages)
candidates = extract_rationale_candidates(messages, decisions, include_same_message=True)

for c in candidates:
    print(c["alternative"])       # the option that was ruled out
    print(c["reason"])            # the clause explaining why
    print(c["confidence"])        # 'high' (explicit causal marker) or 'medium'
    print(c["decision_msg_idx"])  # which final decision this explains
```

To make sure this reasoning survives compression rather than being cut along with the rest of the message, inject it as rationale stubs:

```python
from dagc import inject_rationale_stubs, inject_dropped_rationale_stubs

compressed, report = inject_rationale_stubs(compressed, messages, decisions)
print(report["rationale_stubs_added"])

# Also recover rationale from messages that were dropped entirely
compressed, dropped_report = inject_dropped_rationale_stubs(compressed, messages)
print(dropped_report["dropped_rationale_stubs_added"])
```

Both functions return a report alongside the compressed trace, so you can see exactly how many rationale stubs were added and inspect each one via `rationale_added_detail` / `dropped_rationale_added_detail`.

## Evaluate a trace

`dagc_eval` measures **Decision Reproducibility Rate (DRR)**, which evaluates whether a compressed trace still contains sufficient information to reproduce the original operational decisions. It also reports complementary diagnostics such as Reasoning Chain Integrity (RCI) and artifact retention.

```python
from dagc_eval import TASKS, compute_drr, generate_trace

trace = generate_trace(TASKS[0])

result = compute_drr(trace)

print(result["DRR_soft"])
print(result["RCI"])
```

`compute_drr` also reports artifact retention alongside DRR and RCI — `art_ret` for all recoverable artifacts, and `decision_art_ret` specifically for artifacts tied to a decision, which is the strongest signal that decision-critical information survived compression intact.

```python
print(result["art_ret"])
print(result["decision_art_ret"])
```

The default evaluation pipeline is deterministic and runs entirely offline. Optionally, you can supply your own LLM client for reconstruction-based evaluation.

```python
import openai

from dagc_eval import compute_drr
from dagc_eval.interfaces import OpenAIChatClient

llm = OpenAIChatClient(openai.OpenAI())

result = compute_drr(
    trace,
    llm=llm,
)
```

The command-line interface exposes the same workflows:

```bash
dagc compress trace.json --target-reduction 0.85 -o compressed.json

dagc evaluate trace.json -o report.md

dagc benchmark --n-traces 3 -o benchmark.json
```

## Optional proxy

Install the server extra to run the wire-compatible compression proxy.

```bash
pip install "dagc[server]"

export UPSTREAM_BASE_URL="https://your-llm-provider.example"

dagc-server
```

The proxy automatically detects common request formats (`messages`, `trace`, `conversation`, or `turns`), compresses the conversation, preserves tool-call payloads, and forwards the request to the configured upstream API. If compression fails for any reason, the original request is forwarded unchanged.

## Project layout

```text
src/dagc/        Core compression engine, graph construction, formats, adapters,
                 and decision-rationale extraction (rationale_ext.py)
src/dagc_eval/   Evaluation toolkit, DRR, benchmarking, diagnostics, exports, and proxy server
examples/        Runnable integration examples
tests/           Unit tests and format robustness tests
```

## License

Released under the MIT License.