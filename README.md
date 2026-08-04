```
██████╗   █████╗   ██████╗   ██████╗
██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝
██║  ██║ ███████║ ██║  ███╗██║
██║  ██║ ██╔══██║ ██║   ██║██║
██████╔╝ ██║  ██║ ╚██████╔╝╚██████╗
╚═════╝  ╚═╝  ╚═╝  ╚═════╝  ╚═════╝
        Decision-Anchored Graph Compression
```

**Compress agent traces without losing the decisions they depend on — local-first, deterministic, reversible-by-design.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#install) [![Status](https://img.shields.io/badge/status-early--stage-yellow.svg)](#project-layout)

[Why DAGC](#why-dagc) · [Install](#install) · [Quick start](#quick-start) · [How it works](#how-compression-works) · [Evaluation](#evaluate-a-trace) · [Proxy](#optional-proxy)

---

DAGC compresses conversational and agent traces while preserving the decision-critical information that downstream reasoning depends on — tool arguments, identifiers, file paths, configuration values, metrics, and confirmed choices. Unlike generic summarizers that optimize for semantic similarity, DAGC optimizes for **operational continuity**: can the agent still do the right thing after the trace has been shrunk?

The core package runs entirely locally. It makes no LLM or network calls unless you explicitly wire in an optional embedding or evaluation adapter that does.

## What it does

- **Decision-aware compression** — `compress()` for canonical chat traces, `compress_any()` to normalize, compress, and restore arbitrary message formats (including `tool_call` / `tool_calls` payloads).
- **Protects what matters** — tool arguments, identifiers, config values, file paths, metrics, and confirmed decisions are preserved even under aggressive reduction targets.
- **Decision rationale extraction** — recovers *why* a choice was made, including alternatives that were considered and ruled out, and can inject that rationale back into the compressed trace so it survives.
- **Deterministic, offline evaluation** — `dagc_eval` measures Decision Reproducibility Rate (DRR) plus supporting diagnostics, fully offline by default.
- **Optional HTTP proxy** — wire-compatible compression proxy for production deployments, with automatic passthrough if compression fails.
- **Pluggable adapters** — lightweight local fallback tokenizer/embedder out of the box; optional `tiktoken`, Sentence Transformers, OpenAI, and Cohere adapters for production-matched quality.

## Why DAGC?

Traditional conversation compression and summarization preserve general meaning but can discard the operational details an agent needs to resume work correctly. Identifiers, file paths, config values, tool arguments, and confirmed decisions often matter more than semantic similarity when an agent has to pick a task back up later.

DAGC is built to preserve exactly those decision-critical artifacts while still aggressively reducing context size. Its evaluation toolkit measures this directly with **Decision Reproducibility Rate (DRR)** — whether the original operational decisions are still reproducible after compression.

## How compression works

```
Conversation / agent trace
       │
       ▼
┌───────────────────────────────────────────────────┐
│  1. Extract decision-critical artifacts            │
│     tool calls · identifiers · configs · paths ·   │
│     metrics · confirmed choices                    │
│                                                     │
│  2. Build a dependency graph                        │
│     link each decision to its supporting evidence   │
│                                                     │
│  3. Allocate the compression budget                 │
│     protect decision-critical regions ·             │
│     drop redundant context                          │
│                                                     │
│  4. Validate & restore                              │
│     re-insert any decision evidence that got cut     │
└───────────────────────────────────────────────────┘
       │
       ▼
Compressed trace (target_reduction met, decisions intact)
```

DAGC uses heuristic selection strategies — evaluate compression quality on representative production traces before relying on it in a critical path.

→ [Decision rationale](#decision-rationale) · [Tuning](#tuning) · [Evaluation](#evaluate-a-trace)

## Install

Install from source while developing:

```bash
pip install -e .
```

Once published, install from PyPI:

```bash
pip install dagc
```

The core package only needs NumPy and SciPy. For model-aligned token counts and semantic embeddings, add the relevant extras:

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

# Agent / tool-call example
agent_msg = {
    "role": "assistant",
    "content": "Called system to fetch metrics.",
    "tool_call": {
        "name": "get_metrics",
        "args": {"job_id": "run-42"},
    },
}

compressed_agent = compress([agent_msg], target_reduction=0.5)
```

`compress()` takes a list of message dicts and returns a new list; each returned message carries `_orig_idx`, recording its position in the original trace.

For arbitrary message formats, use `compress_any()` — it normalizes common message/envelope formats, compresses them, and restores the original structure automatically:

```python
from dagc import compress_any

compressed_payload = compress_any(request_payload, target_reduction=0.85)
```

## Configure production adapters

The built-in adapters are fine for getting started, but pointing DAGC at the tokenizer and embedding model that match your production stack generally improves compression quality:

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
| ---------------------- | ------- | --------------------------------------------------- |
| `TARGET_REDUCTION`     | `0.87`  | Requested fraction of tokens to remove.              |
| `KEEP_LAST_K`          | `1`     | Protects the most recent messages.                   |
| `PROTECT_TOOL_CALLS`   | `True`  | Preserves tool-call messages during selection.       |
| `PROTECT_JUDGMENTS`    | `True`  | Preserves assistant judgments and confirmations.     |
| `USE_CAUSAL_SKELETON`  | `True`  | Uses the causal dependency graph during selection.   |

Protected messages can still have their content shortened when `COMPRESS_PROTECTED=True` (default).

> **Note:** `TARGET_REDUCTION` is a soft target, not a guarantee. Decision-critical evidence is hard-protected regardless of how aggressive the requested reduction is, so actual reduction can fall short at very high settings — and can plateau below the target at very conservative settings, once all decision-relevant content is already retained. Evaluate on your own traces rather than assuming the requested percentage will be hit exactly.

## Decision rationale

Beyond preserving *what* was decided, DAGC can extract *why* — including alternatives that were considered and ruled out. Useful when an agent needs the reasoning that eliminated other options, not just the final choice.

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

To make sure this reasoning survives compression instead of being cut with the rest of the message, inject it as rationale stubs:

```python
from dagc import inject_rationale_stubs, inject_dropped_rationale_stubs

compressed, report = inject_rationale_stubs(compressed, messages, decisions)
print(report["rationale_stubs_added"])

# Also recover rationale from messages that were dropped entirely
compressed, dropped_report = inject_dropped_rationale_stubs(compressed, messages)
print(dropped_report["dropped_rationale_stubs_added"])
```

Both functions return a report alongside the compressed trace — how many rationale stubs were added, and each one individually via `rationale_added_detail` / `dropped_rationale_added_detail`.

## Evaluate a trace

`dagc_eval` measures **Decision Reproducibility Rate (DRR)** — whether a compressed trace still contains enough information to reproduce the original operational decisions — plus complementary diagnostics like Reasoning Chain Integrity (RCI) and artifact retention.

```python
from dagc_eval import TASKS, compute_drr, generate_trace

trace = generate_trace(TASKS[0])
result = compute_drr(trace)

print(result["DRR_soft"])
print(result["RCI"])
print(result["art_ret"])           # retention across all recoverable artifacts
print(result["decision_art_ret"])  # retention for artifacts tied to a decision
```

The default evaluation pipeline is deterministic and fully offline. You can optionally supply your own LLM client for reconstruction-based evaluation:

```python
import openai
from dagc_eval import compute_drr
from dagc_eval.interfaces import OpenAIChatClient

llm = OpenAIChatClient(openai.OpenAI())
result = compute_drr(trace, llm=llm)
```

The CLI exposes the same workflows:

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

The proxy auto-detects common request formats (`messages`, `trace`, `conversation`, `turns`), compresses the conversation, preserves tool-call payloads, and forwards to the configured upstream API. If compression fails for any reason, the original request is forwarded unchanged.

## Project layout

```
src/dagc/        Core compression engine, graph construction, formats, adapters,
                 and decision-rationale extraction (rationale_ext.py)
src/dagc_eval/   Evaluation toolkit, DRR, benchmarking, diagnostics, exports, and proxy server
examples/        Runnable integration examples
tests/           Unit tests and format robustness tests
```

## License

MIT — see [LICENSE](LICENSE).
