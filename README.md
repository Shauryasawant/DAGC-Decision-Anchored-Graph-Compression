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
📖 **[Full documentation →](docs/index.md)**

[Paper](#paper) · [Why DAGC](#why-dagc) · [Proof](#proof) · [Install](#install) · [Quick start](#quick-start) · [How it works](#how-compression-works) · [Evaluation](#evaluate-a-trace) · [Proxy](#optional-proxy) · [MCP server](#optional-mcp-server) · [Contributing](#contributing)

---

DAGC compresses conversational and agent traces while preserving the decision-critical information that downstream reasoning depends on — tool arguments, identifiers, file paths, configuration values, metrics, and confirmed choices. Unlike generic summarizers that optimize for semantic similarity, DAGC optimizes for **operational continuity**: can the agent still do the right thing after the trace has been shrunk?

The core package runs entirely locally. It makes no LLM or network calls unless you explicitly wire in an optional embedding or evaluation adapter that does.

## What it does

- **Decision-aware compression** — `compress()` for canonical chat traces, `compress_any()` to normalize, compress, and restore arbitrary message formats (including `tool_call` / `tool_calls` payloads).
- **Protects what matters** — tool arguments, identifiers, config values, file paths, metrics, and confirmed decisions are preserved even under aggressive reduction targets.
- **Decision rationale extraction** — recovers *why* a choice was made, including alternatives that were considered and ruled out, and can inject that rationale back into the compressed trace so it survives.
- **Deterministic, offline evaluation** — `dagc_eval` measures Decision Reproducibility Rate (DRR) plus supporting diagnostics, fully offline by default.
- **Optional HTTP proxy** — wire-compatible compression proxy for production deployments, with automatic passthrough if compression fails.
- **Optional MCP server** — expose compression, evaluation, and rescue as MCP tools for Claude Desktop or any MCP client, with pluggable single-process, file, or Redis-backed persistence for multi-instance deployments.
- **Pluggable adapters** — lightweight local fallback tokenizer/embedder out of the box; optional `tiktoken`, Sentence Transformers, OpenAI, and Cohere adapters for production-matched quality.

## Why DAGC?

Traditional conversation compression and summarization preserve general meaning but can discard the operational details an agent needs to resume work correctly. Identifiers, file paths, config values, tool arguments, and confirmed decisions often matter more than semantic similarity when an agent has to pick a task back up later.

DAGC is built to preserve exactly those decision-critical artifacts while still aggressively reducing context size. Its evaluation toolkit measures this directly with **Decision Reproducibility Rate (DRR)** — whether the original operational decisions are still reproducible after compression.

## Where DAGC fits best

DAGC is most valuable in environments where preserving the correctness of the next action matters more than simply shortening text. It is especially well-suited for:

- Enterprise agent automation, copilots, and workflow assistants.
- Software engineering and debugging agents that rely on tool calls, file paths, identifiers, and configuration values.
- Finance, operations, compliance, and regulated workflows where losing a key detail can cause a wrong decision.
- Customer support, account operations, and multi-step task agents that must retain state across long conversations.
- Any system that needs to compress long traces while still preserving decision-critical evidence, rationale, and tool-call payloads.

This is different from generic summarization or prompt-compression approaches, which often optimize for semantic similarity or shorter prompts but may discard the operational details that an agent needs to act correctly. DAGC is designed for decision-faithful compression: keeping the evidence needed to reproduce the original action, not just the gist of the conversation. Production environments where "the summary looked fine but the agent made the wrong call" is unacceptable are exactly where DAGC earns its keep.

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

Between turns, an optional **rescue engine** extends this same guarantee across a whole session: if a later message references a value that survived only in an earlier, now-uncompressed turn, rescue detects the reference and force-preserves the owning decision on the next compression pass — without re-implementing anything `compress_dagc` already does. See [Rescue feature](#rescue-feature).

→ [Decision rationale](#decision-rationale) · [Tuning](#tuning) · [Evaluation](#evaluate-a-trace)

## Proof

Benchmarked on **951 agent traces (8,895 messages)** across six corpora (WildChat, BFCL, TauBench, OASST1, Nemotron, Qwen) and 14 operational domains; 691 traces contained an extractable decision and were scored. Full methodology, theory, and figures are in the [paper](#paper) — this is the summary.

**Leaderboard** (mean DRR — Decision Reproducibility Rate — vs. 10 baselines, ranked among methods that meaningfully compress the trace):

| Rank | Method | Mean DRR | Reduction | RCI |
| --- | --- | --- | --- | --- |
| 1 | **DAGC** | **0.9958** | **74.0%** | **0.994** |
| 2 | TextRank (sumy) | 0.7934 | 36.5% | 0.922 |
| 3 | PyTextRank (spaCy) | 0.7927 | 35.2% | 0.932 |
| 4 | LexRank (sumy) | 0.7803 | 39.4% | 0.909 |
| 5 | LlamaIndex | 0.3574 | 54.1% | 0.746 |
| 6 | Random Drop | 0.3467 | 65.7% | 0.689 |
| 7 | Tail Truncation | 0.3432 | 77.1% | 0.725 |
| 8 | Token Trim Last N | 0.2757 | 67.2% | 0.636 |
| 9 | Sliding Window | 0.0958 | 80.4% | 0.504 |
| 10 | LangChain Token Buffer Memory | 0.0619 | 84.0% | 0.339 |

DAGC compresses **74.0%** of the trace while keeping **99.6%** decision reproducibility — over 20 points of DRR ahead of the next-best real compressor, which only reduces the trace by half as much. Every baseline that cuts the trace by more than 50% (other than DAGC) drops to a mean DRR at or below 0.36.

**Statistical significance** — paired Wilcoxon signed-rank test on per-trace DRR, Bonferroni-corrected, on the full 691-trace benchmark: DAGC beats every baseline that meaningfully compresses the trace with large positive effect size (Cohen's d from 1.23 up to 7.65 vs. LangChain's token buffer memory), all significant after correction.

**Adversarial robustness** — under a 4-family perturbation suite (prompt injection, noise amplification, decision masking, contradiction injection), DAGC's adversarial-to-clean DRR ratio stays within **0.998–1.009** of parity — no measurable degradation, well clear of the 0.85 robustness threshold.

Reproduce it yourself:

```bash
dagc benchmark --n-traces 3 -o benchmark.json
```

## When to use · When to skip

**Good fit if you…**

- run agents or long chat sessions where tool calls, file paths, config values, or confirmed decisions need to survive into later turns
- need the trace to stay debuggable — decision rationale, not just raw text, has to come through compression intact
- want deterministic, offline evaluation (DRR) before trusting compression in a critical path
- are fine tuning `TARGET_REDUCTION` per-trace rather than expecting an exact percentage every time

**Skip it if you…**

- only need to trim conversation length for display purposes, with no decision-fidelity requirement — plain truncation or summarization is simpler
- need a hard guarantee on the exact reduction percentage (DAGC hard-protects decision evidence first, so aggressive targets can fall short)
- want a hosted/managed compression API rather than a local library — DAGC's core makes no network calls by design

## Compared to

|                                    | Scope                                    | Basis of retention          | Mean DRR @ its own reduction |
| ---------------------------------- | ----------------------------------------- | ---------------------------- | ----------------------------- |
| **DAGC**                           | Decision-critical artifacts, causal graph | Hard-guaranteed dependency graph | **0.9958 @ 74.0%**      |
| TextRank / PyTextRank / LexRank    | Whole document                            | Embedding/lexical centrality | 0.78–0.79 @ 35–39%             |
| LlamaIndex / LangChain buffer      | Conversation history                      | Recency / production heuristics | 0.06–0.36 @ 54–84%          |
| Sliding window / tail truncation   | Conversation history                      | Positional (keep last N)     | 0.10–0.34 @ 77–80%             |
| LLMLingua                          | Prompt tokens                             | Perplexity                   | ~1.00 @ ~0% (barely compresses) |

Numbers are mean DRR and mean token reduction from the [paper's](#paper) 691-trace benchmark, not marketing estimates. Centrality-based summarizers and perplexity filters are theoretically biased against rare, low-frequency tokens (IDs, hashes, config values) — see [Section 5 of the paper](#paper) for the proof.

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

### Rescue feature

The rescue capability is available as part of the installed package. You can import it directly:

```python
from dagc import RescueEngine, ShadowBuffer, reset_rescue_session
```

Or run the bundled helper from the CLI after installation:

```bash
dagc rescue
```

`compress()` calls rescue automatically (`enable_rescue=True` by default) — see [Quick start](#quick-start) for the `session_id` semantics that govern it.

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

> **`session_id` note:** rescue is keyed by `session_id` (default `"default"`), and `messages` must be a growing, append-only prefix across calls that share one `session_id`. Use a distinct `session_id` per independent conversation, or call `reset_rescue_session(session_id)` before reusing one — otherwise unrelated traces will contaminate each other's rescue state. Pass `enable_rescue=False` for one-shot calls with no session semantics.

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

### Wrap supported coding CLIs

After installing the server extra, launch these tools through the proxy without
changing their configuration or source code:

```bash
dagc wrap claude -- --help
dagc wrap codex -- --help
dagc wrap aider -- --help
```

`dagc wrap` selects a free local port, starts the proxy, and sets the one base
URL variable each tool honors: `ANTHROPIC_BASE_URL` for Claude Code,
`OPENAI_BASE_URL` for Codex CLI, and `OPENAI_API_BASE` for Aider. Use
`--upstream URL` to choose a compatible provider; OpenAI-compatible URLs may
include `/v1` or omit it. Cursor and GitHub Copilot CLI are not supported
because they do not offer a user-settable model base URL.

## Optional MCP server

> **Status: prototype, not yet merged.** The `dagc-mcp` command and the tools/scaling options below were designed and drafted in a working session and reflect the intended shape of this feature. They require `store.py`'s `RedisBackend` addition and a new `rescue_redis.py` module to actually be merged into `src/dagc/` before `dagc-mcp` behaves as documented here — check `src/dagc/mcp_server.py` in this repo for what's actually shipped before depending on this section.

Install the MCP extra to expose DAGC as tools for Claude Desktop or any other MCP client:

```bash
pip install "dagc[mcp]"

dagc-mcp   # stdio transport
```

No LLM call happens inside the server by default (BYOK — see [Configure production adapters](#configure-production-adapters)); it does not require an API key to run.

**Tools:**

| Tool | What it does |
| --- | --- |
| `dagc_compress` | Compress a message trace, format-tolerant. Takes `target_reduction`, `force_preserve`, `trace_id`, `session_id`, `enable_rescue`, `strict_no_loss`. |
| `dagc_evaluate` | Score a trace's decision-reproducibility (wraps `compute_drr`) without compressing it. |
| `dagc_get_original` | Resolve one compressed message's `_orig_idx` back to its original content. |
| `dagc_resolve_trace` | Batch version — resolve a whole compressed trace at once, or dump everything stored for a `trace_id`. |
| `dagc_verify` | Independent, from-scratch check that a compressed trace still contains every decision-critical value (wraps `verify_no_decision_loss`). |
| `dagc_frontier` | Empirical rate-distortion curve — output tokens vs. decision loss across a list of candidate budgets (wraps `decision_loss_frontier`). |
| `dagc_reset_session` | Drop all rescue state for a `session_id` — use when a new, unrelated conversation reuses one. |
| `dagc_stats` | Aggregated monitoring: call counts, token totals, achieved reduction, unrescuable evictions — global or scoped to one `session_id`. |

**`session_id` vs. `trace_id`:** these are two different keys. `trace_id` is a storage key, purely for `dagc_get_original`/`dagc_resolve_trace` lookups later. `session_id` is a rescue key — `messages` must be a growing, append-only prefix across calls sharing one `session_id`, same rule as `compress()` itself. If you don't pass `session_id` explicitly, `dagc_compress` defaults it to `trace_id` (not a shared global default), so distinct traces don't silently share rescue state.

**Persistence and scaling**, controlled entirely by environment variables — no code changes required to move between tiers:

| Env var | Effect |
| --- | --- |
| *(none set)* | In-memory only. Fine for local use; state is lost on restart and not shared across processes. |
| `DAGC_MCP_STORE_PATH` | Trace storage persists to disk (`FileBackend`) — survives restarts on a single machine. No cross-process write locking; not safe for multiple server instances sharing the same path. |
| `DAGC_REDIS_URL` | Trace storage **and** rescue session state move to Redis — durable and shared across every server instance, the tier needed for real horizontal scaling. Trace storage uses an atomic per-field hash write (no read-modify-write race). Rescue state is protected by a Redis distributed lock held for the duration of each `dagc_compress` call, so concurrent calls for the same `session_id` across instances are strictly serialized. |
| `DAGC_REDIS_TRACE_TTL_S` | Idle expiry (seconds) for stored traces when using Redis. Defaults to 7 days. |
| `DAGC_MCP_ANALYTICS_PATH` | Persists `dagc_stats` data to a JSON snapshot after every call. |

Two caveats worth knowing before relying on the Redis tier in production: rescue session state is pickled (safe only because the server fleet is the sole writer to that keyspace — never point it at a Redis instance an untrusted party can write to), and the distributed lock is single-Redis-primary locking, not the multi-node RedLock algorithm — revisit if you run Redis Cluster/Sentinel with failover.

## Contributing

```bash
git clone https://github.com/Shauryasawant/DAGC-Decision-Anchored-Graph-Compression.git
cd DAGC-Decision-Anchored-Graph-Compression
pip install -e ".[tiktoken,sentence-transformers]"
pytest
```

Issues and PRs are welcome — see [tests/](tests/) for the format-robustness suite before changing core compression logic.

## Community

- **[Issues](https://github.com/Shauryasawant/DAGC-Decision-Anchored-Graph-Compression/issues)** — bugs, questions, feature requests.

## Paper

DAGC is described in full — theory, algorithm, 951-trace benchmark, adversarial robustness suite, and statistical analysis — in the accompanying paper, *DAGC: Decision-Anchored Graph Compression for Reproducible Context Compression* (Shaurya Sawant).

```bibtex
@misc{sawant_dagc,
  title  = {DAGC: Decision-Anchored Graph Compression for Reproducible Context Compression},
  author = {Sawant, Shaurya},
  note   = {Independent AI Research}
}
```

> https://zenodo.org/records/21621103

## Project layout

```
src/dagc/        Core compression engine, graph construction, formats, adapters,
                 decision-rationale extraction (rationale_ext.py), rescue engine
                 (rescue.py), persistence (store.py), and the optional MCP
                 server (mcp_server.py)
src/dagc_eval/   Evaluation toolkit, DRR, benchmarking, diagnostics, exports, and proxy server
examples/        Runnable integration examples
tests/           Unit tests and format robustness tests
```

## License

MIT — see [LICENSE](LICENSE).

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#install) [![Status](https://img.shields.io/badge/sta[...]
