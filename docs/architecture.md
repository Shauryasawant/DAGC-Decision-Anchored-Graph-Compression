# Architecture

## Package layout

```
src/dagc/        Core compression engine, graph construction, formats, adapters,
                  and decision-rationale extraction (rationale_ext.py)
src/dagc_eval/    Evaluation toolkit, DRR, benchmarking, diagnostics, exports,
                  and proxy server
examples/         Runnable integration examples
tests/            Unit tests and format robustness tests
```

## Why the split

`dagc` and `dagc_eval` are separate for a reason: the compression path and the
measurement path have different network/dependency requirements, and keeping them
apart means you can depend on one without pulling in the other.

| Package | Purpose | Network calls |
|---|---|---|
| `dagc` | Compression: `compress()`, `compress_any()`, extraction, rationale | None, unless you configure an adapter that makes them |
| `dagc_eval` | Measurement: DRR, RCI, benchmarking, the CLI, `dagc-server` | None by default; opt-in if you supply an `llm=` client to `compute_drr()` |

## Offline by default

The core package runs entirely locally — it does not make LLM or network calls
unless you explicitly configure an optional embedding or evaluation adapter that
does. Concretely, that means:

- Calling `compress()` or `compress_any()` with no extra config never leaves the
  process.
- Extraction, the dependency graph, budget allocation, and validation (see
  [How Compression Works](./how-it-works.md)) are all deterministic, local
  computation — no model in the loop.
- Network calls only enter the picture if you explicitly opt in: configuring a
  `SentenceTransformerEmbedder` / `TiktokenTokenizer` adapter, supplying an
  `llm=` client to `dagc_eval.compute_drr()` for reconstruction-based evaluation
  (see [Evaluation](./evaluation.md)), or running the optional proxy
  (see [Optional Proxy](./proxy.md)), which itself only calls your configured
  upstream, not a third party on DAGC's behalf.

## Configuring production adapters

The built-in tokenizer and embedder are local fallbacks, good enough to get
started but not necessarily aligned with your production model's tokenization.
Swapping in matched adapters generally improves compression quality:

```python
import dagc
from dagc.adapters import SentenceTransformerEmbedder, TiktokenTokenizer

dagc.configure(
    tokenizer=TiktokenTokenizer("cl100k_base"),
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
)
```

See `examples/byok_openai_embeddings.py` for a full bring-your-own-key embedding
setup, and `examples/langgraph_style_node.py` for wiring `compress()` into a
LangGraph-style node.

## Design principle

Compression correctness (does the important information survive) and compression
measurement (how do we know) are kept as separate concerns with separate
dependency footprints, rather than bundled into one package that always pulls in
evaluation dependencies just to compress a trace. If you're only ever calling
`compress()` in production, you shouldn't need to install anything `dagc_eval`
depends on.
