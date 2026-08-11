# DAGC Docs

This folder is the deep-dive companion to the [README](../README.md). The README
gets you installed and running in five minutes; these pages go one level deeper
into each part of the system, for anyone extending, tuning, or evaluating DAGC.

If you haven't run `compress()` yet, start with the [README Quick Start](../README.md#quick-start)
first — everything here assumes you've seen the basic call already.

## Core Concepts

| Page | What it covers |
|---|---|
| [How Compression Works](./how-it-works.md) | The four-stage pipeline: extraction → dependency graph → budget allocation → validation |
| [Tuning](./tuning.md) | Every `DAGCConfig` field, what it trades off, and when to change it |
| [Decision Rationale](./decision-rationale.md) | Extracting *why* a decision was made, not just what it was, and making that survive compression |
| [Evaluation](./evaluation.md) | DRR, RCI, artifact retention — what each metric means and how to run them on your own traces |
| [Optional Proxy](./proxy.md) | Running `dagc-server` as a wire-compatible compression gateway |
| [Architecture](./architecture.md) | Package layout, what runs offline vs. what's opt-in network, and the design principles behind the split |

## Interfaces

| Page | What it covers |
|---|---|
| [CLI](./cli.md) | `dagc compress` / `evaluate` / `benchmark` / `compare` / `stats` / `rescue` — every subcommand and flag |
| [MCP Server](./mcp-server.md) | `dagc-mcp` — 8 MCP tools for Claude Desktop or any MCP client, session/rescue semantics, storage backends |

## Reading order

If you're evaluating DAGC for a new use case, this order tends to answer questions
in the sequence they come up:

1. [How Compression Works](./how-it-works.md) — understand what's actually happening to your trace
2. [Tuning](./tuning.md) — get the reduction/fidelity tradeoff right for your traces
3. [Evaluation](./evaluation.md) — measure it, don't take the defaults on faith
4. [Decision Rationale](./decision-rationale.md) — if your downstream task needs *why*, not just *what*
5. [Architecture](./architecture.md) / [Optional Proxy](./proxy.md) — if you're deploying this, not just calling it in-process
6. [CLI](./cli.md) / [MCP Server](docs/CLI.md) — if you're driving DAGC from outside Python directly

## Something not here?

The README's [Project Layout](../README.md#project-layout) section is the source
of truth for what code exists. If a doc page here describes behavior that doesn't
match `src/dagc/`, the code wins — open an issue.
