# CLI

The `dagc` CLI is a thin wrapper around the same Python API described elsewhere
in these docs — it parses arguments and dispatches to the library functions
directly, so behavior always matches calling `compress()` / `compute_drr()` /
etc. yourself. Nothing is reimplemented at the CLI layer.

```bash
dagc <command> [args]
```

## `dagc compress`

Compress a trace stored as a JSON list of messages.

```bash
dagc compress trace.json --target-reduction 0.85 -o compressed.json
```

| Flag | Default | Description |
|---|---|---|
| `input` (positional) | — | Path to input trace JSON |
| `-o`, `--output` | stdout | Path to write compressed trace JSON |
| `--target-reduction` | `0.87` | Same meaning as `TARGET_REDUCTION` — see [Tuning](./tuning.md) |

Without `-o`, the compressed trace prints to stdout as formatted JSON.

## `dagc evaluate`

Score a trace with `compute_drr` — see [Evaluation](./evaluation.md) for what
each metric means.

```bash
dagc evaluate trace.json -o report.md
```

| Flag | Default | Description |
|---|---|---|
| `input` (positional) | — | Path to input trace JSON |
| `-o`, `--output` | none | Report path — extension picks the format: `.json`, `.csv`, `.md`, `.html` |
| `--decision-roles` | `user,assistant` | Comma-separated roles to treat as decision sources |
| `--quiet` | off | Suppress verbose scoring output |

Prints a one-line summary regardless of `-o`:

```
DRR_soft=0.97  DRR_binary=0.94  RCI=0.96  reduction=71.2%
```

## `dagc benchmark`

Runs the synthetic-task DRR sweep across generated traces — the same
`TASKS`-based generation used in [Evaluation](./evaluation.md), swept over
multiple noise levels rather than a single trace.

```bash
dagc benchmark --n-traces 3 -o benchmark.json
```

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output` | none | Report path — same format-by-extension rule as `evaluate` |
| `--n-traces` | `3` | Traces generated per task |
| `--noise-levels` | `1,2,3,4,5` | Comma-separated noise levels to sweep |

## `dagc compare`

Runs DAGC against baseline methods on the same synthetic sweep, for a quick
relative read without setting up your own comparison harness.

```bash
dagc compare --n-traces 2 --noise-levels 3
```

| Flag | Default | Description |
|---|---|---|
| `--n-traces` | `2` | Traces generated per task |
| `--noise-levels` | `3` | Comma-separated noise levels to sweep |

Output is printed directly rather than written to a report file.

## `dagc stats`

Bootstrap confidence interval over a JSON list of `DRR_soft` scores you've
already collected — useful once you have your own set of evaluation runs and
want a CI rather than a single point estimate.

```bash
dagc stats scores.json
```

`scores.json` is a flat JSON list of floats. Output:

```
mean=0.9531  95% CI=[0.9280, 0.9760]  n=42
```

## `dagc rescue`

Runs the rescue validation helper (`validate_rescue`). No flags — invoke it
directly when you need to sanity-check rescue behavior (see the
[MCP Server](./mcp-server.md) page for what "rescue" state means in a
multi-turn session).

```bash
dagc rescue
```

## Error handling

If an `input` path doesn't exist, the CLI exits with a direct message rather
than a raw traceback:

```
Trace file not found: trace.json
Provide a JSON trace file, for example: dagc evaluate your_trace.json
```
