# MCP Server

`dagc.mcp_server` exposes DAGC as MCP tools over stdio — usable from Claude
Desktop or any other MCP client, without writing a client-specific integration.

```bash
pip install dagc[mcp]
dagc-mcp
```

No LLM calls happen anywhere in this server by default (BYOK via
`dagc.configure`, same as the core library — see [Architecture](./architecture.md)).
No API key is required to run it.

## Tools

| Tool | Purpose |
|---|---|
| `dagc_compress` | Compress a message trace, format-tolerant |
| `dagc_evaluate` | Score a trace's decision-reproducibility, no LLM call |
| `dagc_get_original` | Resolve a compressed message's `_orig_idx` back to its source |
| `dagc_resolve_trace` | Batch resolve, or dump a whole stored trace |
| `dagc_verify` | Independent, from-scratch retention-invariant check |
| `dagc_frontier` | Empirical rate-distortion curve across token budgets |
| `dagc_reset_session` | Drop rescue state for a session |
| `dagc_stats` | Aggregated monitoring/analytics |

### `dagc_compress`

The main tool — same behavior as `compress_any()` (see
[How Compression Works](./how-it-works.md)), plus session-aware rescue and an
optional strict-loss mode.

```
dagc_compress(
    messages, target_reduction=None, force_preserve=None,
    trace_id=None, session_id=None, enable_rescue=True, strict_no_loss=False,
)
```

- **`force_preserve`** — extra literal strings to hard-guarantee survive, on
  top of whatever decision extraction already finds.
- **`trace_id`** — registers the trace for later `dagc_get_original` /
  `dagc_resolve_trace` lookups. Purely a storage key.
- **`session_id`** — identifies a *rescue* session (separate concept from
  `trace_id` — see below). If omitted, falls back to `trace_id`; if neither is
  given, falls back to a shared `"default"` session, which is fine for
  one-shot calls but **not safe for concurrent multi-user traffic** — pass an
  explicit `session_id` per user/conversation in that case.
- **`enable_rescue`** — maintain cross-turn rescue state for this call. Set
  `False` for one-shot compression with no session semantics.
- **`strict_no_loss`** — raise instead of silently shipping a trace with any
  unrecoverable decision-critical value. Surfaces as `{"error": ...}` in the
  tool result rather than an exception across the MCP boundary.

Returns:

```json
{
  "compressed": [...],
  "trace_id": "...",
  "session_id": "...",
  "diagnostics": {
    "orig_tokens": 0,
    "output_tokens": 0,
    "achieved_reduction": 0.0,
    "n_stubbed": 0,
    "unrescuable_evictions": [],
    "injection_filtered_msg_idxs": []
  }
}
```

### `session_id` vs `trace_id`

These are two different keys and it's worth being deliberate about both:

- **`trace_id`** — a storage key for later original-content lookups.
- **`session_id`** — identifies a rescue session (`ShadowBuffer` +
  `RescueEngine` + last-compressed cache). `messages` passed under the same
  `session_id` must be a growing, append-only prefix across calls.

If you don't pass `session_id` explicitly, it defaults to `trace_id` rather
than a global default — this means distinct traces get distinct rescue state
without you having to think about it on every call. But if you also omit
`trace_id`, every call shares the same session, and concurrent/multi-user
traces can cross-contaminate each other's rescue state. **Pass an explicit
`session_id` per user or conversation in any multi-user deployment.**

### `dagc_evaluate`

Deterministic scoring only — same metrics as [Evaluation](./evaluation.md)
(`DRR_soft`, `DRR_binary`, `RCI`, achieved reduction), callable before you
commit to compressing a trace for real, or to compare two candidate settings.

```
dagc_evaluate(messages, decision_roles=None)
```

### `dagc_get_original` / `dagc_resolve_trace`

Resolve compressed output back to source content, using the `_orig_idx` every
compressed message carries.

- `dagc_get_original(trace_id, orig_idx)` — single lookup.
- `dagc_resolve_trace(trace_id, compressed_messages=None)` — batch resolve a
  full compressed list (best-effort: falls back to the compressed message
  itself if the original isn't in the store), or omit `compressed_messages`
  to dump every original message stored for that `trace_id`.

Both depend on `trace_id` having been registered via `dagc_compress(...,
trace_id=...)` first, and both are **in-memory only unless `DAGC_MCP_STORE_PATH`
is set** — without it, stored traces don't survive a server restart.

### `dagc_verify`

An independent check, not a reuse of state from whatever produced `compressed`
— re-extracts decisions from `messages` from scratch and confirms every
critical value is recoverable from `compressed`. This is the same retention
invariant DAGC's own test suite checks, exposed here so you can audit any
compressed trace, including ones produced outside this server.

```
dagc_verify(messages, compressed, decision_roles=None)
→ {"ok": bool, "missing": [[msg_idx, value], ...]}
```

Empty `missing` means the invariant holds.

### `dagc_frontier`

Runs compression at each token budget you give it and reports
`(output_tokens, decision_loss)` per point — an empirical rate-distortion
curve. Useful for picking a budget for a new trace type before committing to
it in production, or for showing a user concretely what a tighter budget
costs in decision fidelity on their specific trace.

```
dagc_frontier(messages, budgets, decision_roles=None)
→ {"frontier": [...]}
```

### `dagc_reset_session`

Drops all rescue state for a `session_id` — the `ShadowBuffer`, `RescueEngine`,
and last-compressed cache. Call this when a new, unrelated conversation
happens to reuse a `session_id`; DAGC's length-shrank heuristic will
eventually catch it on its own, but resetting explicitly is cheaper and
unambiguous.

### `dagc_stats`

Aggregated monitoring: total calls, errors, token totals, achieved reduction,
unrescuable evictions, stubbed rationale count. Pass `session_id` to scope to
one session, or omit it for a global summary across everything this server
process has seen.

```
dagc_stats(session_id=None, recent_calls=20)
```

Set `recent_calls=0` to get aggregates only, without the per-call log.

## Storage backends

Three tiers, checked in order, each falling back to the next rather than
crashing the server on a bad or unreachable config:

| Env var | Backend | Durability |
|---|---|---|
| `DAGC_REDIS_URL` | `RedisBackend` | Durable, shared across processes/machines — real horizontal scaling |
| `DAGC_MCP_STORE_PATH` | `FileBackend` | Durable, single-machine only, no cross-process write locking |
| neither set | `DictBackend` | In-memory, process-lifetime only |

## Rescue session backend

Separate from trace storage above. If `DAGC_REDIS_URL` is set, rescue state
(`ShadowBuffer`, `GuaranteedSet`, decayed recurrence) is shared across every
server instance via `RedisRescueSessionStore`. Otherwise it falls back to
DAGC's in-process session dict — **single process only, unsafe for
multi-instance deployment** if you're running more than one server process
behind a load balancer without Redis configured.

## Analytics persistence

Set `DAGC_MCP_ANALYTICS_PATH` to persist a snapshot of aggregated stats and
the recent-call ring buffer (last 500 calls) to disk after every call. Without
it, `dagc_stats` data is in-memory only, same restart caveat as the trace
store.
