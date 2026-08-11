# Optional Proxy

`dagc-server` is a wire-compatible HTTP proxy that sits between your app and your
LLM provider, compressing requests in flight. Use it when you'd rather not add
`compress()` calls into every call site in your codebase.

## Install and run

```bash
pip install "dagc[server]"

export UPSTREAM_BASE_URL="https://your-llm-provider.example"

dagc-server
```

## What it does

The proxy auto-detects common request shapes — `messages`, `trace`, `conversation`,
or `turns` — so it works with requests formatted for different SDKs without
per-client configuration. For each request it:

1. Detects the conversation field
2. Compresses it (same pipeline as calling `compress()` directly — see
   [How Compression Works](./how-it-works.md))
3. Preserves tool-call payloads through the round trip
4. Forwards the (now smaller) request to `UPSTREAM_BASE_URL`

## Fail-open by design

If compression fails for any reason, the proxy forwards the **original,
uncompressed** request rather than blocking it. This means a bug in the
compression path degrades you to "no savings this request," not "request fails."
Worth knowing before you rely on the proxy in a latency- or cost-sensitive path —
monitor for compression failures separately from request failures, since the
proxy won't surface the former as an error.

## When to use the proxy vs. calling `compress()` directly

| | Proxy (`dagc-server`) | Direct (`compress()` / `compress_any()`) |
|---|---|---|
| Integration effort | Point your client at the proxy URL, no code changes | Add a `compress()` call per call site |
| Config granularity | One config for everything behind the proxy | Per-call `DAGCConfig` overrides (see [Tuning](./tuning.md)) |
| Visibility | Compression happens in a separate process | In-process, easier to log/inspect alongside your own code |
| Best fit | Retrofitting compression onto an existing app with many call sites | New code, or anywhere you want fine-grained control per trace |

If you're not sure, start with `compress()` directly — it's easier to reason about
and to evaluate (see [Evaluation](./evaluation.md)) since there's no network hop
between you and the result.
