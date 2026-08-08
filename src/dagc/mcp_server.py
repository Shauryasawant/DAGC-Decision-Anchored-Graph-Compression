"""
dagc.mcp_server — expose dagc as MCP tools.

    pip install dagc[mcp]
    dagc-mcp                      # stdio transport, e.g. for Claude Desktop

Exposes:
  - dagc_compress         -- compress a message trace, format-tolerant
  - dagc_evaluate          -- score a trace's decision-reproducibility
  - dagc_get_original     -- resolve a compressed message's `_orig_idx`
  - dagc_resolve_trace    -- batch resolve, or dump a whole stored trace
  - dagc_verify            -- post-hoc retention-invariant check
  - dagc_frontier          -- empirical rate-distortion curve
  - dagc_reset_session     -- drop rescue state for a session_id
  - dagc_stats             -- aggregated monitoring/analytics

No LLM calls anywhere here by default (BYOK, see dagc.configure) -- this
server does not require an API key to run.

--- session_id vs trace_id -------------------------------------------------
These are two DIFFERENT keys and this version stops conflating them:

  trace_id   -- identifies a message trace for later dagc_get_original
                lookups. Purely a storage key.
  session_id -- identifies a rescue session (ShadowBuffer + RescueEngine +
                last_compressed cache). `messages` passed under the same
                session_id must be a growing, append-only prefix across
                calls (see dagc.rescue module docstring).

Previous versions of this server never forwarded session_id into
compress_any(), so EVERY call (regardless of trace_id) shared dagc's
"default" rescue session -- concurrent/multi-user traces silently
cross-contaminated each other's rescue state (GuaranteedSet, decayed
recurrence, shadow buffer). Fixed below: if the caller doesn't pass an
explicit session_id, we fall back to trace_id (not the global default),
so distinct traces get distinct rescue state without the caller having
to think about it. Pass enable_rescue=False for one-shot / non-session
compressions where this doesn't matter at all.
-----------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from mcp.server import MCPServer

import dagc
from dagc.compressor import DecisionLossError, decision_loss_frontier, verify_no_decision_loss
from dagc.rescue import reset_rescue_session

from .store import MessageStore

mcp = MCPServer("dagc")

# --------------------------------------------------------------------------
# Storage. Three tiers, checked in this order:
#   1. DAGC_REDIS_URL set        -> RedisBackend (durable + shared across
#                                    processes/machines -- real horizontal
#                                    scaling; needs store.py's RedisBackend,
#                                    see store_redis_addition.py)
#   2. DAGC_MCP_STORE_PATH set   -> FileBackend (durable, single-machine
#                                    only, no cross-process write locking --
#                                    see FileBackend's own docstring caveat)
#   3. neither set               -> DictBackend (in-memory, process-lifetime)
# Each tier falls back to the next rather than crashing the server on a
# bad/unreachable config, logging why.
# --------------------------------------------------------------------------
_store_backend = None
_redis_url = os.environ.get("DAGC_REDIS_URL")
_store_path = os.environ.get("DAGC_MCP_STORE_PATH")

if _redis_url:
    try:
        from dagc.store import RedisBackend
        _store_backend = RedisBackend(
            _redis_url, ttl_seconds=int(os.environ.get("DAGC_REDIS_TRACE_TTL_S", "604800")))
    except Exception as exc:  # noqa: BLE001
        print(f"[dagc-mcp] WARNING: could not initialize RedisBackend "
              f"({type(exc).__name__}: {exc}); falling back.")

if _store_backend is None and _store_path:
    try:
        from dagc.store import FileBackend
        _store_backend = FileBackend(_store_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[dagc-mcp] WARNING: could not initialize FileBackend at "
              f"{_store_path!r} ({type(exc).__name__}: {exc}); "
              f"falling back to in-memory MessageStore. Traces will NOT "
              f"survive a restart.")

_store = MessageStore(backend=_store_backend) if _store_backend is not None else MessageStore()

# --------------------------------------------------------------------------
# Rescue session backend. Redis-backed if DAGC_REDIS_URL is set (see
# rescue_redis.py) -- shares rescue state (ShadowBuffer, GuaranteedSet,
# decayed recurrence) across every server instance instead of each process
# keeping its own. Falls back to dagc's built-in in-process
# _rescue_sessions dict (single process only) if Redis isn't configured or
# fails to initialize.
# --------------------------------------------------------------------------
_redis_rescue_store = None
if _redis_url:
    try:
        from dagc.rescue_redis import RedisRescueSessionStore
        _redis_rescue_store = RedisRescueSessionStore(_redis_url)
    except Exception as exc:  # noqa: BLE001
        print(f"[dagc-mcp] WARNING: could not initialize RedisRescueSessionStore "
              f"({type(exc).__name__}: {exc}); rescue state will be "
              f"per-process only -- unsafe for multi-instance deployment.")

# --------------------------------------------------------------------------
# Monitoring / analytics. Built here rather than assumed from dagc internals
# -- diagnostics={} on compress()/compress_any() is a real, confirmed hook
# (see compress_dagc's diagnostics.update(...) calls), so this aggregates
# what that dict already exposes rather than inventing new dagc surface.
# Keyed by session_id so multi-user stats don't blend together. Thread-
# guarded since MCP servers may see concurrent tool calls.
# --------------------------------------------------------------------------
_stats_lock = threading.Lock()
_stats: Dict[str, Dict[str, Any]] = {}   # session_id -> aggregate dict
_call_log: List[Dict[str, Any]] = []      # bounded recent-call ring buffer
_CALL_LOG_MAX = 500

_analytics_path = os.environ.get("DAGC_MCP_ANALYTICS_PATH")


def _persist_analytics_snapshot() -> None:
    if not _analytics_path:
        return
    try:
        with open(_analytics_path, "w") as f:
            json.dump({"stats": _stats, "recent_calls": _call_log[-_CALL_LOG_MAX:]}, f)
    except Exception as exc:  # noqa: BLE001
        print(f"[dagc-mcp] WARNING: failed to persist analytics snapshot: {exc}")


def _record_call(session_id: str, trace_id: Optional[str], diagnostics: Dict[str, Any],
                  ok: bool, error: Optional[str] = None) -> None:
    orig = diagnostics.get("orig_tokens")
    out = diagnostics.get("output_tokens")
    reduction = (1 - out / orig) if (orig and out is not None and orig > 0) else None
    unrescuable = diagnostics.get("unrescuable_evictions") or []

    with _stats_lock:
        agg = _stats.setdefault(session_id, {
            "calls": 0, "errors": 0,
            "total_orig_tokens": 0, "total_output_tokens": 0,
            "total_unrescuable_evictions": 0, "total_stubbed": 0,
            "last_call_ts": None,
        })
        agg["calls"] += 1
        agg["last_call_ts"] = time.time()
        if not ok:
            agg["errors"] += 1
        if orig is not None:
            agg["total_orig_tokens"] += orig
        if out is not None:
            agg["total_output_tokens"] += out
        agg["total_unrescuable_evictions"] += len(unrescuable)
        agg["total_stubbed"] += diagnostics.get("n_stubbed", 0) or 0

        _call_log.append({
            "ts": agg["last_call_ts"], "session_id": session_id, "trace_id": trace_id,
            "ok": ok, "error": error, "orig_tokens": orig, "output_tokens": out,
            "achieved_reduction": reduction, "unrescuable_evictions": len(unrescuable),
        })
        if len(_call_log) > _CALL_LOG_MAX:
            del _call_log[: len(_call_log) - _CALL_LOG_MAX]

    _persist_analytics_snapshot()


@mcp.tool()
def dagc_compress(
    messages: List[Dict[str, Any]],
    target_reduction: Optional[float] = None,
    force_preserve: Optional[List[str]] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    enable_rescue: bool = True,
    strict_no_loss: bool = False,
) -> Dict[str, Any]:
    """
    Compress an agent/chat message trace, preserving every artifact any
    detected decision depends on (tool-call args, confirmed IDs, cited
    metrics/values).

    messages: list of role/content message dicts. Tolerant of a few
        common shapes via dagc's format layer.
    target_reduction: fraction of tokens to remove, 0-1 (default 0.87).
    force_preserve: extra literal strings to hard-guarantee survive, on
        top of whatever dagc's decision extraction already finds.
    trace_id: optional id to register this trace under for later
        dagc_get_original lookups.
    session_id: identifies which rescue session this call belongs to.
        `messages` must be a growing, append-only prefix across calls
        sharing a session_id. If omitted, defaults to trace_id (so
        distinct traces don't share rescue state); if neither is given,
        falls back to dagc's own "default" session -- fine for one-shot
        calls, NOT safe for concurrent multi-user traffic.
    enable_rescue: maintain cross-turn rescue state for this call. Set
        False for a pure one-shot compression with no session semantics.
    strict_no_loss: if True, raises (via DecisionLossError, surfaced as
        an error result) instead of silently shipping a trace where any
        decision-critical value is unrecoverable after every rescue pass.

    Returns {"compressed": [...], "trace_id": ..., "session_id": ...,
    "diagnostics": {...}} on success, or {"error": ...} on failure.
    """
    effective_session_id = session_id or trace_id or "default"
    diagnostics: Dict[str, Any] = {}

    try:
        if enable_rescue and _redis_rescue_store is not None:
            # Manual orchestration path: compress()'s own enable_rescue=True
            # is hardwired to the in-process _rescue_sessions dict (see
            # compress_dagc's `from .rescue import _run_rescue_for_call`) --
            # there's no hook to swap that store from outside. So when Redis
            # rescue is configured, we run rescue ourselves against
            # RedisRescueSessionStore and hand compress_any the resulting
            # force_preserve set with enable_rescue=False, which is
            # behaviorally equivalent to letting compress() do it in-process
            # -- same force_preserve semantics, just computed against shared
            # state instead of a local dict.
            budget_estimate = dagc.DAGCConfig().ABSOLUTE_BUDGET_TOKENS
            if budget_estimate is None:
                from dagc.compressor import _footprint_text, _tok
                orig_toks_estimate = sum(_tok(_footprint_text(m)) for m in messages)
                tr = target_reduction if target_reduction is not None else dagc.DAGCConfig().TARGET_REDUCTION
                budget_estimate = max(1, int(orig_toks_estimate * (1 - tr)))

            with _redis_rescue_store.session(effective_session_id) as ctx:
                rescue_fp, _events, unrescuable = ctx.process_new_messages(messages, budget_estimate)
                merged_fp = (set(force_preserve) if force_preserve else set()) | rescue_fp
                result = dagc.compress_any(
                    messages, target_reduction=target_reduction, force_preserve=merged_fp,
                    enable_rescue=False, diagnostics=diagnostics,
                    ASSERT_NO_DECISION_LOSS=strict_no_loss,
                )
                ctx.set_last_compressed(result)
            if unrescuable:
                diagnostics["unrescuable_evictions"] = unrescuable
        else:
            result = dagc.compress_any(
                messages,
                target_reduction=target_reduction,
                force_preserve=force_preserve,
                enable_rescue=enable_rescue,
                session_id=effective_session_id,
                diagnostics=diagnostics,
                ASSERT_NO_DECISION_LOSS=strict_no_loss,
            )
    except DecisionLossError as exc:
        _record_call(effective_session_id, trace_id, diagnostics, ok=False, error=str(exc))
        return {"error": f"decision loss under strict_no_loss: {exc}",
                "session_id": effective_session_id, "trace_id": trace_id}
    except TimeoutError as exc:
        _record_call(effective_session_id, trace_id, diagnostics, ok=False, error=str(exc))
        return {"error": f"rescue session lock timeout: {exc}",
                "session_id": effective_session_id, "trace_id": trace_id}

    if trace_id:
        _store.save_trace(trace_id, messages)

    _record_call(effective_session_id, trace_id, diagnostics, ok=True)

    return {
        "compressed": result,
        "trace_id": trace_id,
        "session_id": effective_session_id,
        "diagnostics": {
            "orig_tokens": diagnostics.get("orig_tokens"),
            "output_tokens": diagnostics.get("output_tokens"),
            "achieved_reduction": (
                1 - diagnostics["output_tokens"] / diagnostics["orig_tokens"]
                if diagnostics.get("orig_tokens") and diagnostics.get("output_tokens") is not None
                else None
            ),
            "n_stubbed": diagnostics.get("n_stubbed"),
            "unrescuable_evictions": diagnostics.get("unrescuable_evictions", []),
            "injection_filtered_msg_idxs": diagnostics.get("injection_filtered_msg_idxs", []),
        },
    }


@mcp.tool()
def dagc_evaluate(
    messages: List[Dict[str, Any]],
    decision_roles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Score whether a trace's decisions would survive compression, without
    any LLM call (deterministic scoring only). Returns DRR_soft,
    DRR_binary, RCI, and achieved token reduction -- use this to sanity-
    check a trace BEFORE compressing it for real, or to compare two
    candidate compression settings against each other.
    """
    from dagc_eval import compute_drr

    roles = tuple(decision_roles) if decision_roles else ("user", "assistant")
    result = compute_drr(messages, verbose=False, decision_roles=roles)
    return {k: v for k, v in result.items() if not str(k).startswith("_")}


@mcp.tool()
def dagc_get_original(trace_id: str, orig_idx: int) -> Dict[str, Any]:
    """
    Resolve a compressed message's `_orig_idx` back to its original,
    uncompressed content. Only works for trace_ids previously registered
    via dagc_compress(..., trace_id=...). Persistence across restarts
    depends on DAGC_MCP_STORE_PATH being set (see module header) --
    without it, this is in-memory only and resets when the server does.
    """
    msg = _store.get_message(trace_id, orig_idx)
    if msg is None:
        return {"found": False, "trace_id": trace_id, "orig_idx": orig_idx}
    return {"found": True, "trace_id": trace_id, "orig_idx": orig_idx, "message": msg}


@mcp.tool()
def dagc_resolve_trace(
    trace_id: str,
    compressed_messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Batch version of dagc_get_original. Two modes:

    - compressed_messages given: for each message (using its `_orig_idx`),
      return the original if the store has it, else the compressed
      message itself unchanged (best-effort resolve -- mirrors
      MessageStore.resolve()).
    - compressed_messages omitted: return every original message stored
      for trace_id, keyed by index (mirrors MessageStore.get_original_trace()).

    Same persistence caveat as dagc_get_original: only trace_ids saved via
    dagc_compress(..., trace_id=...) are resolvable, and only in-memory
    unless DAGC_MCP_STORE_PATH is set.
    """
    if compressed_messages is not None:
        resolved = _store.resolve(trace_id, compressed_messages)
        return {"trace_id": trace_id, "resolved": resolved}
    original = _store.get_original_trace(trace_id)
    return {"trace_id": trace_id, "original_by_idx": {str(k): v for k, v in original.items()},
            "found": bool(original)}


@mcp.tool()
def dagc_verify(
    messages: List[Dict[str, Any]],
    compressed: List[Dict[str, Any]],
    decision_roles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Independent, from-scratch check: re-extracts decisions from `messages`
    and confirms every critical value is actually recoverable from
    `compressed`. Does not reuse any state from the compress() call that
    produced `compressed` -- this is the same invariant check dagc's own
    test suite runs, exposed here so you can audit any compressed trace,
    including ones produced outside this server.

    Returns {"ok": bool, "missing": [[msg_idx, value], ...]}. Empty
    `missing` means the retention invariant holds.
    """
    roles = tuple(decision_roles) if decision_roles else ("user", "assistant")
    missing = verify_no_decision_loss(messages, compressed, decision_roles=roles)
    return {"ok": len(missing) == 0, "missing": [list(m) for m in missing]}


@mcp.tool()
def dagc_frontier(
    messages: List[Dict[str, Any]],
    budgets: List[int],
    decision_roles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Empirical rate-distortion curve: runs compression at each token budget
    in `budgets` and reports (output_tokens, decision_loss) for each --
    useful for picking a budget for a NEW trace type before committing to
    it in production, or for explaining to a user why a tighter budget
    costs more decision fidelity on their specific trace.
    """
    roles = tuple(decision_roles) if decision_roles else ("user", "assistant")
    points = decision_loss_frontier(messages, budgets=budgets, decision_roles=roles)
    return {"frontier": points}


@mcp.tool()
def dagc_reset_session(session_id: str) -> Dict[str, Any]:
    """
    Drop all rescue state (ShadowBuffer, RescueEngine, last-compressed
    cache) for this session_id. Call this when starting a new,
    unrelated conversation that happens to reuse a session_id --
    otherwise dagc's own length-shrank heuristic will eventually catch
    it, but resetting explicitly is cheaper and unambiguous.
    """
    reset_rescue_session(session_id)  # clears the in-process fallback dict too
    if _redis_rescue_store is not None:
        _redis_rescue_store.reset(session_id)
    with _stats_lock:
        _stats.pop(session_id, None)
    return {"reset": True, "session_id": session_id,
            "backend": "redis" if _redis_rescue_store is not None else "in-process"}


@mcp.tool()
def dagc_stats(session_id: Optional[str] = None, recent_calls: int = 20) -> Dict[str, Any]:
    """
    Monitoring/analytics: aggregated compression stats. Pass session_id
    to scope to one session (e.g. one user/conversation); omit it for a
    global summary across every session this server process has seen.

    recent_calls: how many of the most recent individual calls to include
    (across all sessions if session_id is omitted, else filtered to it).
    Set to 0 to omit the call log and get aggregates only.
    """
    with _stats_lock:
        if session_id is not None:
            agg = dict(_stats.get(session_id, {}))
            log = [c for c in _call_log if c["session_id"] == session_id]
        else:
            agg = {
                "sessions": len(_stats),
                "calls": sum(s["calls"] for s in _stats.values()),
                "errors": sum(s["errors"] for s in _stats.values()),
                "total_orig_tokens": sum(s["total_orig_tokens"] for s in _stats.values()),
                "total_output_tokens": sum(s["total_output_tokens"] for s in _stats.values()),
                "total_unrescuable_evictions": sum(
                    s["total_unrescuable_evictions"] for s in _stats.values()),
                "total_stubbed": sum(s["total_stubbed"] for s in _stats.values()),
            }
            log = list(_call_log)

        orig = agg.get("total_orig_tokens", 0)
        out = agg.get("total_output_tokens", 0)
        agg["overall_achieved_reduction"] = (1 - out / orig) if orig else None

        log = log[-recent_calls:] if recent_calls > 0 else []

    return {"session_id": session_id, "aggregate": agg, "recent_calls": log,
            "persisted_to": _analytics_path}


def run():
    mcp.run()


if __name__ == "__main__":
    run()