"""
DAGC <-> LangGraph adapter.

`examples/langgraph_style_node.py` in this repo shows the idea using plain
dicts. This module makes it correct for real LangGraph graphs, which is
subtler than it looks: LangGraph's default `MessagesState` uses the
`add_messages` reducer, which *appends* whatever a node returns for the
"messages" key rather than replacing it. A node that just does

    state["messages"] = compress(state["messages"])
    return state

will not shrink the conversation at all once the graph merges the update —
the old messages stay and the compressed ones get appended alongside them.

`make_compression_node` handles this correctly by emitting `RemoveMessage`
entries for every original message id alongside the compressed replacements,
which is LangGraph's documented pattern for actually deleting history.

Requires `langgraph` and `langchain-core`
(``pip install "dagc[langgraph]"``).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from dagc import DAGCConfig, compress

from .langchain import dagc_messages_to_lc, lc_messages_to_dagc

try:
    from langgraph.graph.message import RemoveMessage, add_messages
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "dagc.integrations.langgraph requires langgraph. "
        "Install it with: pip install -e '.[langgraph]'"
    ) from e


def make_compression_node(
    target_reduction: float = 0.5,
    messages_key: str = "messages",
    cfg: Optional[DAGCConfig] = None,
    session_id: str = "default",
    enable_rescue: bool = True,
    min_messages_to_compress: int = 4,
    **overrides: Any,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Build a node function you can add to a `StateGraph`.

        from langgraph.graph import StateGraph, MessagesState
        from dagc.integrations.langgraph import make_compression_node

        graph = StateGraph(MessagesState)
        graph.add_node("compress", make_compression_node(target_reduction=0.7))
        graph.add_edge("compress", "agent")

    The returned node replaces `state[messages_key]` with its compressed
    form using `RemoveMessage` + new messages, which is the update shape
    `add_messages` needs to actually shrink history instead of appending
    to it.

    Args:
        target_reduction: fraction of tokens to remove (0-1).
        messages_key: state key holding the message list (default matches
            LangGraph's built-in `MessagesState`).
        cfg: a full DAGCConfig for advanced tuning.
        session_id: forwarded to dagc's rescue engine; use a distinct id
            per independent conversation/thread if you run multiple
            threads through the same process (e.g. the LangGraph
            `thread_id` from config is a good choice — see
            `make_compression_node_from_config` below).
        min_messages_to_compress: skip compression below this many
            messages, so short conversations aren't touched every turn.
        **overrides: any other DAGCConfig field.
    """

    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get(messages_key, [])
        if len(messages) < min_messages_to_compress:
            return {}

        # Messages only get an `.id` once they've passed through the
        # add_messages reducer at least once. A node invoked directly
        # (unit tests) or on the very first turn of a fresh conversation
        # can see id=None on every message -- canonicalize the same way
        # LangGraph itself would, so RemoveMessage always has a real id
        # to target.
        messages = add_messages([], messages)

        dagc_msgs = lc_messages_to_dagc(messages)
        compressed = compress(
            dagc_msgs,
            target_reduction=target_reduction,
            cfg=cfg,
            session_id=session_id,
            enable_rescue=enable_rescue,
            **overrides,
        )
        new_messages = dagc_messages_to_lc(compressed, messages)

        removals = [RemoveMessage(id=m.id) for m in messages]
        return {messages_key: removals + new_messages}

    return node


def make_compression_node_from_config(
    target_reduction: float = 0.5,
    messages_key: str = "messages",
    cfg: Optional[DAGCConfig] = None,
    enable_rescue: bool = True,
    min_messages_to_compress: int = 4,
    **overrides: Any,
) -> Callable[[Dict[str, Any], Any], Dict[str, Any]]:
    """Same as `make_compression_node`, but derives `session_id` from the
    LangGraph run's `thread_id` automatically, so multi-thread deployments
    (one process serving many conversations) don't need a manual session_id
    per thread.

        graph.add_node("compress", make_compression_node_from_config())

    LangGraph passes `config` as the node's second positional argument when
    the node function declares it; StateGraph detects this by signature.
    """

    def node(state: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        messages = state.get(messages_key, [])
        if len(messages) < min_messages_to_compress:
            return {}

        thread_id = "default"
        if config is not None:
            thread_id = (config.get("configurable") or {}).get("thread_id", "default")

        # See make_compression_node for why this canonicalization step
        # is required before RemoveMessage can target every message.
        messages = add_messages([], messages)

        dagc_msgs = lc_messages_to_dagc(messages)
        compressed = compress(
            dagc_msgs,
            target_reduction=target_reduction,
            cfg=cfg,
            session_id=str(thread_id),
            enable_rescue=enable_rescue,
            **overrides,
        )
        new_messages = dagc_messages_to_lc(compressed, messages)

        removals = [RemoveMessage(id=m.id) for m in messages]
        return {messages_key: removals + new_messages}

    return node
