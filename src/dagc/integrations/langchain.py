"""
DAGC <-> LangChain adapter.

Converts LangChain BaseMessage objects to dagc's canonical dict format,
compresses with dagc.compress, and converts the survivors back to
BaseMessage objects with their original type, id, name, and tool_calls
preserved.

Two ways to use this:

1. Drop-in Runnable for LCEL pipelines::

    from dagc.integrations.langchain import DAGCCompressor
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4.1")
    chain = DAGCCompressor(target_reduction=0.85) | model

    chain.invoke(messages)  # messages compressed before hitting the model

2. Wrap an existing chat model so every call is compressed transparently::

    from dagc.integrations.langchain import wrap_chat_model

    compressed_model = wrap_chat_model(ChatOpenAI(model="gpt-4.1"), target_reduction=0.85)
    compressed_model.invoke(messages)

Requires `langchain-core` (``pip install "dagc[langchain]"``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from dagc import DAGCConfig, compress

try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.runnables import Runnable, RunnableConfig
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "dagc.integrations.langchain requires langchain-core. "
        "Install it with: pip install -e '.[langchain]'"
    ) from e


_ROLE_TO_LC = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
    "function": FunctionMessage,
}

_LC_TYPE_TO_ROLE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
    "chat": "chat",
}


def lc_messages_to_dagc(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Convert a list of LangChain BaseMessage objects to dagc's message dicts.

    Each dagc dict carries a `_lc_index` so the compressed output can be
    mapped back to the original message object (and its id/type) without
    guessing from content alone.
    """
    out = []
    for i, m in enumerate(messages):
        role = _LC_TYPE_TO_ROLE.get(getattr(m, "type", None), "user")
        d: Dict[str, Any] = {
            "role": role,
            "content": m.content if isinstance(m.content, str) else str(m.content),
            "_lc_index": i,
        }
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            # LangChain AIMessage.tool_calls: [{"name", "args", "id"}, ...]
            # dagc reads either {"name","args"} or OpenAI-style {"function":{...}}
            d["tool_calls"] = list(tool_calls)
            d["tool_call"] = tool_calls[0]
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            d["tool_call_id"] = tool_call_id
        name = getattr(m, "name", None)
        if name:
            d["name"] = name
        out.append(d)
    return out


def dagc_messages_to_lc(
    compressed: List[Dict[str, Any]], original: List[BaseMessage]
) -> List[BaseMessage]:
    """Rehydrate dagc's compressed dicts back into LangChain BaseMessage objects.

    Uses `_lc_index` (falling back to `_orig_idx`, which dagc always sets)
    to recover the original message's type, id, and metadata, then swaps
    in the (possibly shortened) content dagc produced.
    """
    out = []
    for d in compressed:
        idx = d.get("_lc_index", d.get("_orig_idx"))
        content = d.get("content", "")

        if idx is not None and 0 <= idx < len(original):
            src = original[idx]
            new_msg = src.model_copy(update={"content": content})
            out.append(new_msg)
            continue

        # Fallback: no reliable index (shouldn't normally happen), rebuild
        # from role.
        cls = _ROLE_TO_LC.get(d.get("role", "user"), HumanMessage)
        kwargs: Dict[str, Any] = {"content": content}
        if d.get("name"):
            kwargs["name"] = d["name"]
        if d.get("tool_call_id"):
            kwargs["tool_call_id"] = d["tool_call_id"]
        out.append(cls(**kwargs))
    return out


class DAGCCompressor(Runnable):
    """LCEL Runnable: `List[BaseMessage] -> List[BaseMessage]`, compressed.

    Pipe it directly in front of a chat model::

        chain = DAGCCompressor(target_reduction=0.85) | model
    """

    def __init__(
        self,
        target_reduction: float = 0.85,
        cfg: Optional[DAGCConfig] = None,
        session_id: str = "default",
        enable_rescue: bool = True,
        **overrides: Any,
    ):
        self.target_reduction = target_reduction
        self.cfg = cfg
        self.session_id = session_id
        self.enable_rescue = enable_rescue
        self.overrides = overrides

    def invoke(
        self, input: List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[BaseMessage]:
        if not input:
            return input
        dagc_msgs = lc_messages_to_dagc(input)
        compressed = compress(
            dagc_msgs,
            target_reduction=self.target_reduction,
            cfg=self.cfg,
            session_id=self.session_id,
            enable_rescue=self.enable_rescue,
            **self.overrides,
        )
        return dagc_messages_to_lc(compressed, input)

    async def ainvoke(
        self, input: List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[BaseMessage]:
        # dagc.compress is CPU-bound and synchronous (no network/LLM calls),
        # so async just delegates straight through.
        return self.invoke(input, config, **kwargs)


def wrap_chat_model(model, target_reduction: float = 0.85, **compressor_kwargs):
    """Return `DAGCCompressor | model`, so `.invoke(messages)` compresses first.

    Convenience wrapper for people who don't want to think about LCEL:

        compressed_model = wrap_chat_model(ChatOpenAI(model="gpt-4.1"))
        compressed_model.invoke(messages)
    """
    return DAGCCompressor(target_reduction=target_reduction, **compressor_kwargs) | model
