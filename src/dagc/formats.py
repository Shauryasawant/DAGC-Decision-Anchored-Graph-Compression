"""Helpers for normalizing dagc traces across a few different message formats.

The point here is to keep the trace handling in one place:
- turn odd message shapes into dagc's canonical dict form
- unwrap the common envelope wrappers
- convert compressed output back to the caller's original shape
- let callers register their own per-message adapters
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

REGISTRY: Dict[str, Callable[[Any], Dict]] = {}


def register_adapter(name: str):
    """Register a custom adapter for a specific schema name."""

    def _wrap(fn: Callable[[Any], Dict]) -> Callable[[Any], Dict]:
        REGISTRY[name] = fn
        return fn

    return _wrap


def _flatten_content_blocks(content: Any) -> str:
    """Best-effort conversion of content into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return _flatten_content_blocks(content["content"])
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in ("text", None) and "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(_flatten_content_blocks(block["content"]))
        return " ".join(p for p in parts if p)
    return str(content)


def _normalize_tool_calls(msg: Dict) -> Optional[Dict]:
    """Return the primary tool call in a best-effort way."""
    tc = msg.get("tool_call")
    if isinstance(tc, dict):
        return tc

    tcs = msg.get("tool_calls")
    if isinstance(tcs, list) and tcs:
        first = tcs[0]
        if isinstance(first, dict):
            return first

    return None


def _normalize_role(msg: Dict, default_role: str = "unknown") -> str:
    """Best-effort role extraction from common aliases."""
    if isinstance(msg, dict):
        if isinstance(msg.get("role"), str) and msg["role"].strip():
            return msg["role"]
        for alt_key in ("sender", "speaker", "from"):
            value = msg.get(alt_key)
            if isinstance(value, str) and value.strip():
                return value
        return default_role
    return default_role


def normalize_message(msg: Any, default_role: str = "unknown") -> Dict:
    """Normalize one message into dagc's canonical dict shape."""
    if not isinstance(msg, dict):
        return {
            "role": default_role,
            "content": str(msg),
            "tool_call": None,
            "_extra": {"_original_non_dict": True},
        }

    role = _normalize_role(msg, default_role=default_role)
    content = _flatten_content_blocks(msg.get("content", msg.get("text", "")))
    tool_call = _normalize_tool_calls(msg)

    known_keys = {"role", "content", "text", "tool_call", "tool_calls",
                  "sender", "speaker", "from"}
    extra = {k: v for k, v in msg.items() if k not in known_keys}

    out = {"role": role, "content": content, "tool_call": tool_call}
    if extra:
        out["_extra"] = extra
    return out


def normalize_trace(raw: Union[List[Any], Dict, Any]) -> List[Dict]:
    """Normalize either a bare list or an envelope dict into a message list."""
    if isinstance(raw, dict):
        for key in ("messages", "trace", "conversation", "turns"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            return []

    if not isinstance(raw, list):
        return []

    return [normalize_message(message) for message in raw]


def _rehydrate_content(original_content: Any, new_text: str) -> Any:
    """Put compressed text back into the same content shape as the original."""
    if original_content is None:
        return None
    if isinstance(original_content, str):
        return new_text
    if isinstance(original_content, dict) and "text" in original_content:
        out = dict(original_content)
        out["text"] = new_text
        return out
    if isinstance(original_content, list):
        new_blocks = []
        replaced = False
        for block in original_content:
            is_text_block = (
                isinstance(block, dict)
                and block.get("type") in ("text", None)
                and "text" in block
            )
            if is_text_block and not replaced:
                nb = dict(block)
                nb["text"] = new_text
                new_blocks.append(nb)
                replaced = True
            elif is_text_block:
                continue
            else:
                new_blocks.append(block)
        if not replaced and new_text:
            new_blocks.insert(0, {"type": "text", "text": new_text})
        return new_blocks
    return new_text


def _rehydrate_tool_call(original_msg: Dict, new_tool_call: Optional[Dict]) -> Dict:
    """Put the compressed tool call back into the original message shape."""
    out = dict(original_msg)
    if new_tool_call is None:
        return out

    if "tool_call" in original_msg:
        out["tool_call"] = new_tool_call
        return out

    if isinstance(original_msg.get("tool_calls"), list) and original_msg["tool_calls"]:
        new_list = list(original_msg["tool_calls"])
        new_list[0] = new_tool_call
        out["tool_calls"] = new_list
        return out

    out["tool_call"] = new_tool_call
    return out


def denormalize_message(original_raw: Any, compressed_canonical: Dict) -> Any:
    """Return the original message shape with compressed content/tool calls swapped in."""
    if not isinstance(original_raw, dict):
        return compressed_canonical.get("content", str(original_raw))

    out = dict(original_raw)
    out["content"] = _rehydrate_content(
        original_raw.get("content"), compressed_canonical.get("content", "")
    )
    out = _rehydrate_tool_call(out, compressed_canonical.get("tool_call"))
    return out


def denormalize_trace(original_raw_messages: List[Any], compressed_canonical_messages: List[Dict]) -> List[Any]:
    """Return compressed messages in the same wire shape as the original input."""
    out = []
    for compressed in compressed_canonical_messages:
        idx = compressed.get("_orig_idx")
        if idx is None or not (0 <= idx < len(original_raw_messages)):
            out.append({k: v for k, v in compressed.items() if k != "_orig_idx"})
            continue
        out.append(denormalize_message(original_raw_messages[idx], compressed))
    return out


__all__ = [
    "REGISTRY",
    "normalize_message",
    "normalize_trace",
    "denormalize_message",
    "denormalize_trace",
    "register_adapter",
]
