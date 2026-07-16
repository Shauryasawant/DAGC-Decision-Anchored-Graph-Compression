"""
Convert arbitrary orchestrator/agent traces into dagc's minimal message
schema: {'role': str, 'content': str, 'tool_call': Optional[dict]}.
No dependency on any agent framework -- this is dagc's own format.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

_REGISTRY: Dict[str, Callable[[Any], Dict]] = {}


def _unwrap_events(events: Any) -> List[Any]:
    """Accept a bare trace, a list, a tuple, or a common envelope dict."""
    if isinstance(events, dict):
        for key in ('messages', 'trace', 'conversation', 'turns', 'events'):
            value = events.get(key)
            if isinstance(value, list):
                return value
        if any(k in events for k in ('role', 'content', 'text', 'message', 'speaker', 'sender')):
            return [events]
        return []
    if isinstance(events, (list, tuple)):
        return list(events)
    if events is None:
        return []
    return [events]


def _coerce_content(content: Any) -> str:
    """Best-effort normalization for common content shapes."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if 'text' in content:
            return str(content['text'])
        if 'content' in content:
            return _coerce_content(content['content'])
        return ''
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if 'text' in block:
                    parts.append(str(block['text']))
                elif 'content' in block:
                    parts.append(_coerce_content(block['content']))
                elif block.get('type') in ('text', None) and 'text' in block:
                    parts.append(str(block['text']))
        return ' '.join(p for p in parts if p)
    return str(content)


def _normalize_tool_call(tool_call: Any) -> Optional[Dict]:
    """Normalize common tool-call shapes to a simple {'name', 'args'} dict."""
    if tool_call is None:
        return None

    if isinstance(tool_call, dict):
        if 'function' in tool_call and isinstance(tool_call['function'], dict):
            fn = tool_call['function']
            return {
                'name': fn.get('name'),
                'args': fn.get('arguments', {}),
            }
        if 'name' in tool_call:
            return {
                'name': tool_call.get('name'),
                'args': tool_call.get('args', tool_call.get('arguments', {})),
            }
        return tool_call

    if isinstance(tool_call, list) and tool_call:
        return _normalize_tool_call(tool_call[0])

    return {'name': str(tool_call), 'args': {}}


def _normalize_role(event: Any, default_role: str) -> str:
    if isinstance(event, dict):
        for key in ('role', 'sender', 'speaker', 'from'):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return default_role

    role = getattr(event, 'role', None) or getattr(event, 'type', None)
    if role:
        return str(role)
    for key in ('sender', 'speaker', 'from'):
        value = getattr(event, key, None)
        if value:
            return str(value)
    return default_role


def _normalize_message(event: Any, default_role: str = 'assistant') -> Dict:
    """Best-effort conversion for a single event into dagc's canonical shape."""
    if isinstance(event, dict):
        role = _normalize_role(event, default_role)
        content = _coerce_content(event.get('content', event.get('text', '')))
        tool_call = _normalize_tool_call(event.get('tool_call') or event.get('tool_calls'))
        return {'role': str(role), 'content': content, 'tool_call': tool_call}

    if isinstance(event, (tuple, list)) and len(event) >= 2:
        role, text = event[0], event[1]
        return {'role': str(role), 'content': str(text), 'tool_call': None}

    if isinstance(event, str):
        return {'role': default_role, 'content': event, 'tool_call': None}

    if event is None:
        return {'role': default_role, 'content': '', 'tool_call': None}

    role = _normalize_role(event, default_role)
    content = getattr(event, 'content', None)
    if content is None:
        content = getattr(event, 'text', '')
    tool_call = getattr(event, 'tool_call', None) or getattr(event, 'tool_calls', None)
    return {
        'role': str(role),
        'content': _coerce_content(content),
        'tool_call': _normalize_tool_call(tool_call),
    }


def register_adapter(name: str):
    """Decorator to register a custom converter for your orchestrator's
    event format. The function receives one raw event and must return
    a dict with at least 'role' and 'content'."""
    def _wrap(fn):
        _REGISTRY[name] = fn
        return fn
    return _wrap


def _passthrough(event) -> Dict:
    if isinstance(event, dict) and 'content' in event:
        return {
            'role': event.get('role', 'assistant'),
            'content': str(event.get('content', '')),
            'tool_call': event.get('tool_call'),
        }
    raise ValueError(f"Cannot auto-convert event: {event!r}")


def _from_tuple(event) -> Dict:
    role, text = event[0], event[1]
    return {'role': role, 'content': str(text), 'tool_call': None}


def _from_object(event) -> Dict:
    # Support common message objects as well as dictionaries.
    role = getattr(event, 'role', None) or getattr(event, 'type', 'assistant')
    content = getattr(event, 'content', '')
    tool_call = getattr(event, 'tool_call', None) or getattr(event, 'tool_calls', None)
    if isinstance(tool_call, list) and tool_call:
        tool_call = tool_call[0]
    return {'role': str(role), 'content': str(content), 'tool_call': tool_call}


def to_dagc_format(events: List[Any], schema: str = 'auto',
                    default_role: str = 'assistant') -> List[Dict]:
    """
    Convert a raw trace (any shape) into dagc's message format.

    schema:
      "auto"   - infer per-event: dict, tuple, object, or envelope
      "string" - flat list of strings, alternating user/assistant
                 (or all `default_role` if only one side exists)
      <name>   - a custom converter registered via @register_adapter
                 or one of the built-in format aliases: openai, anthropic,
                 langchain, langgraph, autogen.
    """
    normalized_events = _unwrap_events(events)

    if schema in _REGISTRY:
        return [_REGISTRY[schema](e) for e in normalized_events]

    if schema == 'string':
        out = []
        for i, text in enumerate(normalized_events):
            role = 'user' if i % 2 == 0 else 'assistant'
            out.append({'role': role, 'content': str(text), 'tool_call': None})
        return out

    if schema in {'openai', 'anthropic', 'langchain', 'langgraph', 'autogen', 'auto'}:
        return [_normalize_message(e, default_role=default_role) for e in normalized_events]

    # Unknown schemas use best-effort normalization.
    return [_normalize_message(e, default_role=default_role) for e in normalized_events]
