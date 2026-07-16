"""
BYOK LLM client protocol for dagc_eval. dagc_eval only needs an LLM to
*validate* that a compressed trace still lets a reader reconstruct each
decision -- the compressor itself (the `dagc` package) never calls one.

Bring any provider by implementing `.complete(system, user, temperature,
max_tokens) -> str`. Ready-made thin wrappers for a few common SDKs are
below (each lazily imported so installing dagc_eval doesn't require every
vendor SDK).
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 400) -> str: ...


class OpenAIChatClient:
    """Wrap your own openai.OpenAI(...) client. dagc_eval never sees your key."""
    def __init__(self, client, model: str = "gpt-4o-mini"):
        self._client = client
        self._model = model

    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 400) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


class AnthropicChatClient:
    """Wrap your own anthropic.Anthropic(...) client."""
    def __init__(self, client, model: str = "claude-haiku-4-5-20251001"):
        self._client = client
        self._model = model

    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 400) -> str:
        resp = self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


class MistralChatClient:
    """Wrap your own mistralai client (kept for parity with the original
    prototype, which used Mistral)."""
    def __init__(self, client, model: str = "mistral-small-latest"):
        self._client = client
        self._model = model

    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 400) -> str:
        resp = self._client.chat.complete(
            model=self._model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
