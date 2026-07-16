"""
BYOK runtime configuration for dagc.

dagc needs exactly two capabilities to do its job: a tokenizer (to count/
truncate tokens) and an embedder (to score semantic relevance for MMR
selection). Neither requires an LLM call. Both are swappable so you can
plug in whatever you already have -- OpenAI, Cohere, sentence-transformers,
tiktoken, your own in-house model -- without touching compressor code.

Zero-dependency defaults are provided so `import dagc; dagc.compress(msgs)`
works out of the box with no API keys and no extra installs. Quality goes
up if you swap in a real tokenizer/embedder via `dagc.configure(...)`, but
nothing is required to get started.
"""
from __future__ import annotations
import hashlib
import re
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...


@runtime_checkable
class Embedder(Protocol):
    def encode(self, texts: List[str], normalize_embeddings: bool = False) -> np.ndarray: ...


class SimpleWordTokenizer:
    """
    Dependency-free fallback tokenizer. Treats whitespace-delimited words
    as the countable unit instead of real BPE tokens. Token *counts* will
    differ somewhat from a real tokenizer's, but every downstream
    consumer (budgeting, head/tail truncation, dedup) only needs
    encode/decode to round-trip consistently -- which this does exactly.

    For accurate token-budget math against a specific model, install the
    `tiktoken` extra and use `dagc.adapters.TiktokenTokenizer` instead.
    """
    _RE = re.compile(r'\S+|\s+')

    def encode(self, text: str) -> List[str]:
        return self._RE.findall(str(text))

    def decode(self, tokens: List[str]) -> str:
        return ''.join(tokens)


class HashingEmbedder:
    """
    Dependency-free fallback embedder using the hashing trick (signed
    feature hashing into a fixed-size bag-of-words vector). No model
    download, no API call, no external package.

    This is meaningfully lower quality than a real sentence embedding
    model for semantic relevance ranking (Phase 2 / MMR in the
    compressor) -- artifact-anchored decision preservation (Phase 1,
    the hard guarantee) is UNAFFECTED since that phase is purely
    string-matching and never touches embeddings.

    For production quality, install `sentence-transformers` and use
    `dagc.adapters.SentenceTransformerEmbedder`, or wrap your own
    OpenAI/Cohere/Voyage embedding client -- see dagc.adapters for
    ready-made examples of both.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: List[str], normalize_embeddings: bool = False) -> np.ndarray:
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for w in re.findall(r'[a-z0-9]+', str(t).lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                v[idx] += sign
            n = np.linalg.norm(v)
            out.append(v / n if n > 0 else v)
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)


class Runtime:
    def __init__(self, tokenizer: Optional[Tokenizer] = None,
                 embedder: Optional[Embedder] = None):
        self.tokenizer: Tokenizer = tokenizer or SimpleWordTokenizer()
        self.embedder: Embedder = embedder or HashingEmbedder()


# Keep this object stable: other modules import it directly.
runtime = Runtime()


def configure(tokenizer: Optional[Tokenizer] = None,
              embedder: Optional[Embedder] = None) -> None:
    """
    Swap the global tokenizer/embedder used by dagc.compress(). Call once
    at startup, e.g.:

        import dagc
        from dagc.adapters import TiktokenTokenizer, OpenAIEmbedder

        dagc.configure(
            tokenizer=TiktokenTokenizer("cl100k_base"),
            embedder=OpenAIEmbedder(client=my_openai_client),
        )

    Anything implementing the Tokenizer / Embedder protocol works --
    no dependency on a specific vendor.
    """
    if tokenizer is not None:
        runtime.tokenizer = tokenizer
    if embedder is not None:
        runtime.embedder = embedder
