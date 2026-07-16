"""
Optional, real-quality Tokenizer/Embedder adapters. Each is lazily
imported so that importing `dagc.adapters` never requires having every
vendor SDK installed -- only the one you actually instantiate.

Install what you need:
    pip install dagc[tiktoken]
    pip install dagc[sentence-transformers]
    pip install dagc[openai]

Or write your own -- any object with .encode()/.decode() (Tokenizer) or
.encode() -> np.ndarray (Embedder) works, see dagc.config.Tokenizer /
dagc.config.Embedder for the exact protocol.
"""
from __future__ import annotations
from typing import List
import numpy as np


class TiktokenTokenizer:
    """Real BPE token counts via OpenAI's tiktoken. pip install tiktoken."""
    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "TiktokenTokenizer needs tiktoken. Install it with "
                "pip install -e '.[tiktoken]'"
            ) from e
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(str(text))

    def decode(self, tokens: List[int]) -> str:
        return self._enc.decode(tokens)


class SentenceTransformerEmbedder:
    """Local, free, good-quality embeddings. pip install sentence-transformers."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbedder needs sentence-transformers. "
                "Install it with pip install -e '.[sentence-transformers]'"
            ) from e
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], normalize_embeddings: bool = False) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=normalize_embeddings)
        )


class OpenAIEmbedder:
    """
    BYOK wrapper around any OpenAI-compatible embeddings endpoint.
    Pass your own already-configured client (openai.OpenAI(...)) --
    dagc never sees or stores your API key.
    """
    def __init__(self, client, model: str = "text-embedding-3-small"):
        self._client = client
        self._model = model

    def encode(self, texts: List[str], normalize_embeddings: bool = False) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs


class CohereEmbedder:
    """BYOK wrapper around a Cohere client (cohere.Client(...))."""
    def __init__(self, client, model: str = "embed-english-v3.0",
                 input_type: str = "search_document"):
        self._client = client
        self._model = model
        self._input_type = input_type

    def encode(self, texts: List[str], normalize_embeddings: bool = False) -> np.ndarray:
        resp = self._client.embed(texts=list(texts), model=self._model,
                                   input_type=self._input_type)
        vecs = np.array(resp.embeddings, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs
