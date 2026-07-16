import os

import numpy as np

import dagc
from dagc.adapters import TiktokenTokenizer
from mistralai.client import Mistral

# Set MISTRAL_API_KEY in your environment before running this example.

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError(
        "Set the MISTRAL_API_KEY environment variable before running this example."
    )

client = Mistral(api_key=MISTRAL_API_KEY)


class MistralEmbedder:
    def __init__(self, client, model="mistral-embed"):
        self._client = client
        self._model = model

    def encode(self, texts, normalize_embeddings=False):
        resp = self._client.embeddings.create(
            model=self._model,
            inputs=list(texts),
        )

        vecs = np.array(
            [d.embedding for d in resp.data],
            dtype=np.float32,
        )

        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs = vecs / norms

        return vecs


dagc.configure(
    tokenizer=TiktokenTokenizer("cl100k_base"),
    embedder=MistralEmbedder(client),
)

from dagc import compress


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Experiment ID is EXP-4471."
    },
    {
        "role": "assistant",
        "content": "Okay, I'll remember EXP-4471."
    },
    {
        "role": "user",
        "content": "The results file is results_exp4471.json."
    },
    {
        "role": "assistant",
        "content": "Got it."
    },
    {
        "role": "user",
        "content": "Recommend the best vector database."
    },
    {
        "role": "assistant",
        "content": "Qdrant performs best."
    },
    {
        "role": "user",
        "content": "When answering, include the experiment ID and filename."
    }
]

print("=" * 70)
print("ORIGINAL")
print("=" * 70)

for m in messages:
    print(f"{m['role']:10} {m['content']}")

compressed = compress(
    messages,
    target_reduction=0.85,
)

print("\n")
print("=" * 70)
print("COMPRESSED")
print("=" * 70)

for m in compressed:
    print(f"{m['role']:10} {m['content']}")

combined = " ".join(m["content"] for m in compressed)

print("\n")
print("=" * 70)
print("CHECK")
print("=" * 70)

print("EXP-4471 preserved?           ", "EXP-4471" in combined)
print("results_exp4471.json present?", "results_exp4471.json" in combined)
print("Qdrant preserved?            ", "Qdrant" in combined)

print("\n")
print("=" * 70)

if "EXP-4471" not in combined:
    print("⚠️ Compressor removed the experiment ID.")

if "results_exp4471.json" not in combined:
    print("⚠️ Compressor removed the filename.")

print("=" * 70)
