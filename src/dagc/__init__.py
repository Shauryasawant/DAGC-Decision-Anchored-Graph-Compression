"""
dagc — Decision-Anchored Graph Compression.

Compress long agent/chat message histories while guaranteeing that every
artifact a decision depends on (tool-call arguments, confirmed IDs, cited
metrics) survives compression. Zero required dependencies, zero LLM calls.

    from dagc import compress
    compressed = compress(messages, target_reduction=0.85)
    response = client.chat.completions.create(model="gpt-4", messages=compressed)

Bring your own tokenizer/embedder for production-grade quality:

    import dagc
    from dagc.adapters import TiktokenTokenizer, SentenceTransformerEmbedder
    dagc.configure(
        tokenizer=TiktokenTokenizer("cl100k_base"),
        embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
    )

"""
from .compressor import compress, compress_any, compress_dagc, DAGCConfig, DAGC_CFG
from .extraction import extract_decisions
from .graph import (
    build_dependency_graph, attach_dependencies, compute_rci, compute_chain_rci,
    CausalMessageGraph, CausalGraphConfig, SpectralCompressor,
)
from .config import configure, Runtime, Tokenizer, Embedder, runtime
from .formats import (
    normalize_message, normalize_trace,
    denormalize_message, denormalize_trace,
    register_adapter as register_format_adapter,
)
from .schema import register_adapter, to_dagc_format

__version__ = "0.2.0"

__all__ = [
    "compress", "compress_any", "compress_dagc", "DAGCConfig", "DAGC_CFG",
    "extract_decisions",
    "build_dependency_graph", "attach_dependencies", "compute_rci", "compute_chain_rci",
    "CausalMessageGraph", "CausalGraphConfig", "SpectralCompressor",
    "configure", "Runtime", "Tokenizer", "Embedder", "runtime",
    "normalize_message", "normalize_trace",
    "denormalize_message", "denormalize_trace",
    "register_adapter", "register_format_adapter", "to_dagc_format",
]
