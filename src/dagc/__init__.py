"""
dagc — Decision-Anchored/Aware Graph Compression.

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
from .sv_dagc import compress_dagc_sv
from .rationale_ext import (
    extract_rationale_candidates, inject_rationale_stubs, inject_dropped_rationale_stubs,
)
from .graph import (
    build_dependency_graph, attach_dependencies, compute_rci, compute_chain_rci,
    CausalMessageGraph, CausalGraphConfig, SpectralCompressor,
)
from .config import configure, Runtime, Tokenizer, Embedder, runtime
from .convmem import Memory
from .formats import (
    normalize_message, normalize_trace,
    denormalize_message, denormalize_trace,
    register_adapter as register_format_adapter,
)
from .schema import register_adapter, to_dagc_format
from .rescue import RescueEngine, ShadowBuffer, reset_rescue_session

__version__ = "0.1.8"

__all__ = [
    "compress", "compress_any", "compress_dagc", "DAGCConfig", "DAGC_CFG",
    "compress_dagc_sv",
    "extract_decisions",
    "extract_rationale_candidates", "inject_rationale_stubs", "inject_dropped_rationale_stubs",
    "build_dependency_graph", "attach_dependencies", "compute_rci", "compute_chain_rci",
    "CausalMessageGraph", "CausalGraphConfig", "SpectralCompressor",
    "configure", "Runtime", "Tokenizer", "Embedder", "runtime",
    "Memory",
    "normalize_message", "normalize_trace",
    "denormalize_message", "denormalize_trace",
    "register_adapter", "register_format_adapter", "to_dagc_format",
    "RescueEngine", "ShadowBuffer", "reset_rescue_session",
]