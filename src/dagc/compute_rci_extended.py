"""
test_rci_extended.py — sanity check that compute_rci_extended's
recoverability fix doesn't regress RCI on a real trace.
Run: python -m dagc.test_rci_extended <trace_name>
"""
import sys
import dagc
from dagc.adapters import TiktokenTokenizer, SentenceTransformerEmbedder
from .calibrate_filler_scorer import load_traces, TRACES_PATH_DEFAULT
from .compressor import compress_dagc, _decision_critical_values
from .extraction import extract_decisions
from .graph_ext import compute_rci_extended, build_dependency_graph_extended

dagc.configure(
    tokenizer=TiktokenTokenizer("cl100k_base"),
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
)

name = sys.argv[1] if len(sys.argv) > 1 else None
all_traces = load_traces(TRACES_PATH_DEFAULT)

if name is None:
    name = next(iter(all_traces))
    print(f"[no trace name given, using first available: {name!r}]")

messages = all_traces[name]
result = compress_dagc(messages)
compressed = result[0] 
decisions = extract_decisions(messages)

result = compute_rci_extended(messages, compressed, decisions, include_critical_values=True)
print(f"Trace: {name}")
print(f"RCI: {result['RCI']}")
print(f"edges_total: {result['edges_total']}  edges_preserved: {result['edges_preserved']}")

# Show any edge that's marked NOT preserved, so you can eyeball whether
# it's a genuine miss or something the old raw-substring check would
# have also missed vs. now correctly catches.
missed = [e for e in result['edges'] if not e['preserved']]
print(f"\n{len(missed)} edge(s) not preserved:")
for e in missed[:10]:
    print(f"  artifact={e['artifact']!r}  decision_msg_idx={e['decision_msg_idx']}")