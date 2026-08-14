"""
dagc_eval — optional decision-reproducibility scoring harness for dagc.

Not required to use dagc.compress(). Bundled with the `dagc` distribution --
`import dagc_eval` works as soon as `dagc` is installed, no extra needed --
if you want to measure whether a compressor (dagc's or your own) preserves
enough information for decisions to be reconstructed.

Fully BYOK for the LLM-assisted path, and fully usable with NO LLM at all
(reproduce_decision(..., llm=None) does deterministic-only scoring):

    from dagc_eval import compute_drr, generate_trace, TASKS
    trace = generate_trace(TASKS[0])
    result = compute_drr(trace)
    print(result['DRR_soft'], result['RCI'])

    from dagc_eval.interfaces import OpenAIChatClient
    import openai
    llm = OpenAIChatClient(openai.OpenAI(api_key="..."))
    result = compute_drr(trace, llm=llm)
"""
from .benchmark import compute_drr, generate_trace, TASKS
from .interfaces import LLMClient, OpenAIChatClient, AnthropicChatClient, MistralChatClient
from .match import match_decision, DRR_THRESHOLD
from .reproduce import reproduce_decision
from .adversarial import run_adversarial_suite, ADVERSARIAL_ATTACKS
from .stats import bootstrap_drr, wilcoxon_test, cohen_d
from .benchmark import run_benchmark, run_method_comparison, run_statistical_comparison
from .diagnostics import explain_drr, diff_trace, diff_trace_report
from .export import to_json, to_csv, to_markdown, to_html
from .normalize import normalize_message, normalize_trace
from .leaderboard import efficiency_score, rank_leaderboard, print_leaderboard

__version__ = "0.1.7"

__all__ = ["normalize_message", "normalize_trace", "explain_drr", "diff_trace", "diff_trace_report",
           "to_json", "to_csv", "to_markdown", "to_html",
           "AnthropicChatClient", "DRR_THRESHOLD", "LLMClient", "MistralChatClient", "OpenAIChatClient",
           "TASKS", "ADVERSARIAL_ATTACKS", "bootstrap_drr", "cohen_d", "compute_drr", "generate_trace",
           "match_decision", "reproduce_decision", "run_adversarial_suite", "run_benchmark",
           "run_method_comparison", "run_statistical_comparison", "wilcoxon_test",
           "efficiency_score", "rank_leaderboard", "print_leaderboard"]
