"""
dagc CLI: dagc compress | benchmark | compare | evaluate | stats

Wraps the existing Python API -- no logic duplicated here, just argument
parsing and dispatch, so behavior always matches the library functions.
"""
from __future__ import annotations
import argparse
import json
import sys


def _load_trace(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(f"Trace file not found: {path}\nProvide a JSON trace file, for example: dagc evaluate your_trace.json") from exc


def cmd_compress(args):
    import dagc
    messages = _load_trace(args.input)
    result = dagc.compress(messages, target_reduction=args.target_reduction)
    out = json.dumps(result, indent=2, default=str)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(out)
        print(f"Wrote compressed trace to {args.output}")
    else:
        print(out)


def cmd_evaluate(args):
    from dagc_eval import compute_drr
    messages = _load_trace(args.input)
    result = compute_drr(messages, verbose=not args.quiet,
                          decision_roles=tuple(args.decision_roles.split(',')))
    print(f"\nDRR_soft={result['DRR_soft']}  DRR_binary={result['DRR_binary']}  "
          f"RCI={result['RCI']}  reduction={result.get('reduction', 0):.1f}%")
    if args.output:
        from dagc_eval.export import to_json, to_csv, to_markdown, to_html
        ext = args.output.rsplit('.', 1)[-1].lower()
        {'json': to_json, 'csv': to_csv, 'md': to_markdown, 'html': to_html}.get(
            ext, to_json)(result, args.output)
        print(f"Wrote report to {args.output}")


def cmd_benchmark(args):
    from dagc_eval import run_benchmark
    results = run_benchmark(n_traces_per_task=args.n_traces,
                             noise_levels=[int(x) for x in args.noise_levels.split(',')])
    if args.output:
        from dagc_eval.export import to_json, to_csv, to_markdown, to_html
        ext = args.output.rsplit('.', 1)[-1].lower()
        {'json': to_json, 'csv': to_csv, 'md': to_markdown, 'html': to_html}.get(
            ext, to_json)(results, args.output)
        print(f"Wrote report to {args.output}")


def cmd_compare(args):
    from dagc_eval import run_method_comparison
    run_method_comparison(n_traces_per_task=args.n_traces,
                           noise_levels=[int(x) for x in args.noise_levels.split(',')])


def cmd_stats(args):
    from dagc_eval import bootstrap_drr
    with open(args.input) as f:
        scores = json.load(f)
    ci = bootstrap_drr([{'DRR_soft': s} for s in scores])
    print(f"mean={ci['mean']:.4f}  95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]  n={ci['n']}")


def main():
    parser = argparse.ArgumentParser(prog='dagc', description='DAGC — Decision-Anchored Graph Compression CLI')
    sub = parser.add_subparsers(dest='command', required=True)

    p_compress = sub.add_parser('compress', help='Compress a trace (JSON list of messages)')
    p_compress.add_argument('input', help='Path to input trace JSON')
    p_compress.add_argument('-o', '--output', help='Path to write compressed trace JSON')
    p_compress.add_argument('--target-reduction', type=float, default=0.87)
    p_compress.set_defaults(func=cmd_compress)

    p_eval = sub.add_parser('evaluate', help='Score a trace with compute_drr')
    p_eval.add_argument('input', help='Path to input trace JSON')
    p_eval.add_argument('-o', '--output', help='Path to write report (.json/.csv/.md/.html)')
    p_eval.add_argument('--decision-roles', default='assistant')
    p_eval.add_argument('--quiet', action='store_true')
    p_eval.set_defaults(func=cmd_evaluate)

    p_bench = sub.add_parser('benchmark', help='Run the synthetic-task DRR benchmark sweep')
    p_bench.add_argument('-o', '--output', help='Path to write report (.json/.csv/.md/.html)')
    p_bench.add_argument('--n-traces', type=int, default=3)
    p_bench.add_argument('--noise-levels', default='1,2,3,4,5')
    p_bench.set_defaults(func=cmd_benchmark)

    p_compare = sub.add_parser('compare', help='Compare DAGC vs BASELINES')
    p_compare.add_argument('--n-traces', type=int, default=2)
    p_compare.add_argument('--noise-levels', default='3')
    p_compare.set_defaults(func=cmd_compare)

    p_stats = sub.add_parser('stats', help='Bootstrap CI over a JSON list of DRR_soft scores')
    p_stats.add_argument('input', help='Path to JSON list of floats')
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
