"""Small runnable example: generate a synthetic trace, compress it,
and compute DRR metrics. Run with the repo venv active.

Usage:
  python examples/basic_usage.py
"""
import sys
from pathlib import Path

# Ensure local src is importable when running this example directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from dagc_eval.benchmark import generate_trace, TASKS
from dagc import compress
from dagc_eval import compute_drr


def main():
    trace = generate_trace(TASKS[0], noise_level=3, rng_seed=0)
    print('Generated trace with', len(trace), 'messages')

    compressed = compress(trace, target_reduction=0.85)
    print('Compressed trace has', len(compressed), 'messages')

    result = compute_drr(trace)
    print('\nSummary:')
    print('DRR_soft=', result.get('DRR_soft'))
    print('RCI=', result.get('RCI'))


if __name__ == '__main__':
    main()
