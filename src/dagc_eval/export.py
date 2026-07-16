"""
Evaluation report export: serialize compute_drr() or run_benchmark()
results to JSON, CSV, Markdown, or HTML for CI/CD integration or
sharing outside a Python session.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, List, Union


def _flatten_result(r: Dict) -> Dict:
    """Flatten a single compute_drr()/run_benchmark() result row to
    scalar fields only, dropping nested decision detail (use
    dagc_eval.diagnostics.explain_drr for the detailed version)."""
    return {
        'DRR_soft': r.get('DRR_soft'),
        'DRR_binary': r.get('DRR_binary'),
        'RCI': r.get('RCI'),
        'SP': r.get('SP'),
        'reduction': r.get('reduction'),
        'art_ret': r.get('art_ret'),
        'decision_art_ret': r.get('decision_art_ret'),
        'orig_tokens': r.get('orig_tokens'),
        'comp_tokens': r.get('comp_tokens'),
        'task_label': r.get('task_label'),
        'noise': r.get('noise'),
        'seed': r.get('seed'),
    }


def to_json(results: Union[Dict, List[Dict]], path: str) -> None:
    rows = results if isinstance(results, list) else [results]
    Path(path).write_text(json.dumps([_flatten_result(r) for r in rows], indent=2))


def to_csv(results: Union[Dict, List[Dict]], path: str) -> None:
    rows = [_flatten_result(r) for r in (results if isinstance(results, list) else [results])]
    if not rows:
        Path(path).write_text('')
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_markdown(results: Union[Dict, List[Dict]], path: str) -> None:
    rows = [_flatten_result(r) for r in (results if isinstance(results, list) else [results])]
    if not rows:
        Path(path).write_text('_No results._\n')
        return
    cols = list(rows[0].keys())
    lines = ['| ' + ' | '.join(cols) + ' |',
             '| ' + ' | '.join('---' for _ in cols) + ' |']
    for r in rows:
        lines.append('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    Path(path).write_text('\n'.join(lines) + '\n')


def to_html(results: Union[Dict, List[Dict]], path: str) -> None:
    rows = [_flatten_result(r) for r in (results if isinstance(results, list) else [results])]
    cols = list(rows[0].keys()) if rows else []
    thead = ''.join(f'<th>{c}</th>' for c in cols)
    trs = ''.join(
        '<tr>' + ''.join(f'<td>{r.get(c, "")}</td>' for c in cols) + '</tr>'
        for r in rows
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DAGC Evaluation Report</title>
<style>
table {{ border-collapse: collapse; font-family: sans-serif; font-size: 14px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #f0f0f0; }}
</style></head><body>
<h1>DAGC Evaluation Report</h1>
<table><thead><tr>{thead}</tr></thead><tbody>{trs}</tbody></table>
</body></html>"""
    Path(path).write_text(html)
