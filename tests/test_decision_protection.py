"""
Regression tests for two bugs found and fixed during hardening:

1. Decision ACTION verbs (e.g. "recommend", "best") were never added to
   the protected-artifact set -- only targets/rationale were -- so under
   aggressive compression a decision's verb could be silently dropped
   even when its target survived. Fixed in
   dagc.compressor._decision_critical_values.

2. reproduce_decision's deterministic extractor didn't know how to read
   the compressor's own `[preserved: ...]` fallback tag, so a target
   that WAS successfully rescued by the compressor still scored as a
   miss during DRR evaluation. Fixed in
   dagc.extraction._preserved_tag_candidates / _extract_target.
"""
import pytest

from dagc import compress
from dagc.compressor import compress_dagc, DAGCConfig, _decision_critical_values
from dagc.extraction import extract_decisions, _preserved_tag_candidates
from dagc_eval import compute_drr


def test_preserved_tag_single_value():
    text = "Recommend hash index. [preserved: DBOPS-4471]"
    assert _preserved_tag_candidates(text) == ["DBOPS-4471"]


def test_preserved_tag_multiple_values():
    text = "Task truncated. [preserved: DB, DB index]"
    assert _preserved_tag_candidates(text) == ["DB", "DB index"]


def test_preserved_tag_absent_returns_empty():
    text = "Nothing to see here."
    assert _preserved_tag_candidates(text) == []


def test_action_verb_included_in_critical_values():
    decisions = [{
        'type': 'judgment', 'action': 'recommend', 'target': 'hash index',
        'rationale': [], 'artifacts': {'paths': [], 'ids': [], 'errors': []},
        'verbatim': '...', 'msg_idx': 0,
    }]
    crit = _decision_critical_values(decisions)
    assert 'recommend' in crit
    assert 'hash index' in crit


def test_weak_signal_decision_survives_aggressive_compression():
    trace = [
        {'role': 'user', 'content': 'What should we do?'},
        {'role': 'assistant',
         'content': 'Task: pick the best DB index strategy for the users table.'},
        {'role': 'assistant', 'content': 'btree: 120ms avg. hash: 45ms avg.'},
        {'role': 'assistant',
         'content': 'Recommend hash index. Confirmed: ticket DBOPS-4471 preserved.'},
    ]
    cfg = DAGCConfig(TARGET_REDUCTION=0.87)
    compressed = compress_dagc(trace, cfg=cfg)
    compressed_text = ' '.join(m.get('content', '') for m in compressed)
    assert 'best' in compressed_text or '[preserved:' in compressed_text

def test_tool_free_conversational_decision_survives():
    """A decision made in a purely conversational trace (zero tool
    messages) must not be penalized for lacking tool evidence -- there
    was nothing to corroborate against. Regression test for a bug where
    removing the whole-trace tool-fraction check accidentally made
    tools_before < min_corr fire unconditionally on tool-free traces."""
    trace = [
        {'role': 'user', 'content': 'Should we roll back deployment DEP-4021?'},
        {'role': 'assistant', 'content': 'Filler analysis text. ' * 15},
        {'role': 'assistant', 'content': 'Recommend rollback of DEP-4021. Confirmed: error rate=5.2%.'},
    ]
    from dagc_eval import compute_drr
    result = compute_drr(trace, verbose=False)
    assert result['DRR_soft'] > 0.5, (
        f"tool-free conversational decision scored {result['DRR_soft']} -- "
        f"evidence gate is wrongly penalizing traces with no tool calls"
    )
    
def test_drr_scores_recover_preserved_target():
    trace = [
        {'role': 'user', 'content': 'What should we do?'},
        {'role': 'assistant',
         'content': 'Task: pick the best DB index strategy for the users table.'},
        {'role': 'assistant', 'content': 'btree: 120ms avg. hash: 45ms avg.'},
        {'role': 'assistant',
         'content': 'Recommend hash index. Confirmed: ticket DBOPS-4471 preserved.'},
    ]
    result = compute_drr(trace, decision_roles=('assistant',), verbose=False)
    assert result['DRR_soft'] is not None
    assert result['DRR_soft'] > 0.0


def test_strong_judgment_outweighs_confirmation_when_recommend_is_present():
    text = 'Please confirm the deployment plan, but recommend rollback of DEP-4021.'
    decisions = extract_decisions([
        {'role': 'assistant', 'content': text},
    ])
    assert len(decisions) == 1
    assert decisions[0]['type'] == 'judgment'
    assert decisions[0]['action'] == 'recommend'


def test_plain_confirmation_still_uses_confirmation_type():
    text = 'Please confirm ticket DBOPS-4471.'
    decisions = extract_decisions([
        {'role': 'assistant', 'content': text},
    ])
    assert len(decisions) == 1
    assert decisions[0]['type'] == 'confirmation'


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
