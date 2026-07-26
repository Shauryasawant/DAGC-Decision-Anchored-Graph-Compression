"""
Adversarial robustness regression suite. Confirms DAGC's DRR score
doesn't collapse under injection, noise, redaction, or contradiction
attacks relative to clean compression.
"""
import pytest
from dagc_eval.adversarial import _atk_long_context, run_adversarial_suite
from dagc_eval.benchmark import TASKS, generate_trace


@pytest.mark.parametrize("task_spec", TASKS, ids=[t['task'][:30] for t in TASKS])
def test_adversarial_robustness(task_spec):
    summary = run_adversarial_suite(task_spec, n_seeds=3, verbose=False)
    for attack_name, row in summary.items():
        assert not (row['ratio'] != row['ratio']), (
            f"{attack_name} produced NaN ratio (clean_drr or adv_drr empty)"
        )
        assert row['robust'], (
            f"{attack_name} FRAGILE: adv_DRR/clean_DRR = {row['ratio']} "
            f"(clean={row['clean_drr']}, adv={row['adv_drr']}) — below 0.85 threshold"
        )


def test_long_context_attack_keeps_the_original_final_message_last():
    trace = generate_trace(TASKS[0])
    attacked = _atk_long_context(trace, TASKS[0], seed=0)
    assert attacked[-1] == trace[-1]


def test_long_context_uses_the_same_budget_and_evidence_floor_as_clean_trace():
    row = run_adversarial_suite(TASKS[0], n_seeds=1, verbose=False)['E_long_context']
    assert row['clean_budget'] == row['adv_budget']
    assert row['clean_evidence_floor'] == row['adv_evidence_floor']
