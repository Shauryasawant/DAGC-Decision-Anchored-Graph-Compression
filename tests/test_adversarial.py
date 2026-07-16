"""
Adversarial robustness regression suite. Confirms DAGC's DRR score
doesn't collapse under injection, noise, redaction, or contradiction
attacks relative to clean compression.
"""
import pytest
from dagc_eval.adversarial import run_adversarial_suite
from dagc_eval.benchmark import TASKS


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
