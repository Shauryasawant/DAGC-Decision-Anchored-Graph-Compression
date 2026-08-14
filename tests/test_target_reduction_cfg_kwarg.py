"""
Regression test for a bug in compress()'s target_reduction handling.

compress()'s `target_reduction` parameter defaults to the sentinel `...`
(Ellipsis), meaning "not passed, defer to cfg.TARGET_REDUCTION". The
original guard was `if target_reduction is not None`, which is always True
for the Ellipsis sentinel -- so any call passing `cfg=` without ALSO
passing `target_reduction=` explicitly had its cfg's TARGET_REDUCTION
silently overwritten with `...`, crashing downstream with:
    TypeError: unsupported operand type(s) for -: 'int' and 'ellipsis'

This test locks in the fix and pins the additive guarantee: passing only
`cfg=` must behave identically to passing the equivalent `target_reduction=`
kwarg, and an explicit `target_reduction=` must still win over `cfg`'s
value when both are supplied.
"""
from dagc import compress, DAGCConfig

_MESSAGES = [
    {"role": "user", "content": "Hey, can you help me debug our nightly ETL job?"},
    {
        "role": "assistant",
        "content": "Sure. Fetching metrics.",
        "tool_call": {"name": "get_metrics", "args": {"job_id": "run-42"}},
    },
    {"role": "tool", "content": "Run run-42 failed, memory limit was 4096MB."},
    {"role": "user", "content": "Bump memory to 8192."},
    {
        "role": "assistant",
        "content": "Updated.",
        "tool_call": {"name": "update_job_config", "args": {"job_id": "run-42", "memory_mb": 8192}},
    },
]


def test_cfg_only_does_not_raise():
    """Previously raised TypeError: 'int' and 'ellipsis'."""
    cfg = DAGCConfig(TARGET_REDUCTION=0.5)
    compress(_MESSAGES, cfg=cfg)


def test_cfg_only_matches_equivalent_target_reduction_kwarg():
    cfg = DAGCConfig(TARGET_REDUCTION=0.5)
    via_cfg = compress(_MESSAGES, cfg=cfg)
    via_kwarg = compress(_MESSAGES, target_reduction=0.5)
    assert via_cfg == via_kwarg


def test_explicit_target_reduction_overrides_cfg():
    cfg = DAGCConfig(TARGET_REDUCTION=0.9)
    result = compress(_MESSAGES, target_reduction=0.5, cfg=cfg)
    expected = compress(_MESSAGES, target_reduction=0.5)
    assert result == expected


def test_run_method_comparison_default_includes_dagc():
    """
    run_method_comparison() (and the `dagc compare` CLI command built on
    it) is documented as comparing DAGC against BASELINES. Previously its
    default `methods` fell back to BASELINES alone, which never
    contains a 'DAGC' entry -- so a bare call compared baselines against
    each other and never against DAGC, and the downstream
    run_statistical_comparison(..., reference_name='DAGC') branch was
    unreachable dead code. This locks in that the default methods set now
    actually includes DAGC.
    """
    from dagc_eval.benchmark import run_method_comparison
    summary = run_method_comparison(n_traces_per_task=1, noise_levels=[3],
                                     verbose=False, run_stats=False)
    assert 'DAGC' in summary
