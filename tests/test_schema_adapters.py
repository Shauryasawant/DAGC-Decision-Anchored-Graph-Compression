from dagc import register_adapter, to_dagc_format


def test_to_dagc_format_accepts_trace_envelopes():
    envelope = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
    }

    out = to_dagc_format(envelope)

    assert len(out) == 2
    assert out[0]["role"] == "user"
    assert out[1]["content"] == "hi there"


def test_register_adapter_is_available_from_package_root():
    @register_adapter("custom_orchestrator")
    def _convert(event):
        return {
            "role": event["speaker"],
            "content": event["text"],
            "tool_call": None,
        }

    out = to_dagc_format(
        [{"speaker": "agent", "text": "delegate"}],
        schema="custom_orchestrator",
    )

    assert out[0]["role"] == "agent"
    assert out[0]["content"] == "delegate"


def test_openai_style_content_blocks_are_normalized():
    event = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two"},
        ],
    }

    out = to_dagc_format([event], schema="openai")

    assert out[0]["content"] == "part one part two"
    assert out[0]["tool_call"] is None
