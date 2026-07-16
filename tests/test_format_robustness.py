"""
Edge-case coverage for dagc_proxy.normalize: every shape of malformed,
unusual, or non-standard input the proxy might realistically receive.
Nothing here should ever raise -- a proxy in someone's live request
path must degrade gracefully, never 500.
"""
try:
    import pytest  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional in manual demo mode
    pytest = None

from dagc import compress_any
from dagc_eval.normalize import normalize_message, normalize_trace



def test_plain_string_content():
    out = normalize_message({'role': 'user', 'content': 'hello'})
    assert out['content'] == 'hello'


def test_none_content():
    out = normalize_message({'role': 'assistant', 'content': None,
                              'tool_call': {'name': 'x', 'args': {}}})
    assert out['content'] == ''


def test_missing_content_key_entirely():
    out = normalize_message({'role': 'assistant'})
    assert out['content'] == ''


def test_content_as_text_block_list():
    out = normalize_message({
        'role': 'assistant',
        'content': [{'type': 'text', 'text': 'hi there'}],
    })
    assert out['content'] == 'hi there'


def test_content_as_multiple_text_blocks():
    out = normalize_message({
        'role': 'assistant',
        'content': [{'type': 'text', 'text': 'part one'},
                    {'type': 'text', 'text': 'part two'}],
    })
    assert 'part one' in out['content'] and 'part two' in out['content']


def test_content_block_with_image_is_skipped_not_crashed():
    out = normalize_message({
        'role': 'user',
        'content': [{'type': 'image', 'source': {'data': 'base64...'}},
                    {'type': 'text', 'text': 'describe this'}],
    })
    assert out['content'] == 'describe this'


def test_content_as_bare_dict_with_text_key():
    out = normalize_message({'role': 'user', 'content': {'text': 'wrapped'}})
    assert out['content'] == 'wrapped'


def test_content_as_unexpected_type_stringifies_not_crashes():
    out = normalize_message({'role': 'user', 'content': 42})
    assert out['content'] == '42'



def test_tool_call_format_one():
    out = normalize_message({
        'role': 'assistant', 'content': '',
        'tool_call': {'name': 'search', 'args': {'q': 'x'}},
    })
    assert out['tool_call']['name'] == 'search'


def test_tool_call_format_two_function_wrapper():
    out = normalize_message({
        'role': 'assistant', 'content': '',
        'tool_call': {'function': {'name': 'search', 'arguments': '{"q":"x"}'}},
    })
    assert out['tool_call']['function']['name'] == 'search'


def test_tool_calls_plural_list_takes_first():
    out = normalize_message({
        'role': 'assistant', 'content': '',
        'tool_calls': [
            {'name': 'search', 'args': {'q': 'x'}},
            {'name': 'other_call', 'args': {}},
        ],
    })
    assert out['tool_call']['name'] == 'search'


def test_no_tool_call_at_all():
    out = normalize_message({'role': 'user', 'content': 'just chat'})
    assert out['tool_call'] is None


def test_tool_calls_empty_list_does_not_crash():
    out = normalize_message({'role': 'assistant', 'content': '', 'tool_calls': []})
    assert out['tool_call'] is None



def test_role_missing_entirely():
    out = normalize_message({'content': 'no role here'})
    assert out['role'] == 'unknown'


def test_role_aliased_as_sender():
    out = normalize_message({'sender': 'orchestrator', 'content': 'delegate this'})
    assert out['role'] == 'orchestrator'


def test_role_aliased_as_speaker():
    out = normalize_message({'speaker': 'worker_1', 'content': 'done'})
    assert out['role'] == 'worker_1'


def test_role_empty_string_falls_back():
    out = normalize_message({'role': '', 'content': 'hmm'})
    assert out['role'] == 'unknown'



def test_message_is_a_bare_string_not_dict():
    out = normalize_message("this is not even a dict")
    assert out['role'] == 'unknown'
    assert 'this is not even a dict' in out['content']


def test_message_is_none():
    out = normalize_message(None)
    assert out['role'] == 'unknown'


def test_trace_is_bare_list():
    trace = [{'role': 'user', 'content': 'hi'}]
    out = normalize_trace(trace)
    assert len(out) == 1


def test_trace_wrapped_in_messages_envelope():
    envelope = {'messages': [{'role': 'user', 'content': 'hi'}], 'meta': {'id': 1}}
    out = normalize_trace(envelope)
    assert len(out) == 1
    assert out[0]['content'] == 'hi'


def test_trace_wrapped_in_alternate_envelope_keys():
    for key in ('trace', 'conversation', 'turns'):
        envelope = {key: [{'role': 'user', 'content': 'hi'}]}
        out = normalize_trace(envelope)
        assert len(out) == 1, f"failed for envelope key={key}"


def test_trace_envelope_with_no_recognized_key_returns_empty():
    envelope = {'unrelated_key': [1, 2, 3]}
    out = normalize_trace(envelope)
    assert out == []


def test_trace_is_not_list_or_dict():
    assert normalize_trace("not a trace") == []
    assert normalize_trace(42) == []
    assert normalize_trace(None) == []


def test_trace_with_mixed_malformed_and_valid_messages():
    trace = [
        {'role': 'user', 'content': 'valid message'},
        "a bare string message",
        None,
        {'sender': 'agent_2', 'content': [{'type': 'text', 'text': 'block content'}]},
        {'role': 'assistant', 'tool_calls': [{'name': 'lookup', 'args': {}}]},
    ]
    out = normalize_trace(trace)
    assert len(out) == 5
    assert out[0]['content'] == 'valid message'
    assert out[3]['role'] == 'agent_2'
    assert out[3]['content'] == 'block content'
    assert out[4]['tool_call']['name'] == 'lookup'



def test_unknown_extra_fields_preserved_under_extra_key():
    out = normalize_message({
        'role': 'user', 'content': 'hi',
        'timestamp': '2026-07-14T00:00:00Z',
        'custom_metadata': {'session_id': 'abc123'},
    })
    assert out['_extra']['timestamp'] == '2026-07-14T00:00:00Z'
    assert out['_extra']['custom_metadata']['session_id'] == 'abc123'


def test_no_extra_key_when_nothing_unrecognized():
    out = normalize_message({'role': 'user', 'content': 'hi'})
    assert '_extra' not in out


def test_compress_any_accepts_trace_envelopes_and_returns_messages():
    trace = {
        'messages': [
            {'role': 'user', 'content': 'alpha beta gamma delta epsilon'},
            {'role': 'assistant', 'content': 'done'},
        ]
    }

    out = compress_any(trace, target_reduction=0.0)

    assert len(out) == 2
    assert out[0]['role'] == 'user'
    assert out[1]['role'] == 'assistant'


def test_compress_any_handles_empty_or_unknown_envelopes():
    assert compress_any({'meta': {'id': 1}}, target_reduction=0.0) == []
    assert compress_any({'messages': []}, target_reduction=0.0) == []


def test_compress_any_handles_mixed_malformed_messages():
    trace = [
        {'role': 'user', 'content': 'one two three four five six seven'},
        'raw',
        None,
        {'sender': 'agent', 'content': [{'type': 'text', 'text': 'x'}],
         'tool_calls': [{'name': 'do', 'args': {'a': 1}}]},
    ]

    out = compress_any(trace, target_reduction=0.0)

    assert len(out) <= len(trace)
    assert out[0]['role'] == 'user'
    last = out[-1]
    tool_call = last.get('tool_call') or (last.get('tool_calls', [{}])[0] if last.get('tool_calls') else None)
    assert tool_call is None or tool_call.get('name') == 'do'


if __name__ == "__main__":
    demo_cases = [
        {'role': 'user', 'content': 'hello world'},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'part one'}, {'type': 'text', 'text': 'part two'}]},
        {'sender': 'agent', 'content': [{'type': 'text', 'text': 'x'}], 'tool_calls': [{'name': 'do', 'args': {'a': 1}}]},
        'raw string message',
        None,
    ]

    print("Running manual robustness demo...")
    for idx, case in enumerate(demo_cases, start=1):
        try:
            result = compress_any([case], target_reduction=0.0)
            print(f"Case {idx}: {type(case).__name__} -> {result}")
        except Exception as exc:
            print(f"Case {idx}: {type(case).__name__} -> ERROR: {exc}")
