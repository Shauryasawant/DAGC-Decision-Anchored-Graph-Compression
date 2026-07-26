from dagc.extraction import extract_decisions
from dagc.extraction import _extract_rationale

out = _extract_rationale(
    'Only 100% credit was applied. [preserved: 150]',
    {'errors': [], 'ids': []},
    decision_values={'100', 'confirm'},
)
print(out)
assert 'preserved:150' in out, "D-fix did not land"
print("D-fix OK")
messages = [
    {"role": "user", "content": "placeholder 0", "_orig_idx": 0},
    {"role": "assistant",
     "content": "[preserved: confirm#d3, 150#d3]",
     "_orig_idx": 3},
]
decisions = extract_decisions(messages)
print(decisions[0]['msg_idx'], decisions[0]['rationale'])