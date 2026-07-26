import sys, json
sys.path.insert(0, 'src')
from dagc.compressor import compress_dagc
from dagc.extraction import extract_decisions
from dagc.utils import _get_text

with open('/Users/shaurya/python/DDR/Tau_bench.json') as f:
    data = json.load(f)
item = next(i for i in data if i['trace_id'] == 'tau_airline_0080')
trace = item['trace']

print('=== ORIGINAL trace, messages 14-20 ===')
for i, m in enumerate(trace[14:20], start=14):
    print(f'{i}: [{m.get("role")}] {_get_text(m)[:250]}')

print()
print('=== decisions found on ORIGINAL trace ===')
decs_orig = extract_decisions(trace, decision_roles=('user', 'assistant'))
for d in decs_orig:
    print(f"  msg_idx={d['msg_idx']} action={d['action']!r} target={d['target']!r} verbatim={d['verbatim'][:150]!r}")

print()
compressed = compress_dagc(trace, decision_roles=('user', 'assistant'))
print('=== COMPRESSED trace, all messages ===')
for i, m in enumerate(compressed):
    print(f'{i}: [{m.get("role")}] {_get_text(m)[:250]}')

print()
print('=== decisions found on COMPRESSED trace ===')
decs_comp = extract_decisions(compressed, decision_roles=('user', 'assistant'))
for d in decs_comp:
    print(f"  msg_idx={d['msg_idx']} action={d['action']!r} target={d['target']!r} verbatim={d['verbatim'][:150]!r}")
