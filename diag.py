import dagc
from dagc.compressor import compress_dagc
from dagc.tool_artifacts import extract_tool_artifacts

messages = [
    {'role':'user','content':'What version?'},
    {'role':'assistant','content':'Checking...','tool_call':{'name':'get','args':{}}},
    {'role':'tool','content':'{"name":"requests","version":"2.32.3"}'},
    {'role':'assistant','content':'Let me check.'},
]

# Step 1: does extraction even find it?
arts = extract_tool_artifacts(messages)
print("1. extract_tool_artifacts found:", arts)
print("   '2.32.3' in extracted set:", '2.32.3' in arts)

# Step 2: does compress_dagc directly (bypassing compress()'s wiring) preserve it
# if we hand it force_preserve manually? Isolates: is the bug in compress()'s
# wiring, or deeper in compress_dagc/pool/rescue itself?
diag = {}
out = compress_dagc(messages, force_preserve={'2.32.3'}, diagnostics=diag)
full = str(out)
print("\n2. compress_dagc(force_preserve={'2.32.3'}) directly:")
print("   survives:", '2.32.3' in full)
print("   output:", out)
print("   diagnostics keys:", list(diag.keys()))

# Step 3: full compress() path, exactly as before
diag2 = {}
out2 = dagc.compress(messages, target_reduction=0.5, diagnostics=diag2)
print("\n3. dagc.compress() full path:")
print("   survives:", '2.32.3' in str(out2))
print("   output:", out2)
print("   diagnostics:", {k: v for k, v in diag2.items() if k not in ('valid_decisions','all_decisions')})