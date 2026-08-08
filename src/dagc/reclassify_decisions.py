#!/usr/bin/env python3
"""
reclassify_decisions.py — Post-processing gate that tightens
"load_bearing" labels down to genuine decisions.

Design: POSITIVE SIGNAL REQUIRED. A clause is only kept as a real
decision if it contains a concrete committed action, an explicit
choice/judgment, or a confirmation/correction of a fact. Everything
else (generic task requests, brainstorms, translations, non-English
text, confused/hedged messages, quiz stems, system prompts) is
rejected by default -- this naturally filters non-English text too,
since it won't match any of these English patterns.

CHANGELOG: the earlier version hand-maintained its own "positive
signal" verb regexes (_COMMIT_ACTION_VERBS / _JUDGMENT_CHOICE_VERBS)
as a standalone subset. That subset silently drifted from the
canonical decisive-verb vocabulary the rest of the pipeline actually
uses (dagc.extraction._JUDGMENT_VERBS / _verb_match_is_decisive) --
missing common verbs like "use", "keep", "remove", "move", "target",
"implement", "suggest". Any genuine decision phrased with one of those
verbs was silently demoted from load_bearing -> safely_droppable here,
which corrupts ground truth for scoring even though nothing is
actually wrong with compression/reproduction downstream. Root-caused
after a DRR_soft regression (824 traces scored, 409 below 0.95) that
did not move at all after fixing an unrelated (real, but not-the-
culprit) recoverability gap in compressor.py/reproduce.py.

Fix: defer to the canonical vocabulary instead of re-implementing it.
_verb_match_is_decisive already exists specifically to distinguish a
decisive use of a verb ("I'll use the cached version") from a
non-decisive one ("you can use this file however you like") -- reusing
it here means this gate can never again drift out of sync with what
extraction.py and compressor.py treat as a decision.
"""

import argparse
import glob
import json
import os
import re
from dagc.extraction import _mask_code_fences

_QUIZ_STEM = re.compile(r'\bselect one\b', re.IGNORECASE)

# Real "stakes" actions -- committing to or changing something concrete.
# Kept as a fast-path/backstop for verbs outside the canonical judgment
# vocabulary (refunds, bookings, etc. are transactional actions, not
# "judgment" verbs in extraction.py's sense).
_COMMIT_ACTION_VERBS = re.compile(
    r'\b(cancel\w*|exchang\w*|refund\w*|rebook\w*|reschedul\w*|return(?:s|ed|ing)?|'
    r'book(?:s|ed|ing)?|delet\w*|renam\w*|revert\w*|migrat\w*|deploy\w*|switch\w*|'
    r'adopt\w*)\b', re.IGNORECASE)

# Explicit choice/judgment among options. Kept as a fast-path so the
# common cases don't pay a cross-module import + full decisive-match
# scan; _JUDGMENT_VERBS (below) is the authoritative backstop for
# anything this misses (suggest/implement/use/target/keep/remove/move
# and any future additions to the canonical vocabulary).
_JUDGMENT_CHOICE_VERBS = re.compile(
    r'\b(recommend\w*|conclude\w*|decid\w*|choos\w*|chose|select\w*|prefer\w*|'
    r'winner|best option|optimal|final decision|'
    r'(?:go(?:es|ing)?|went)\s+with)\b', re.IGNORECASE)

# Confirming or correcting a concrete fact/value
_CONFIRM_CORRECT_SIGNALS = re.compile(
    r'\b(confirm\w*|verified?|actually|correction|i mean|i meant|no wait|'
    r'my bad|my mistake|i definitely|'
    r"i'?m (?:absolutely |completely |100% |definitely )?(?:certain|sure|positive)|"
    r"i distinctly remember|it should be|should actually be)\b", re.IGNORECASE)

# Genuine uncertainty/confusion -- NOT a decision, unless overridden by
# an explicit confirm/correct signal in the same message.
_HEDGE = re.compile(
    r"\b(maybe|might|probably|not sure|can'?t remember|not certain|no idea|"
    r"don'?t know|not (?:yet )?decided|undecided)\b", re.IGNORECASE)

_TOOL_CALL_SHAPE = re.compile(
    r'\[calls\s+\w+\(|'                          # [calls func_name(...)]
    r'"name"\s*:\s*"[\w.]+".*"args"\s*:|'        # {"name": "...", "args": {...}}
    r'"args"\s*:\s*\{.*\}.*"name"\s*:',           # args-before-name variant
    re.IGNORECASE | re.DOTALL)

_EXACT_DECISION_PHRASES = re.compile(
    r"\b(let'?s (?:just )?keep the \w+ class|"       # "let's keep the economy class"
    r"add(?:ing)? the \d+ checked bags?|"             # "add the 2 checked bags"
    r"task (?:is )?completed\.?$)\b",                 # "Task completed."
    re.IGNORECASE)


def _canonical_decisive_verb_hit(text: str) -> bool:
    """
    Authoritative backstop: defer to dagc.extraction's own decisive-verb
    detection (_JUDGMENT_VERBS + _verb_match_is_decisive) rather than
    re-implementing a parallel subset of it. This is a lazy import
    (dagc.extraction is a heavier module and this script should stay
    usable standalone / without a full dagc install for quick label
    audits) and is deliberately soft-failing: if dagc isn't importable
    in this environment, this backstop simply contributes nothing and
    the fast-path regexes above still apply -- it never turns an
    ImportError into a crash of the whole reclassification run.
    """
    try:
        from dagc.extraction import _JUDGMENT_VERBS, _verb_match_is_decisive
    except Exception:
        return False
    return any(_verb_match_is_decisive(text, m) for m in _JUDGMENT_VERBS.finditer(text))


def is_genuine_decision(text: str, role: str) -> bool:
    if role == 'system':
        return False
    if _TOOL_CALL_SHAPE.search(text):
        return True
    if _EXACT_DECISION_PHRASES.search(text):      # <-- new, isolated check
        return True
    if _QUIZ_STEM.search(text):
        return False
    if _HEDGE.search(text) and not _CONFIRM_CORRECT_SIGNALS.search(text):
        return False
    return bool(
        _COMMIT_ACTION_VERBS.search(text)
        or _JUDGMENT_CHOICE_VERBS.search(text)
        or _CONFIRM_CORRECT_SIGNALS.search(text)
        or _canonical_decisive_verb_hit(text)
    )


def reclassify_file(in_path: str, out_path: str) -> int:
    with open(in_path) as f:
        data = json.load(f)

    changed = 0
    for l in data.get('labels', []):
        if l.get('label') == 'load_bearing':
            text = l.get('clause_text_preview', '')
            role = l.get('role', '')
            if not is_genuine_decision(text, role):
                l['label'] = 'safely_droppable'
                l['_reclassify_reason'] = 'not_a_genuine_decision'
                changed += 1

    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    return changed


def find_decision_evidence(text: str, role: str):
    if role == 'system':
        return None
    text = _mask_code_fences(text)   # <-- add this
    if _QUIZ_STEM.search(text):
        return None
    if _HEDGE.search(text) and not _CONFIRM_CORRECT_SIGNALS.search(text):
        return None
    for pattern in (_TOOL_CALL_SHAPE, _EXACT_DECISION_PHRASES,
                    _COMMIT_ACTION_VERBS, _JUDGMENT_CHOICE_VERBS,
                    _CONFIRM_CORRECT_SIGNALS):
        m = pattern.search(text)
        if m:
            return m
    try:
        from dagc.extraction import _JUDGMENT_VERBS, _verb_match_is_decisive
        for m in _JUDGMENT_VERBS.finditer(text):
            if _verb_match_is_decisive(text, m):
                return m
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out-dir", default="./eval3_results_by_source_fixed")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"No files matched: {args.glob}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    total_changed = 0
    for in_path in files:
        out_path = os.path.join(args.out_dir, os.path.basename(in_path))
        changed = reclassify_file(in_path, out_path)
        total_changed += changed
        print(f"  {os.path.basename(in_path)}: reclassified {changed} row(s) -> {out_path}")

    print(f"\nTotal reclassified across {len(files)} file(s): {total_changed}")
    print(f"Fixed files written to: {args.out_dir}")


if __name__ == "__main__":
    main()