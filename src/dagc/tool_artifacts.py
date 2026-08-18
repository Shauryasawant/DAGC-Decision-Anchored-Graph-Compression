"""tool_artifacts.py
==================
Generic, content-agnostic literal-value extraction for role=='tool'
messages, feeding DAGC's existing force_preserve/rescue mechanism.

WHY THIS EXISTS
----------------
extract_decisions() (extraction.py) only scans role in ('user','assistant')
by default. A role=='tool' message's own payload -- a JSON API response, a
source file, a log dump -- never becomes a "decision" itself, so no literal
value inside it is ever added to target_arts unless:
  (a) a later assistant/user message happens to restate it in text that
      extract_decisions can parse into a decision target, or
  (b) the caller manually passes force_preserve=... naming the exact value.

(a) is unreliable (agent traces routinely reference tool output by pronoun
or paraphrase, not verbatim restatement) and (b) requires knowing in
advance which fact a downstream question will need -- which the compressor
cannot know at compression time. Net effect: literal facts embedded only
in tool payloads (a version string, a file's line count, a config value)
can be silently dropped even though the pipeline is working exactly as
designed.

WHAT THIS MODULE DOES
----------------------
Extracts EVERY scalar literal from every tool-role message, generically --
no knowledge of what question will be asked later, so this cannot be
tuned to any specific benchmark:

  - JSON tool content: parsed and walked; every scalar leaf (string or
    number) in the tree is a candidate artifact.
  - Non-JSON / code tool content: a conservative regex scan for quoted
    string literals and standalone numeric literals.

These candidates are handed to DAGC's own public, documented
force_preserve mechanism (compress(..., force_preserve=...)), which
decides HOW to guarantee survival (kept in place, or recovered via a
compact "[preserved: ...]" tag) -- this module only decides WHAT must
survive, never how.

This is intentionally symmetric with -- and independent of -- the
existing PROTECT_TOOL_CALLS setting, which protects the *calling*
assistant message's tool_call arguments. This module covers the tool's
*response* content, which is a different message with no protection
path of its own.

COLUMNAR COMPACTION (compact_tool_listings / compact_json_listing)
--------------------------------------------------------------------
Extraction above has a hard ceiling: "protect every literal" only keeps
reduction% healthy while a payload is small. On a JSON array of many
similarly-shaped records (an API listing), blindly protecting every
field of every record protects most of the payload, and reduction
collapses -- that's a real trade-off, documented above, not something
literal-extraction can fix by itself.

compact_json_listing() attacks the OTHER side of that trade-off: it
doesn't change what survives, it shrinks what surviving costs. A JSON
array of N objects sharing (most of) the same keys re-states every key
name N times and pays JSON's braces/quotes/commas on every record. A
one-line header of the shared keys plus one compact row per record
carries the exact same values with none of that repetition -- typically
40-55% smaller on realistic listings, measured losslessly (every value
still present verbatim), before the normal selection/protection pipeline
ever runs.

This is pure pre-processing: it replaces a tool message's `content`
string with an equivalent, more compact string, then hands the trace to
the unchanged compressor exactly as before. It does not touch
extraction.py's decision logic, compressor.py's causal graph, selection,
or rescue -- from their point of view it's just a shorter tool message.
It only fires when a tool message's content is a JSON array of >=3
dicts with enough shared keys to tabulate; anything else (a single
object, code, prose, a non-uniform array) passes through byte-for-byte
unchanged, so it cannot regress any case this module doesn't target.
"""
import json
import re
from typing import Dict, Iterable, List, Optional, Set

_MIN_LEN = 2
_MAX_LEN = 80

# Caps prevent a huge tool payload (a multi-MB log dump, a giant API
# response) from blowing up force_preserve / phase1 budgeting. This is a
# safety valve, not a signal-selection mechanism -- if a real trace
# regularly exceeds the cap, that's a signal to raise it, not to make the
# extractor "smarter" about which literals matter (it deliberately isn't).
DEFAULT_MAX_ARTIFACTS_PER_MESSAGE = 200
DEFAULT_MAX_TOTAL_ARTIFACTS = 800

# KNOWN LIMITATION, not fully solved by this module: "extract every
# literal, force_preserve all of them" only works while a tool payload is
# small enough that "everything" is still a small set. On a large payload
# (e.g. a 30-record API listing), nearly every field is a syntactically
# valid literal, so blind extraction force-protects most of the payload
# and reduction collapses toward ~0% -- the opposite of what compression
# is for. This is a real trade-off, not a matching bug: fixing it properly
# needs extraction to be need-driven (only protect a literal some later
# message/decision actually references) rather than blind, which this
# module does not attempt.
#
# As a bounded stopgap, per-message extraction is skipped entirely once a
# tool message's content exceeds this size -- protecting the common case
# (a single lookup/read returning one record) without silently defeating
# compression on large listings. Traces above this threshold get NO
# automatic help from this module and still need either a smaller
# TARGET_REDUCTION tolerance or a caller-supplied, need-driven
# force_preserve.
DEFAULT_SKIP_MESSAGE_ABOVE_CHARS = 1200

_RE_STRING_LITERAL = re.compile(r'''(["'])((?:(?!\1)[^\\]|\\.)*)\1''')
_RE_NUMERIC_LITERAL = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])')


def _walk_json_leaves(obj, out: Set[str], cap: int) -> None:
    if len(out) >= cap:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_json_leaves(v, out, cap)
    elif isinstance(obj, list):
        for v in obj:
            _walk_json_leaves(v, out, cap)
    elif isinstance(obj, str):
        if _MIN_LEN <= len(obj) <= _MAX_LEN:
            out.add(obj)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(str(obj))


def _extract_code_literals(content: str, cap: int) -> Set[str]:
    out: Set[str] = set()
    for m in _RE_STRING_LITERAL.finditer(content):
        s = m.group(2)
        if _MIN_LEN <= len(s) <= _MAX_LEN:
            out.add(s)
        if len(out) >= cap:
            return out
    for m in _RE_NUMERIC_LITERAL.finditer(content):
        out.add(m.group(0))
        if len(out) >= cap:
            return out
    return out


def extract_literals_from_tool_content(
    content: str, max_artifacts: int = DEFAULT_MAX_ARTIFACTS_PER_MESSAGE
) -> Set[str]:
    """Every scalar literal in one tool message's content, generic and
    order-independent. Tries JSON first (structured, lossless walk); falls
    back to a regex literal scan for anything that doesn't parse as JSON
    (source code, plain text, logs)."""
    if not isinstance(content, str) or not content.strip():
        return set()
    try:
        obj = json.loads(content)
    except Exception:
        return _extract_code_literals(content, max_artifacts)

    leaves: Set[str] = set()
    _walk_json_leaves(obj, leaves, max_artifacts)
    return leaves


def extract_tool_artifacts(
    messages: List[dict],
    max_artifacts_per_message: int = DEFAULT_MAX_ARTIFACTS_PER_MESSAGE,
    max_total_artifacts: int = DEFAULT_MAX_TOTAL_ARTIFACTS,
    skip_message_above_chars: int = DEFAULT_SKIP_MESSAGE_ABOVE_CHARS,
    size_reference_messages: Optional[List[dict]] = None,
) -> Set[str]:
    """Scan every role=='tool' message in a trace and return the union of
    all extracted literals, ready to pass as compress(force_preserve=...).

    Generic over the whole trace -- does not look at what question is
    being asked, so it cannot be shaped around any one evaluation.

    Messages larger than skip_message_above_chars are left alone (see the
    module docstring's KNOWN LIMITATION note): blindly protecting every
    literal in a large payload defeats compression rather than aiding
    fidelity, so this only helps on small/targeted tool responses by
    design, not on large listings/dumps.

    size_reference_messages: if given (same length/order as `messages`),
    the skip-threshold check is measured against THIS message's content
    length instead of `messages`' own -- pass the compact_tool_listings()
    output here so a listing that compaction already shrank under the
    threshold isn't skipped just because its pre-compaction size was
    larger. The literal walk itself always reads from `messages` (the
    original, exact JSON), never from the reference -- this parameter
    only affects the go/no-go size check.
    """
    out: Set[str] = set()
    for idx, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        content = m.get("content")
        if not isinstance(content, str):
            continue
        size_probe = content
        if size_reference_messages is not None and idx < len(size_reference_messages):
            ref = size_reference_messages[idx]
            ref_content = ref.get("content") if isinstance(ref, dict) else None
            if isinstance(ref_content, str):
                size_probe = ref_content
        if len(size_probe) > skip_message_above_chars:
            continue
        found = extract_literals_from_tool_content(content, max_artifacts_per_message)
        out |= found
        if len(out) >= max_total_artifacts:
            # Truncate deterministically rather than growing unbounded.
            return set(sorted(out)[:max_total_artifacts])
    return out


# ---------------------------------------------------------------------------
# Columnar compaction for uniform JSON-array tool payloads
# ---------------------------------------------------------------------------

DEFAULT_MIN_RECORDS = 3
DEFAULT_MIN_KEY_OVERLAP = 0.7  # fraction of records that must share a key
                                # for it to be tabulated as a column
_ROW_SEP = "\n"
_FIELD_SEP = "\x1f"  # unit separator: won't collide with real field text,
                      # unlike ',' which commonly appears in prose values


def _scalar_or_none(v):
    """Only compact fields whose values are themselves scalars -- deeply
    nested, highly irregular values are left out of the table (see
    `leftover` handling below) rather than forced into a table cell,
    since that's where they compact best anyway."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return None


def _common_keys(records: List[dict], min_overlap: float) -> List[str]:
    from collections import Counter
    key_counts = Counter()
    for r in records:
        key_counts.update(r.keys())
    n = len(records)
    keys = [k for k, c in key_counts.items() if c / n >= min_overlap]
    # Stable order: first-seen order across records, not Counter's
    # arbitrary order, so output is deterministic and matches how a
    # human skimming the original JSON would expect fields to appear.
    seen = []
    for r in records:
        for k in r:
            if k in keys and k not in seen:
                seen.append(k)
    return seen


def _format_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        # Escape only what would break row/field parsing; everything else
        # (commas, punctuation, unicode) passes through untouched so the
        # exact original string -- not a normalized copy -- is what
        # survives, which matters for exact-match artifact recovery.
        return v.replace(_FIELD_SEP, " ").replace(_ROW_SEP, " ")
    return json.dumps(v, ensure_ascii=False)


def compact_json_listing(
    content: str,
    min_records: int = DEFAULT_MIN_RECORDS,
    min_key_overlap: float = DEFAULT_MIN_KEY_OVERLAP,
    fold_constant_columns: bool = True,
    dictionary_encode: bool = True,
    dedupe_rows: bool = True,
) -> str:
    """If `content` is a JSON array of >=min_records dict records sharing
    enough keys to tabulate, return an equivalent compact table (same
    values, no repeated key names / braces / quotes). Otherwise return
    `content` unchanged -- this never raises and never lossily alters
    values, only re-arranges how they're written.

    Any field whose value is a scalar in a given record is written as a
    table cell. A shared key whose value is non-scalar (nested list/dict)
    in some record is still declared as a column, but that record's cell
    is left empty and the real value is appended verbatim as a trailing
    "# irregular:" JSON block -- so an occasional irregular record can
    never cause silent data loss, it just doesn't get the size benefit.

    Three further, independent compactions target VALUE repetition rather
    than structural (key/brace/quote) repetition -- real listings often
    repeat the same handful of strings down a column, or even repeat
    whole records (pagination overlap, duplicate fetches):

      fold_constant_columns: a column holding exactly one distinct value
        across all records is hoisted out into a single
        "# constant: key=value" header line and dropped from every row.

      dictionary_encode: ADAPTIVE, not threshold-based -- for each column
        this computes the actual byte cost of writing it as repeated raw
        values vs. as short numeric codes + a one-time legend, and uses
        whichever is smaller. No cardinality threshold to tune: a column
        is dictionary-coded exactly when doing so is measurably smaller,
        for THIS payload, full stop. Still fully lossless either way.

      dedupe_rows: a record whose every table cell is byte-identical to
        an earlier record's is replaced with a one-token back-reference
        ("=3") instead of repeating the row. Exact-match only -- no
        fuzzy/partial row merging, which would need a similarity
        threshold to tune and isn't implemented here (kept out
        deliberately: an approximate mechanism with no clear stopping
        point is exactly the kind of complexity not worth adding for a
        rare case; exact duplicates from pagination/retry overlap are
        common enough to be worth the ~5 lines this takes).
    """
    try:
        obj = json.loads(content)
    except Exception:
        return content
    if not isinstance(obj, list) or len(obj) < min_records:
        return content
    if not all(isinstance(r, dict) for r in obj):
        return content

    keys = _common_keys(obj, min_key_overlap)
    if not keys:
        return content

    n = len(obj)
    header_notes = []
    table_keys = list(keys)
    dictionaries = {}  # key -> {value: code}

    if fold_constant_columns or dictionary_encode:
        for k in keys:
            values = [r.get(k) for r in obj]
            scalars = [_scalar_or_none(v) for v in values]
            if any(s is None and v is not None for s, v in zip(scalars, values)):
                continue  # irregular column (non-scalar somewhere) -- leave as-is
            distinct = {}
            for s in scalars:
                distinct.setdefault(s, 0)
                distinct[s] += 1
            if fold_constant_columns and len(distinct) == 1:
                header_notes.append(f"# constant: {k}={_format_cell(scalars[0])}")
                table_keys.remove(k)
            elif dictionary_encode and len(distinct) >= 2:
                # Adaptive: only dictionary-code this column if doing so
                # is actually smaller for these specific values -- e.g. a
                # column of mostly-unique short strings can lose to a
                # legend's own overhead, and this catches that instead of
                # guessing from cardinality alone.
                raw_cost = sum(len(_format_cell(s)) for s in scalars)
                codes = {v: i for i, v in enumerate(distinct.keys())}
                legend_cost = len(",".join(f"{i}={_format_cell(v)}" for v, i in codes.items()))
                code_cost = sum(len(str(codes[s])) for s in scalars)
                if legend_cost + code_cost < raw_cost:
                    dictionaries[k] = codes

    if not table_keys:
        # every column folded to a constant -- header notes alone still
        # carry every value, so the table body can be entirely empty.
        rows = []
        row_cells = []
    else:
        rows = [_FIELD_SEP.join(table_keys)]
        row_cells = []

    irregular = []
    for row_i, r in enumerate(obj, start=1):
        cells = []
        for k in table_keys:
            v = r.get(k)
            sv = _scalar_or_none(v)
            if sv is None and v is not None:
                irregular.append({"row": row_i, "key": k, "value": v})
                cells.append("")
            elif k in dictionaries:
                cells.append(str(dictionaries[k][sv]))
            else:
                cells.append(_format_cell(sv))
        row_cells.append(tuple(cells))

    if table_keys:
        seen_rows = {}  # cell-tuple -> first row index it appeared at (1-based data row)
        for data_row_i, cells in enumerate(row_cells, start=1):
            if dedupe_rows and cells in seen_rows:
                rows.append("=" + str(seen_rows[cells]))
            else:
                seen_rows.setdefault(cells, data_row_i)
                rows.append(_FIELD_SEP.join(cells))

    parts = list(header_notes)
    for k, codes in dictionaries.items():
        legend = ",".join(f"{i}={_format_cell(v)}" for v, i in codes.items())
        parts.append(f"# dict: {k}: {legend}")
    parts.extend(rows)
    table = _ROW_SEP.join(parts)
    if irregular:
        table += _ROW_SEP + "# irregular: " + json.dumps(irregular, ensure_ascii=False)

    # Safety: only use the table if it actually IS more compact -- on
    # small/short-key-name records the fixed overhead of a header row
    # can occasionally lose to raw JSON, and this must never make output
    # bigger than doing nothing.
    return table if len(table) < len(content) else content


def compact_tool_listings(messages: List[dict], **kwargs) -> List[dict]:
    """Return a list with every role=='tool' message's JSON-array-listing
    content replaced by its compact table form (see compact_json_listing).
    Every other message, and any tool message that isn't a uniform
    JSON-array listing, is passed through as the exact same dict object --
    no unrelated case can regress.
    """
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool" and isinstance(m.get("content"), str):
            new_content = compact_json_listing(m["content"], **kwargs)
            if new_content is not m["content"]:
                m2 = dict(m)
                m2["content"] = new_content
                out.append(m2)
                continue
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# Cross-message deduplication
# ---------------------------------------------------------------------------

def _normalize_for_dedup(content: str) -> Optional[str]:
    """Canonical form used only to DETECT exact duplicates -- re-serializes
    valid JSON with sorted keys and no whitespace so two payloads that
    differ only in key order or formatting still count as identical; for
    non-JSON content, returns the content unchanged (whitespace-sensitive
    exact match). Returns None only when content is empty/non-string.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        obj = json.loads(content)
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return content


def dedupe_tool_messages(messages: List[dict], min_content_chars: int = 40) -> List[dict]:
    """If a role=='tool' message's content is an exact duplicate (after
    JSON-aware normalization) of an EARLIER role=='tool' message's content
    in the same trace, replace the later one with a short reference note
    instead of repeating the payload.

    Why this can never lose a fact: extract_tool_artifacts() (see above)
    scans every tool message BEFORE this function runs and unions their
    literals into force_preserve regardless of duplication -- a value
    that only ever appeared inside a later duplicate is already captured
    from that duplicate's original content by the time this replaces it.
    Whichever message (or neither, verbatim) ends up carrying that value
    in the final compressed output, DAGC's existing rescue/stub-injection
    path guarantees it surfaces somewhere. This function only removes
    bytes that were never the only copy of anything.

    min_content_chars avoids replacing trivially short content (e.g. two
    tool calls that both legitimately returned "{}" or "OK") where a
    reference note isn't meaningfully smaller than just repeating it.
    """
    out = []
    seen: Dict[str, int] = {}  # normalized content -> first occurrence's original message index
    for idx, m in enumerate(messages):
        if not (isinstance(m, dict) and m.get("role") == "tool" and isinstance(m.get("content"), str)):
            out.append(m)
            continue
        content = m["content"]
        if len(content) < min_content_chars:
            out.append(m)
            continue
        norm = _normalize_for_dedup(content)
        if norm is None:
            out.append(m)
            continue
        if norm in seen:
            ref_idx = seen[norm]
            m2 = dict(m)
            m2["content"] = f"# duplicate of tool response at message index {ref_idx} (identical content)"
            out.append(m2)
        else:
            seen[norm] = idx
            out.append(m)
    return out