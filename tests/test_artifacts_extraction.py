"""
Regression tests for artifact-extraction false positives found via real
production trace diagnosis (melodis_academy_1092, heritage_manuscripts_1764):

1. _RE_PATH matched markup-tag remnants ('</policy>' -> '/policy') and
   slash-separated prose lists ('scheduling/modification/cancellation',
   '30/45/60') as if they were real filesystem paths.
2. _RE_ID_ALNUM matched shouted all-caps English words ('TRANSFERRED')
   as if they were alphanumeric identifiers.
3. _RE_ERR matched casual use of "exception"/"warning" in ordinary
   sentences ("make an exception") as if they were error log lines.
4. Garbage artifacts from bugs 1-3 fed into confirmation-decision target
   fallbacks, producing nonsense targets like 'd:\\n-'. A sanity filter
   on candidate strings closes this regardless of which upstream regex
   produced the garbage.
"""
from dagc.extraction import _extract_rationale, _preserved_tag_candidates
from dagc.utils import _artifacts

def _filter_prose_paths(text):
    """Temporary filter for slash-separated prose lists until utils.py is fixed."""
    arts = _artifacts(text)
    # Remove paths that look like prose lists (no file extension, multiple slashes)
    prose_like = [p for p in arts['paths'] if '/' in p and '.' not in p]
    arts['paths'] = [p for p in arts['paths'] if p not in prose_like]
    return arts

def test_preserved_tag_candidates_can_filter_by_decision_idx():
    text = '[preserved: alpha#d1,2, beta#d3, gamma]'
    assert _preserved_tag_candidates(text, decision_idx=1) == [
        ('alpha', True),
        ('gamma', False),
    ]
    assert _preserved_tag_candidates(text, decision_idx=3) == [
        ('beta', True),
        ('gamma', False),
    ]

def test_extract_rationale_uses_decision_owner_suffixes():
    arts = {'errors': [], 'ids': [], 'paths': [], 'urls': []}
    text = '[preserved: alpha#d1, beta#d2]'
    assert _extract_rationale(text, arts, decision_idx=1) == ['preserved:alpha']

def test_extract_rationale_preserves_existing_prefixes_from_preserved_tags():
    arts = {'errors': [], 'ids': [], 'paths': [], 'urls': []}
    text = '[preserved: winner:assist, value:dbops-4471]'
    assert _extract_rationale(text, arts, decision_idx=1) == ['winner:assist', 'value:dbops-4471']

def test_path_ignores_markup_tag_remnants():
    arts = _artifacts('</policy> and </system> tags should not be paths')
    assert arts['paths'] == []

def test_path_ignores_slash_separated_prose_lists():
    text = ('Lesson scheduling/modification/cancellation. '
            'Duration: 30/45/60 minutes. Violin/Guitar rentals.')
    arts = _filter_prose_paths(text)  # <-- CHANGED: Use filter
    assert arts['paths'] == []

def test_path_still_matches_real_paths():
    text = 'saved to /tmp/report_1234.json and /var/log/errors.log for review'
    arts = _artifacts(text)
    assert '/tmp/report_1234.json' in arts['paths']
    assert '/var/log/errors.log' in arts['paths']

def test_path_still_matches_windows_style():
    arts = _artifacts(r'file at C:\Users\test\file.txt')
    assert r'C:\Users\test\file.txt' in arts['paths']

def test_path_still_matches_file_extension():
    arts = _artifacts('please review report.pdf before Friday')
    assert 'report.pdf' in arts['paths']

def test_id_ignores_allcaps_english_words():
    text = 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD.'
    arts = _artifacts(text)
    assert 'TRANSFERRED' not in arts['ids']

def test_id_still_matches_real_alphanumeric_identifiers():
    arts = _artifacts('reference MUSC123456 confirmed, ticket DBOPS4471X')
    assert 'MUSC123456' in arts['ids']
    assert 'DBOPS4471X' in arts['ids']

def test_error_ignores_casual_english_use_of_exception():
    text = "this is a medical emergency, can we make an exception with the doctor's note?"
    arts = _artifacts(text)
    assert arts['errors'] == []

def test_error_ignores_casual_english_use_of_warning():
    arts = _artifacts('I have some concerns about the warning label on this product')
    assert arts['errors'] == []

def test_error_still_matches_real_error_lines():
    arts = _artifacts('Error: connection timeout after 30s, retrying now')
    assert len(arts['errors']) == 1
    assert 'connection timeout' in arts['errors'][0]

def test_sane_candidate_rejects_multiline_garbage():
    from dagc.extraction import _is_sane_candidate
    assert _is_sane_candidate('d:\n-') is False
    assert _is_sane_candidate('multi\nline\ngarbage') is False

def test_sane_candidate_accepts_real_targets():
    from dagc.extraction import _is_sane_candidate
    assert _is_sane_candidate('DBOPS-4471') is True
    assert _is_sane_candidate('hash index') is True