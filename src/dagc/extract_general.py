
from __future__ import annotations

from typing import Dict, List

import spacy

_NLP = None

_FIRST_PERSON = {"i", "we", "my", "our", "me", "us"}
# Closed-class intent/modal verbs -- grammatical category, not topic
# vocabulary. This is the one list left, and it's a small closed set
# in English (unlike nouns, which are open-ended).
_INTENT_LEMMAS = {
    "want", "need", "plan", "hope", "intend", "try", "consider", "might",
    "look",  # "look for/into" (prep-checked below to avoid "look at that")
}
_INTENT_PREPS = {"for", "into", "towards", "toward"}


def _get_text(m: Dict) -> str:
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return str(c)


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _is_first_person_sentence(sent) -> bool:
    for tok in sent:
        if tok.lower_ in _FIRST_PERSON and tok.dep_ in (
            "nsubj", "nsubjpass", "poss", "dobj", "pobj",
        ):
            return True
    return False


def _is_goal(sent) -> bool:
    for tok in sent:
        if tok.dep_ == "ROOT" and tok.lemma_.lower() in _INTENT_LEMMAS:
            if tok.lemma_.lower() != "look":
                return True
            # "look for a job" (goal) vs "look tired" (not) -- require
            # a following preposition from the intent set
            for child in tok.children:
                if child.dep_ == "prep" and child.lower_ in _INTENT_PREPS:
                    return True
    return False


def classify_general(messages: List[Dict]) -> Dict[str, List[Dict]]:
    """Drop-in replacement for convmem.classify with the same return
    shape: {'state': [...], 'goal': [...]}, each {'msg_idx', 'clause'}."""
    nlp = _nlp()
    out: Dict[str, List[Dict]] = {"state": [], "goal": []}
    for i, m in enumerate(messages):
        text = _get_text(m).strip()
        if not text:
            continue
        doc = nlp(text)
        for sent in doc.sents:
            if len(sent.text.strip()) < 4:
                continue
            if not _is_first_person_sentence(sent):
                continue
            cat = "goal" if _is_goal(sent) else "state"
            out[cat].append({"msg_idx": i, "clause": sent.text.strip()})
    return out