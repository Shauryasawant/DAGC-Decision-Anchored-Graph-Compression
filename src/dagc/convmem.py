
from __future__ import annotations

import itertools
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_RE_STATE_CUE = re.compile(
    r"\b("
    r"i(?:'m| am)|i live|i work|i use|i have|i moved to|i graduated from|"
    r"i studied|i majored in|i'?m allergic to|i can'?t|i don'?t|"
    r"my (?:\w+ ){0,2}is|my (?:\w+ ){0,2}are|i'?m based in|i was born in|"
    r"i grew up in|i'?ve been (?:using|working)|our (?:team|company|budget|"
    r"deadline) is|"
    r"i like|i love|i hate|i enjoy|i dislike|i prefer|i'?m into|"
    r"my (?:wife|husband|partner|girlfriend|boyfriend|son|daughter|kids|"
    r"mother|father|mom|dad|sister|brother|friend|boss|manager|colleague)|"
    r"yesterday i|last week i|last year i|recently i|a few days ago i"
    r")\b",
    re.IGNORECASE)

_RE_GOAL_CUE = re.compile(
    r"\b(?:i (?:want|need|plan|hope|intend|would like) to|i'?m trying to|"
    r"remind me to|don'?t let me forget|i'?m planning to|i'?m working "
    r"towards|my goal is|our goal is|my target is|our target is|"
    r"my priority is|our priority is|my objective is|our objective is|"
    r"i'?d like to eventually|next i want to|"
    r"i'?m looking for|i'?m searching for|i'?m considering|i might)\b",
    re.IGNORECASE)

_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|,\s+(?=\w)")

_STOPWORDS = {
    "i", "im", "i'm", "am", "is", "are", "was", "were", "my", "the", "a",
    "an", "to", "of", "in", "on", "at", "and", "or", "for", "it", "that",
    "this", "be", "have", "has", "had", "do", "does", "did", "not", "no",
    "so", "as", "with", "we", "you", "your", "me", "our",
    # generic template/connective words that show up across many
    # different "my X is Y" facts without carrying the actual topic
    # (e.g. "my favorite color" vs "my favorite season" share only
    # "favorite" -- that's not enough to call them the same fact).
    "favorite", "now", "actually", "correction", "instead", "still",
    "recently", "currently", "new", "name",
}


def _get_text(m: Dict) -> str:
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return str(c)


def _clauses(text: str) -> List[str]:
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]


def _content_tokens(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in toks if t not in _STOPWORDS and len(t) > 2}


def classify(messages: List[Dict]) -> Dict[str, List[Dict]]:
    """Default extractor. Returns {'state': [...], 'goal': [...]}, each
    entry {'msg_idx', 'clause'}. A clause matching both cue sets is
    filed as 'goal' only (goal cues are more specific and would
    otherwise be swallowed by the very broad generic "i'm" state cue)."""
    out: Dict[str, List[Dict]] = {"state": [], "goal": []}
    for i, m in enumerate(messages):
        for cl in _clauses(_get_text(m)):
            if _RE_GOAL_CUE.search(cl):
                out["goal"].append({"msg_idx": i, "clause": cl})
            elif _RE_STATE_CUE.search(cl):
                out["state"].append({"msg_idx": i, "clause": cl})
    return out


# ---------------------------------------------------------------------------
# Embedding backend (pluggable). Default: TF-IDF refit over the store.
# ---------------------------------------------------------------------------

class TfidfEmbedder:
    """Stateful default embedder: refits a TF-IDF vectorizer over the
    full set of texts it's given each call. Fine for stores up to a
    few thousand facts; swap in a real embedding model for anything
    bigger or for better semantic (non-lexical) recall."""

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
        try:
            mat = vec.fit_transform(texts)
        except ValueError:
            # all-stopword / empty vocabulary edge case
            return np.zeros((len(texts), 1))
        return mat.toarray()

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

THETA_RESOLVE = 0.25  # cosine threshold for candidate supersession -- kept low
# on purpose because the content-word gate (MIN_SHARED_CONTENT_TOKENS below)
# is what actually blocks false positives now, not this number. A true
# same-topic update ("I live in Pune" -> "I live in Mumbai now") scores
# ~0.3-0.4 on plain TF-IDF cosine, well below where a naive threshold like
# 0.65 would ever fire -- see bench_conflict.py for the calibration.
MIN_SHARED_CONTENT_TOKENS = 1  # gate: require >=1 shared non-stopword token


class Memory:
    """Extraction + conflict resolution + persistence + retrieval.

    m = Memory()
    m.add(messages)
    m.recall("where does the user live?", k=3)
    m.save("store.json")
    """

    def __init__(
        self,
        path: Optional[str] = None,
        autoload: bool = True,
        embedder: Optional[Any] = None,
        theta_resolve: float = THETA_RESOLVE,
    ):
        self.path = path
        self.embedder = embedder or TfidfEmbedder()
        self.theta_resolve = theta_resolve
        self._facts: List[Dict[str, Any]] = []
        self._raw_messages: List[str] = []
        self._seq = itertools.count()
        if autoload and path and os.path.exists(path):
            self.load(path)

    def _reembed_all(self) -> None:
        """Refit the embedder over every stored clause and cache each
        fact's current vector. Called after every add() so recall()
        and future resolution both operate on the same fitted space."""
        if not self._facts:
            return
        texts = [f["clause"] for f in self._facts]
        vecs = self.embedder.encode(texts)
        for f, v in zip(self._facts, vecs):
            f["_vec"] = v

    def add(
        self,
        messages: List[Dict],
        extractor: Optional[Callable[[List[Dict]], Dict[str, List[Dict]]]] = None,
        theta_resolve: Optional[float] = None,
    ) -> List[Dict]:
        extractor = extractor or classify
        theta = self.theta_resolve if theta_resolve is None else theta_resolve
        self._raw_messages.extend(_get_text(m) for m in messages if _get_text(m).strip())
        anchors = extractor(messages)
        new_items = [
            (cat, a["clause"], a["msg_idx"])
            for cat in ("state", "goal")
            for a in anchors.get(cat, [])
        ]
        if not new_items:
            return []

        added = []
        for cat, clause, msg_idx in new_items:
            new_id = next(self._seq)
            self._facts.append({
                "id": new_id, "category": cat, "clause": clause,
                "source_msg_idx": msg_idx, "active": True,
                "superseded_by": None, "created_at": time.time(),
                "_vec": None,
            })
            added.append(self._facts[-1])

        # Refit embeddings over the whole (now-updated) store once,
        # then run resolution using the fresh vectors.
        self._reembed_all()
        new_ids = {f["id"] for f in added}
        for new_fact in added:
            new_toks = _content_tokens(new_fact["clause"])
            for old in self._facts:
                if old["id"] == new_fact["id"] or old["id"] in new_ids:
                    continue
                if old["category"] != new_fact["category"] or not old["active"]:
                    continue
                sim = self.embedder.cosine(new_fact["_vec"], old["_vec"])
                shared = new_toks & _content_tokens(old["clause"])
                if sim > theta and len(shared) >= MIN_SHARED_CONTENT_TOKENS:
                    old["active"] = False
                    old["superseded_by"] = new_fact["id"]
        return added

    def recall(
        self,
        query: str,
        k: int = 3,
        include_inactive: bool = False,
        category: Optional[str] = None,
        recency_weight: float = 0.0,
        include_raw: bool = False,
    ) -> List[Dict]:
        """include_raw=True adds the raw message history as extra
        candidates alongside extracted facts, ranked jointly. This is
        the hybrid mode: extraction gives you clean, deduplicated,
        conflict-resolved facts, but its cue set is finite and will
        always miss things a real conversation says. Raw-message
        candidates are tagged category='raw' with no id/active state."""
        pool = [f for f in self._facts if f["active"] or include_inactive]
        if category:
            pool = [f for f in pool if f["category"] == category]
        if include_raw:
            pool = pool + [
                {"id": None, "category": "raw", "clause": msg, "active": True,
                 "superseded_by": None, "created_at": 0.0}
                for msg in self._raw_messages
            ]
        if not pool:
            return []
        # Refit embedder jointly over [query] + pool so the query lands
        # in the same vector space as the facts (TF-IDF vocab needs to
        # see the query terms too).
        texts = [query] + [f["clause"] for f in pool]
        vecs = self.embedder.encode(texts)
        q_vec, fact_vecs = vecs[0], vecs[1:]
        sims = [self.embedder.cosine(q_vec, v) for v in fact_vecs]

        if recency_weight > 0:
            times = [f["created_at"] for f in pool]
            t_min, t_max = min(times), max(times)
            span = (t_max - t_min) or 1.0
            recency = [(t - t_min) / span for t in times]
            final = [(1 - recency_weight) * s + recency_weight * r
                     for s, r in zip(sims, recency)]
        else:
            final = sims

        ranked = sorted(zip(pool, sims, final), key=lambda x: x[2], reverse=True)
        return [{**{k: v for k, v in f.items() if k != "_vec"}, "score": s}
                for f, s, _ in ranked[:k]]

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        clean = [{k: v for k, v in f.items() if k != "_vec"} for f in self._facts]
        with open(path, "w") as fh:
            json.dump(clean, fh)

    def load(self, path: Optional[str] = None) -> None:
        path = path or self.path
        with open(path) as fh:
            self._facts = json.load(fh)
        for f in self._facts:
            f["_vec"] = None
        max_id = max((f["id"] for f in self._facts), default=-1)
        self._seq = itertools.count(max_id + 1)

    def all(self, active_only: bool = True) -> List[Dict]:
        return [f for f in self._facts if f["active"] or not active_only]


if __name__ == "__main__":
    m = Memory(autoload=False)
    m.add([{"role": "user", "content": "I live in Pune. I love hiking. My manager is Priya."}])
    m.add([{"role": "user", "content": "Actually, I live in Mumbai now, not Pune."}])
    m.add([{"role": "user", "content": "My deadline is next Friday. I'm looking for a new apartment."}])
    print("Recall 'where do I live':")
    for f in m.recall("where do I live", k=3):
        print(f"  [{f['score']:.2f}] active={f['active']} {f['clause']!r}")
    print("\nAll facts:")
    for f in m.all(active_only=False):
        print(f"  id={f['id']} cat={f['category']} active={f['active']} superseded_by={f['superseded_by']} {f['clause']!r}")