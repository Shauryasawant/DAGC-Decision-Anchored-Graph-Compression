"""
Dependency graph, Reasoning-Chain Integrity (RCI), the causal message
graph used for the minimal-sufficient-set (M*), and the spectral
compressor. No LLM dependency.
"""
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

from .utils import STOPWORDS, _artifacts, _get_text


def build_dependency_graph(messages, decisions):
    first_seen = {}
    for i, m in enumerate(messages):
        arts = _artifacts(_get_text(m))
        for k in ('paths', 'ids'):
            for a in arts[k]:
                if a not in first_seen:
                    first_seen[a] = i

    from .compressor import _decision_critical_values
    crit_vals = _decision_critical_values(decisions)
    for a in crit_vals:
        if a in first_seen:
            continue
        for i, m in enumerate(messages):
            if a in _get_text(m):
                first_seen[a] = i
                break

    edges = []
    for dec_idx, d in enumerate(decisions):
        cited = set(d['artifacts'].get('paths', []) + d['artifacts'].get('ids', []))
        cited |= _decision_critical_values([d])
        for a in cited:
            origin = first_seen.get(a)
            if origin is not None and origin < d['msg_idx']:
                edges.append({'to_decision_idx': dec_idx, 'artifact': a,
                              'origin_msg_idx': origin, 'decision_msg_idx': d['msg_idx']})
    return edges


def attach_dependencies(messages, decisions):
    from .compressor import _decision_critical_values
    art_origin = {}
    for dec_idx, d in enumerate(decisions):
        produced = set(d['artifacts'].get('paths', []) + d['artifacts'].get('ids', []))
        produced |= _decision_critical_values([d])
        for a in produced:
            if a not in art_origin:
                art_origin[a] = dec_idx
    for dec_idx, d in enumerate(decisions):
        cited = set(d['artifacts'].get('paths', []) + d['artifacts'].get('ids', []))
        cited |= _decision_critical_values([d])
        d['depends_on'] = sorted({art_origin[a] for a in cited
                                   if a in art_origin and art_origin[a] < dec_idx})
    return decisions


def compute_rci(messages, compressed, decisions):
    edges = build_dependency_graph(messages, decisions)
    if not edges:
        return {'RCI': None, 'edges_total': 0, 'edges_preserved': 0, 'edges': []}
    comp_text = ' '.join(_get_text(m) for m in compressed)
    preserved = 0
    detail = []
    for e in edges:
        ok = e['artifact'] in comp_text
        preserved += int(ok)
        detail.append({**e, 'preserved': ok})
    return {'RCI': round(preserved / len(edges), 4),
            'edges_total': len(edges), 'edges_preserved': preserved, 'edges': detail}


def compute_chain_rci(messages, compressed, decisions):
    decisions = attach_dependencies(messages, decisions)
    comp_idx = {cm['_orig_idx'] for cm in compressed if '_orig_idx' in cm}
    chains_total = chains_intact = 0
    detail = []
    for dec_idx, d in enumerate(decisions):
        if not d.get('depends_on'):
            continue
        chains_total += 1
        own = d['msg_idx'] in comp_idx
        deps = all(decisions[dep]['msg_idx'] in comp_idx for dep in d['depends_on'])
        chains_intact += int(own and deps)
        detail.append({'decision_idx': dec_idx, 'depends_on': d['depends_on'],
                       'own_present': own, 'deps_present': deps, 'intact': own and deps})
    if chains_total == 0:
        return {'chain_RCI': None, 'chains_total': 0, 'chains_intact': 0, 'detail': detail}
    return {'chain_RCI': round(chains_intact / chains_total, 4),
            'chains_total': chains_total, 'chains_intact': chains_intact, 'detail': detail}



@dataclass
class CausalGraphConfig:
    CAUSAL_TAU: float = 5.0
    ADD_SEQ_EDGES: bool = False
    SEQ_KEYWORD_OVERLAP: int = 4
    MIN_KEYWORD_LEN: int = 5
    IDF_SMOOTH: float = 1.0


class CausalMessageGraph:
    def __init__(self, messages, decisions, cfg=None):
        self.cfg = cfg or CausalGraphConfig()
        self.n = len(messages)
        self.messages = messages
        self.decisions = decisions
        self.adj = defaultdict(set)
        self.rev_adj = defaultdict(set)
        self.art_producer = {}
        self.art_idf = {}
        self._build()
        self._idf()

    def _build(self):
        for i, m in enumerate(self.messages):
            for kind in ('paths', 'ids', 'errors'):
                for a in _artifacts(_get_text(m))[kind]:
                    if a not in self.art_producer:
                        self.art_producer[a] = i

        for j, m in enumerate(self.messages):
            consumed = set()
            for kind in ('paths', 'ids', 'errors'):
                consumed |= set(_artifacts(_get_text(m))[kind])
            for a in consumed:
                i = self.art_producer.get(a, j)
                if i < j:
                    self.adj[i].add(j)
                    self.rev_adj[j].add(i)

        if self.cfg.ADD_SEQ_EDGES:
            import re
            kl = self.cfg.MIN_KEYWORD_LEN
            pat = re.compile(rf'\b[a-z]{{{kl},}}\b')
            for i in range(self.n - 1):
                wi = set(pat.findall(_get_text(self.messages[i]).lower())) - STOPWORDS
                wj = set(pat.findall(_get_text(self.messages[i + 1]).lower())) - STOPWORDS
                if len(wi & wj) >= self.cfg.SEQ_KEYWORD_OVERLAP:
                    self.adj[i].add(i + 1)
                    self.rev_adj[i + 1].add(i)

    def _idf(self):
        freq = defaultdict(int)
        for m in self.messages:
            seen = set()
            for kind in ('paths', 'ids', 'errors'):
                for a in _artifacts(_get_text(m))[kind]:
                    if a not in seen:
                        freq[a] += 1
                        seen.add(a)
        s = self.cfg.IDF_SMOOTH
        self.art_idf = {a: math.log((self.n + s) / (f + s)) + 1.0 for a, f in freq.items()}

    def ancestors(self, node):
        if not hasattr(self, '_anc_cache'):
            self._anc_cache = {}
        if node in self._anc_cache:
            return self._anc_cache[node]
        visited, queue = set(), list(self.rev_adj.get(node, set()))
        while queue:
            c = queue.pop()
            if c not in visited:
                visited.add(c)
                queue.extend(self.rev_adj.get(c, set()))
        self._anc_cache[node] = visited
        return visited

    def minimal_sufficient_set(self):
        D_nodes = {d['msg_idx'] for d in self.decisions if d['msg_idx'] < self.n}
        M_star = set(D_nodes)
        for dn in D_nodes:
            M_star |= self.ancestors(dn)
        for i, m in enumerate(self.messages):
            if m.get('role') == 'system':
                M_star.add(i)
        for i in range(self.n - 1, -1, -1):
            if self.messages[i].get('role') == 'user':
                M_star.add(i)
                break
        return M_star

    def causal_flow_matrix(self):
        if not self.decisions:
            return np.zeros((self.n, 1), dtype=np.float32)
        tau = self.cfg.CAUSAL_TAU
        F = np.zeros((self.n, len(self.decisions)), dtype=np.float32)
        for j, d in enumerate(self.decisions):
            dn = d['msg_idx']
            if dn >= self.n:
                continue
            anc_d = self.ancestors(dn) | {dn}
            d_arts = set()
            for kind in ('paths', 'ids', 'errors'):
                d_arts |= set(d['artifacts'].get(kind, []))
            for i in anc_d:
                if i >= self.n:
                    continue
                m_arts = set()
                for kind in ('paths', 'ids', 'errors'):
                    m_arts |= set(_artifacts(_get_text(self.messages[i]))[kind])
                rel = (m_arts & d_arts) if d_arts else m_arts
                if rel:
                    F[i, j] = (sum(self.art_idf.get(a, 1.0) for a in rel)
                                * math.exp(-(dn - i) / max(tau, 1.0)))
        return F

    def compression_bound(self):
        M_star = self.minimal_sufficient_set()
        rate = len(M_star) / max(1, self.n)
        return {
            'achievable_zero_loss_compression': round(1.0 - rate, 4),
            'M_star_fraction': round(rate, 4),
            'M_star_size': len(M_star),
            'n_messages': self.n,
            'droppable_messages': self.n - len(M_star),
        }


class SpectralCompressor:
    """Optional: ranks messages by an SVD-derived causal-importance score.
    Off by default in DAGCConfig (USE_SPECTRAL=False); purely linear
    algebra, no LLM."""
    def __init__(self, graph: CausalMessageGraph):
        self.graph = graph
        self.F = graph.causal_flow_matrix()
        self.U = None
        self.S = None
        self._imp = np.zeros(graph.n, dtype=np.float32)
        self._decompose()

    def _decompose(self):
        if self.F.size == 0 or self.F.max() == 0:
            return
        try:
            U, S, _ = np.linalg.svd(self.F, full_matrices=False)
            self.U, self.S = U, S
            self._imp = np.sum((U * (S[np.newaxis, :] ** 2)) * U, axis=1).astype(np.float32)
        except np.linalg.LinAlgError:
            self._imp = np.linalg.norm(self.F, axis=1).astype(np.float32)

    def score(self, i):
        return float(self._imp[i]) if i < len(self._imp) else 0.0

    def normalized_scores(self):
        mx = float(self._imp.max())
        if mx == 0:
            return {i: 0.0 for i in range(self.graph.n)}
        return {i: float(self._imp[i] / mx) for i in range(self.graph.n)}

    def explained_variance_ratio(self, k):
        if self.S is None or self.S.sum() == 0:
            return 0.0
        s2 = self.S ** 2
        return float(s2[:k].sum() / s2.sum())
