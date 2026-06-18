"""IR metrics for the retrieval eval harness — pure functions over per-rank binary relevance.

Each takes ``relevances``: the ranked results' relevance flags, best-first (``relevances[i]`` is whether
the rank-``i`` result is relevant). Binary relevance keeps the harness's content-based labels honest (a
result is relevant iff its text contains a gold phrase — see ``dataset``) and the dataset format simple.
"""

from __future__ import annotations

import math


def hit_at_k(relevances: list[bool], k: int) -> float:
    """1.0 if any of the top-``k`` results is relevant, else 0.0 — the RAG "did we retrieve a relevant
    chunk at all" notion of recall@k."""
    return 1.0 if any(relevances[:k]) else 0.0


def reciprocal_rank(relevances: list[bool]) -> float:
    """1 / (1-based rank of the first relevant result), or 0.0 if none is relevant (drives MRR)."""
    for i, relevant in enumerate(relevances):
        if relevant:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(relevances: list[bool], k: int) -> float:
    """Discounted cumulative gain over the top-``k`` (binary gains: 1 per relevant result)."""
    return sum(1.0 / math.log2(i + 2) for i, relevant in enumerate(relevances[:k]) if relevant)


def ndcg_at_k(relevances: list[bool], k: int) -> float:
    """DCG@k normalized by the ideal ordering (all relevant results first); 0.0 if none is relevant."""
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg_at_k(relevances, k) / ideal if ideal > 0 else 0.0
