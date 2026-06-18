"""The IR metrics — pure functions over per-rank binary relevance (best-first)."""

import math

import pytest

from tarnrag.eval import metrics


def test_hit_at_k():
    assert metrics.hit_at_k([False, True, False], 3) == 1.0
    assert metrics.hit_at_k([False, True, False], 1) == 0.0  # relevant at rank 2, not in top-1
    assert metrics.hit_at_k([], 5) == 0.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank([True, False]) == 1.0
    assert metrics.reciprocal_rank([False, True, True]) == 0.5  # first relevant at rank 2
    assert metrics.reciprocal_rank([False, False]) == 0.0


def test_ndcg_at_k():
    assert metrics.ndcg_at_k([True, False, False], 3) == 1.0  # relevant first = the ideal ordering
    assert metrics.ndcg_at_k([False, False], 2) == 0.0  # nothing relevant
    # one relevant at rank 2: DCG = 1/log2(3); IDCG (ideal: relevant first) = 1/log2(2) = 1
    assert metrics.ndcg_at_k([False, True], 2) == pytest.approx(1 / math.log2(3))
    # k truncates: a relevant result past k doesn't count
    assert metrics.ndcg_at_k([False, False, True], 2) == 0.0
