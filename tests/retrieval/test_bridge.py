"""The Phase-2 bridge retrieval components: multi-query expansion (MultiQueryRetriever) + the LLM
relevance judge (LlmJudgeReranker). Driven by a deterministic ``StaticLanguageModel`` + tiny fakes — no
network, no key, no model."""

import json

import pytest

from tarnrag.contracts import Candidate, ChunkProvenance, RetrievalResult
from tarnrag.contracts.structure import Span
from tarnrag.core.exceptions import RetrievalError
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.retrieval.components.reranker import LlmJudgeReranker
from tarnrag.retrieval.components.retriever import MultiQueryRetriever, RetrievalContext
from tarnrag.retrieval.types import Query


class _FakeStore:
    """Records dense_knn calls and returns a canned candidate list each time."""

    def __init__(self, results):
        self._results = results
        self.calls = 0

    async def dense_knn(self, vec, k, chunk_filter):
        self.calls += 1
        return list(self._results)


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.0, 0.0, 0.0]


def _ctx(store, llm=None):
    return RetrievalContext(store=store, embedder=_FakeEmbedder(), llm=llm)


def _result(chunk_id, text):
    return RetrievalResult(
        chunk_id=chunk_id, text=text, score=0.0, component_scores={}, document_id="d",
        source_kind="text", standard_id=None, locator=None, license_class="open",
        provenance=ChunkProvenance(geometry=[Span(start=0, end=len(text))], header_path=[], content_hash="h"),
    )


# ---------------- MultiQueryRetriever ----------------


async def test_multi_query_expands_and_rrf_fuses():
    store = _FakeStore([Candidate("a", 1, 0.1), Candidate("b", 2, 0.2)])
    llm = StaticLanguageModel(json.dumps({"queries": ["rewrite one", "rewrite two"]}))
    out = await MultiQueryRetriever(MultiQueryRetriever.Config(num_variants=2)).retrieve(
        Query(text="q"), _ctx(store, llm)
    )
    assert store.calls == 3  # original + 2 rewrites, each dense-retrieved
    assert [c.chunk_id for c in out] == ["a", "b"]  # deduped + RRF-fused (a out-ranks b)
    assert [c.rank for c in out] == [1, 2]  # re-ranked, 1-based


async def test_multi_query_caps_at_num_variants():
    store = _FakeStore([Candidate("a", 1, 0.1)])
    llm = StaticLanguageModel(json.dumps({"queries": ["r1", "r2", "r3", "r4", "r5"]}))
    await MultiQueryRetriever(MultiQueryRetriever.Config(num_variants=2)).retrieve(Query(text="q"), _ctx(store, llm))
    assert store.calls == 3  # original + only num_variants (2) rewrites


async def test_multi_query_without_llm_is_single_dense():
    store = _FakeStore([Candidate("a", 1, 0.1)])
    out = await MultiQueryRetriever(MultiQueryRetriever.Config()).retrieve(Query(text="q"), _ctx(store, llm=None))
    assert store.calls == 1 and [c.chunk_id for c in out] == ["a"]


async def test_multi_query_unparseable_expansion_uses_original_only():
    store = _FakeStore([Candidate("a", 1, 0.1)])
    await MultiQueryRetriever(MultiQueryRetriever.Config()).retrieve(
        Query(text="q"), _ctx(store, StaticLanguageModel("not json at all"))
    )
    assert store.calls == 1  # no parseable rewrites -> just the original query


# ---------------- LlmJudgeReranker ----------------


async def test_llm_judge_rescores_and_reorders():
    results = [_result("a", "alpha"), _result("b", "beta"), _result("c", "gamma")]
    reply = json.dumps(
        {"scores": [{"passage": 1, "score": 1}, {"passage": 2, "score": 9}, {"passage": 3, "score": 5}]}
    )
    out = await LlmJudgeReranker(LlmJudgeReranker.Config()).rerank(
        Query(text="q"), results, _ctx(store=None, llm=StaticLanguageModel(reply))
    )
    assert [r.chunk_id for r in out] == ["b", "c", "a"]  # by judged score, desc
    assert out[0].score == 9.0 and out[0].component_scores["llm_judge"] == 9.0


async def test_llm_judge_unscored_get_zero_with_deterministic_tiebreak():
    results = [_result("b", "x"), _result("a", "y")]  # neither scored -> 0/0, tie broken by chunk_id asc
    out = await LlmJudgeReranker(LlmJudgeReranker.Config()).rerank(
        Query(text="q"), results, _ctx(store=None, llm=StaticLanguageModel("{}"))
    )
    assert [r.chunk_id for r in out] == ["a", "b"]


async def test_llm_judge_ignores_out_of_range_and_non_numeric_scores():
    results = [_result("a", "x"), _result("b", "y")]
    reply = json.dumps({"scores": [{"passage": 5, "score": 9}, {"passage": 1, "score": "high"}, {"passage": 2, "score": 3}]})
    out = await LlmJudgeReranker(LlmJudgeReranker.Config()).rerank(
        Query(text="q"), results, _ctx(store=None, llm=StaticLanguageModel(reply))
    )
    assert [r.chunk_id for r in out] == ["b", "a"]  # b=3, a stayed 0 (non-numeric ignored; 5 out of range)


async def test_llm_judge_requires_an_llm():
    with pytest.raises(RetrievalError, match="no LLM"):
        await LlmJudgeReranker(LlmJudgeReranker.Config()).rerank(
            Query(text="q"), [_result("a", "x")], _ctx(store=None, llm=None)
        )


async def test_llm_judge_empty_results_is_noop():
    out = await LlmJudgeReranker(LlmJudgeReranker.Config()).rerank(Query(text="q"), [], _ctx(store=None))
    assert out == []
