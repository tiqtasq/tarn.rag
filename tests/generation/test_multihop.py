"""The multi-hop reasoners: IterativeReasoner (retrieve<->read loop) + DecompositionReasoner (sub-questions)."""

import json

import pytest

from bausatz import ComponentFactory
from tarnrag.core.exceptions import GenerationError
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.generation import (
    DecompositionReasoner,
    GenerationContext,
    GenerationPipeline,
    GroundedRetrievalReasoner,
    IterativeReasoner,
)
from tarnrag.retrieval.types import Query


class _RoutedRetrieval:
    """A retrieval-port double that returns different results per query (first substring match) and records
    the queries it saw — so a multi-hop reasoner actually gathers new evidence across hops."""

    def __init__(self, routes):  # list of (substring, results); "" matches anything
        self._routes = routes
        self.queries: list[str] = []

    async def search(self, query):
        self.queries.append(query.text)
        for key, results in self._routes:
            if key in query.text.lower():
                return list(results)
        return []


def _scripted(*replies):
    """A StaticLanguageModel that returns the given replies in order, one per ``complete`` call."""
    it = iter(replies)
    return StaticLanguageModel(lambda _prompt: next(it))


# ---------------- IterativeReasoner ----------------


async def test_iterative_follows_up_and_accumulates_evidence(make_result):
    a, b = make_result("a", "Alpha fact."), make_result("b", "Beta fact.")
    retrieval = _RoutedRetrieval([("alpha", [a]), ("beta", [b])])
    llm = _scripted(
        json.dumps({"done": False, "follow_up": "beta detail"}),  # hop 1: needs more
        json.dumps({"done": True, "answer": "alpha then beta", "steps": [{"claim": "c", "cited": [1, 2]}]}),
    )
    out = await IterativeReasoner(IterativeReasoner.Config()).reason(
        Query(text="alpha question"), GenerationContext(retrieval, llm)
    )
    assert out.answer == "alpha then beta"
    assert [r.chunk_id for r in out.evidence] == ["a", "b"]  # accumulated across the two hops
    assert out.steps[0].cited == [0, 1]  # cites both gathered passages
    assert retrieval.queries == ["alpha question", "beta detail"]  # hop 2 used the follow-up


async def test_iterative_stops_at_the_hop_budget(make_result):
    a = make_result("a", "Alpha.")
    retrieval = _RoutedRetrieval([("", [a])])  # any query -> [a]
    llm = _scripted(json.dumps({"done": False, "follow_up": "more"}))  # always wants more
    out = await IterativeReasoner(IterativeReasoner.Config(max_hops=1)).reason(
        Query(text="q"), GenerationContext(retrieval, llm)
    )
    assert len(retrieval.queries) == 1 and len(out.evidence) == 1  # forced to answer at the budget


async def test_iterative_cleans_the_answer(make_result):
    a = make_result("a", "Alpha.")
    retrieval = _RoutedRetrieval([("", [a])])
    llm = _scripted(json.dumps({"done": True, "answer": "The answer is alpha", "steps": [{"claim": "c", "cited": [1]}]}))
    out = await IterativeReasoner(IterativeReasoner.Config()).reason(
        Query(text="q"), GenerationContext(retrieval, llm)
    )
    assert out.answer == "alpha"  # the short-answer lead-in is stripped


async def test_iterative_dedups_evidence_across_hops(make_result):
    a = make_result("a", "Alpha.")
    retrieval = _RoutedRetrieval([("", [a])])  # every hop retrieves the same chunk
    llm = _scripted(
        json.dumps({"done": False, "follow_up": "again"}),
        json.dumps({"done": True, "answer": "ok", "steps": [{"claim": "c", "cited": [1]}]}),
    )
    out = await IterativeReasoner(IterativeReasoner.Config()).reason(
        Query(text="q"), GenerationContext(retrieval, llm)
    )
    assert len(out.evidence) == 1  # the repeat hit is not added twice


# ---------------- DecompositionReasoner ----------------


async def test_decomposition_retrieves_subquestions_then_synthesizes(make_result):
    a, b = make_result("a", "Alpha fact."), make_result("b", "Beta fact.")
    retrieval = _RoutedRetrieval([("alpha", [a]), ("beta", [b])])
    llm = _scripted(
        json.dumps({"sub_questions": ["the alpha part", "the beta part"]}),  # decompose
        json.dumps({"answer": "combined", "steps": [{"claim": "c", "cited": [1, 2]}]}),  # synthesize
    )
    out = await DecompositionReasoner(DecompositionReasoner.Config()).reason(
        Query(text="alpha and beta?"), GenerationContext(retrieval, llm)
    )
    assert out.answer == "combined"
    assert [r.chunk_id for r in out.evidence] == ["a", "b"]
    assert out.steps[0].cited == [0, 1]
    assert retrieval.queries == ["the alpha part", "the beta part"]


async def test_decomposition_falls_back_to_the_original_question(make_result):
    a = make_result("a", "Alpha.")
    retrieval = _RoutedRetrieval([("", [a])])
    llm = _scripted("not json", json.dumps({"answer": "ok", "steps": [{"claim": "c", "cited": [1]}]}))
    out = await DecompositionReasoner(DecompositionReasoner.Config()).reason(
        Query(text="original q"), GenerationContext(retrieval, llm)
    )
    assert retrieval.queries == ["original q"] and out.answer == "ok"  # decompose failed -> the question


# ---------------- DecompositionReasoner + evidence rerank (P7) ----------------


class _ScriptedCrossEncoder:
    """A ``CrossEncoder`` double: scores each passage by a canned map (0.0 if unlisted) and records the
    query it scored against."""

    def __init__(self, scores_by_text):
        self._scores = dict(scores_by_text)
        self.seen_query = None

    def score(self, query, passages):
        self.seen_query = query
        return [self._scores.get(p, 0.0) for p in passages]


def _rerank_fixture(make_result):
    """Three passages pooled in retrieval order a, b, c; the CE ranks them c > b > a against the
    ORIGINAL question. Scripted LLM: decompose into two sub-questions, then synthesize."""
    a, b, c = make_result("a", "Alpha fact."), make_result("b", "Beta fact."), make_result("c", "Gamma fact.")
    retrieval = _RoutedRetrieval([("alpha", [a, b]), ("beta", [c])])
    ce = _ScriptedCrossEncoder({"Alpha fact.": 0.1, "Beta fact.": 0.5, "Gamma fact.": 0.9})
    llm = _scripted(
        json.dumps({"sub_questions": ["the alpha part", "the beta part"]}),
        json.dumps({"answer": "combined", "steps": [{"claim": "c", "cited": [1]}]}),
    )
    return retrieval, ce, llm


async def test_decomposition_reranks_the_pool_against_the_original_question(make_result):
    retrieval, ce, llm = _rerank_fixture(make_result)
    out = await DecompositionReasoner(DecompositionReasoner.Config(rerank_evidence=True)).reason(
        Query(text="alpha and beta?"), GenerationContext(retrieval, llm, ce)
    )
    assert ce.seen_query == "alpha and beta?"  # scored against the ORIGINAL question, not a sub-question
    assert [r.chunk_id for r in out.evidence] == ["c", "b", "a"]  # CE order replaces retrieval order
    assert out.steps[0].cited == [0]  # cited passage 1 = the reranked head ('c') — numbering is post-rerank


async def test_decomposition_evidence_top_k_truncates_after_the_rerank(make_result):
    retrieval, ce, llm = _rerank_fixture(make_result)
    out = await DecompositionReasoner(
        DecompositionReasoner.Config(rerank_evidence=True, evidence_top_k=2)
    ).reason(Query(text="alpha and beta?"), GenerationContext(retrieval, llm, ce))
    assert [r.chunk_id for r in out.evidence] == ["c", "b"]  # best two survive; 'a' never reaches the read


async def test_decomposition_rerank_requires_a_cross_encoder(make_result):
    retrieval, _, llm = _rerank_fixture(make_result)
    with pytest.raises(GenerationError, match="no cross-encoder"):
        await DecompositionReasoner(DecompositionReasoner.Config(rerank_evidence=True)).reason(
            Query(text="alpha and beta?"), GenerationContext(retrieval, llm)  # no CE in the context
        )


async def test_decomposition_rerank_off_by_default_never_touches_the_model(make_result):
    retrieval, ce, llm = _rerank_fixture(make_result)
    out = await DecompositionReasoner(DecompositionReasoner.Config()).reason(
        Query(text="alpha and beta?"), GenerationContext(retrieval, llm, ce)
    )
    assert ce.seen_query is None  # default config: the CE is never invoked
    assert [r.chunk_id for r in out.evidence] == ["a", "b", "c"]  # retrieval order preserved


# ---------------- GroundedRetrievalReasoner (γ-driven re-retrieval) ----------------


async def test_grounded_retrieval_reretrieves_for_an_ungrounded_claim(make_result):
    a, b = make_result("a", "Alpha is a 1990 movie."), make_result("b", "Beta directed Alpha.")
    retrieval = _RoutedRetrieval([("beta", [b]), ("alpha", [a])])  # question->a; follow-up 'beta...'->b
    llm = _scripted(
        # hop 1 over [a]: a claim about Beta that [a] doesn't support -> heuristic UNGROUNDED
        json.dumps({"answer": "?", "steps": [{"claim": "beta directed", "cited": [1]}]}),
        # hop 2 over [a, b]: now grounded
        json.dumps({"answer": "Beta", "steps": [{"claim": "Beta directed Alpha", "cited": [1, 2]}]}),
    )
    out = await GroundedRetrievalReasoner(GroundedRetrievalReasoner.Config()).reason(
        Query(text="who directed alpha?"), GenerationContext(retrieval, llm)
    )
    assert out.answer == "Beta"
    assert [r.chunk_id for r in out.evidence] == ["a", "b"]  # the follow-up pulled in the bridge passage
    assert retrieval.queries == ["who directed alpha?", "beta directed"]  # re-retrieved for the claim


async def test_grounded_retrieval_stops_when_grounded(make_result):
    a = make_result("a", "Beta directed Alpha in 1990.")
    retrieval = _RoutedRetrieval([("", [a])])  # any query -> [a]
    # one read whose claim is fully covered by [a] -> GROUNDED -> no follow-up retrieval
    llm = _scripted(json.dumps({"answer": "Beta", "steps": [{"claim": "Beta directed Alpha", "cited": [1]}]}))
    out = await GroundedRetrievalReasoner(GroundedRetrievalReasoner.Config()).reason(
        Query(text="who directed alpha?"), GenerationContext(retrieval, llm)
    )
    assert out.answer == "Beta" and len(retrieval.queries) == 1  # stopped after one read (already grounded)


async def test_grounded_retrieval_stops_when_followup_finds_nothing_new(make_result):
    a = make_result("a", "Alpha is a movie.")
    retrieval = _RoutedRetrieval([("", [a])])  # every query -> the same [a] (no new evidence)
    llm = _scripted(json.dumps({"answer": "x", "steps": [{"claim": "totally unrelated zzz", "cited": [1]}]}))
    out = await GroundedRetrievalReasoner(GroundedRetrievalReasoner.Config(max_hops=5)).reason(
        Query(text="q"), GenerationContext(retrieval, llm)
    )
    # claim is ungrounded, but re-retrieval returns only the already-seen [a] -> stop (no infinite loop)
    assert len(retrieval.queries) == 2 and len(out.evidence) == 1


# ---------------- through the pipeline (swap the reasoner spec) ----------------


async def test_pipeline_runs_a_multihop_reasoner(make_result):
    a, b = make_result("a", "Alpha fact."), make_result("b", "Beta fact.")
    retrieval = _RoutedRetrieval([("alpha", [a]), ("beta", [b])])
    llm = _scripted(
        json.dumps({"done": False, "follow_up": "beta"}),
        json.dumps({"done": True, "answer": "done", "steps": [{"claim": "c", "cited": [1, 2]}]}),
    )
    pipeline = ComponentFactory.get().create_as(
        {"class_name": "generation_pipeline", "reasoner": {"class_name": "iterative"}}, GenerationPipeline
    )
    res = await pipeline.answer(Query(text="alpha?"), GenerationContext(retrieval, llm))
    assert res.answer == "done" and len(res.evidence) == 2
    assert [c.chunk_id for c in res.proof[0].citations] == ["a", "b"]  # proof tree spans both hops
