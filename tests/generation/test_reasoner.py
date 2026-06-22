"""The SingleHopReasoner: retrieve -> numbered passages -> strict-JSON answer with per-claim citations."""

import json

import pytest

from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.generation import SingleHopReasoner
from tarnrag.generation.components.reasoner import Reasoner
from tarnrag.generation.context import GenerationContext
from tarnrag.retrieval.types import Query


def _reasoner(**cfg) -> SingleHopReasoner:
    return SingleHopReasoner(SingleHopReasoner.Config(**cfg))


def _ctx(retrieval, reply: str) -> GenerationContext:
    return GenerationContext(retrieval, StaticLanguageModel(reply))


async def test_parses_answer_and_per_claim_citations(make_result, fake_retrieval):
    results = [
        make_result("a", "Coatings resist corrosion.", header_path=["Maint"]),
        make_result("b", "Quokkas are marsupials."),
    ]
    reply = json.dumps(
        {"answer": "Apply a coating.", "steps": [{"claim": "coatings resist corrosion", "cited": [1]}]}
    )
    out = await _reasoner().reason(Query(text="prevent rust?"), _ctx(fake_retrieval(results), reply))

    assert out.answer == "Apply a coating."
    assert len(out.steps) == 1
    assert out.steps[0].claim == "coatings resist corrosion"
    assert out.steps[0].cited == [0]  # 1-based passage 1 -> evidence index 0
    assert [r.chunk_id for r in out.evidence] == ["a", "b"]


async def test_falls_back_to_cite_all_on_non_json(make_result, fake_retrieval):
    results = [make_result("a", "x"), make_result("b", "y")]
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), "Just a plain sentence."))
    assert out.answer == "Just a plain sentence."
    assert len(out.steps) == 1 and out.steps[0].cited == [0, 1]  # the cite-all fallback


async def test_extracts_json_embedded_in_prose(make_result, fake_retrieval):
    results = [make_result("a", "x")]
    reply = 'Sure!\n{"answer": "ok", "steps": [{"claim": "c", "cited": [1]}]}\nThanks'
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), reply))
    assert out.answer == "ok" and out.steps[0].cited == [0]


async def test_drops_out_of_range_and_non_int_citations(make_result, fake_retrieval):
    results = [make_result("a", "x"), make_result("b", "y")]
    reply = json.dumps({"answer": "ok", "steps": [{"claim": "c", "cited": [2, 5, 0, "x", True, 2]}]})
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), reply))
    # passage 2 -> idx 1 (kept, deduped); 5 -> out of range; 0 -> idx -1; "x"/True -> not a valid int.
    assert out.steps[0].cited == [1]


async def test_well_formed_answer_without_steps_gets_one_cite_all_step(make_result, fake_retrieval):
    results = [make_result("a", "x"), make_result("b", "y")]
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), json.dumps({"answer": "ok"})))
    assert len(out.steps) == 1 and out.steps[0].claim == "ok" and out.steps[0].cited == [0, 1]


async def test_read_width_comes_from_config_top_k(make_result, fake_retrieval):
    fr = fake_retrieval([make_result("a", "x")])
    ctx = GenerationContext(fr, StaticLanguageModel(json.dumps({"answer": "ok"})))
    await _reasoner(top_k=3).reason(Query(text="q", top_k=99), ctx)
    assert fr.seen.top_k == 3  # the reasoner's config drives the read, not the caller's query.top_k


@pytest.mark.parametrize(
    "raw, clean",
    [
        ("The answer is Paris", "Paris"),
        ("Answer: 1889", "1889"),
        ('"Eiffel Tower"', "Eiffel Tower"),
        ("Final answer: yes", "yes"),
        ("Eiffel Tower", "Eiffel Tower"),  # already minimal — untouched
        ("Apply a coating.", "Apply a coating."),  # trailing punctuation left to the scorer
        ("answer", "answer"),  # bare word (no 'is'/':') is not stripped
    ],
)
def test_clean_answer_strips_leadins_and_quotes(raw, clean):
    assert Reasoner._clean_answer(raw) == clean


async def test_reason_applies_short_answer_extraction(make_result, fake_retrieval):
    """The reasoner trims the model's lead-in so the answer is the span (what token-F1/EM score)."""
    results = [make_result("a", "x")]
    reply = json.dumps({"answer": "The answer is Paris", "steps": [{"claim": "c", "cited": [1]}]})
    out = await _reasoner().reason(Query(text="q"), _ctx(fake_retrieval(results), reply))
    assert out.answer == "Paris"


async def test_numbered_passages_and_header_label_in_the_prompt(make_result, fake_retrieval):
    results = [make_result("a", "Coatings help.", header_path=["Maint", "Corrosion"])]
    captured = {}

    def capture(prompt):
        captured["user"] = prompt.user
        return json.dumps({"answer": "ok"})

    ctx = GenerationContext(fake_retrieval(results), StaticLanguageModel(capture))
    await _reasoner().reason(Query(text="how?"), ctx)
    assert "[1] Maint > Corrosion — Coatings help." in captured["user"]
    assert "Question: how?" in captured["user"]
