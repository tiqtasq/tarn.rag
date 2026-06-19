"""The GroundingChecker seam: the heuristic (content-word overlap) + LLM (batched verdicts) checkers."""

import json

from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.generation import (
    GenerationContext,
    HeuristicGroundingChecker,
    LLMGroundingChecker,
    ReasonedAnswer,
    ReasonedStep,
)


def _reasoned(steps, evidence):
    return ReasonedAnswer(answer="ans", steps=steps, evidence=evidence)


def _heuristic(**cfg):
    return HeuristicGroundingChecker(HeuristicGroundingChecker.Config(**cfg))


def _llm(reply):
    return GenerationContext(None, StaticLanguageModel(reply))


async def test_heuristic_grounds_a_supported_claim(make_result):
    ev = [make_result("a", "Coatings resist corrosion on steel tanks.")]
    reasoned = _reasoned([ReasonedStep(claim="coatings resist corrosion", cited=[0])], ev)
    assert await _heuristic().check(reasoned, None) == [True]


async def test_heuristic_flags_an_unsupported_claim(make_result):
    ev = [make_result("a", "Coatings resist corrosion."), make_result("b", "Quokkas are marsupials.")]
    # the claim cites the coatings passage but is about quokkas -> no overlap -> ungrounded
    reasoned = _reasoned([ReasonedStep(claim="quokkas are marsupials", cited=[0])], ev)
    assert await _heuristic().check(reasoned, None) == [False]


async def test_heuristic_threshold_is_configurable(make_result):
    ev = [make_result("a", "Coatings resist corrosion.")]
    # claim shares 2 of its 3 content words (coatings, corrosion) with the passage -> 0.67
    reasoned = _reasoned([ReasonedStep(claim="coatings prevent corrosion", cited=[0])], ev)
    assert await _heuristic(min_overlap=0.6).check(reasoned, None) == [True]
    assert await _heuristic(min_overlap=0.9).check(reasoned, None) == [False]


async def test_heuristic_claim_with_no_content_words_is_ungrounded(make_result):
    ev = [make_result("a", "anything at all")]
    reasoned = _reasoned([ReasonedStep(claim="the a of", cited=[0])], ev)  # all stopwords / single chars
    assert await _heuristic().check(reasoned, None) == [False]


async def test_llm_parses_batched_verdicts(make_result):
    ev = [make_result("a", "x"), make_result("b", "y")]
    reasoned = _reasoned([ReasonedStep(claim="c1", cited=[0]), ReasonedStep(claim="c2", cited=[1])], ev)
    out = await LLMGroundingChecker(LLMGroundingChecker.Config()).check(
        reasoned, _llm(json.dumps({"verdicts": [True, False]}))
    )
    assert out == [True, False]


async def test_llm_falls_back_to_grounded_on_bad_json(make_result):
    ev = [make_result("a", "x")]
    reasoned = _reasoned([ReasonedStep(claim="c", cited=[0])], ev)
    out = await LLMGroundingChecker(LLMGroundingChecker.Config()).check(reasoned, _llm("not json at all"))
    assert out == [True]  # unparseable -> never spuriously refuse


async def test_llm_pads_missing_verdicts_as_grounded(make_result):
    ev = [make_result("a", "x"), make_result("b", "y")]
    reasoned = _reasoned([ReasonedStep(claim="c1", cited=[0]), ReasonedStep(claim="c2", cited=[1])], ev)
    out = await LLMGroundingChecker(LLMGroundingChecker.Config()).check(
        reasoned, _llm(json.dumps({"verdicts": [False]}))  # only one verdict for two steps
    )
    assert out == [False, True]
