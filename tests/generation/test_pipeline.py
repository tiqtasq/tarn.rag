"""The GenerationPipeline + GenerationEngine: reason -> assemble proof tree -> GenerationResult."""

import json

from tarnrag.core.components import ComponentFactory
from tarnrag.core.llm import StaticLanguageModel
from tarnrag.generation import GenerationContext, GenerationEngine, GenerationPipeline
from tarnrag.retrieval.types import Query


def _pipeline(spec=None) -> GenerationPipeline:
    return ComponentFactory.get().create_as(spec or {"class_name": "generation_pipeline"}, GenerationPipeline)


async def test_pipeline_builds_proof_tree_with_provenance(make_result, fake_retrieval):
    results = [
        make_result("a", "Coatings resist corrosion.", header_path=["Maint", "Corrosion"], start=0, end=24),
        make_result("b", "Quokkas grin."),
    ]
    reply = json.dumps(
        {"answer": "Apply a coating.", "steps": [{"claim": "coatings resist corrosion", "cited": [1]}]}
    )
    ctx = GenerationContext(fake_retrieval(results), StaticLanguageModel(reply))
    res = await _pipeline().answer(Query(text="prevent rust?"), ctx)

    assert res.answer == "Apply a coating."
    assert res.grounded is True and res.abstained is False  # slice-2 defaults (grounding is slice 3)
    assert [r.chunk_id for r in res.evidence] == ["a", "b"]

    [step] = res.proof
    [cite] = step.citations
    assert cite.chunk_id == "a"
    assert cite.header_path == ["Maint", "Corrosion"]
    assert cite.geometry[0].start == 0 and cite.geometry[0].end == 24  # highlightable span from provenance


async def test_engine_answer_text_delegates(make_result, fake_retrieval):
    results = [make_result("a", "x")]
    llm = StaticLanguageModel(json.dumps({"answer": "ok"}))
    eng = GenerationEngine(fake_retrieval(results), llm, _pipeline())
    res = await eng.answer_text("q")
    assert res.answer == "ok" and len(res.evidence) == 1


async def test_engine_is_an_async_context_manager(make_result, fake_retrieval):
    # aclose is best-effort: the fake port exposes no aclose, so __aexit__ must be a no-op (not error).
    async with GenerationEngine(fake_retrieval([]), StaticLanguageModel("{}"), _pipeline()) as eng:
        assert eng.retrieval is not None


async def test_no_results_yields_an_answer_with_empty_evidence(make_result, fake_retrieval):
    # The model is told there are no passages; the pipeline still returns a (cite-nothing) result.
    ctx = GenerationContext(fake_retrieval([]), StaticLanguageModel(json.dumps({"answer": "No passages found."})))
    res = await _pipeline().answer(Query(text="q"), ctx)
    assert res.answer == "No passages found." and res.evidence == []
    assert res.proof and res.proof[0].citations == []  # one step, nothing to cite
