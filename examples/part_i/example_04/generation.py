"""Example 04 — End-to-end generation (config-driven): question -> grounded answer + proof tree.

The whole Goal-3 stack, assembled from JSON. Three files next to this script define the composition; the
Python just loads them, wires the engine, and prints results:

  retrieval.json    a hybrid retrieval pipeline (dense + sparse, fused by RRF) — what fetches the passages.
  generation.json   the generation pipeline: a single-hop reasoner, a CASCADING grounding checker (a cheap
                    overlap heuristic that escalates only the *uncertain* claims to the LLM), and an
                    abstention policy (refuse if fewer than half the claims hold up).
  questions.json    the questions to ask (two answerable from the corpus, one that isn't).

For each question it prints the answer, whether the system abstained, and the **proof tree**: each
reasoning step, whether grounding found it supported, and the cited passages (by header path / locator) —
the evidence the answer rests on.

The reasoner + grounding need an LLM. Set ``ANTHROPIC_API_KEY`` (the provider + model live in
``Settings.llm``; the default is Claude). Without a key the example still ingests + retrieves and previews
the passages a question would feed the model — so you can see everything but the generation itself.

Run (from the repo root, after the one-time ingestion):

    python -m examples.part_i.example_04.ingestion
    ANTHROPIC_API_KEY=sk-...  python -m examples.part_i.example_04.generation
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from tarnrag import RetrievalEngine
from tarnrag.core.components import ComponentFactory
from tarnrag.core.engine.config import GENERATION_PIPELINE, RETRIEVAL_PIPELINE
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.generation import GenerationEngine, GenerationPipeline, GenerationResult

from examples.common import base_settings, example_db, require_model

HERE = Path(__file__).resolve().parent
RETRIEVAL_SPEC = json.loads((HERE / "retrieval.json").read_text())
GENERATION_SPEC = json.loads((HERE / "generation.json").read_text())
QUESTIONS = [q["question"] for q in json.loads((HERE / "questions.json").read_text())]


async def main(llm: LanguageModel | None = None) -> list[GenerationResult]:
    """Answer every question in ``questions.json`` over the example's store. ``llm`` is injected by the
    smoke test (a canned model, no key); the CLI builds the real one from ``Settings.llm`` (needs a key)."""
    require_model()
    db_path = example_db(__file__)
    if not db_path.exists():
        raise SystemExit(f"No store at {db_path}.\nRun:  python -m examples.part_i.example_04.ingestion")

    # Everything composable comes from JSON — the retrieval pipeline and the generation pipeline.
    settings = base_settings(
        db_path, components={RETRIEVAL_PIPELINE: RETRIEVAL_SPEC, GENERATION_PIPELINE: GENERATION_SPEC}
    )

    # Wire the engine explicitly — what ``GenerationEngine.create()`` does under the hood, spelled out:
    # the retrieval engine (consumed through the ``RetrievalEngineProtocol`` port) + the LLM + the pipeline.
    retrieval = await RetrievalEngine.create(settings)
    pipeline = ComponentFactory.get().create_as(settings.components[GENERATION_PIPELINE], GenerationPipeline)
    has_key = bool(settings.llm.api_key or os.environ.get("ANTHROPIC_API_KEY"))
    model = llm if llm is not None else (LanguageModel.create(settings.llm) if has_key else None)

    async with retrieval:  # closes the store on exit
        if model is None:
            await _preview_without_a_key(retrieval)
            return []
        engine = GenerationEngine(retrieval, model, pipeline)
        results: list[GenerationResult] = []
        for question in QUESTIONS:
            result = await engine.answer_text(question)
            _print(question, result)
            results.append(result)
        return results


def _print(question: str, result: GenerationResult) -> None:
    """One question's answer + its proof tree (steps with grounding verdicts + cited passages)."""
    print(f"Q: {question}")
    if result.abstained:
        print(f"A: [abstained] {result.answer}\n")
        return
    print(f"A: {result.answer}    (grounded={result.grounded})")
    for step in result.proof:
        mark = "grounded" if step.grounded else "UNSUPPORTED"
        print(f"   - [{mark}] {step.claim}")
        for c in step.citations:
            where = " > ".join(c.header_path) or c.locator or c.document_id  # header path for structured docs
            print(f"       cite: {where}")
    print()


async def _preview_without_a_key(retrieval: RetrievalEngine) -> None:
    """No key: show the generation config + the passages retrieval would feed the model for one question."""
    print("ANTHROPIC_API_KEY not set — ingestion + retrieval only (no generation).\n")
    print("The generation pipeline that WOULD run (generation.json):")
    print(json.dumps(GENERATION_SPEC, indent=2), "\n")
    question = QUESTIONS[0]
    print(f"Passages retrieval would feed the model for:  {question}")
    for i, r in enumerate(await retrieval.search_text(question, top_k=4), 1):
        head = " > ".join(r.provenance.header_path) if r.provenance and r.provenance.header_path else (
            r.locator or r.document_id
        )
        print(f"  [{i}] {head}: {r.text[:90]}")
    print("\nSet ANTHROPIC_API_KEY and re-run to generate grounded answers over these passages.")


if __name__ == "__main__":
    asyncio.run(main())
