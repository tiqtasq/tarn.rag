"""Run the MOTHRAG multi-hop QA benchmarks through tarn.rag's generation engine and score them.

Distractor protocol: for each question we ingest *only its own* candidate passages into an ephemeral store
(isolated, so retrieval can't leak across questions), run the configured ``GenerationPipeline``, and score
the answer with the generation harness's token-F1 / exact-match (the metric MOTHRAG reports). One embedder
and one store are reused across questions (the store is cleared each question); the LLM reader is injected
(an OpenAI-compatible Llama endpoint to match MOTHRAG, or a ``StaticLanguageModel`` for offline validation).

This is **not** a controlled MOTHRAG replication — different embedder/reader and the lean reasoners (no
ensemble / bridge / ChainFilter) — but a baseline of where tarn.rag's current stack stands. See
``MOTHRAG_PUBLISHED`` for the numbers to compare against.
"""

from __future__ import annotations

import tempfile

from tarnrag.core.engine.config import DatabaseSettings, Settings
from tarnrag.core.resources.embedder import Embedder
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.eval.benchmarks import BenchItem
from tarnrag.eval.generation import GenEvalReport, _aggregate, _score
from tarnrag.generation.engine.engine import GenerationEngine
from tarnrag.ingestion.engine.engine import IngestionEngine
from tarnrag.retrieval.engine.engine import RetrievalEngine
from tarnrag.retrieval.types import Query
from tarnrag.storage.repository import DocumentRepository

# MOTHRAG's published F1 / EM (paper Table 1 + the EM line), Llama-3.3-70B reader, n=1000/dataset.
MOTHRAG_PUBLISHED: dict[str, dict[str, float]] = {
    "hotpotqa": {"f1": 0.781, "em": 0.648},
    "2wiki": {"f1": 0.763, "em": 0.682},
    "musique": {"f1": 0.505, "em": 0.406},
}


async def run_benchmark(
    items: list[BenchItem], llm: LanguageModel, *, settings: Settings | None = None
) -> GenEvalReport:
    """Answer each ``BenchItem`` over its own passages and return the aggregate generation report
    (token-F1 / EM / grounded-rate / citation-coverage). ``llm`` is the injected reader; ``settings`` drives
    the embedder + the ``GENERATION_PIPELINE`` spec (reasoner / grounding)."""
    settings = settings or Settings(_env_file=None)
    embedder = Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
    with tempfile.TemporaryDirectory() as tmp:
        repo = await DocumentRepository.create(
            DatabaseSettings(document_url=f"sqlite:///{tmp}/eval.db"), settings.EMBEDDING_DIMENSION
        )
        try:
            await IngestionEngine.ensure_index_meta(repo, embedder)
            ingest = await IngestionEngine.create(settings, repository=repo, embedder=embedder)
            retrieval = await RetrievalEngine.create(settings, repository=repo, embedder=embedder)
            generation = GenerationEngine.assemble(retrieval, llm, settings)
            per_query = []
            for item in items:
                await _clear(ingest)
                await ingest.ingest_content(
                    [{"content": text, "title": title, "source_type": "text"} for title, text in item.passages]
                )
                result = await generation.answer(Query(text=item.query.text, purpose=item.query.purpose))
                per_query.append(_score(item.query, result))
            return _aggregate(per_query)
        finally:
            await repo.disconnect()  # idempotent dispose; shared by the engines (injected)


async def _clear(ingest: IngestionEngine) -> None:
    """Drop the previous question's passages so each question retrieves over only its own (distractor)."""
    for summary in await ingest.list_documents():
        await ingest.delete_document(summary.document_id)


def format_comparison(reports: dict[str, GenEvalReport]) -> str:
    """A table of tarn.rag's F1 / EM per dataset against MOTHRAG's published numbers (+ deltas + averages).
    Keys are dataset names (``hotpotqa`` / ``2wiki`` / ``musique``)."""
    header = f"{'dataset':<12}{'n':>6}{'F1':>8}{'F1*':>8}{'ΔF1':>8}{'EM':>8}{'EM*':>8}{'ΔEM':>8}"
    lines = [header, "-" * len(header), "(* = MOTHRAG published, Llama-3.3-70B reader)"]
    f1s, ems, mf1s, mems = [], [], [], []
    for name, r in reports.items():
        m = MOTHRAG_PUBLISHED.get(name, {})
        mf1, mem = m.get("f1", float("nan")), m.get("em", float("nan"))
        lines.append(
            f"{name:<12}{r.n:>6}{r.token_f1:>8.3f}{mf1:>8.3f}{r.token_f1 - mf1:>+8.3f}"
            f"{r.exact_match:>8.3f}{mem:>8.3f}{r.exact_match - mem:>+8.3f}"
        )
        f1s.append(r.token_f1)
        ems.append(r.exact_match)
        mf1s.append(mf1)
        mems.append(mem)
    if reports:
        lines.append(
            f"{'AVG':<12}{'':>6}{_avg(f1s):>8.3f}{_avg(mf1s):>8.3f}{_avg(f1s) - _avg(mf1s):>+8.3f}"
            f"{_avg(ems):>8.3f}{_avg(mems):>8.3f}{_avg(ems) - _avg(mems):>+8.3f}"
        )
    return "\n".join(lines)


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
