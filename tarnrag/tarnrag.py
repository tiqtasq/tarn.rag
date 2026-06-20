"""TarnRag — the high-level facade over tarn.rag's ingestion + retrieval + generation engines.

A small, **output-free** session API over one store: ``ingest`` / ``docs`` / ``delete`` / ``retrieve`` /
``ask``. Each returns an :class:`~tarnrag.report.Outcome` — the value it produced, plus a
:class:`~tarnrag.report.Report` of any non-fatal issues encountered (empty when all went well). Nothing is
printed and nothing is silently skipped: a UI over the facade (the console, the tiqtasq REST API) renders
the value and surfaces the report. ``TarnRag`` is the composition root: it builds the one store (the
repository) and the embedder once and injects both into the engines that run over them (one connection —
retrieval sees ingests live, and works on an empty store); the generation engine is built lazily on the
first ``ask``. Use it directly::

    async with TarnRag("config.json") as tarn:
        ingested = await tarn.ingest(["docs/"])
        for issue in ingested.report.issues:
            ...  # e.g. a path that wasn't found
        answer = await tarn.ask("…")
        print(answer.value.answer)

…or behind a UI. ``llm`` may be injected (tests / a custom backend).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tarnrag.contracts import RetrievalResult
from tarnrag.core.components import ComponentFactory
from tarnrag.core.engine.config import GENERATION_PIPELINE, Settings
from tarnrag.core.resources.embedder import Embedder
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.generation.engine.engine import GenerationEngine
from tarnrag.generation.pipeline.pipeline import GenerationPipeline
from tarnrag.generation.types import GenerationResult
from tarnrag.ingestion.engine.engine import IngestionEngine
from tarnrag.ingestion.engine.types import DocumentStatus, DocumentSummary
from tarnrag.report import Issue, Outcome, Report, Severity
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.engine.engine import RetrievalEngine
from tarnrag.storage.repository import DocumentRepository


class TarnRag:
    """An ingestion + retrieval + generation session over one store (see the module docstring)."""

    def __init__(self, settings: Settings | str | Path, llm: LanguageModel | None = None) -> None:
        """``settings`` is a ready ``Settings`` or the path to a JSON config to load (the ambient ``.env``
        is ignored — the file is authoritative; OS env vars still supplement, e.g. the LLM key). ``llm``
        may be injected (tests / a custom backend)."""
        if not isinstance(settings, Settings):
            settings = Settings(_env_file=None, **json.loads(Path(settings).read_text()))
        self.settings = settings
        self._injected_llm = llm
        self._repository: DocumentRepository | None = None
        self._embedder: Embedder | None = None
        self._ingestion: IngestionEngine | None = None
        self._retrieval: RetrievalEngine | None = None
        self._generation: GenerationEngine | None = None

    async def open(self) -> TarnRag:
        """Build the shared resources once — the repository (the one store) and the passage/query
        embedder — plus the retrieval engine over them, and establish/confirm the index identity. The
        composition root: engines receive their dependencies. The ingestion engine is built lazily (on
        the first ingest / docs / delete) and generation on the first ask, so a read-only / eval session
        never spins up the ingestion machinery."""
        self._repository = await DocumentRepository.create(
            self.settings.database, self.settings.EMBEDDING_DIMENSION
        )
        try:
            self._embedder = Embedder.create(self.settings.embedding, self.settings.EMBEDDING_DIMENSION)
            # Establish/confirm the index identity (cheap, repository-only) so retrieval opens even on a
            # fresh store — without building the ingestion machinery, which waits until something ingests.
            await IngestionEngine.ensure_index_meta(self._repository, self._embedder)
            self._retrieval = await RetrievalEngine.create(
                self.settings, repository=self._repository, embedder=self._embedder
            )
        except BaseException:
            await self.close()  # release the store we opened before re-raising (e.g. an index mismatch)
            raise
        return self

    async def close(self) -> None:
        if self._repository is not None:
            await self._repository.disconnect()  # the one store, shared by every engine

    async def __aenter__(self) -> TarnRag:
        return await self.open()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ---------------- high-level API (each returns an Outcome[value] + Report) ----------------

    async def ingest(self, paths: list[str]) -> Outcome[list[DocumentStatus]]:
        """Ingest (or re-ingest) the given files / directories — a directory contributes the files in it.
        The document id is each file's stem, so re-ingesting a file upserts in place (under
        ``ID_POLICY='caller'``). The value is the resulting per-document statuses; the report flags paths
        that matched nothing and documents that failed to ingest — neither is dropped silently."""
        files, missing = self._expand(paths)
        issues = [Issue("not found", Severity.WARNING, subject=raw) for raw in missing]
        if not files:
            return Outcome([], Report(tuple(issues)))
        ingestion = await self._ingestion_engine()
        source_ids = [p.stem for p in files] if self.settings.ID_POLICY == "caller" else None
        doc_ids = await ingestion.ingest_paths([str(p) for p in files], source_ids=source_ids)
        statuses = [s for s in [await ingestion.status(d) for d in doc_ids] if s is not None]
        issues += [
            Issue("ingestion failed", Severity.ERROR, subject=s.document_id)
            for s in statuses
            if s.status == "failed"
        ]
        return Outcome(statuses, Report(tuple(issues)))

    async def docs(self) -> Outcome[list[DocumentSummary]]:
        """Every ingested document (id + chunk / embedding counts)."""
        return Outcome(await (await self._ingestion_engine()).list_documents())

    async def delete(self, document_id: str) -> Outcome[bool]:
        """Delete a document and everything derived from it. The value is False if it wasn't known."""
        return Outcome(await (await self._ingestion_engine()).delete_document(document_id))

    async def retrieve(self, query: str, *, top_k: int = 8) -> Outcome[list[RetrievalResult]]:
        """Retrieval only — the ranked, provenance-bearing passages. ``top_k`` mirrors the engine's
        ``search_text`` default."""
        return Outcome(await self._retrieval.search_text(query, top_k=top_k))

    async def ask(self, query: str) -> Outcome[GenerationResult]:
        """Retrieval + generation — a grounded answer with a proof tree. Needs an LLM."""
        return Outcome(await self._gen().answer_text(query))

    def retrieval_context(self) -> RetrievalContext:
        """The shared ``(store, embedder)`` the retrieval pipelines run against — the seam the eval
        harness's ``sweep`` uses to score *alternative* pipeline specs over this one index."""
        return RetrievalContext(self._repository, self._embedder)

    async def _ingestion_engine(self) -> IngestionEngine:
        """The ingestion engine, built lazily on first use (ingest / docs / delete) — a read-only or eval
        session never spins up the orchestrator + queue + worker + sinks. Shares the open store."""
        if self._ingestion is None:
            self._ingestion = await IngestionEngine.create(
                self.settings, repository=self._repository, embedder=self._embedder
            )
        return self._ingestion

    def _gen(self) -> GenerationEngine:
        """The generation engine, built once over the shared retrieval engine + the configured LLM."""
        if self._generation is None:
            llm = self._injected_llm or self._build_llm()
            spec = self.settings.components.get(GENERATION_PIPELINE) or {"class_name": "generation_pipeline"}
            pipeline = ComponentFactory.get().create_as(spec, GenerationPipeline)
            self._generation = GenerationEngine(self._retrieval, llm, pipeline)
        return self._generation

    def _build_llm(self) -> LanguageModel:
        if not (self.settings.llm.api_key or os.environ.get("ANTHROPIC_API_KEY")):
            raise RuntimeError(
                "generation needs an LLM key — set ANTHROPIC_API_KEY (the provider/model are in the config)"
            )
        return LanguageModel.create(self.settings.llm)

    @staticmethod
    def _expand(paths: list[str]) -> tuple[list[Path], list[str]]:
        """Resolve each path to file(s): a directory contributes the files it directly contains. Returns
        the resolved files and the raw paths that matched nothing (reported, not skipped silently)."""
        files: list[Path] = []
        missing: list[str] = []
        for raw in paths:
            path = Path(raw).expanduser()
            if path.is_dir():
                files.extend(sorted(p for p in path.iterdir() if p.is_file()))
            elif path.is_file():
                files.append(path)
            else:
                missing.append(raw)
        return files, missing
