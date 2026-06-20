"""TarnRag — the high-level facade over tarn.rag's ingestion + retrieval + generation engines.

A small, **output-free** session API over one store: ``ingest`` / ``docs`` / ``delete`` / ``retrieve`` /
``ask``. The ingestion engine owns the repository; the retrieval engine *shares* it (one connection — it
sees ingests live, and works on an empty store); the generation engine is built lazily on the first
``ask``. Use it directly::

    async with TarnRag(load_settings("config.json")) as tarn:
        await tarn.ingest(["docs/"])
        result = await tarn.ask("…")

…or behind a UI — the interactive console (``tarnrag.console``) is one such UI, owning all rendering and
delegating the work here. ``llm`` may be injected (tests / a custom backend).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tarnrag.contracts import RetrievalResult
from tarnrag.core.components import ComponentFactory
from tarnrag.core.engine.config import GENERATION_PIPELINE, Settings
from tarnrag.core.resources.embedder import build_embedder
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.generation.engine.engine import GenerationEngine
from tarnrag.generation.pipeline.pipeline import GenerationPipeline
from tarnrag.generation.types import GenerationResult
from tarnrag.ingestion.engine.engine import IngestionEngine
from tarnrag.ingestion.engine.types import DocumentStatus, DocumentSummary
from tarnrag.retrieval.engine.engine import RetrievalEngine


def load_settings(config_path: str | Path) -> Settings:
    """Build ``Settings`` from a JSON config file (ambient ``.env`` ignored — the file is authoritative;
    OS env vars still supplement, e.g. the LLM key)."""
    config = json.loads(Path(config_path).read_text())
    return Settings(_env_file=None, **config)


class TarnRag:
    """An ingestion + retrieval + generation session over one store (see the module docstring)."""

    def __init__(self, settings: Settings, llm: LanguageModel | None = None) -> None:
        self.settings = settings
        self._injected_llm = llm
        self._ingestion: IngestionEngine | None = None
        self._retrieval: RetrievalEngine | None = None
        self._generation: GenerationEngine | None = None

    async def open(self) -> TarnRag:
        """Build the engines. The ingestion engine owns the repository (and stamps the index metadata);
        the retrieval engine shares it — one connection, and it sees ingests live."""
        self._ingestion = await IngestionEngine.create(self.settings)
        embedder = build_embedder(self.settings.embedding, self.settings.EMBEDDING_DIMENSION)
        self._retrieval = RetrievalEngine(self._ingestion.repository, embedder, self.settings)
        return self

    async def close(self) -> None:
        if self._ingestion is not None:
            await self._ingestion.aclose()  # owns the shared repository; retrieval just borrows it

    async def __aenter__(self) -> TarnRag:
        return await self.open()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ---------------- high-level API ----------------

    async def ingest(self, paths: list[str]) -> list[DocumentStatus]:
        """Ingest (or re-ingest) the given files / directories — a directory contributes the files in it,
        missing paths are skipped. The document id is each file's stem, so re-ingesting a file upserts in
        place (under ``ID_POLICY='caller'``). Returns the resulting per-document statuses."""
        files = _expand(paths)
        if not files:
            return []
        source_ids = [p.stem for p in files] if self.settings.ID_POLICY == "caller" else None
        doc_ids = await self._ingestion.ingest_paths([str(p) for p in files], source_ids=source_ids)
        return [s for s in [await self._ingestion.status(d) for d in doc_ids] if s is not None]

    async def docs(self) -> list[DocumentSummary]:
        """Every ingested document (id + chunk / embedding counts)."""
        return await self._ingestion.list_documents()

    async def delete(self, document_id: str) -> bool:
        """Delete a document and everything derived from it; False if it wasn't known."""
        return await self._ingestion.delete_document(document_id)

    async def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieval only — the ranked, provenance-bearing passages."""
        return await self._retrieval.search_text(query, top_k=top_k)

    async def ask(self, query: str) -> GenerationResult:
        """Retrieval + generation — a grounded answer with a proof tree. Needs an LLM."""
        return await self._gen().answer_text(query)

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


def _expand(paths: list[str]) -> list[Path]:
    """Resolve each path to file(s): a directory contributes the files it directly contains; a missing
    path is skipped."""
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.is_file()))
        elif path.is_file():
            files.append(path)
    return files
