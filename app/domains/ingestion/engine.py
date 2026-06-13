"""IngestionEngine — the high-level facade for the ingestion pipeline.

The public surface is **document-centric**: callers submit documents and poll by
``document_id`` (== ``source_id``). Jobs are an internal detail and never leak here — the
only exception is the debug-only ``verbose`` breakdown on status reads.

Use :meth:`create` for ready-to-use wiring from ``Settings``:

* ``MODE='embedded'`` (default) runs the whole pipeline **in-process** (InMemory queue, no
  Postgres/pgQueuer) — each ingest call processes to completion before returning.
* ``MODE='distributed'`` enqueues to pgQueuer; separate worker processes (``run_worker``)
  consume the queue.

The bare constructor is the low-level seam (inject your own pipeline/orchestrator — used by
tests and advanced wiring), mirroring ``RetrievalEngine.open``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from app.core.config import Settings, get_settings
from app.core.observability import NoOpObservability
from app.domains.base.embedder import OnnxEmbedder
from app.domains.base.index_store import SqliteIndexStore
from app.domains.base.models import PipelineItem
from app.domains.base.repository import DocumentRepository
from app.domains.base.status import DocumentFactsSource, DocumentStatusReader
from app.domains.ingestion.orchestrator import PipelineDAG, PipelineOrchestrator
from app.domains.ingestion.pipeline import Pipeline
from app.domains.ingestion.queue import (
    InMemoryJobQueue,
    JobConsumer,
    JobEnqueuer,
    PgQueuerJobQueue,
)
from app.domains.ingestion.types import DocumentRef, DocumentStatus, IngestSubmission
from app.domains.ingestion.worker import IngestionWorker
from app.factories import create_ingestion_pipeline, create_sink_registry

logger = logging.getLogger(__name__)

_SOURCE_TYPES = {"pdf": "pdf", "txt": "text", "text": "text", "html": "html", "htm": "html"}


class IngestionEngine:
    """High-level facade for the ingestion pipeline; document-centric public surface."""

    def __init__(
        self,
        pipeline: Pipeline,
        orchestrator: PipelineOrchestrator,
        repository: DocumentRepository,
        observability: Any = None,  # Phase 5 Observability plugs in here
        staging_dir: str | None = None,  # where streamed bytes are staged for the worker
        facts_source: DocumentFactsSource | None = None,  # data store for status counts
        *,
        queue: JobEnqueuer | None = None,  # held for embedded draining / the worker loop
        auto_drain: bool = False,  # embedded: process the whole DAG inline on each ingest
    ):
        self.pipeline = pipeline
        self.orchestrator = orchestrator
        self.repository = repository  # persistence + job_status projection
        self.obs = observability
        self.staging_dir = staging_dir
        # Status composes the repo's job_status with the data facts. In the classic path the
        # repo is both; in retrieval mode the §8 index store supplies the facts (counts).
        self._status = DocumentStatusReader(repository, facts_source or repository)
        self._queue = queue
        self._auto_drain = auto_drain
        self._index_store: SqliteIndexStore | None = None  # set by create(), closed by aclose()

    # ----- construction -----

    @classmethod
    async def create(cls, settings: Settings | None = None) -> IngestionEngine:
        """Build a ready-to-use engine from ``Settings`` (mode-driven). The easy entry point;
        the constructor is the lower-level seam for injecting your own wiring."""
        settings = settings or get_settings()
        obs = NoOpObservability() if settings.observability.enabled else None
        repository = await DocumentRepository.create(
            settings.database, settings.EMBEDDING_DIMENSION
        )
        embedder = OnnxEmbedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
        # The engine is the index producer: it records index_meta and the sinks persist into
        # the §8 index. job_status stays on the repository.
        index_store = SqliteIndexStore.create(
            settings.index, settings.EMBEDDING_DIMENSION, embedder=embedder
        )
        pipeline = create_ingestion_pipeline(settings)

        if settings.MODE == "distributed":
            queue: JobEnqueuer = await PgQueuerJobQueue.connect(settings.database.queue_url)
            auto_drain = False
        else:  # embedded — run the whole pipeline in this process
            queue = InMemoryJobQueue()
            auto_drain = True

        orchestrator = PipelineOrchestrator(
            PipelineDAG(pipeline.stages), queue, repository, create_sink_registry(),
            observability=obs, chunk_store=index_store,
        )
        engine = cls(
            pipeline, orchestrator, repository, observability=obs,
            staging_dir=settings.UPLOAD_DIR, facts_source=index_store,
            queue=queue, auto_drain=auto_drain,
        )
        engine._index_store = index_store
        if auto_drain:  # register the in-process worker that ingest() will drive
            worker = IngestionWorker(orchestrator, observability=obs)
            queue.set_handler(worker.handle_batch)
        return engine

    # ----- ingest -----

    async def ingest_paths(
        self, paths: list[str], *, parser: str | None = None
    ) -> IngestSubmission:
        """Ingest documents loaded from file paths (LoadAndParse reads them). ``parser``
        selects the PDF backend for this request (None → default)."""
        items = [
            self._item(
                source_id=str(uuid.uuid4()),
                content="",  # loaded by LoadAndParseStage
                extra={
                    "source_path": path,
                    "source_type": self._infer_source_type(path),
                    **({"parser": parser} if parser else {}),
                },
            )
            for path in paths
        ]
        return await self._submit(items)

    async def ingest_content(
        self, documents: list[dict[str, str]], *, parser: str | None = None
    ) -> IngestSubmission:
        """Ingest pre-loaded content. A client-supplied ``source_id`` becomes the
        ``document_id``; otherwise one is assigned."""
        items = []
        for doc in documents:
            doc = dict(doc)  # don't mutate the caller's dict
            content = doc.pop("content")
            source_id = doc.pop("source_id", None) or str(uuid.uuid4())
            if parser:
                doc.setdefault("parser", parser)
            items.append(self._item(source_id, content=content, extra=doc))
        return await self._submit(items)

    async def ingest_streams(
        self, streams: list[tuple[str, BinaryIO]], *, parser: str | None = None
    ) -> IngestSubmission:
        """Ingest binary streams (e.g. HTTP uploads): **stream** each ``(filename, source)``
        to the staging dir, then ingest by path (parsing happens in the worker). ``source`` is
        any binary file-like; it is copied in chunks (never fully held in memory). Requires
        ``staging_dir``; in ``distributed`` mode that must be a volume the worker can read."""
        if not self.staging_dir:
            raise RuntimeError("streams are not configured (no staging_dir / UPLOAD_DIR)")
        # Offload the blocking disk copy so the event loop isn't stalled on large files.
        paths = [
            await asyncio.to_thread(self._stage_stream, filename, source)
            for filename, source in streams
        ]
        return await self.ingest_paths(paths, parser=parser)

    async def status(
        self, document_id: str, verbose: bool = False
    ) -> DocumentStatus | None:
        """Document-level status derived from persisted data (pending | in_progress |
        complete | failed). ``None`` if the document is unknown. ``verbose=True`` adds the
        debug-only ``jobs`` breakdown — the only place per-job state is exposed."""
        raw = await self._status.document_status(document_id, verbose=verbose)
        if raw is None:
            return None
        return DocumentStatus(
            document_id=raw["document_id"],
            status=raw["status"],
            chunk_count=raw["chunk_count"],
            embedding_count=raw["embedding_count"],
            jobs=raw.get("jobs"),
        )

    # ----- distributed worker loop -----

    async def run_worker(self) -> None:
        """Run the consumer loop — the worker-process entry for ``MODE='distributed'``.
        Registers the compute worker on the queue and runs until it stops, then releases
        resources. (In ``embedded`` mode the queue drains per ingest, so this isn't needed.)"""
        if not isinstance(self._queue, JobConsumer):
            raise RuntimeError("this engine has no consumable queue to run a worker on")
        worker = IngestionWorker(self.orchestrator, observability=self.obs)
        self._queue.set_handler(worker.handle_batch)
        logger.info("Ingestion worker %s started", worker.worker_id)
        try:
            await self._queue.run()
        finally:
            await self.aclose()

    # ----- lifecycle -----

    async def aclose(self) -> None:
        """Release resources owned by the engine (DB engine pool, index store)."""
        engine = getattr(self.repository, "engine", None)
        if engine is not None:
            await engine.dispose()
        if self._index_store is not None:
            self._index_store.close()

    async def __aenter__(self) -> IngestionEngine:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ----- internals -----

    async def _submit(self, items: list[PipelineItem]) -> IngestSubmission:
        document_ids = await self.orchestrator.ingest_documents(items)
        if self.obs:
            self.obs.counter("ingestion.documents_queued", len(items))
            await self.obs.log("info", "documents queued", count=len(document_ids))
        if self._auto_drain and isinstance(self._queue, JobConsumer):
            await self._queue.run()  # embedded: process the whole DAG in-process
        return IngestSubmission(
            documents=[DocumentRef(document_id=d) for d in document_ids],
            queued=len(document_ids),
        )

    def _item(self, source_id: str, content: str, extra: dict[str, Any]) -> PipelineItem:
        return PipelineItem(
            id=source_id,
            content=content,
            metadata={
                "source_id": source_id,  # public handle == document_id
                "created_at": datetime.now(UTC).isoformat(),
                **extra,
            },
        )

    def _infer_source_type(self, path: str) -> str:
        return _SOURCE_TYPES.get(path.lower().rsplit(".", 1)[-1], "unknown")

    def _stage_stream(self, filename: str, source: BinaryIO) -> str:
        """Stream a binary file-like under a unique name (extension preserved so the loader
        dispatches correctly) and return the path. ``copyfileobj`` copies in chunks — no
        full-memory load. An object store (S3/blob) would plug in behind this method."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in (filename or "") else "bin"
        dest = Path(self.staging_dir) / f"{uuid.uuid4()}.{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as out:
            shutil.copyfileobj(source, out)
        return str(dest)
