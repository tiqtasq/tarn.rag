"""IngestionEngine — the high-level facade for the ingestion pipeline.

The public surface is **document-centric**: callers submit documents and poll by
``document_id`` (== ``source_id``). Jobs are an internal detail and never leak here — the
only window is the debug-gated :meth:`document_jobs` (raises unless ``APP__DEBUG``).

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
import hashlib
import logging
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from tarnrag.core.config import IdPolicy, Settings, get_settings
from tarnrag.core.observability import NoOpObservability
from tarnrag.embedder import OnnxEmbedder
from tarnrag.storage.index_store import SqliteIndexStore
from tarnrag.storage.models import PipelineItem
from tarnrag.storage.repository import DocumentRepository
from tarnrag.storage.status import DocumentFactsSource, DocumentStatusReader
from tarnrag.ingestion.orchestrator import PipelineDAG, PipelineOrchestrator
from tarnrag.ingestion.pipeline import Pipeline
from tarnrag.ingestion.queue import (
    InMemoryJobQueue,
    JobConsumer,
    JobEnqueuer,
    PgQueuerJobQueue,
)
from tarnrag.ingestion.types import DocumentStatus, DocumentSummary
from tarnrag.ingestion.worker import IngestionWorker
from tarnrag.ingestion.factories import create_ingestion_pipeline, create_sink_registry

logger = logging.getLogger(__name__)

_SOURCE_TYPES = {"pdf": "pdf", "txt": "text", "text": "text", "html": "html", "htm": "html"}

_HASH_BLOCK = 1 << 16  # 64 KiB — stream files through the hasher (no full-memory load)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_HASH_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


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
        debug: bool = False,  # gates the debug-only methods (set from APP__DEBUG)
        id_policy: IdPolicy = "uuid",  # how document ids are assigned (set from ID_POLICY)
    ):
        self.pipeline = pipeline
        self.orchestrator = orchestrator
        self.repository = repository  # persistence + job_status projection
        self.obs = observability
        self.staging_dir = staging_dir
        # The document data store (repo in the classic path, §8 index store in retrieval mode):
        # supplies status facts AND the content_hash dedup lookup.
        self._facts_source = facts_source or repository
        self._status = DocumentStatusReader(repository, self._facts_source)
        self._queue = queue
        self._auto_drain = auto_drain
        self._debug = debug
        self._id_policy = id_policy
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
            queue=queue, auto_drain=auto_drain, debug=settings.app.debug,
            id_policy=settings.ID_POLICY,
        )
        engine._index_store = index_store
        if auto_drain:  # register the in-process worker that ingest() will drive
            worker = IngestionWorker(orchestrator, observability=obs)
            queue.set_handler(worker.handle_batch)
        return engine

    # ----- ingest -----

    async def ingest_paths(
        self, paths: list[str], *, source_ids: list[str] | None = None, parser: str | None = None
    ) -> list[str]:
        """Ingest documents loaded from file paths (LoadAndParse reads them). ``parser``
        selects the PDF backend (None → default).

        ``source_ids`` (parallel to ``paths``) sets each document's ``source_id`` (==
        ``document_id``), which is **stable**: re-ingesting the same id upserts in place. Under
        ``ID_POLICY='caller'`` it is required; under ``'uuid'`` it must be omitted (the engine
        assigns ids). Content dedup is separate — every document stores a ``content_hash``
        (query :meth:`find_by_content_hash`). Returns the document IDs (poll :meth:`status`)."""
        ids = self._resolve_document_ids(source_ids, len(paths))
        items = [
            self._item(
                source_id=ids[i],
                content="",  # loaded by LoadAndParseStage
                extra={
                    "source_path": path,
                    "source_type": self._infer_source_type(path),
                    "content_hash": _sha256_file(path),  # dedup key (sha256 of the file bytes)
                    **({"parser": parser} if parser else {}),
                },
            )
            for i, path in enumerate(paths)
        ]
        return await self._submit(items)

    async def ingest_content(
        self, documents: list[dict[str, str]], *, parser: str | None = None
    ) -> list[str]:
        """Ingest pre-loaded content. Each doc's ``source_id`` is governed by ``ID_POLICY``
        (required under ``'caller'``, forbidden under ``'uuid'``). The document's
        ``content_hash`` is the sha256 of its ``content``. Returns the document IDs."""
        docs = [dict(d) for d in documents]  # don't mutate the caller's dicts
        contents = [d.pop("content") for d in docs]
        ids = self._resolve_document_ids([d.pop("source_id", None) for d in docs], len(docs))
        items = []
        for i, doc in enumerate(docs):
            if parser:
                doc.setdefault("parser", parser)
            doc["content_hash"] = _sha256_bytes(contents[i].encode("utf-8"))
            items.append(self._item(ids[i], content=contents[i], extra=doc))
        return await self._submit(items)

    async def ingest_streams(
        self, streams: list[tuple[str, BinaryIO]], *, source_ids: list[str] | None = None,
        parser: str | None = None,
    ) -> list[str]:
        """Ingest binary streams (e.g. HTTP uploads): **stream** each ``(filename, source)``
        to the staging dir, then ingest by path (parsing happens in the worker). ``source`` is
        any binary file-like; it is copied in chunks (never fully held in memory). Requires
        ``staging_dir``; in ``distributed`` mode that must be a volume the worker can read.

        ``source_ids`` (parallel to ``streams``) follows ``ID_POLICY`` exactly as in
        :meth:`ingest_paths`; the ``content_hash`` is the sha256 of the streamed bytes.
        Returns the document IDs."""
        if not self.staging_dir:
            raise RuntimeError("streams are not configured (no staging_dir / UPLOAD_DIR)")
        # Validate the id supply against the policy BEFORE staging, so a bad call leaves no
        # orphaned files; ingest_paths then resolves ids + hashes from the staged files.
        self._check_supply(self._align_source_ids(source_ids, len(streams)))
        # Offload the blocking disk copy so the event loop isn't stalled on large files.
        paths = [
            await asyncio.to_thread(self._stage_stream, filename, source)
            for filename, source in streams
        ]
        return await self.ingest_paths(paths, source_ids=source_ids, parser=parser)

    async def status(self, document_id: str) -> DocumentStatus | None:
        """Document-level status derived from persisted data (pending | in_progress |
        complete | failed). ``None`` if the document is unknown."""
        raw = await self._status.document_status(document_id)
        if raw is None:
            return None
        return DocumentStatus(
            document_id=raw["document_id"],
            status=raw["status"],
            chunk_count=raw["chunk_count"],
            embedding_count=raw["embedding_count"],
        )

    async def find_by_content_hash(self, content_hash: str) -> list[str]:
        """Document IDs whose stored content matches ``content_hash`` — the content-dedup query.
        Empty if that content was never ingested; possibly several (same content under different
        ids). Build the key with :meth:`content_hash` / :meth:`content_hash_of_file`. Identity is
        independent: a document keeps its id when its content (and this hash) changes."""
        return await self._facts_source.documents_by_content_hash(content_hash)

    @staticmethod
    def content_hash(data: bytes) -> str:
        """The sha256 hex the engine stores for submitted content — use it to build a dedup key
        (e.g. ``await engine.find_by_content_hash(engine.content_hash(data))``)."""
        return _sha256_bytes(data)

    @staticmethod
    def content_hash_of_file(path: str) -> str:
        """:meth:`content_hash` for a file, read in chunks (constant memory)."""
        return _sha256_file(path)

    async def list_documents(self) -> list[DocumentSummary]:
        """Inventory of every ingested document — id, ``content_hash``, and chunk/embedding
        counts. Order is unspecified; pair with :meth:`status` for per-document state."""
        return [
            DocumentSummary(
                document_id=r["document_id"],
                content_hash=r["content_hash"],
                chunk_count=r["chunk_count"],
                embedding_count=r["embedding_count"],
            )
            for r in await self._facts_source.list_documents()
        ]

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and everything derived from it — chunks, embeddings, retrieval-index
        rows, and its job-status records. Returns True if the document was known (had data or
        in-flight jobs), False if there was nothing to delete.

        This spans **two independent stores** (the document data store and the job_status
        projection — separate databases in any real deployment), so it can't be one transaction.
        Each call is individually atomic; they run **job-status-first** on purpose, so a failure
        partway leaves a *consistent* residue — the document stays fully present (its status and
        inventory unchanged), losing only debug job rows, never a 'ghost' status pointing at
        deleted data. It is idempotent: a retry completes the delete and heals any residue."""
        # job_status first (operational exhaust). If the data delete then fails, the document is
        # left intact and consistent rather than gone-with-a-stale-status; a retry finishes it.
        removed_jobs = await self.repository.delete_document_jobs(document_id)
        removed_data = await self._facts_source.delete_document(document_id)
        return removed_data or removed_jobs

    # ----- debug (gated on APP__DEBUG) -----

    async def document_jobs(self, document_id: str) -> list[dict]:
        """Per-job breakdown for a document — the one window into the otherwise-internal job
        state. **Debug only:** raises ``RuntimeError`` unless ``APP__DEBUG`` is set."""
        self._require_debug()
        return await self.repository.document_jobs(document_id)

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

    def _require_debug(self) -> None:
        if not self._debug:
            raise RuntimeError("debug-only operation; set APP__DEBUG=true to enable")

    def _resolve_document_ids(
        self, source_ids: list[str | None] | None, count: int
    ) -> list[str]:
        """Document ids per ``ID_POLICY``: 'caller' uses the supplied ids (every one required);
        'uuid' assigns fresh uuid4s (and forbids supplied ids). Raises on a policy mismatch."""
        supplied = self._align_source_ids(source_ids, count)
        self._check_supply(supplied)
        if self._id_policy == "caller":
            return [s for s in supplied]  # all present (validated by _check_supply)
        return [str(uuid.uuid4()) for _ in range(count)]

    def _align_source_ids(
        self, source_ids: list[str | None] | None, count: int
    ) -> list[str | None]:
        """Normalize caller ids to one slot per document (None where unset); validate length."""
        if source_ids is None:
            return [None] * count
        if len(source_ids) != count:
            raise ValueError(
                f"source_ids has {len(source_ids)} entries but {count} documents were given"
            )
        return list(source_ids)

    def _check_supply(self, supplied: list[str | None]) -> None:
        """Validate the caller's id supply against ``ID_POLICY`` (presence/absence)."""
        count = len(supplied)
        given = sum(1 for s in supplied if s)
        if self._id_policy == "caller" and given != count:
            raise ValueError(
                f"ID_POLICY='caller' requires a source_id for every document "
                f"({count - given} of {count} missing)"
            )
        if self._id_policy == "uuid" and given:
            raise ValueError(
                f"ID_POLICY='uuid' assigns document ids automatically; remove the "
                f"{given} caller-supplied source_id(s) (or set ID_POLICY='caller')"
            )

    async def _submit(self, items: list[PipelineItem]) -> list[str]:
        document_ids = await self.orchestrator.ingest_documents(items)
        if self.obs:
            self.obs.counter("ingestion.documents_queued", len(items))
            await self.obs.log("info", "documents queued", count=len(document_ids))
        if self._auto_drain and isinstance(self._queue, JobConsumer):
            await self._queue.run()  # embedded: process the whole DAG in-process
        return document_ids

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


async def run_worker(settings: Settings | None = None) -> None:
    """Consumer entry point for ``MODE='distributed'`` — build the wiring via
    :meth:`IngestionEngine.create` and run the consume loop until the queue stops, then release
    resources. (In ``embedded`` mode ingestion runs in-process, so there is no separate worker.)"""
    engine = await IngestionEngine.create(settings)
    if not isinstance(engine._queue, JobConsumer):
        raise RuntimeError("this engine has no consumable queue to run a worker on")
    worker = IngestionWorker(engine.orchestrator, observability=engine.obs)
    engine._queue.set_handler(worker.handle_batch)
    logger.info("Ingestion worker %s started", worker.worker_id)
    try:
        await engine._queue.run()
    finally:
        await engine.aclose()
