"""IngestionService — the high-level facade used by the API and CLI.

The public surface is **document-centric**: callers queue documents and poll by
``document_id`` (== ``source_id``). Jobs are an internal detail and never leak here — the
only exception is the debug-only ``verbose`` breakdown on status reads. The facade is thin:
it shapes ``PipelineItem``s, delegates queueing to the orchestrator, and derives status from
persisted data via the repository.
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from app.domains.base.models import PipelineItem
from app.domains.base.repository import DocumentRepository
from app.domains.base.status import DocumentFactsSource, DocumentStatusReader
from app.domains.ingestion.orchestrator import PipelineOrchestrator
from app.domains.ingestion.pipeline import Pipeline

_SOURCE_TYPES = {"pdf": "pdf", "txt": "text", "text": "text", "html": "html", "htm": "html"}


class IngestionService:
    """High-level facade for the ingestion pipeline; document-centric public surface."""

    def __init__(
        self,
        pipeline: Pipeline,
        orchestrator: PipelineOrchestrator,
        repository: DocumentRepository,
        observability: Any = None,  # Phase 5 Observability plugs in here
        staging_dir: str | None = None,  # where uploaded bytes are staged for the worker
        facts_source: DocumentFactsSource | None = None,  # data store for status counts
    ):
        self.pipeline = pipeline
        self.orchestrator = orchestrator
        self.repository = repository  # persistence + job_status projection
        self.obs = observability
        self.staging_dir = staging_dir
        # Status composes the repo's job_status with the data facts. In the classic path the
        # repo is both; in retrieval mode the §8 index store supplies the facts (counts).
        self._status = DocumentStatusReader(repository, facts_source or repository)

    async def ingest_from_paths(
        self, file_paths: list[str], parser: str | None = None
    ) -> dict[str, Any]:
        """Queue ingestion of documents loaded from file paths (LoadAndParse reads them).
        ``parser`` selects the PDF backend for this request (None → default)."""
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
            for path in file_paths
        ]
        return await self._queue(items)

    async def ingest_from_content(
        self, documents: list[dict[str, str]], parser: str | None = None
    ) -> dict[str, Any]:
        """Queue ingestion of pre-loaded content. A client-supplied ``source_id`` becomes
        the ``document_id``; otherwise one is assigned. ``parser`` as in ingest_from_paths."""
        items = []
        for doc in documents:
            doc = dict(doc)  # don't mutate the caller's dict
            content = doc.pop("content")
            source_id = doc.pop("source_id", None) or str(uuid.uuid4())
            if parser:
                doc.setdefault("parser", parser)
            items.append(self._item(source_id, content=content, extra=doc))
        return await self._queue(items)

    async def ingest_from_uploads(
        self, uploads: list[tuple[str, BinaryIO]], parser: str | None = None
    ) -> dict[str, Any]:
        """Queue ingestion of uploaded files: **stream** each ``(filename, source)`` to the
        shared upload dir, then ingest by path (parsing happens in the worker). ``source`` is
        any binary file-like; it is copied in chunks (never fully held in memory). ``parser``
        as elsewhere. Requires ``staging_dir`` to be configured."""
        if not self.staging_dir:
            raise RuntimeError("uploads are not configured (no staging_dir / UPLOAD_DIR)")
        # Offload the blocking disk copy so the event loop isn't stalled on large files.
        paths = [
            await asyncio.to_thread(self._stage_upload, filename, source)
            for filename, source in uploads
        ]
        return await self.ingest_from_paths(paths, parser)

    async def get_document_status(
        self, document_id: str, verbose: bool = False
    ) -> dict[str, Any] | None:
        """Document-level status derived from persisted data (pending | in_progress |
        complete | failed). ``None`` if the document is unknown. ``verbose=True`` adds a
        debug-only ``jobs`` breakdown — the only place per-job state is exposed."""
        return await self._status.document_status(document_id, verbose=verbose)

    # ----- internals -----

    async def _queue(self, items: list[PipelineItem]) -> dict[str, Any]:
        document_ids = await self.orchestrator.ingest_documents(items)
        if self.obs:
            self.obs.counter("ingestion.documents_queued", len(items))
            await self.obs.log("info", "documents queued", count=len(document_ids))
        return {
            "documents": [{"document_id": d, "status": "queued"} for d in document_ids],
            "documents_queued": len(document_ids),
        }

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

    def _stage_upload(self, filename: str, source: BinaryIO) -> str:
        """Stream an uploaded file-like under a unique name (extension preserved so the
        loader dispatches correctly) and return the path. ``copyfileobj`` copies in chunks —
        no full-memory load. Local-filesystem staging — an object store (S3/blob) would plug
        in behind this same method."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in (filename or "") else "bin"
        dest = Path(self.staging_dir) / f"{uuid.uuid4()}.{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as out:
            shutil.copyfileobj(source, out)
        return str(dest)
