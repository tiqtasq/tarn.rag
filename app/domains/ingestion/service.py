"""IngestionService — the high-level facade used by the API and CLI.

The public surface is **document-centric**: callers queue documents and poll by
``document_id`` (== ``source_id``). Jobs are an internal detail and never leak here — the
only exception is the debug-only ``verbose`` breakdown on status reads. The facade is thin:
it shapes ``PipelineItem``s, delegates queueing to the orchestrator, and derives status from
persisted data via the repository.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domains.base.models import PipelineItem
from app.domains.base.repository import DocumentRepository
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
    ):
        self.pipeline = pipeline
        self.orchestrator = orchestrator
        self.repository = repository  # persistence + document-status reads
        self.obs = observability
        self.staging_dir = staging_dir

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
        self, uploads: list[tuple[str, bytes]], parser: str | None = None
    ) -> dict[str, Any]:
        """Queue ingestion of uploaded files: stage each ``(filename, data)`` to the shared
        upload dir, then ingest by path (parsing happens in the worker). ``parser`` as
        elsewhere. Requires ``staging_dir`` to be configured."""
        if not self.staging_dir:
            raise RuntimeError("uploads are not configured (no staging_dir / UPLOAD_DIR)")
        paths = [self._stage_upload(filename, data) for filename, data in uploads]
        return await self.ingest_from_paths(paths, parser)

    async def get_document_status(
        self, document_id: str, verbose: bool = False
    ) -> dict[str, Any] | None:
        """Document-level status derived from persisted data (pending | in_progress |
        complete | failed). ``None`` if the document is unknown. ``verbose=True`` adds a
        debug-only ``jobs`` breakdown — the only place per-job state is exposed."""
        status = await self.repository.document_status(document_id)
        if status is None:
            return None
        if verbose:
            status["jobs"] = await self.repository.document_jobs(document_id)
        return status

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

    def _stage_upload(self, filename: str, data: bytes) -> str:
        """Persist uploaded bytes under a unique name (extension preserved so the loader
        dispatches correctly) and return the path. Local-filesystem staging — an object
        store (S3/blob) would plug in behind this same method."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in (filename or "") else "bin"
        dest = Path(self.staging_dir) / f"{uuid.uuid4()}.{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return str(dest)
