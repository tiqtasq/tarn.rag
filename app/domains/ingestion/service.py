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
    ):
        self.pipeline = pipeline
        self.orchestrator = orchestrator
        self.repository = repository  # persistence + document-status reads
        self.obs = observability

    async def ingest_from_paths(self, file_paths: list[str]) -> dict[str, Any]:
        """Queue ingestion of documents loaded from file paths (LoadAndParse reads them)."""
        items = [
            self._item(
                source_id := str(uuid.uuid4()),
                content="",  # loaded by LoadAndParseStage
                extra={"source_path": path, "source_type": self._infer_source_type(path)},
            )
            for path in file_paths
        ]
        return await self._queue(items)

    async def ingest_from_content(self, documents: list[dict[str, str]]) -> dict[str, Any]:
        """Queue ingestion of pre-loaded content. A client-supplied ``source_id`` becomes
        the ``document_id``; otherwise one is assigned."""
        items = []
        for doc in documents:
            doc = dict(doc)  # don't mutate the caller's dict
            content = doc.pop("content")
            source_id = doc.pop("source_id", None) or str(uuid.uuid4())
            items.append(self._item(source_id, content=content, extra=doc))
        return await self._queue(items)

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
