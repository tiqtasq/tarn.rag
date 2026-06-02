"""API request/response schemas. Document-centric — jobs never appear in the contract
(except the debug-only ``jobs`` field under ``?verbose=true``)."""

from __future__ import annotations

from pydantic import BaseModel


class IngestRequest(BaseModel):
    """Ingest documents from file paths."""

    file_paths: list[str]


class IngestFromContentRequest(BaseModel):
    """Ingest pre-loaded content: ``[{"content": "...", "source_id": "..."}]`` (source_id
    optional; extra keys flow into the document metadata)."""

    documents: list[dict[str, str]]


class DocumentRef(BaseModel):
    """A document handle returned to the client (jobs are never exposed)."""

    document_id: str
    status: str  # "queued"


class IngestResponse(BaseModel):
    """Response from an ingest endpoint — document-centric."""

    documents: list[DocumentRef]
    documents_queued: int


class DocumentStatusResponse(BaseModel):
    """Document-level ingestion status (derived from persisted data)."""

    document_id: str
    status: str  # pending | in_progress | complete | failed
    chunk_count: int
    embedding_count: int
    jobs: list[dict] | None = None  # debug-only, present when ?verbose=true
