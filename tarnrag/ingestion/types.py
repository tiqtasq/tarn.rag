"""Public result types for the ingestion engine (document-centric — no job internals)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentStatus:
    """
    Document-level status derived from persisted data.
    """

    document_id: str
    status: str  # pending | in_progress | complete | failed
    chunk_count: int
    embedding_count: int


@dataclass(frozen=True)
class DocumentSummary:
    """
    One row of the document inventory returned by ``IngestionEngine.list_documents``.
    """

    document_id: str
    content_hash: str | None
    chunk_count: int
    embedding_count: int
