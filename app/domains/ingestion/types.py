"""Public result types for the ingestion engine (document-centric — no job internals)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentRef:
    """A handle returned when a document is accepted for ingestion."""

    document_id: str
    status: str = "queued"


@dataclass(frozen=True)
class IngestSubmission:
    """The outcome of an ingest call: one ref per document plus the count accepted."""

    documents: list[DocumentRef]
    queued: int


@dataclass(frozen=True)
class DocumentStatus:
    """Document-level status derived from persisted data."""

    document_id: str
    status: str  # pending | in_progress | complete | failed
    chunk_count: int
    embedding_count: int
    jobs: list[dict] | None = None  # debug-only per-job breakdown (verbose reads)
