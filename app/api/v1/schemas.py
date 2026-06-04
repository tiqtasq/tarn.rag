"""API request/response schemas. Document-centric — jobs never appear in the contract
(except the debug-only ``jobs`` field under ``?verbose=true``)."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from app.domains.ingestion.stages.parsers import AVAILABLE_PDF_PARSERS


def validate_parser(value: str | None) -> str | None:
    """Reject unknown parser names at the API edge (→ 422), before anything is queued.
    Reused by the multipart upload endpoint (which has no Pydantic body)."""
    if value is not None and value not in AVAILABLE_PDF_PARSERS:
        raise ValueError(
            f"unknown parser {value!r}; available: {sorted(AVAILABLE_PDF_PARSERS)}"
        )
    return value


class IngestRequest(BaseModel):
    """Ingest documents from file paths. ``parser`` (optional) selects the PDF backend for
    the whole request; omit it for the default (applies to PDFs only)."""

    file_paths: list[str]
    parser: str | None = None

    _check_parser = field_validator("parser")(validate_parser)


class IngestFromContentRequest(BaseModel):
    """Ingest pre-loaded content: ``[{"content": "...", "source_id": "..."}]`` (source_id
    optional; extra keys flow into the document metadata). ``parser`` as in IngestRequest."""

    documents: list[dict[str, str]]
    parser: str | None = None

    _check_parser = field_validator("parser")(validate_parser)


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
