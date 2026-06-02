"""Ingestion API — document-centric. Clients get a ``document_id`` and poll its status;
jobs are internal (surfaced only under ``?verbose=true`` for debugging)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.dependencies import get_ingestion_service
from app.api.v1.schemas import (
    DocumentStatusResponse,
    IngestFromContentRequest,
    IngestRequest,
    IngestResponse,
)
from app.domains.ingestion.service import IngestionService

router = APIRouter(prefix="/v1/ingest", tags=["ingestion"])


@router.post("/", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Ingest documents from file paths. Returns a ``document_id`` per document."""
    return IngestResponse(**await service.ingest_from_paths(req.file_paths))


@router.post("/content", response_model=IngestResponse)
async def ingest_content(
    req: IngestFromContentRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Ingest pre-loaded document content. Returns a ``document_id`` per document."""
    return IngestResponse(**await service.ingest_from_content(req.documents))


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def document_status(
    document_id: str,
    verbose: bool = False,  # debug only: include the per-job breakdown
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentStatusResponse:
    """Document-level ingestion status. 404 if the document is unknown."""
    status = await service.get_document_status(document_id, verbose=verbose)
    if status is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(**status)
