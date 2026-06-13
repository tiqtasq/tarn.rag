"""Ingestion domain — public surface.

``IngestionEngine`` is the high-level facade (``await IngestionEngine.create()``); the result
types describe what its calls return. Everything else (orchestrator, queue, worker, stages) is
internal wiring.
"""

from app.domains.ingestion.engine import IngestionEngine
from app.domains.ingestion.types import DocumentRef, DocumentStatus, IngestSubmission

__all__ = ["IngestionEngine", "IngestSubmission", "DocumentRef", "DocumentStatus"]
