"""Ingestion domain — public surface.

``IngestionEngine`` is the high-level producer/query facade (``await IngestionEngine.create()``);
``run_worker`` is the consumer entry point for distributed mode. The result types describe what
the engine's calls return. Everything else (orchestrator, queue, worker, stages) is internal.
"""

from tarnrag.ingestion.engine import IngestionEngine, run_worker
from tarnrag.ingestion.types import DocumentStatus, DocumentSummary

__all__ = ["IngestionEngine", "run_worker", "DocumentStatus", "DocumentSummary"]
