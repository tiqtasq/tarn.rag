"""tarnrag — RAG ingestion + retrieval engines.

Public API::

    from tarnrag import IngestionEngine, RetrievalEngine, run_worker, Query, DocumentStatus
"""

from tarnrag.ingestion import DocumentStatus, IngestionEngine, run_worker
from tarnrag.retrieval import MethodRef, Query, RetrievalEngine, RetrievalError, RetrievalResult

__all__ = [
    "IngestionEngine",
    "run_worker",
    "DocumentStatus",
    "RetrievalEngine",
    "RetrievalError",
    "Query",
    "RetrievalResult",
    "MethodRef",
]
