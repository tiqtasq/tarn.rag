"""Cross-boundary contracts — the shared kernel.

The DTOs, ports, retrieval-result types, and the ``index_meta`` build record that ingestion,
storage, and retrieval all depend on — and which depends on none of them (a leaf package). Import
from here: ``from tarnrag.contracts import Document, ChunkStore, Candidate, RetrievalResult``.
"""

from tarnrag.contracts.dtos import (
    Chunk,
    Document,
    DocumentFacts,
    Embedding,
    MethodRef,
    PipelineItem,
)
from tarnrag.contracts.index_meta import (
    FTS_TOKENIZER,
    INGESTION_VERSION,
    SCHEMA_VERSION,
    build_index_meta,
)
from tarnrag.contracts.ports import ChunkStore, DocumentFactsSource, JobStatusSource
from tarnrag.contracts.results import Candidate, ChunkRecord, RetrievalResult

__all__ = [
    # dtos
    "PipelineItem",
    "Document",
    "Chunk",
    "Embedding",
    "DocumentFacts",
    "MethodRef",
    # ports
    "ChunkStore",
    "DocumentFactsSource",
    "JobStatusSource",
    # results
    "Candidate",
    "ChunkRecord",
    "RetrievalResult",
    # index_meta
    "SCHEMA_VERSION",
    "INGESTION_VERSION",
    "FTS_TOKENIZER",
    "build_index_meta",
]
