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
    index_meta_conflict,
)
from tarnrag.contracts.ports import ChunkStore, DocumentFactsSource, JobStatusSource, RetrievalStore
from tarnrag.contracts.results import Candidate, ChunkRecord, RetrievalResult
from tarnrag.contracts.structure import (
    Annotation,
    ChunkProvenance,
    Element,
    ElementKind,
    Geometry,
    PageBox,
    Span,
    StructuredDocument,
    Table,
    TableCell,
)

__all__ = [
    # dtos
    "PipelineItem",
    "Document",
    "Chunk",
    "Embedding",
    "DocumentFacts",
    "MethodRef",
    # structure (layout-aware extraction)
    "StructuredDocument",
    "Element",
    "ElementKind",
    "Table",
    "TableCell",
    "Annotation",
    "Span",
    "PageBox",
    "Geometry",
    "ChunkProvenance",
    # ports
    "ChunkStore",
    "RetrievalStore",
    "DocumentFactsSource",
    "JobStatusSource",
    # results
    "Candidate",
    "ChunkRecord",
    "RetrievalResult",
    # index_meta
    "SCHEMA_VERSION",
    "index_meta_conflict",
    "INGESTION_VERSION",
    "FTS_TOKENIZER",
    "build_index_meta",
]
