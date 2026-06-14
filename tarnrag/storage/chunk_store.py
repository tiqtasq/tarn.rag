"""The persistence seam the ResultSinks (D4) write through.

``ChunkStore`` is the surface a sink needs — *where* document/chunk/embedding data lands. It is
implemented by ``DocumentRepository`` (SQLAlchemy), which is the single store: documents/chunks,
the §8 retrieval index (sqlite-vec/FTS5 on SQLite, pgvector on Postgres), and the ``job_status``
projection. Methods are async to match the sinks' ``await``ed ``_persist``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tarnrag.storage.models import Chunk, Document, Embedding


class ChunkStore(ABC):
    """
    The persistence port a ResultSink writes through — implemented by ``DocumentRepository``
    (see the module docstring).
    """

    @abstractmethod
    async def store_document(self, doc: Document) -> str:
        """Upsert a document; return its id (threaded forward as ``metadata['doc_id']``)."""

    @abstractmethod
    async def store_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Persist chunks atomically; return their ids (in order)."""

    @abstractmethod
    async def store_embeddings(self, embeddings: list[Embedding]) -> list[str]:
        """Persist chunk embeddings; return the chunk ids written."""

    @abstractmethod
    async def update_chunk_metadata(self, chunk_id: str, updates: dict[str, Any]) -> None:
        """Merge enrichment into a chunk (no-op for stores without a chunk-metadata column)."""
