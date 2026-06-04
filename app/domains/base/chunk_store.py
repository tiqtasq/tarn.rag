"""The persistence seam the ResultSinks (D4) write through.

``ChunkStore`` is the surface a sink needs — *where* document/chunk/embedding data lands. Two
implementations: ``DocumentRepository`` (SQLAlchemy; the operational store that also backs the
``job_status`` projection) and ``SqliteIndexStore`` (the §8 sqlite-vec/FTS5 retrieval index).
Methods are async to match the sinks' ``await``ed ``_persist``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.domains.base.models import Chunk, Document, Embedding


class ChunkStore(ABC):
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
