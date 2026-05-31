"""PostgreSQL adapter (SQLAlchemy Core + asyncpg + pgvector).

Real vector search via the pgvector ``<=>`` cosine-distance operator. Requires the
``postgres`` extra (``asyncpg``, ``pgvector``); not imported on the SQLite path.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection
from pgvector.sqlalchemy import Vector

from app.domains.base.models import Chunk
from app.domains.base.repository import DocumentRepository


class PostgresRepository(DocumentRepository):
    """PostgreSQL adapter. Inherits the portable upsert + CRUD from the base; supplies
    the asyncpg driver URL, the pgvector column, and pgvector cosine search."""

    def _driver_url(self, url: str) -> str:
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)

    def _vector_type(self):
        return Vector(self.embedding_dimension)

    def _encode_vector(self, vector: list[float]):
        return list(vector)  # pgvector accepts a Python list directly

    def _decode_vector(self, stored) -> list[float]:
        return list(stored)

    async def _before_create_schema(self, conn: AsyncConnection) -> None:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")

    async def _after_create_schema(self, conn: AsyncConnection) -> None:
        await conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_id "
            "ON documents ((metadata->>'source_id'))"
        )
        await conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_vector "
            "ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 100)"
        )

    async def vector_search(
        self,
        vector: list[float],
        k: int = 10,
        model: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        dist = self.embeddings.c.vector.cosine_distance(vector)
        stmt = (
            select(self.chunks, (1 - dist).label("similarity"))
            .join(self.embeddings, self.embeddings.c.chunk_id == self.chunks.c.id)
            .order_by(dist)
            .limit(k)
        )
        if model:
            stmt = stmt.where(self.embeddings.c.model == model)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        return [(self._row_to_chunk(r), r["similarity"]) for r in rows]
