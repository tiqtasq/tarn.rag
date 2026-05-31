"""SQLite adapter (SQLAlchemy Core + aiosqlite).

Vectors are stored as JSON text and searched with in-memory cosine similarity —
fine for development and small-scale use, not for production-scale retrieval.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sqlalchemy import Text, select
from sqlalchemy.ext.asyncio import AsyncConnection

from app.domains.base.models import Chunk
from app.domains.base.repository import DocumentRepository


class SqliteRepository(DocumentRepository):
    """SQLite adapter; vectors as JSON, in-memory cosine search."""

    def _driver_url(self, url: str) -> str:
        path = url.replace("sqlite:///", "").replace("sqlite://", "")
        return f"sqlite+aiosqlite:///{path}"

    def _vector_type(self):
        return Text

    def _encode_vector(self, vector: list[float]):
        return json.dumps(list(vector))

    def _decode_vector(self, stored) -> list[float]:
        return json.loads(stored)

    async def _after_create_schema(self, conn: AsyncConnection) -> None:
        await conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_id "
            "ON documents (json_extract(metadata, '$.source_id'))"
        )

    async def vector_search(
        self,
        vector: list[float],
        k: int = 10,
        model: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        stmt = select(self.chunks, self.embeddings.c.vector).join(
            self.embeddings, self.embeddings.c.chunk_id == self.chunks.c.id
        )
        if model:
            stmt = stmt.where(self.embeddings.c.model == model)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        q = np.asarray(vector, dtype=float)
        qn = np.linalg.norm(q) or 1.0
        scored: list[tuple[Chunk, float]] = []
        for r in rows:
            v = np.asarray(self._decode_vector(r["vector"]), dtype=float)
            sim = float(q @ v / (qn * (np.linalg.norm(v) or 1.0)))
            scored.append((self._row_to_chunk(r), sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]
