"""SQLite adapter (SQLAlchemy Core + aiosqlite).

Vectors are stored as JSON text and searched with in-memory cosine similarity —
fine for development and small-scale use, not for production-scale retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import Text, event, select
from sqlalchemy.ext.asyncio import AsyncConnection

from tarnrag.storage.models import Chunk
from tarnrag.storage.repository.base import DocumentRepository


class SqliteRepository(DocumentRepository):
    """SQLite adapter; vectors as JSON, in-memory cosine search."""

    def __init__(self, connection_url: str, embedding_dimension: int = 384):
        super().__init__(connection_url, embedding_dimension)
        # SQLite enforces foreign keys / ON DELETE CASCADE only when
        # ``PRAGMA foreign_keys=ON``, which is per-connection.
        @event.listens_for(self.engine.sync_engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record):  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    async def connect(self) -> None:
        # SQLite won't create the parent directory — ensure it exists (no-op for in-memory).
        db_file = self.engine.url.database
        if db_file and db_file != ":memory:":
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        await super().connect()

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
