"""SQLite adapter (SQLAlchemy Core + aiosqlite).

Vectors are stored as JSON text and searched with in-memory cosine similarity —
fine for development and small-scale use, not for production-scale retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec
from sqlalchemy import Text, event, select
from sqlalchemy.ext.asyncio import AsyncConnection

from tarnrag.storage.models import Chunk
from tarnrag.storage.repository.base import DocumentRepository


class SqliteRepository(DocumentRepository):
    """
    SQLite adapter; vectors as JSON, in-memory cosine search.
    """

    def __init__(self, connection_url: str, embedding_dimension: int = 384):
        super().__init__(connection_url, embedding_dimension)

        @event.listens_for(self.engine.sync_engine, "connect")
        def _on_connect(dbapi_connection, _record):  # noqa: ANN001
            # SQLite enforces foreign keys / ON DELETE CASCADE only when
            # ``PRAGMA foreign_keys=ON``, which is per-connection.
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
            # Load sqlite-vec so vec0 KNN is available on every connection (see
            # ``_load_sqlite_vec`` for the ``._conn`` rationale). This is the foundation for
            # folding the §8 retrieval index into this repository.
            self._load_sqlite_vec(dbapi_connection)

    @staticmethod
    def _load_sqlite_vec(dbapi_connection) -> None:  # noqa: ANN001
        """
        Load the sqlite-vec extension onto a freshly-opened connection (called from the
        per-connection ``connect`` event), so vec0 dense-KNN is available on every connection.

        Why this reaches a **private aiosqlite attribute** (``._conn``)
        --------------------------------------------------------------
        Loading a SQLite loadable extension needs the C-API calls
        ``conn.enable_load_extension(True)`` / ``conn.load_extension(path)`` on the **real,
        synchronous** ``sqlite3.Connection``. Neither layer above us offers that synchronously:

        * ``dbapi_connection`` here is SQLAlchemy's async adapter
          (``AsyncAdapt_aiosqlite_connection``) — it does **not** expose ``enable_load_extension``.
        * ``dbapi_connection.driver_connection`` is the ``aiosqlite.Connection`` (public SQLAlchemy
          API), but aiosqlite's ``enable_load_extension`` / ``load_extension`` are **async**
          coroutines — unusable from this *synchronous* ``connect`` event.

        So we descend one more level, to the ``sqlite3.Connection`` that aiosqlite wraps, and load
        on it directly. aiosqlite opens that connection with ``check_same_thread=False``, so calling
        its sync methods from this hook is safe::

            dbapi_connection                  # AsyncAdapt_aiosqlite_connection  (SQLAlchemy adapter)
                .driver_connection            # aiosqlite.Connection             (public SQLAlchemy API)
                ._conn                        # sqlite3.Connection               (PRIVATE aiosqlite attr) <-- here

        ``._conn`` is a **private aiosqlite attribute** and is therefore the single fragile point
        of this integration: an aiosqlite-internal rename would silently break extension loading.
        It is guarded by ``tests/storage/test_sqlite_vec_under_aiosqlite.py``, which fails loudly
        if the path changes. If aiosqlite ever exposes a public *sync* accessor (or SQLAlchemy
        gains an async ``connect`` hook), switch to it and delete this note.
        """
        raw_sqlite3_connection = dbapi_connection.driver_connection._conn  # PRIVATE aiosqlite attr
        raw_sqlite3_connection.enable_load_extension(True)
        sqlite_vec.load(raw_sqlite3_connection)
        raw_sqlite3_connection.enable_load_extension(False)

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
        """
        Top-k by cosine similarity computed in-memory with numpy over all stored vectors
        (dev/small-scale; no ANN index).
        """
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
