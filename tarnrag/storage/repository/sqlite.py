"""
SQLite adapter (SQLAlchemy Core + aiosqlite) — also the §8 retrieval store.

The normal §8 tables (documents / chunks / method_chunks / index_meta) are SQLAlchemy ``Table``s
defined in the base; the dense + sparse retrieval indexes are SQLite-extension **virtual tables**:
``vec_chunks`` (sqlite-vec, dense KNN) and ``fts_chunks`` (FTS5, sparse).

Those virtual-table operations are intentionally **raw SQL** (``exec_driver_sql``), not SQLAlchemy
Core, for two reasons:

* SQLAlchemy doesn't model virtual tables — no ``Table`` emits ``CREATE VIRTUAL TABLE ... USING
  ...`` (true of built-in FTS5, not just the proprietary sqlite-vec).
* The KNN query uses sqlite-vec's proprietary ``MATCH`` operator + synthetic ``distance`` column,
  which has no SQLAlchemy expression.

A few statements that touch only the normal tables (e.g. ``hydrate``'s join, the ``chunk_id``
lookups) are kept raw too, deliberately, so the whole sqlite-vec/FTS layer reads as one block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sqlite_vec
from sqlalchemy import Text, event
from sqlalchemy.ext.asyncio import AsyncConnection

from tarnrag.storage.models import Chunk, Embedding
from tarnrag.storage.repository.base import DocumentRepository
from tarnrag.storage.retrieval import Candidate, ChunkRecord


class SqliteRepository(DocumentRepository):
    """
    SQLite adapter and the §8 retrieval store: dense vectors in ``vec_chunks`` (sqlite-vec) and
    sparse text in ``fts_chunks`` (FTS5). See the module docstring for why that layer is raw SQL.
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

    async def _after_create_schema(self, conn: AsyncConnection) -> None:
        """
        Build the §8 sqlite-vec dense index + FTS5 sparse index (virtual tables; the sqlite-vec
        extension is loaded per connection by the connect hook above).
        """
        await conn.exec_driver_sql(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_id TEXT PRIMARY KEY, embedding float[{self.embedding_dimension}])"
        )
        await conn.exec_driver_sql(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5("
            "chunk_id UNINDEXED, text, tokenize='unicode61')"
        )

    def _driver_url(self, url: str) -> str:
        path = url.replace("sqlite:///", "").replace("sqlite://", "")
        return f"sqlite+aiosqlite:///{path}"

    def _vector_type(self):
        return Text

    def _encode_vector(self, vector: list[float]):
        return json.dumps(list(vector))

    def _decode_vector(self, stored) -> list[float]:
        return json.loads(stored)

    async def vector_search(
        self,
        vector: list[float],
        k: int = 10,
        model: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Top-k by sqlite-vec dense KNN over ``vec_chunks``, mapped to the ``(Chunk, similarity)``
        shape (similarity = -distance, so nearest ranks highest).
        """
        results: list[tuple[Chunk, float]] = []
        for cand in await self.dense_knn(vector, k):
            chunk = await self.get_chunk(cand.chunk_id)
            if chunk is not None:
                results.append((chunk, -cand.raw_score))
        return results

    async def store_embeddings(self, embeddings: list[Embedding]) -> list[str]:
        """
        §8: dense vectors live in the sqlite-vec ``vec_chunks`` virtual table, not the embeddings
        table (which stays empty on SQLite).
        """
        if not embeddings:
            return []
        rows = [(e.chunk_id, sqlite_vec.serialize_float32(e.vector)) for e in embeddings]
        async with self.engine.begin() as conn:
            await conn.exec_driver_sql(
                "INSERT OR REPLACE INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)", rows
            )
        return [e.chunk_id for e in embeddings]

    async def dense_knn(self, query_vec: list[float], k: int) -> list[Candidate]:
        """
        Exact KNN over ``vec_chunks`` (sqlite-vec), nearest first. Raw SQL: ``MATCH`` and the
        synthetic ``distance`` column are sqlite-vec's proprietary query API (see module docstring).
        """
        q = sqlite_vec.serialize_float32(query_vec)
        async with self.engine.connect() as conn:
            rows = (
                await conn.exec_driver_sql(
                    "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? "
                    "ORDER BY distance LIMIT ?",
                    (q, k),
                )
            ).fetchall()
        return [
            Candidate(chunk_id=cid, rank=i + 1, raw_score=dist)
            for i, (cid, dist) in enumerate(rows)
        ]

    async def hydrate(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        """
        Canonical text + provenance for the given chunk ids, preserving input order.
        """
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        async with self.engine.connect() as conn:
            rows = (
                await conn.exec_driver_sql(
                    "SELECT c.chunk_id, c.text, c.document_id, d.source_kind, d.standard_id, "
                    "c.locator, c.license_class FROM chunks c "
                    "JOIN documents d ON c.document_id = d.document_id "
                    f"WHERE c.chunk_id IN ({marks})",
                    tuple(chunk_ids),
                )
            ).fetchall()
            by_id = {r[0]: r for r in rows}
            records: list[ChunkRecord] = []
            for cid in chunk_ids:
                r = by_id.get(cid)
                if r is None:
                    continue
                methods = (
                    await conn.exec_driver_sql(
                        "SELECT method_id, method_version FROM method_chunks WHERE chunk_id = ?",
                        (cid,),
                    )
                ).fetchall()
                records.append(
                    ChunkRecord(
                        chunk_id=r[0], text=r[1], document_id=r[2], source_kind=r[3],
                        standard_id=r[4], locator=r[5], license_class=r[6],
                        methods=[(m, v) for m, v in methods],
                    )
                )
        return records

    # ----- §8 search-index hooks (vec0 + FTS5 live outside the FK graph) -----

    async def _index_chunk_text(
        self, conn: AsyncConnection, ids: list[str], chunks: list[Chunk]
    ) -> None:
        rows = [(cid, ch.content) for cid, ch in zip(ids, chunks)]
        if rows:
            await conn.exec_driver_sql("INSERT INTO fts_chunks(chunk_id, text) VALUES (?, ?)", rows)

    async def _clear_chunk_index(self, conn: AsyncConnection, document_id: str) -> None:
        old = [
            r[0]
            for r in (
                await conn.exec_driver_sql(
                    "SELECT chunk_id FROM chunks WHERE document_id = ?", (document_id,)
                )
            ).fetchall()
        ]
        if not old:
            return
        marks = ",".join("?" * len(old))
        await conn.exec_driver_sql(f"DELETE FROM vec_chunks WHERE chunk_id IN ({marks})", tuple(old))
        await conn.exec_driver_sql(f"DELETE FROM fts_chunks WHERE chunk_id IN ({marks})", tuple(old))

    async def _count_doc_embeddings(self, conn: AsyncConnection, document_id: str) -> int:
        row = (
            await conn.exec_driver_sql(
                "SELECT count(*) FROM vec_chunks WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE document_id = ?)",
                (document_id,),
            )
        ).fetchone()
        return row[0]

    async def _embedding_counts_by_document(self, conn: AsyncConnection) -> dict[str, int]:
        rows = (
            await conn.exec_driver_sql(
                "SELECT c.document_id, count(v.chunk_id) FROM chunks c "
                "JOIN vec_chunks v ON v.chunk_id = c.chunk_id GROUP BY c.document_id"
            )
        ).fetchall()
        return {doc_id: cnt for doc_id, cnt in rows}
