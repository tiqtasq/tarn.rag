"""SQLAlchemy 2.0 Core repository shared by ingestion and retrieval.

All dialect-agnostic table definitions and CRUD live in ``DocumentRepository``;
subclasses supply only the Postgres/SQLite specifics via a small set of hooks.
"""

from __future__ import annotations

import uuid
from abc import abstractmethod
from typing import Any

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from tarnrag.core.config import DatabaseSettings
from tarnrag.core.exceptions import ChunkNotFoundError
from tarnrag.storage.chunk_store import ChunkStore
from tarnrag.storage.models import Chunk, Document, Embedding
from tarnrag.storage.status import (
    DocumentFacts,
    DocumentFactsSource,
    DocumentStatusReader,
    JobStatusSource,
)


class DocumentRepository(ChunkStore, JobStatusSource, DocumentFactsSource):
    """SQLAlchemy Core repository shared by ingestion and retrieval.

    Subclasses supply only the dialect specifics: the async driver URL, the vector
    column type + value encoding, the upsert, dialect-only schema objects (pgvector
    extension/index, expression indexes), and the vector-search query. Everything
    else — table definitions, lifecycle, and the rest of the CRUD — is shared here.

    Guarantees:
    - Atomicity: multi-row writes (store_document_with_chunks / store_chunks /
      store_embeddings) run in one transaction (engine.begin) — all or nothing.
    - Idempotency: documents are keyed by metadata['source_id'] (UNIQUE). Re-ingesting
      a source UPSERTS the document and REPLACES its chunks/embeddings (cascade);
      never duplicates.
    """

    @classmethod
    async def create(
        cls, database: DatabaseSettings, embedding_dimension: int
    ) -> DocumentRepository:
        """Build and connect the repository selected by ``database.document_url`` (Postgres on
        a ``postgres`` URL, else SQLite). Heavy backends are imported lazily."""
        if "postgres" in database.document_url:
            from tarnrag.storage.repository.postgres import PostgresRepository

            repo: DocumentRepository = PostgresRepository(
                database.document_url, embedding_dimension=embedding_dimension
            )
        else:
            from tarnrag.storage.repository.sqlite import SqliteRepository

            repo = SqliteRepository(
                database.document_url, embedding_dimension=embedding_dimension
            )
        await repo.connect()
        return repo

    def __init__(self, connection_url: str, embedding_dimension: int = 384):
        self.embedding_dimension = embedding_dimension
        self.engine: AsyncEngine = create_async_engine(self._driver_url(connection_url))
        self.metadata = MetaData()
        self._define_tables()

    # ---------------- dialect hooks (subclasses implement) ----------------

    @abstractmethod
    def _driver_url(self, url: str) -> str:
        """Map a user URL to its async driver URL (e.g. postgresql+asyncpg://...)."""

    @abstractmethod
    def _vector_type(self):
        """Column type for an embedding vector (pgvector Vector vs Text/JSON)."""

    @abstractmethod
    def _encode_vector(self, vector: list[float]):
        """Adapt a vector for storage (list for pgvector, JSON string for SQLite)."""

    @abstractmethod
    def _decode_vector(self, stored) -> list[float]:
        """Inverse of _encode_vector when reading rows back."""

    @abstractmethod
    async def vector_search(
        self,
        vector: list[float],
        k: int = 10,
        model: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Semantic search: pgvector cosine on Postgres, in-memory cosine on SQLite."""

    # Schema hooks: default no-ops. The pgvector extension must exist *before* the
    # tables (the embeddings vector column needs it); expression/vector indexes must
    # be created *after* the tables exist — hence two hooks rather than one.
    async def _before_create_schema(self, conn: AsyncConnection) -> None:
        return None

    async def _after_create_schema(self, conn: AsyncConnection) -> None:
        return None

    # ---------------- shared schema ----------------

    def _define_tables(self) -> None:
        json_type = JSON().with_variant(JSONB(), "postgresql")
        self.documents = Table(
            "documents",
            self.metadata,
            Column("id", Text, primary_key=True),
            Column("content", Text, nullable=False),
            Column("metadata", json_type, nullable=False, default=dict),
            # sha256 of the document's submitted content — the content-dedup key (independent
            # of the source_id identity, which is stable across content replacement).
            Column("content_hash", Text, index=True),
            Column("created_at", TIMESTAMP, server_default=func.now()),
        )
        self.chunks = Table(
            "chunks",
            self.metadata,
            Column("id", Text, primary_key=True),
            Column(
                "parent_doc_id",
                Text,
                ForeignKey("documents.id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("content", Text, nullable=False),
            Column("chunk_index", Integer, nullable=False),
            Column("total_chunks", Integer, nullable=False),
            Column("metadata", json_type, nullable=False, default=dict),
            Column("created_at", TIMESTAMP, server_default=func.now()),
            Index("idx_chunks_parent", "parent_doc_id"),
        )
        self.embeddings = Table(
            "embeddings",
            self.metadata,
            Column("id", Text, primary_key=True),
            Column(
                "chunk_id",
                Text,
                ForeignKey("chunks.id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("vector", self._vector_type(), nullable=False),
            Column("model", Text, nullable=False),
            Column("dimension", Integer, nullable=False),
            Column("metadata", json_type, nullable=False, default=dict),
            Index("idx_embeddings_chunk", "chunk_id"),
        )
        # Document-keyed status PROJECTION for the API (the queue itself is pgQueuer's).
        self.job_status = Table(
            "job_status",
            self.metadata,
            Column("job_id", Text, primary_key=True),
            Column("document_id", Text, nullable=False, index=True),
            Column("stage_name", Text, nullable=False),
            Column("status", Text, nullable=False, default="queued"),
            Column("error", Text),
            Column("created_at", TIMESTAMP, server_default=func.now()),
            Column("updated_at", TIMESTAMP, server_default=func.now()),
        )

    # ---------------- lifecycle ----------------

    async def connect(self) -> None:
        async with self.engine.begin() as conn:
            await self._before_create_schema(conn)
            await conn.run_sync(self.metadata.create_all)
            await self._after_create_schema(conn)

    async def disconnect(self) -> None:
        await self.engine.dispose()

    async def health_check(self) -> bool:
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    # ---------------- writes (shared Core) ----------------

    async def store_document(self, doc: Document) -> str:
        # Atomic: upsert + chunk-delete share one engine.begin() transaction — if the
        # delete fails, the upsert is rolled back too (commits only on clean exit).
        async with self.engine.begin() as conn:
            doc_id = await self._upsert_document(conn, self._doc_values(doc))
            # Re-storing a document REPLACES its derived data (idempotency): drop its
            # chunks (ON DELETE CASCADE removes their embeddings). A new doc has none.
            await conn.execute(
                self.chunks.delete().where(self.chunks.c.parent_doc_id == doc_id)
            )
            return doc_id

    async def store_document_with_chunks(
        self, doc: Document, chunks: list[Chunk]
    ) -> tuple[str, list[str]]:
        async with self.engine.begin() as conn:
            doc_id = await self._upsert_document(conn, self._doc_values(doc))
            # Re-ingest replaces chunks (ON DELETE CASCADE removes their embeddings).
            await conn.execute(
                self.chunks.delete().where(self.chunks.c.parent_doc_id == doc_id)
            )
            chunk_ids = await self._insert_chunks(conn, doc_id, chunks)
            return doc_id, chunk_ids

    async def store_chunks(self, chunks: list[Chunk]) -> list[str]:
        async with self.engine.begin() as conn:
            return await self._insert_chunks(conn, None, chunks)

    async def store_embeddings(self, embeddings: list[Embedding]) -> list[str]:
        rows, ids = [], []
        for emb in embeddings:
            eid = emb.id or str(uuid.uuid4())
            ids.append(eid)
            rows.append(
                {
                    "id": eid,
                    "chunk_id": emb.chunk_id,
                    "vector": self._encode_vector(emb.vector),
                    "model": emb.model,
                    "dimension": emb.dimension,
                    "metadata": emb.metadata,
                }
            )
        async with self.engine.begin() as conn:
            if rows:
                await conn.execute(insert(self.embeddings), rows)
        return ids

    async def update_chunk_metadata(self, chunk_id: str, updates: dict[str, Any]) -> None:
        async with self.engine.begin() as conn:
            row = (
                await conn.execute(
                    select(self.chunks.c.metadata).where(self.chunks.c.id == chunk_id)
                )
            ).first()
            if row is None:
                raise ChunkNotFoundError(chunk_id)
            merged = {**(row[0] or {}), **updates}
            await conn.execute(
                update(self.chunks)
                .where(self.chunks.c.id == chunk_id)
                .values(metadata=merged)
            )

    # ---------------- reads (shared Core) ----------------

    async def get_document(self, doc_id: str) -> Document | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    select(self.documents).where(self.documents.c.id == doc_id)
                )
            ).mappings().first()
        return Document(**self._pick(row, Document)) if row else None

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    select(self.chunks).where(self.chunks.c.id == chunk_id)
                )
            ).mappings().first()
        return self._row_to_chunk(row) if row else None

    async def get_chunks_by_document(self, doc_id: str) -> list[Chunk]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(self.chunks)
                    .where(self.chunks.c.parent_doc_id == doc_id)
                    .order_by(self.chunks.c.chunk_index)
                )
            ).mappings().all()
        return [self._row_to_chunk(r) for r in rows]

    async def query_chunks(
        self, filters: dict[str, Any], limit: int = 100
    ) -> list[Chunk]:
        stmt = select(self.chunks)
        if "source_id" in filters:
            stmt = stmt.where(
                self.chunks.c.metadata["source_id"].as_string() == filters["source_id"]
            )
        if "source_type" in filters:
            stmt = stmt.where(
                self.chunks.c.metadata["source_type"].as_string()
                == filters["source_type"]
            )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt.limit(limit))).mappings().all()
        return [self._row_to_chunk(r) for r in rows]

    # ---------------- job-status projection (for the document-status API) ----------------

    async def record_job(
        self,
        document_id: str,
        job_id: str,
        stage_name: str,
        status: str,
        error: str | None = None,
    ) -> None:
        """Upsert a job's status row (portable: update, else insert)."""
        async with self.engine.begin() as conn:
            res = await conn.execute(
                update(self.job_status)
                .where(self.job_status.c.job_id == job_id)
                .values(status=status, error=error, updated_at=func.now())
            )
            if res.rowcount == 0:
                await conn.execute(
                    insert(self.job_status).values(
                        job_id=job_id,
                        document_id=document_id,
                        stage_name=stage_name,
                        status=status,
                        error=error,
                    )
                )

    async def document_jobs(self, document_id: str) -> list[dict[str, Any]]:
        """Debug-only per-job breakdown for one document."""
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(
                        self.job_status.c.job_id,
                        self.job_status.c.stage_name,
                        self.job_status.c.status,
                        self.job_status.c.error,
                    ).where(self.job_status.c.document_id == document_id)
                )
            ).mappings().all()
        return [dict(r) for r in rows]

    async def document_facts(self, document_id: str) -> DocumentFacts:
        """Persisted-data facts (presence + chunk/embedding counts) for this document."""
        src = self.documents.c.metadata["source_id"].as_string()
        async with self.engine.connect() as conn:
            doc = (
                await conn.execute(select(self.documents.c.id).where(src == document_id))
            ).first()
            if doc is None:
                return DocumentFacts(present=False, chunk_count=0, embedding_count=0)
            doc_id = doc[0]
            chunk_count = (
                await conn.execute(
                    select(func.count())
                    .select_from(self.chunks)
                    .where(self.chunks.c.parent_doc_id == doc_id)
                )
            ).scalar_one()
            embedding_count = (
                await conn.execute(
                    select(func.count())
                    .select_from(self.embeddings)
                    .join(self.chunks, self.embeddings.c.chunk_id == self.chunks.c.id)
                    .where(self.chunks.c.parent_doc_id == doc_id)
                )
            ).scalar_one()
        return DocumentFacts(True, chunk_count, embedding_count)

    async def documents_by_content_hash(self, content_hash: str) -> list[str]:
        """Public document_ids (== source_id) whose stored content_hash matches — content dedup."""
        src = self.documents.c.metadata["source_id"].as_string()
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(src).where(self.documents.c.content_hash == content_hash)
                )
            ).all()
        return [r[0] for r in rows]

    async def document_status(self, document_id: str) -> dict[str, Any] | None:
        """Convenience: status over this repo alone (job_status + data both here). The
        retrieval path composes a reader over the repo (jobs) + the index store (facts)."""
        return await DocumentStatusReader(self, self).document_status(document_id)

    # ---------------- shared helpers ----------------

    async def _upsert_document(self, conn: AsyncConnection, values: dict) -> str:
        """Upsert on metadata['source_id']: update if present, else insert. Portable
        (no dialect ON CONFLICT); the UNIQUE source_id index is the safety net.
        Returns the (existing) document id so chunks resolve parent_doc_id."""
        source_id = (values.get("metadata") or {}).get("source_id")
        if source_id is not None:
            src = self.documents.c.metadata["source_id"].as_string()
            existing = (
                await conn.execute(
                    select(self.documents.c.id).where(src == source_id)
                )
            ).scalar()
            if existing is not None:
                await conn.execute(
                    update(self.documents)
                    .where(self.documents.c.id == existing)
                    .values(
                        content=values["content"],
                        metadata=values["metadata"],
                        content_hash=values.get("content_hash"),
                    )
                )
                return existing
        await conn.execute(insert(self.documents).values(**values))
        return values["id"]

    def _doc_values(self, doc: Document) -> dict:
        return {
            "id": doc.id or str(uuid.uuid4()),
            "content": doc.content,
            "metadata": doc.metadata,
            "content_hash": (doc.metadata or {}).get("content_hash"),
        }

    async def _insert_chunks(
        self, conn: AsyncConnection, parent_doc_id: str | None, chunks: list[Chunk]
    ) -> list[str]:
        rows, ids = [], []
        for ch in chunks:
            cid = ch.id or str(uuid.uuid4())
            ids.append(cid)
            rows.append(
                {
                    "id": cid,
                    "parent_doc_id": parent_doc_id or ch.parent_doc_id,
                    "content": ch.content,
                    "chunk_index": ch.chunk_index,
                    "total_chunks": ch.total_chunks,
                    "metadata": ch.metadata,
                }
            )
        if rows:
            await conn.execute(insert(self.chunks), rows)
        return ids

    def _row_to_chunk(self, r) -> Chunk:
        return Chunk(
            id=r["id"],
            parent_doc_id=r["parent_doc_id"],
            content=r["content"],
            chunk_index=r["chunk_index"],
            total_chunks=r["total_chunks"],
            metadata=r["metadata"],
        )

    def _pick(self, r, model) -> dict:
        return {k: r[k] for k in model.model_fields if k in r}
