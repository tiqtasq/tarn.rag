"""SQLAlchemy 2.0 Core repository shared by ingestion and retrieval.

All dialect-agnostic table definitions and CRUD live in ``DocumentRepository``;
subclasses supply only the Postgres/SQLite specifics via a small set of hooks.
"""

from __future__ import annotations

import hashlib
import uuid
from abc import abstractmethod
from typing import Any

from sqlalchemy import (
    TIMESTAMP,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from tarnrag.core.config import DatabaseSettings
from tarnrag.storage.chunk_store import ChunkStore
from tarnrag.storage.models import Chunk, Document, Embedding
from tarnrag.storage.retrieval import Candidate, ChunkRecord
from tarnrag.storage.status import (
    DocumentFacts,
    DocumentFactsSource,
    DocumentStatusReader,
    JobStatusSource,
)

# §8 license_class is a closed enum (matches the strategy doc). Single source of truth for the
# CHECK constraints on documents.license_class / chunks.license_class.
LICENSE_CLASSES = (
    "customer_licensed",
    "public_domain",
    "modusq_authored",
    "third_party_copyrighted",
    "third_party_licensed",
)
_LICENSE_CHECK = "license_class IN (" + ", ".join(f"'{c}'" for c in LICENSE_CLASSES) + ")"


class DocumentRepository(ChunkStore, JobStatusSource, DocumentFactsSource):
    """
    SQLAlchemy Core repository shared by ingestion and retrieval.

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
    async def dense_knn(self, query_vec: list[float], k: int) -> list[Candidate]:
        """
        §8 dense retrieval: the nearest ``k`` chunks to ``query_vec`` as ranked ``Candidate``s
        (sqlite-vec on SQLite, pgvector on Postgres).
        """

    @abstractmethod
    async def hydrate(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        """
        §8 hydration: canonical text + provenance + license for the given chunk ids, preserving
        input order. Touches only the normal tables, so each dialect renders it natively (raw SQL
        on SQLite, SQLAlchemy on Postgres).
        """

    # Schema hooks: default no-ops. The pgvector extension must exist *before* the
    # tables (the embeddings vector column needs it); expression/vector indexes must
    # be created *after* the tables exist — hence two hooks rather than one.
    async def _before_create_schema(self, conn: AsyncConnection) -> None:
        return None

    async def _after_create_schema(self, conn: AsyncConnection) -> None:
        return None

    # ---------------- shared schema ----------------

    def _define_tables(self) -> None:
        # §8 documents (ModusQ §8.1): typed provenance + ``document_id`` (== source_id) as the PK.
        # ``content`` (full doc text) and ``content_hash`` are Python-side extras beyond the bare
        # §8 columns — harmless to the C++ reader, which reads only the columns it knows.
        self.documents = Table(
            "documents",
            self.metadata,
            Column("document_id", Text, primary_key=True),  # == source_id (the public handle)
            Column("content", Text, nullable=False),         # full doc text (Python-side; not §8)
            Column("title", Text),
            Column("source_kind", Text, nullable=False, default="document"),
            Column("standard_id", Text),
            Column("doc_version", Text),
            Column("license_class", Text, nullable=False, default="public_domain"),
            Column("content_hash", Text, index=True),        # content-dedup key
            CheckConstraint(_LICENSE_CHECK, name="ck_documents_license_class"),
        )
        # §8 chunks: typed provenance + license denormalized for fast filtering. No metadata bag —
        # a positional/metadata field returns later (see the rag-chunk-metadata-deferred note).
        self.chunks = Table(
            "chunks",
            self.metadata,
            Column("chunk_id", Text, primary_key=True),
            Column(
                "document_id",
                Text,
                ForeignKey("documents.document_id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("ordinal", Integer, nullable=False),      # position within the document
            Column("text", Text, nullable=False),            # canonical chunk text (returned verbatim)
            Column("locator", Text),                         # citable locator, e.g. '§6.4.2'
            Column("license_class", Text, nullable=False, default="public_domain"),
            Column("ai_grounding_allowed", Integer, nullable=False, default=1),
            Column("available", Integer, nullable=False, default=1),
            Column("content_hash", Text, nullable=False),    # sha256 of the chunk text
            CheckConstraint(_LICENSE_CHECK, name="ck_chunks_license_class"),
            CheckConstraint("ai_grounding_allowed IN (0, 1)", name="ck_chunks_ai_grounding"),
            CheckConstraint("available IN (0, 1)", name="ck_chunks_available"),
            Index("idx_chunks_document", "document_id"),
            Index("idx_chunks_license", "license_class", "available"),
        )
        # §8 method_chunks: resolved reference bundles (method version -> chunk).
        self.method_chunks = Table(
            "method_chunks",
            self.metadata,
            Column("method_id", Text, nullable=False),
            Column("method_version", Text, nullable=False),
            Column(
                "chunk_id",
                Text,
                ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
                nullable=False,
            ),
            PrimaryKeyConstraint("method_id", "method_version", "chunk_id"),
            Index("idx_method_chunks_chunk", "chunk_id"),
        )
        self.embeddings = Table(
            "embeddings",
            self.metadata,
            Column("id", Text, primary_key=True),
            Column(
                "chunk_id",
                Text,
                ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("vector", self._vector_type(), nullable=False),
            Column("model", Text, nullable=False),
            Column("dimension", Integer, nullable=False),
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
        # §8 build/compatibility metadata (key/value): schema/ingestion versions and the
        # embedding-pipeline fingerprint the retrieval store validates at open(). Dialect-agnostic —
        # the first §8 table to live in the base as the repository takes over the retrieval index.
        self.index_meta_table = Table(
            "index_meta",
            self.metadata,
            Column("key", Text, primary_key=True),
            Column("value", Text, nullable=False),
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

    # ---------------- §8 index metadata ----------------

    async def write_index_meta(self, meta: dict[str, str]) -> None:
        """
        Upsert §8 build/compatibility metadata (key/value) — e.g. the embedding-pipeline
        fingerprint the retrieval store validates at ``open()``. Portable upsert (update, else
        insert); takes a plain dict so the producer owns how the fingerprint is assembled.
        """
        async with self.engine.begin() as conn:
            for key, value in meta.items():
                res = await conn.execute(
                    update(self.index_meta_table)
                    .where(self.index_meta_table.c.key == key)
                    .values(value=value)
                )
                if res.rowcount == 0:
                    await conn.execute(
                        insert(self.index_meta_table).values(key=key, value=value)
                    )

    async def index_meta(self) -> dict[str, str]:
        """
        All §8 build/compatibility metadata as a dict (empty before anything is written).
        """
        async with self.engine.connect() as conn:
            rows = (await conn.execute(select(self.index_meta_table))).all()
        return {key: value for key, value in rows}

    # ---------------- writes (shared Core) ----------------

    async def store_document(self, doc: Document) -> str:
        """
        Atomic: upsert + chunk-delete share one engine.begin() transaction — if the delete
        fails, the upsert is rolled back too (commits only on clean exit).
        """
        async with self.engine.begin() as conn:
            doc_id = await self._upsert_document(conn, self._doc_values(doc))
            # Re-storing REPLACES derived data (idempotency): clear the search indexes (vec0/FTS
            # don't cascade) then drop chunks (ON DELETE CASCADE removes embeddings).
            await self._clear_chunk_index(conn, doc_id)
            await conn.execute(
                self.chunks.delete().where(self.chunks.c.document_id == doc_id)
            )
            return doc_id

    async def store_document_with_chunks(
        self, doc: Document, chunks: list[Chunk]
    ) -> tuple[str, list[str]]:
        async with self.engine.begin() as conn:
            doc_id = await self._upsert_document(conn, self._doc_values(doc))
            # Re-ingest replaces chunks; clear non-cascading search indexes first.
            await self._clear_chunk_index(conn, doc_id)
            await conn.execute(
                self.chunks.delete().where(self.chunks.c.document_id == doc_id)
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
                }
            )
        async with self.engine.begin() as conn:
            if rows:
                await conn.execute(insert(self.embeddings), rows)
        return ids

    async def update_chunk_metadata(self, chunk_id: str, updates: dict[str, Any]) -> None:
        # §8 chunks carry no metadata column — enrichment is not persisted (the metadata bag is
        # deferred; see the rag-chunk-metadata-deferred note). No-op by design.
        return None

    # ---------------- dialect search-index hooks (vec0 / FTS live outside the FK graph) ----------------

    async def _index_chunk_text(
        self, conn: AsyncConnection, ids: list[str], chunks: list[Chunk]
    ) -> None:
        """
        Index chunk text for sparse search. No-op by default; SQLite writes ``fts_chunks``.
        """
        return None

    async def _clear_chunk_index(self, conn: AsyncConnection, document_id: str) -> None:
        """
        Drop a document's chunks from search indexes the FK CASCADE can't reach (vec0/FTS
        virtual tables). No-op by default (the embeddings table cascades).
        """
        return None

    async def _count_doc_embeddings(self, conn: AsyncConnection, document_id: str) -> int:
        """
        Vector count for a document — default: the embeddings table; SQLite counts ``vec_chunks``.
        """
        return (
            await conn.execute(
                select(func.count())
                .select_from(self.embeddings)
                .join(self.chunks, self.embeddings.c.chunk_id == self.chunks.c.chunk_id)
                .where(self.chunks.c.document_id == document_id)
            )
        ).scalar_one()

    async def _embedding_counts_by_document(self, conn: AsyncConnection) -> dict[str, int]:
        """
        Per-document vector counts — default: the embeddings table; SQLite counts ``vec_chunks``.
        """
        return dict(
            (
                await conn.execute(
                    select(self.chunks.c.document_id, func.count(self.embeddings.c.id))
                    .select_from(
                        self.chunks.join(
                            self.embeddings,
                            self.embeddings.c.chunk_id == self.chunks.c.chunk_id,
                        )
                    )
                    .group_by(self.chunks.c.document_id)
                )
            ).all()
        )

    # ---------------- reads (shared Core) ----------------

    async def get_document(self, doc_id: str) -> Document | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    select(self.documents).where(self.documents.c.document_id == doc_id)
                )
            ).mappings().first()
        if row is None:
            return None
        return Document(id=row["document_id"], content=row["content"], metadata=self._doc_metadata(row))

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    select(self.chunks).where(self.chunks.c.chunk_id == chunk_id)
                )
            ).mappings().first()
        return self._row_to_chunk(row) if row else None

    async def get_chunks_by_document(self, doc_id: str) -> list[Chunk]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(self.chunks)
                    .where(self.chunks.c.document_id == doc_id)
                    .order_by(self.chunks.c.ordinal)
                )
            ).mappings().all()
        return [self._row_to_chunk(r) for r in rows]

    async def query_chunks(
        self, filters: dict[str, Any], limit: int = 100
    ) -> list[Chunk]:
        stmt = select(self.chunks)
        if "source_id" in filters:  # source_id is the document_id under §8
            stmt = stmt.where(self.chunks.c.document_id == filters["source_id"])
        if "license_class" in filters:
            stmt = stmt.where(self.chunks.c.license_class == filters["license_class"])
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

    async def delete_document_jobs(self, document_id: str) -> bool:
        """Remove a document's job_status rows (used when deleting a document)."""
        async with self.engine.begin() as conn:
            res = await conn.execute(
                self.job_status.delete().where(self.job_status.c.document_id == document_id)
            )
        return res.rowcount > 0

    async def document_facts(self, document_id: str) -> DocumentFacts:
        """Persisted-data facts (presence + chunk/embedding counts) for this document."""
        async with self.engine.connect() as conn:
            present = (
                await conn.execute(
                    select(self.documents.c.document_id)
                    .where(self.documents.c.document_id == document_id)
                )
            ).first() is not None
            if not present:
                return DocumentFacts(present=False, chunk_count=0, embedding_count=0)
            chunk_count = (
                await conn.execute(
                    select(func.count())
                    .select_from(self.chunks)
                    .where(self.chunks.c.document_id == document_id)
                )
            ).scalar_one()
            embedding_count = await self._count_doc_embeddings(conn, document_id)
        return DocumentFacts(True, chunk_count, embedding_count)

    async def documents_by_content_hash(self, content_hash: str) -> list[str]:
        """Public document_ids whose stored content_hash matches — content dedup."""
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(self.documents.c.document_id)
                    .where(self.documents.c.content_hash == content_hash)
                )
            ).all()
        return [r[0] for r in rows]

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete the document; its chunks (and their embeddings) cascade off the FKs, and the
        non-cascading search indexes (vec0/FTS) are cleared first. Returns True if it existed.
        """
        async with self.engine.begin() as conn:
            await self._clear_chunk_index(conn, document_id)
            res = await conn.execute(
                self.documents.delete().where(self.documents.c.document_id == document_id)
            )
        return res.rowcount > 0

    async def list_documents(self) -> list[dict[str, Any]]:
        """Inventory of all documents with chunk/embedding counts (three grouped queries — no
        per-document N+1)."""
        async with self.engine.connect() as conn:
            docs = (
                await conn.execute(
                    select(self.documents.c.document_id, self.documents.c.content_hash)
                )
            ).all()
            chunk_counts = dict(
                (
                    await conn.execute(
                        select(self.chunks.c.document_id, func.count())
                        .group_by(self.chunks.c.document_id)
                    )
                ).all()
            )
            embedding_counts = await self._embedding_counts_by_document(conn)
        return [
            {
                "document_id": document_id,
                "content_hash": content_hash,
                "chunk_count": chunk_counts.get(document_id, 0),
                "embedding_count": embedding_counts.get(document_id, 0),
            }
            for document_id, content_hash in docs
        ]

    async def document_status(self, document_id: str) -> dict[str, Any] | None:
        """Convenience: status over this repo alone (job_status + data both here). The
        retrieval path composes a reader over the repo (jobs) + the index store (facts)."""
        return await DocumentStatusReader(self, self).document_status(document_id)

    # ---------------- shared helpers ----------------

    async def _upsert_document(self, conn: AsyncConnection, values: dict) -> str:
        """Upsert on the ``document_id`` primary key (portable: update, else insert). Returns
        the document_id so chunks resolve their FK."""
        document_id = values["document_id"]
        res = await conn.execute(
            update(self.documents)
            .where(self.documents.c.document_id == document_id)
            .values({k: v for k, v in values.items() if k != "document_id"})
        )
        if res.rowcount == 0:
            await conn.execute(insert(self.documents).values(**values))
        return document_id

    def _doc_values(self, doc: Document) -> dict:
        """Map a Document DTO to §8 document columns — provenance read from its metadata bag with
        safe defaults; the full ``content`` is kept Python-side."""
        md = doc.metadata or {}
        return {
            "document_id": md.get("source_id") or doc.id or str(uuid.uuid4()),
            "content": doc.content,
            "title": md.get("title"),
            "source_kind": md.get("source_kind") or "document",
            "standard_id": md.get("standard_id"),
            "doc_version": md.get("doc_version"),
            "license_class": md.get("license_class") or "public_domain",
            "content_hash": md.get("content_hash"),
        }

    async def _insert_chunks(
        self, conn: AsyncConnection, document_id: str | None, chunks: list[Chunk]
    ) -> list[str]:
        rows, ids = [], []
        for ch in chunks:
            cid = ch.id or str(uuid.uuid4())
            ids.append(cid)
            md = ch.metadata or {}
            rows.append(
                {
                    "chunk_id": cid,
                    "document_id": document_id or ch.parent_doc_id,
                    "ordinal": ch.chunk_index,
                    "text": ch.content,
                    "locator": md.get("locator"),
                    "license_class": md.get("license_class") or "public_domain",
                    "ai_grounding_allowed": int(md.get("ai_grounding_allowed", 1)),
                    "available": int(md.get("available", 1)),
                    "content_hash": hashlib.sha256(ch.content.encode("utf-8")).hexdigest(),
                }
            )
        if rows:
            await conn.execute(insert(self.chunks), rows)
            await self._index_chunk_text(conn, ids, chunks)
        return ids

    def _row_to_chunk(self, r) -> Chunk:
        return Chunk(
            id=r["chunk_id"],
            parent_doc_id=r["document_id"],
            content=r["text"],
            chunk_index=r["ordinal"],
            total_chunks=0,  # not stored in §8 (derive from a count if ever needed)
            metadata=self._chunk_metadata(r),
        )

    @staticmethod
    def _doc_metadata(r) -> dict[str, Any]:
        """Reconstruct a metadata dict from §8 document columns (for the Document DTO)."""
        return {
            "source_id": r["document_id"],
            "title": r["title"],
            "source_kind": r["source_kind"],
            "standard_id": r["standard_id"],
            "doc_version": r["doc_version"],
            "license_class": r["license_class"],
            "content_hash": r["content_hash"],
        }

    @staticmethod
    def _chunk_metadata(r) -> dict[str, Any]:
        """Reconstruct a metadata dict from §8 chunk columns (for the Chunk DTO)."""
        return {
            "source_id": r["document_id"],
            "locator": r["locator"],
            "license_class": r["license_class"],
            "ai_grounding_allowed": r["ai_grounding_allowed"],
            "available": r["available"],
            "content_hash": r["content_hash"],
        }
