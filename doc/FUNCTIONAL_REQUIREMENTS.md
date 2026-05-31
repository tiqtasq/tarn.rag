# RAG Ingestion Pipeline - Comprehensive Software Specification

**For Implementation with Claude Code**

---

## Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Project Structure](#project-structure)
3. [Core Models](#core-models)
4. [Database & Repository Layer](#database--repository-layer)
5. [Pipeline & Stages](#pipeline--stages)
6. [Orchestration & Workers](#orchestration--workers)
7. [Services (Facade Pattern)](#services-facade-pattern)
8. [API Layer](#api-layer)
9. [Configuration](#configuration)
10. [Observability](#observability)
11. [Execution Flow](#execution-flow)
12. [Implementation Checklist](#implementation-checklist)

---

## Overview & Architecture

### Purpose
Transform raw documents into queryable vector embeddings via a composable, DAG-based ingestion pipeline. Designed for parallel execution, database agnosticism, and future extensibility (NLP preprocessing, retrieval).

### Key Design Principles

1. **Separation of Concerns**: Pipeline logic, storage, orchestration, and observability are independent layers.
2. **Composability**: Stages are independent; new stages are added without modifying existing code.
3. **Parallelism**: Workers pull jobs from a queue (pgQueuer) and process independently.
4. **Database Agnosticism**: Storage abstracted via repository; supports PostgreSQL and SQLite seamlessly.
5. **Extensibility**: Metadata schema and stage types accommodate future NLP preprocessing without refactoring.
6. **Optional Observability**: Instrumentation interfaces are optional; core logic doesn't depend on them.

### High-Level Data Flow

```
Data Sources (files, APIs, DBs)
    ↓
API: POST /v1/ingest
    ↓
IngestionService (high-level facade)
    ↓
PipelineOrchestrator (walks DAG, enqueues jobs, owns lifecycle)
    ↓
JobQueue port → pgQueuer (distributed job queue)
    ↓
IngestionWorker(s) (registered handlers; compute only, parallel)
    ↓
PipelineStages (load → clean → chunk → enrich → embed)
    ↓
DocumentRepository (stores documents, chunks, vectors)
    ↓
PostgreSQL/SQLite (persistent storage)
    ↓
Online Retrieval (future pipeline, same repository)
```

---

## Resolved Architecture Decisions (Authoritative)

> These decisions were resolved during design review and are **authoritative**.
> Where illustrative code in later sections still reflects an earlier model, these
> decisions win. Affected sections carry a pointer back here (D1–D6).

### D1 — Execution model: distributed, one job per (item, stage)

A pgQueuer job is a single `(PipelineItem, stage)` pair. Stages that fan out
enqueue **one downstream job per output item**: a chunker turning a document into
*m* chunks produces *m* embed-stage jobs, not one. The in-flight `PipelineItem`
travels **inline in the job payload** (content + metadata); there is no separate
temporary item store.

### D2 — Job granularity & dispatch

Each chunk is its own job; **pgQueuer** claims and dispatches them and owns the queue
mechanics (SKIP LOCKED, retries, concurrency, `LISTEN/NOTIFY`). pgQueuer hands the
handler one job per call by default, or a list under batch dispatch — which is what
enables cross-document model/persistence batching (a list of items → `process_batch`).
The compute batch size stays independent of any persistence batch size (see D4).

### D3 — Three-layer responsibility split

| Layer | Owns | Must NOT do |
|---|---|---|
| **Worker** | Running the pure stage transform; grouping work into model/compute calls "as it sees fit"; handing results to a `ResultSink`; signalling completion | persist, ack jobs, enqueue downstream, retry |
| **ResultSink** | Batching + persisting results; cleanup; reporting the persistence outcome | compute, recovery policy |
| **Orchestrator** | Walking the DAG; finalizing the sink; recording status; enqueuing downstream jobs; raising on failure so pgQueuer requeues | compute, persistence mechanics, the queue itself |

A worker's job is "done" (from the worker's view) once it has **produced all
results → handed them off via `submit()` → called `close()`**. Whether the bytes
reached the DB is the orchestrator's concern, reconciled via `finalize()`.

### D4 — The `ResultSink` interface (general — used by every stage)

Every stage's worker receives a `ResultSink` instance and pushes to it. The sink
owns persistence and decides whether/how to batch writes. Its **persistence batch
size is independent of the worker's compute batch size** (two-tier batching).

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class FinalizationOutcome:
    """Result of orchestrator-driven finalization."""
    persisted: bool
    detail: str | None = None   # e.g. which results were/weren't written on failure

class ResultSink(ABC):
    """Receives results a worker produced and owns their persistence.
    Per-stage implementations differ in WHAT they persist (Document after Load,
    Chunks after Chunk, Embeddings after Embed) behind this single interface."""

    @abstractmethod
    def submit(self, results: list) -> None:
        """Worker hands off a batch of produced results. Synchronous: BUFFER only,
        no I/O — persistence happens in finalize(). Called by the WORKER, possibly
        many times per job."""

    @abstractmethod
    def close(self) -> None:
        """Worker signals it has produced everything. Synchronous; no outcome, nothing
        persisted yet. Called by the WORKER only."""

    @abstractmethod
    async def finalize(self) -> FinalizationOutcome:
        """Async: durably persist all submitted results, then report the outcome.
        Called by the ORCHESTRATOR only — never the worker."""
```

This **supersedes the earlier input-side `BatchingWrapper`** (which buffered
*inputs* and fed them to a wrapped stage). `ResultSink` sits on the **output**
side: the worker computes and streams results in; the sink batches the writes out.

### D5 — Lifecycle & recovery

1. The orchestrator enqueues root `(item, stage)` jobs via pgQueuer.
2. pgQueuer dispatches a job (or batch) to the worker handler, which runs the pure
   stage, `submit()`s results to the sink, and `close()`s.
3. The orchestrator's `complete_batch` calls `sink.finalize()`:
   - **success** → record `completed` in the status projection **and** enqueue the
     downstream-stage jobs (one per produced item, per D1). pgQueuer acks the job
     when the handler returns.
   - **failure** (results created but not persisted) → record `failed` and **raise**;
     pgQueuer requeues the job (= recovery) and dead-letters it past its retry limit.
4. Job *dependencies* are implicit: a downstream job is enqueued only after its
   upstream persisted, so ordering holds by construction (no explicit gating).

### D6 — Purity & supporting tables

Stages stay **pure** (no DB, no queue access). Persistence is the sink's job, driven
by the orchestrator. **pgQueuer owns the queue tables**; for the document-status API
we keep a small, document-keyed **`job_status` projection** in the repository (a
read-model, not a queue).

---

## Project Structure

```
rag-ingestion/
│
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── endpoints/
│   │       │   ├── __init__.py
│   │       │   └── ingestion.py                 # POST /v1/ingest, GET /v1/jobs/{id}
│   │       ├── schemas.py                       # Pydantic request/response models
│   │       └── dependencies.py                  # FastAPI dependency injection
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                            # Settings, environment variables
│   │   ├── exceptions.py                        # Custom exception classes
│   │   └── observability.py                     # Observability interface & adapters
│   │
│   ├── domains/
│   │   │
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   ├── models.py                        # Shared data models (Document, Chunk, etc.)
│   │   │   ├── repository.py                    # DocumentRepository interface (ABC)
│   │   │   ├── postgres_repository.py           # PostgreSQL implementation
│   │   │   └── sqlite_repository.py             # SQLite implementation
│   │   │
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── models.py                        # IngestionJob model
│   │   │   ├── pipeline.py                      # PipelineStage classes, Pipeline (BatchingWrapper superseded, see D4)
│   │   │   ├── result_sink.py                   # ResultSink interface + per-stage sinks (D4)
│   │   │   ├── stages/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── load_parse.py                # LoadAndParseStage
│   │   │   │   ├── clean_normalize.py           # CleanAndNormalizeStage
│   │   │   │   ├── chunk.py                     # ChunkStage
│   │   │   │   ├── enrich.py                    # EnrichMetadataStage
│   │   │   │   └── embed.py                     # EmbedStage (simple 1-to-1, no batching)
│   │   │   ├── orchestrator.py                  # PipelineOrchestrator, PipelineDAG
│   │   │   ├── queue.py                         # JobQueue port + PgQueuer/InMemory adapters (D2/D5)
│   │   │   ├── worker.py                        # IngestionWorker (compute only; pgQueuer handler)
│   │   │   └── service.py                       # IngestionService (high-level facade)
│   │   │
│   │   └── retrieval/
│   │       ├── __init__.py
│   │       ├── models.py
│   │       ├── retriever.py                     # Future: retrieval logic
│   │       └── service.py                       # Future: RetrievalService
│   │
│   ├── main.py                                  # FastAPI app definition
│   └── __init__.py
│
├── tests/
│   ├── conftest.py                              # Shared pytest fixtures
│   │
│   ├── unit/
│   │   ├── core/
│   │   │   ├── test_config.py
│   │   │   └── test_observability.py
│   │   │
│   │   ├── domains/
│   │   │   ├── base/
│   │   │   │   ├── test_models.py
│   │   │   │   └── test_repository.py
│   │   │   │
│   │   │   └── ingestion/
│   │   │       ├── test_pipeline.py
│   │   │       ├── test_stages/
│   │   │       │   ├── test_chunk.py
│   │   │       │   ├── test_embed.py
│   │   │       │   └── test_clean_normalize.py
│   │   │       ├── test_orchestrator.py
│   │   │       ├── test_worker.py
│   │   │       └── test_service.py
│   │   │
│   │   └── api/
│   │       └── test_ingestion_endpoints.py
│   │
│   ├── integration/
│   │   ├── test_ingestion_e2e.py                # Full pipeline test
│   │   └── test_repository_transactional.py
│   │
│   └── fixtures/
│       ├── sample_documents/
│       │   ├── sample_1.txt
│       │   └── sample_2.pdf
│       └── mock_data.py
│
├── doc/
│   ├── FUNCTIONAL_REQUIREMENTS.md               # What the system does
│   ├── ARCHITECTURE.md                          # System design decisions
│   ├── API_SPEC.md                              # API endpoints (future)
│   └── DATABASE_SCHEMA.md                       # Full DB schema diagrams
│
├── docker/
│   ├── Dockerfile                               # FastAPI app container
│   ├── Dockerfile.worker                        # Ingestion worker container (optional)
│   └── docker-compose.yml                       # Local dev setup
│
├── scripts/
│   ├── init_db.sh                               # Initialize databases (create tables)
│   └── run_worker.py                            # CLI to run ingestion workers
│
├── .env.example                                 # Example environment variables
├── requirements.txt                             # Python dependencies
├── pyproject.toml                               # Project metadata
├── pytest.ini                                   # Pytest configuration
└── README.md
```

### Folder Rationale

- **`app/`** — All application code; mirrors typical FastAPI structure.
- **`app/core/`** — Infrastructure: config, exceptions, observability. No business logic.
- **`app/domains/base/`** — Shared by ingestion and retrieval. Repository interfaces, data models.
- **`app/domains/ingestion/`** — Ingestion-specific: stages, orchestrator, worker, service.
- **`app/domains/retrieval/`** — Retrieval-specific (future). Will reuse `base.repository`.
- **`tests/`** — Mirrors `app/` structure for parallel test organization.

---

## Core Models

### 1. PipelineItem (`app/domains/base/models.py`)

The fundamental unit flowing through the ingestion pipeline.

```python
from typing import Any
from pydantic import BaseModel, Field, ConfigDict

class PipelineItem(BaseModel):
    """A document or chunk flowing through the ingestion pipeline."""
    
    id: str | None = None  # Assigned by storage layer
    content: str              # The actual text
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Metadata conventions (recommended but flexible):
# {
#   "source_id": str,             # Upstream data source ID
#   "source_type": str,           # "file", "api", "database", etc.
#   "source_path": str,           # Original file path or URL
#   "doc_id": str,                # Parent document ID (set by load stage)
#   "chunk_index": int,           # Position within parent document
#   "total_chunks": int,          # Total chunks from same document
#   "chunk_size": int,            # Bytes in this chunk
#   "created_at": str,            # ISO timestamp
#   "nlp_entities": list[Dict],   # NER results (future NLP stage)
#   "nlp_noun_phrases": list[str],# Extracted noun phrases (future)
#   "summary": str,               # Document-level summary (optional)
#   "[stage_name]_applied": bool, # Mark which stages have processed this item
# }
```

### 2. Domain Models (`app/domains/base/models.py`)

Pydantic **domain/transfer objects** at the repository boundary — in-memory, not the
stored rows. The persistent schema is the SQLAlchemy `Table`s in `DocumentRepository`,
which maps each model to/from its table (`Document` ↔ `documents`, etc.).

```python
class Document(BaseModel):
    """A source document. Mapped to/from the ``documents`` table by the repository."""
    
    id: str | None = None
    content: str
    metadata: dict[str, Any]  # source_id, source_type, source_path, created_at, etc.
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class Chunk(BaseModel):
    """A text chunk extracted from a document. Mapped to/from the ``chunks`` table."""
    
    id: str | None = None
    parent_doc_id: str          # Foreign key to Document.id
    content: str
    chunk_index: int            # Position within parent
    total_chunks: int           # Total chunks from same parent
    metadata: dict[str, Any]    # chunk_size, nlp_entities, nlp_noun_phrases, etc.
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class Embedding(BaseModel):
    """A chunk's dense vector. Mapped to/from the ``embeddings`` table."""
    
    id: str | None = None
    chunk_id: str               # Foreign key to Chunk.id
    vector: list                # e.g., list[float] or numpy array
    model: str                  # e.g., "sentence-transformers/all-minilm-l6-v2"
    dimension: int              # Vector dimension (e.g., 384)
    metadata: dict[str, Any]    # generation_time, etc.
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
```

### 3. Ingestion Job Model (`app/domains/ingestion/models.py`)

Unit of work in the job queue (pgQueuer).

```python
from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict
from app.domains.base.models import PipelineItem

class IngestionJob(BaseModel):
    """Internal unit of work enqueued in pgQueuer — never exposed to API clients.

    Runtime state (queued/processing/…/retries) is owned by pgQueuer; for the status
    API a document-keyed `job_status` projection is kept in the repository.
    """

    job_id: str                                     # logical id (used by the status projection)
    document_id: str                                # public document handle (== source_id)
    item: PipelineItem                              # in-flight item, INLINE in the payload (D1)
    stage_name: str                                 # stage to execute
    stage_config: dict[str, Any]                    # stage config (to reconstruct the stage)
    created_at: datetime

    model_config = ConfigDict(arbitrary_types_allowed=True)
```

### 4. Exceptions (`app/core/exceptions.py`)

```python
class IngestionError(Exception):
    """A stage's results could not be produced or persisted. The worker propagates
    it so the queue requeues the job (recovery, D5)."""

class DocumentStorageError(IngestionError):
    """A document / chunk / embedding write failed."""

class ChunkNotFoundError(IngestionError):
    """update_chunk_metadata targeted a chunk id that does not exist."""
```

---

## Database & Repository Layer

### 1. Repository base class (`app/domains/base/repository.py`)

SQLAlchemy 2.0 **Core** (async). Shared by ingestion and retrieval. All
dialect-agnostic table definitions and CRUD live in this base class; subclasses
supply only the Postgres/SQLite specifics via a small set of hooks.

```python
from abc import ABC, abstractmethod
from typing import Any
import uuid
from sqlalchemy import (
    MetaData, Table, Column, Text, Integer, TIMESTAMP, ForeignKey,
    JSON, Index, select, update, insert, func, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncConnection
from app.domains.base.models import Document, Chunk, Embedding
from app.core.exceptions import ChunkNotFoundError


class DocumentRepository(ABC):
    """SQLAlchemy Core repository shared by ingestion and retrieval.

    Subclasses supply only the dialect specifics: the async driver URL, the vector
    column type + value encoding, the upsert, dialect-only schema objects (pgvector
    extension/index, expression indexes), and the vector-search query. Everything
    else — table definitions, lifecycle, and the rest of the CRUD — is shared here.

    Guarantees:
    - Atomicity: multi-row writes (store_document_with_chunks / store_chunks /
      store_embeddings) run in one transaction (engine.begin) — all or nothing.
    - Idempotency: documents are keyed by metadata['source_id'] (UNIQUE). Re-ingesting
      a source UPSERTS the document and REPLACES its chunks/embeddings (cascade); never
      duplicates.
    """

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
    async def _upsert_document(self, conn: AsyncConnection, values: dict) -> str:
        """Dialect upsert on metadata['source_id']; returns the (existing) document id."""

    @abstractmethod
    async def _create_dialect_objects(self, conn: AsyncConnection) -> None:
        """Create dialect-only schema objects (extension, expression/vector indexes)."""

    @abstractmethod
    async def vector_search(self, vector: list[float], k: int = 10,
                            model: str | None = None,
                            filters: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]:
        """Semantic search: pgvector cosine on Postgres, in-memory cosine on SQLite."""

    # ---------------- shared schema ----------------

    def _define_tables(self) -> None:
        json_type = JSON().with_variant(JSONB(), "postgresql")
        self.documents = Table(
            "documents", self.metadata,
            Column("id", Text, primary_key=True),
            Column("content", Text, nullable=False),
            Column("metadata", json_type, nullable=False, default=dict),
            Column("created_at", TIMESTAMP, server_default=func.now()),
        )
        self.chunks = Table(
            "chunks", self.metadata,
            Column("id", Text, primary_key=True),
            Column("parent_doc_id", Text,
                   ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
            Column("content", Text, nullable=False),
            Column("chunk_index", Integer, nullable=False),
            Column("total_chunks", Integer, nullable=False),
            Column("metadata", json_type, nullable=False, default=dict),
            Column("created_at", TIMESTAMP, server_default=func.now()),
            Index("idx_chunks_parent", "parent_doc_id"),
        )
        self.embeddings = Table(
            "embeddings", self.metadata,
            Column("id", Text, primary_key=True),
            Column("chunk_id", Text,
                   ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
            Column("vector", self._vector_type(), nullable=False),
            Column("model", Text, nullable=False),
            Column("dimension", Integer, nullable=False),
            Column("metadata", json_type, nullable=False, default=dict),
            Index("idx_embeddings_chunk", "chunk_id"),
        )
        # Document-keyed status PROJECTION for the API (pgQueuer owns the real queue).
        self.job_status = Table(
            "job_status", self.metadata,
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
            await self._create_dialect_objects(conn)        # extension first on PG
            await conn.run_sync(self.metadata.create_all)

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
        async with self.engine.begin() as conn:
            return await self._upsert_document(conn, self._doc_values(doc))

    async def store_document_with_chunks(self, doc: Document,
                                         chunks: list[Chunk]) -> tuple[str, list[str]]:
        async with self.engine.begin() as conn:
            doc_id = await self._upsert_document(conn, self._doc_values(doc))
            # Re-ingest replaces chunks (ON DELETE CASCADE removes their embeddings).
            await conn.execute(
                self.chunks.delete().where(self.chunks.c.parent_doc_id == doc_id))
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
            rows.append({"id": eid, "chunk_id": emb.chunk_id,
                         "vector": self._encode_vector(emb.vector), "model": emb.model,
                         "dimension": emb.dimension, "metadata": emb.metadata})
        async with self.engine.begin() as conn:
            if rows:
                await conn.execute(insert(self.embeddings), rows)
        return ids

    async def update_chunk_metadata(self, chunk_id: str, updates: dict[str, Any]) -> None:
        async with self.engine.begin() as conn:
            row = (await conn.execute(
                select(self.chunks.c.metadata).where(self.chunks.c.id == chunk_id))).first()
            if row is None:
                raise ChunkNotFoundError(chunk_id)
            merged = {**(row[0] or {}), **updates}
            await conn.execute(update(self.chunks)
                               .where(self.chunks.c.id == chunk_id).values(metadata=merged))

    # ---------------- reads (shared Core) ----------------

    async def get_document(self, doc_id: str) -> Document | None:
        async with self.engine.connect() as conn:
            row = (await conn.execute(
                select(self.documents).where(self.documents.c.id == doc_id))).mappings().first()
        return Document(**self._pick(row, Document)) if row else None

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        async with self.engine.connect() as conn:
            row = (await conn.execute(
                select(self.chunks).where(self.chunks.c.id == chunk_id))).mappings().first()
        return self._row_to_chunk(row) if row else None

    async def get_chunks_by_document(self, doc_id: str) -> list[Chunk]:
        async with self.engine.connect() as conn:
            rows = (await conn.execute(
                select(self.chunks).where(self.chunks.c.parent_doc_id == doc_id)
                .order_by(self.chunks.c.chunk_index))).mappings().all()
        return [self._row_to_chunk(r) for r in rows]

    async def query_chunks(self, filters: dict[str, Any], limit: int = 100) -> list[Chunk]:
        # Common skeleton; metadata predicates (source_id, created_at, has_key) are
        # applied with the dialect's JSON accessor — override in a subclass if needed.
        async with self.engine.connect() as conn:
            rows = (await conn.execute(select(self.chunks).limit(limit))).mappings().all()
        return [self._row_to_chunk(r) for r in rows]

    # ---------------- shared helpers ----------------

    def _doc_values(self, doc: Document) -> dict:
        return {"id": doc.id or str(uuid.uuid4()), "content": doc.content,
                "metadata": doc.metadata}

    async def _insert_chunks(self, conn: AsyncConnection, parent_doc_id: str | None,
                             chunks: list[Chunk]) -> list[str]:
        rows, ids = [], []
        for ch in chunks:
            cid = ch.id or str(uuid.uuid4())
            ids.append(cid)
            rows.append({"id": cid, "parent_doc_id": parent_doc_id or ch.parent_doc_id,
                         "content": ch.content, "chunk_index": ch.chunk_index,
                         "total_chunks": ch.total_chunks, "metadata": ch.metadata})
        if rows:
            await conn.execute(insert(self.chunks), rows)
        return ids

    def _row_to_chunk(self, r) -> Chunk:
        return Chunk(id=r["id"], parent_doc_id=r["parent_doc_id"], content=r["content"],
                     chunk_index=r["chunk_index"], total_chunks=r["total_chunks"],
                     metadata=r["metadata"])

    def _pick(self, r, model) -> dict:
        return {k: r[k] for k in model.model_fields if k in r}

    # ---------------- job-status projection (for the document-status API) ----------------

    async def record_job(self, document_id: str, job_id: str, stage_name: str,
                         status: str, error: str | None = None) -> None:
        """Upsert a job's status row (portable: update, else insert)."""
        async with self.engine.begin() as conn:
            res = await conn.execute(update(self.job_status)
                .where(self.job_status.c.job_id == job_id)
                .values(status=status, error=error, updated_at=func.now()))
            if res.rowcount == 0:
                await conn.execute(insert(self.job_status).values(
                    job_id=job_id, document_id=document_id, stage_name=stage_name,
                    status=status, error=error))

    async def document_jobs(self, document_id: str) -> list[dict[str, Any]]:
        """Debug-only per-job breakdown for one document."""
        async with self.engine.connect() as conn:
            rows = (await conn.execute(select(
                self.job_status.c.job_id, self.job_status.c.stage_name,
                self.job_status.c.status, self.job_status.c.error)
                .where(self.job_status.c.document_id == document_id))).mappings().all()
        return [dict(r) for r in rows]

    async def document_status(self, document_id: str) -> dict[str, Any] | None:
        """Document-level status derived from persisted data, rolled up to 'failed' if
        any of the document's jobs failed. None if the document_id is unknown."""
        src = self.documents.c.metadata["source_id"].as_string()
        async with self.engine.connect() as conn:
            doc = (await conn.execute(
                select(self.documents.c.id).where(src == document_id))).first()
            job_states = (await conn.execute(select(self.job_status.c.status)
                .where(self.job_status.c.document_id == document_id))).scalars().all()
            if doc is None and not job_states:
                return None
            doc_id = doc[0] if doc else None
            chunk_count = embedding_count = 0
            if doc_id is not None:
                chunk_count = (await conn.execute(select(func.count())
                    .select_from(self.chunks)
                    .where(self.chunks.c.parent_doc_id == doc_id))).scalar_one()
                embedding_count = (await conn.execute(select(func.count())
                    .select_from(self.embeddings)
                    .join(self.chunks, self.embeddings.c.chunk_id == self.chunks.c.id)
                    .where(self.chunks.c.parent_doc_id == doc_id))).scalar_one()
        if "failed" in job_states:
            status = "failed"
        elif doc_id is None:
            status = "pending"
        elif chunk_count and embedding_count >= chunk_count:
            status = "complete"
        else:
            status = "in_progress"
        return {"document_id": document_id, "status": status,
                "chunk_count": chunk_count, "embedding_count": embedding_count}
```

### 2. PostgreSQL adapter (`app/domains/base/postgres_repository.py`)

asyncpg driver + pgvector. Real vector search via the `<=>` cosine-distance
operator; upsert via `ON CONFLICT`.

```python
from typing import Any
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection
from pgvector.sqlalchemy import Vector
from app.domains.base.repository import DocumentRepository
from app.domains.base.models import Chunk


class PostgresRepository(DocumentRepository):
    """PostgreSQL adapter (SQLAlchemy Core + asyncpg + pgvector)."""

    def _driver_url(self, url: str) -> str:
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)

    def _vector_type(self):
        return Vector(self.embedding_dimension)

    def _encode_vector(self, vector: list[float]):
        return vector                      # pgvector accepts a Python list directly

    def _decode_vector(self, stored) -> list[float]:
        return list(stored)

    async def _upsert_document(self, conn: AsyncConnection, values: dict) -> str:
        stmt = pg_insert(self.documents).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[self.documents.c.metadata["source_id"].astext],
            set_={"content": stmt.excluded.content, "metadata": stmt.excluded.metadata},
        ).returning(self.documents.c.id)
        return (await conn.execute(stmt)).scalar_one()

    async def _create_dialect_objects(self, conn: AsyncConnection) -> None:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_id "
            "ON documents ((metadata->>'source_id'))"))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_vector "
            "ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 100)"))

    async def vector_search(self, vector: list[float], k: int = 10,
                            model: str | None = None,
                            filters: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]:
        dist = self.embeddings.c.vector.cosine_distance(vector)
        stmt = (select(self.chunks, (1 - dist).label("similarity"))
                .join(self.embeddings, self.embeddings.c.chunk_id == self.chunks.c.id)
                .order_by(dist).limit(k))
        if model:
            stmt = stmt.where(self.embeddings.c.model == model)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        return [(self._row_to_chunk(r), r["similarity"]) for r in rows]
```

### 3. SQLite adapter (`app/domains/base/sqlite_repository.py`)

aiosqlite driver. Vectors are stored as JSON text and searched with in-memory
cosine similarity (dev / small-scale only).

```python
from typing import Any
import json
import numpy as np
from sqlalchemy import select, text, func, Text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncConnection
from app.domains.base.repository import DocumentRepository
from app.domains.base.models import Chunk


class SqliteRepository(DocumentRepository):
    """SQLite adapter (SQLAlchemy Core + aiosqlite); vectors as JSON, in-memory search."""

    def _driver_url(self, url: str) -> str:
        path = url.replace("sqlite:///", "").replace("sqlite://", "")
        return f"sqlite+aiosqlite:///{path}"

    def _vector_type(self):
        return Text

    def _encode_vector(self, vector: list[float]):
        return json.dumps(list(vector))

    def _decode_vector(self, stored) -> list[float]:
        return json.loads(stored)

    async def _upsert_document(self, conn: AsyncConnection, values: dict) -> str:
        src = func.json_extract(self.documents.c.metadata, "$.source_id")
        stmt = sqlite_insert(self.documents).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[src],
            set_={"content": stmt.excluded.content, "metadata": stmt.excluded.metadata},
        ).returning(self.documents.c.id)
        return (await conn.execute(stmt)).scalar_one()  # SQLite 3.35+ RETURNING

    async def _create_dialect_objects(self, conn: AsyncConnection) -> None:
        await conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_id "
            "ON documents (json_extract(metadata, '$.source_id'))"))

    async def vector_search(self, vector: list[float], k: int = 10,
                            model: str | None = None,
                            filters: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]:
        stmt = (select(self.chunks, self.embeddings.c.vector)
                .join(self.embeddings, self.embeddings.c.chunk_id == self.chunks.c.id))
        if model:
            stmt = stmt.where(self.embeddings.c.model == model)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        q = np.asarray(vector, dtype=float)
        scored = []
        for r in rows:
            v = np.asarray(self._decode_vector(r["vector"]), dtype=float)
            sim = float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v)))
            scored.append((self._row_to_chunk(r), sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]
```

### 4. Database Schema

The document/chunk/embedding tables are created from the SQLAlchemy `MetaData` in
the repository base (`metadata.create_all`), and the dialect-only objects (pgvector
extension, the `vector_cosine_ops` ivfflat index, and the `source_id` expression
unique index) by each adapter's `_create_dialect_objects`. The DDL below is the
**equivalent reference** (also handy for `scripts/init_db*.sql`). The `job_status`
table is the API's read-model; pgQueuer owns the real queue tables (its own migrations).

**PostgreSQL** (`scripts/init_db.sql`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- UNIQUE on source_id makes re-ingestion idempotent (upsert via ON CONFLICT)
CREATE UNIQUE INDEX idx_documents_source_id ON documents ((metadata->>'source_id'));

CREATE TABLE chunks (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    parent_doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_parent ON chunks(parent_doc_id);
CREATE INDEX idx_chunks_source ON chunks USING GIN ((metadata->'source_id'));

CREATE TABLE embeddings (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    vector vector(384),  -- dimension MUST equal settings.EMBEDDING_DIMENSION (templated by _init_schema)
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embeddings_chunk ON embeddings(chunk_id);
CREATE INDEX idx_embeddings_vector ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);

-- Document-status PROJECTION for the API. pgQueuer owns the real queue tables
-- (installed via pgQueuer's own migrations); this is NOT the queue.
CREATE TABLE job_status (
    job_id      TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,                    -- source_id; the public document handle
    stage_name  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',   -- queued | processing | completed | failed
    error       TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_job_status_document ON job_status (document_id);
```

**SQLite** (`scripts/init_db_sqlite.sql`)

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_created ON documents(created_at);
-- UNIQUE on source_id makes re-ingestion idempotent (upsert / INSERT OR REPLACE)
CREATE UNIQUE INDEX idx_documents_source_id ON documents (json_extract(metadata, '$.source_id'));

CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    parent_doc_id TEXT NOT NULL REFERENCES documents(id),
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_parent ON chunks(parent_doc_id);

CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL REFERENCES chunks(id),
    vector TEXT NOT NULL,  -- JSON-encoded list
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embeddings_chunk ON embeddings(chunk_id);

-- Document-status PROJECTION for the API (pgQueuer owns the real queue).
CREATE TABLE job_status (
    job_id      TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,                    -- source_id; the public document handle
    stage_name  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',   -- queued | processing | completed | failed
    error       TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_job_status_document ON job_status (document_id);
```

---

## Pipeline & Stages

### 1. Base Classes (`app/domains/ingestion/pipeline.py`)

```python
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any
from app.domains.base.models import PipelineItem

class PipelineStage(ABC):
    """
    Base class for any custom transformation stage.
    Subclass this for stages that don't fit standard patterns.
    """
    
    def __init__(self, name: str, **config):
        self.name = name
        self.config = config
        self.validate()
    
    @abstractmethod
    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        """Transform one or more items. Yields zero or more results."""
        pass
    
    def process_batch(self, items: list[PipelineItem]) -> Iterator[Any]:
        """Batch entrypoint the worker calls (D3). Default: map process() over the
        items one at a time. Stages that benefit from batching (e.g. EmbedStage)
        override this to size their own compute/model calls. Yields stage-typed
        results: PipelineItem for transform stages, Embedding for the embed stage."""
        for item in items:
            yield from self.process(item)
    
    @abstractmethod
    def validate(self) -> None:
        """Validate configuration at initialization. Raise if invalid."""
        pass
    
    def __call__(self, item: PipelineItem) -> Iterator[PipelineItem]:
        """Allow stages to be callable."""
        yield from self.process(item)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class MapperStage(PipelineStage):
    """
    Transforms exactly 1 item → 1 item (1-to-1).
    Examples: CleanAndNormalize, EnrichMetadata.
    
    Implementers override map(); metadata handling is automatic.
    """
    
    @abstractmethod
    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Transform text and optionally update metadata.
        
        Args:
            text: The document content
            metadata: Upstream metadata (don't mutate; return updates)
            
        Returns:
            (transformed_text, metadata_updates)
            where updates will be merged with upstream metadata.
        """
        pass
    
    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        """Execute map and yield the result."""
        new_text, updates = self.map(item.content, item.metadata)
        yield PipelineItem(
            content=new_text,
            metadata={**item.metadata, **updates}
        )


class ChunkerStage(PipelineStage):
    """
    Splits exactly 1 item → N items (1-to-N, N ≥ 1).
    Example: ChunkStage splits documents into chunks.
    
    Implementers override chunk(); metadata handling is automatic.
    """
    
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """
        Split text into chunks.
        
        Args:
            text: The document content
            metadata: Upstream metadata
            
        Returns:
            List of (chunk_text, chunk_metadata_updates) tuples.
            Must return at least one chunk.
        """
        pass
    
    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        """Execute chunk and yield results."""
        chunks = self.chunk(item.content, item.metadata)
        
        if not chunks:
            raise ValueError(f"{self.name} produced no chunks for {item.metadata.get('doc_id', 'unknown')}")
        
        for chunk_idx, (chunk_text, updates) in enumerate(chunks):
            yield PipelineItem(
                content=chunk_text,
                metadata={
                    **item.metadata,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    **updates,
                }
            )


class FilterStage(PipelineStage):
    """
    Optionally keeps or discards items (1 → {0, 1}).
    Example: DeduplicatorStage removes near-duplicates.
    """
    
    @abstractmethod
    def should_keep(self, text: str, metadata: dict[str, Any]) -> bool:
        """Return True to keep, False to discard."""
        pass
    
    def maybe_transform(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Optionally transform before keeping. Default: no transformation."""
        return text, {}
    
    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        """Filter and optionally transform."""
        if self.should_keep(item.content, item.metadata):
            new_text, updates = self.maybe_transform(item.content, item.metadata)
            yield PipelineItem(
                content=new_text,
                metadata={**item.metadata, **updates}
            )


# NOTE: SUPERSEDED — see "Resolved Architecture Decisions" (D4). The class below
# is an INPUT-side buffer that feeds a wrapped stage. The resolved design instead
# uses an OUTPUT-side `ResultSink` (D4): the worker computes results and pushes
# them to the sink, which owns persistence batching. Implement ResultSink, not
# this. Retained here only for historical context.
class BatchingWrapper(PipelineStage):
    """
    Buffers items and yields them in batches to the wrapped stage.
    Used for stages that benefit from batch processing (e.g., embedding).
    
    NOT a typical PipelineStage; handles its own buffering and flushing.
    Call flush() explicitly when done to process remaining items.
    """
    
    def __init__(self, wrapped_stage: PipelineStage, batch_size: int = 32):
        self.wrapped_stage = wrapped_stage
        self.batch_size = batch_size
        self.buffer: list[PipelineItem] = []
        self.name = f"Batching({wrapped_stage.name})"
    
    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        """Buffer item. Yields results when batch is full."""
        self.buffer.append(item)
        if len(self.buffer) >= self.batch_size:
            yield from self._flush_buffer()
    
    def _flush_buffer(self) -> Iterator[PipelineItem]:
        """Process all buffered items through wrapped stage."""
        for item in self.buffer:
            yield from self.wrapped_stage.process(item)
        self.buffer.clear()
    
    def flush(self) -> Iterator[PipelineItem]:
        """
        Explicitly flush remaining items. Call this when done with the batch wrapper
        to ensure all buffered items are processed.
        """
        yield from self._flush_buffer()
    
    def validate(self) -> None:
        """Validate wrapped stage."""
        self.wrapped_stage.validate()


class Pipeline:
    """
    Chains stages sequentially.
    Each item flows through all stages in order.
    If a stage yields N items, each is fed to the next stage independently.
    """
    
    def __init__(self, stages: list[PipelineStage]):
        if not stages:
            raise ValueError("Pipeline must have at least one stage")
        self.stages = stages
    
    def run(self, items: Iterator[PipelineItem]) -> Iterator[PipelineItem]:
        """
        Thread items through the entire pipeline.
        
        Args:
            items: Iterator of input items (e.g., loaded documents)
            
        Yields:
            Transformed items after passing through all stages
        """
        for item in items:
            current = [item]
            
            # Pass through each stage
            for stage in self.stages:
                next_items = []
                for current_item in current:
                    next_items.extend(stage.process(current_item))
                current = next_items
            
            # Yield all final items
            for final_item in current:
                yield final_item
    
    def __repr__(self) -> str:
        stages_str = "\n  ".join(repr(s) for s in self.stages)
        return f"Pipeline([\n  {stages_str}\n])"
```

### 2. Concrete Stages

**LoadAndParseStage** (`app/domains/ingestion/stages/load_parse.py`)

```python
from app.domains.ingestion.pipeline import MapperStage
from typing import Any
import uuid

class LoadAndParseStage(MapperStage):
    """Load documents from files and extract text."""
    
    def __init__(self, supported_types: list[str] = ["txt", "pdf", "html"], **config):
        super().__init__(name="LoadAndParse", supported_types=supported_types, **config)
        self.supported_types = supported_types
    
    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        If metadata['source_path'] is set, load and parse the file.
        Otherwise, treat text as already-loaded content.
        """
        source_path = metadata.get("source_path")
        
        if source_path:
            content = self._load_file(source_path)
        else:
            content = text
        
        # Assign a unique document ID
        doc_id = str(uuid.uuid4())
        
        return content, {
            "doc_id": doc_id,
            "loaded": True,
        }
    
    def _load_file(self, path: str) -> str:
        """Load and parse a file based on extension."""
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in ("txt", "text", "md"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        if ext == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        if ext in ("html", "htm"):
            import html2text
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return html2text.html2text(f.read())
        raise ValueError(f"Unsupported file type: {ext!r}")
    
    def validate(self) -> None:
        pass
```

**CleanAndNormalizeStage** (`app/domains/ingestion/stages/clean_normalize.py`)

```python
from app.domains.ingestion.pipeline import MapperStage
from typing import Any
import re

class CleanAndNormalizeStage(MapperStage):
    """Normalize whitespace and strip control characters before chunking."""

    def __init__(self, collapse_whitespace: bool = True, **config):
        super().__init__(name="CleanAndNormalize",
                         collapse_whitespace=collapse_whitespace, **config)
        self.collapse_whitespace = collapse_whitespace

    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        cleaned = text.replace("\x00", "")
        cleaned = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)  # drop C0 controls
        if self.collapse_whitespace:
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip(), {"cleaned": True}

    def validate(self) -> None:
        pass
```

**ChunkStage** (`app/domains/ingestion/stages/chunk.py`)

```python
from app.domains.ingestion.pipeline import ChunkerStage
from typing import Any

class ChunkStage(ChunkerStage):
    """Split documents into retrieval-sized chunks."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50, **config):
        super().__init__(name="Chunk", chunk_size=chunk_size, overlap=overlap, **config)
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """Split text using recursive chunking."""
        chunks = self._split_recursive(text)
        return [(chunk_text, {"chunk_size": len(chunk_text)}) for chunk_text in chunks]
    
    def _split_recursive(self, text: str, separators: list[str] = None) -> list[str]:
        """Recursively split on coarse→fine separators, then pack pieces into windows
        of chunk_size with `overlap` characters carried between adjacent chunks."""
        separators = ["\n\n", "\n", ". ", " ", ""] if separators is None else separators
        pieces = self._split_to_pieces(text, separators)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) > self.chunk_size:
                chunks.append(current)
                current = (current[-self.overlap:] if self.overlap else "") + piece
            else:
                current += piece
        if current.strip():
            chunks.append(current)
        return chunks

    def _split_to_pieces(self, text: str, separators: list[str]) -> list[str]:
        """Break text into pieces each <= chunk_size, recursing to finer separators
        when a piece is still too large."""
        if len(text) <= self.chunk_size or not separators:
            return [text]
        sep, rest = separators[0], separators[1:]
        parts = text.split(sep) if sep else list(text)
        pieces: list[str] = []
        for part in parts:
            unit = part + sep if sep else part
            if len(unit) <= self.chunk_size:
                pieces.append(unit)
            else:
                pieces.extend(self._split_to_pieces(unit, rest))
        return pieces
    
    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
```

**EnrichMetadataStage** (`app/domains/ingestion/stages/enrich.py`)

```python
from app.domains.ingestion.pipeline import MapperStage
from typing import Any

class EnrichMetadataStage(MapperStage):
    """Attach lightweight, content-derived metadata to each chunk. Richer NLP
    enrichment (NER, noun phrases) is a future extension — see the metadata
    conventions in Core Models."""

    def __init__(self, **config):
        super().__init__(name="EnrichMetadata", **config)

    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return text, {
            "char_count": len(text),
            "word_count": len(text.split()),
        }

    def validate(self) -> None:
        pass
```

**EmbedStage** (`app/domains/ingestion/stages/embed.py`)

```python
from app.domains.ingestion.pipeline import PipelineStage
from app.domains.base.models import PipelineItem, Embedding
from collections.abc import Iterator

class EmbedStage(PipelineStage):
    """
    Vectorize chunks into Embedding results — the embedding compute lives here (D3).
    process_batch() groups items into model_batch_size-sized model calls; the worker
    streams the Embeddings to an EmbeddingResultSink for batched persistence (D4).
    Terminal stage: yields Embeddings (not PipelineItems), so nothing runs downstream.
    """

    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-minilm-l6-v2",
                 model_batch_size: int = 32, **config):
        super().__init__(name="Embed", embedding_model=embedding_model,
                         model_batch_size=model_batch_size, **config)
        self.embedding_model = embedding_model
        self.model_batch_size = model_batch_size
        self._model = None  # lazy load

    def process(self, item: PipelineItem) -> Iterator[Embedding]:
        """Embed a single chunk (one model call)."""
        yield from self.process_batch([item])

    def process_batch(self, items: list[PipelineItem]) -> Iterator[Embedding]:
        """Embed items in model_batch_size-sized model calls (compute batching).
        chunk_id is threaded via metadata['chunk_id'] (set by ChunkResultSink, D4)."""
        model = self._get_model()
        for i in range(0, len(items), self.model_batch_size):
            sub = items[i:i + self.model_batch_size]
            vectors = model.encode([it.content for it in sub], convert_to_tensor=False)
            for it, vec in zip(sub, vectors):
                vector = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                yield Embedding(
                    chunk_id=it.metadata["chunk_id"],
                    vector=vector,
                    model=self.embedding_model,
                    dimension=len(vector),
                    metadata={"source_id": it.metadata.get("source_id")},
                )

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def validate(self) -> None:
        if self.model_batch_size <= 0:
            raise ValueError("model_batch_size must be positive")
```

---

## Result Sinks

Output-side sinks (D4). A worker `submit()`s produced results and `close()`s — both
**synchronous, buffer-only** — and the orchestrator later `await`s `finalize()` to
persist. One sink per stage; `PipelineOrchestrator.make_sink(stage_name)` builds them
from a registry.

**ID threading.** Storage ids are threaded forward via **metadata**, which stages
propagate (they merge metadata): `DocumentResultSink` writes `metadata['doc_id']`,
`ChunkResultSink` writes `metadata['chunk_id']`. Downstream sinks/stages read those,
so the FK chain (chunk→doc, embedding→chunk) resolves without relying on `item.id`
(stages create fresh items) or on the repository honoring caller-supplied ids.

### Interface (`app/domains/ingestion/result_sink.py`)

```python
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
from app.domains.base.models import Document, Chunk, Embedding
from app.domains.base.repository import DocumentRepository


@dataclass
class FinalizationOutcome:
    persisted: bool
    detail: str | None = None


class ResultSink(ABC):
    """submit()/close() are sync (buffer only); finalize() is async (persists)."""

    @abstractmethod
    def submit(self, results: list[Any]) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    async def finalize(self) -> FinalizationOutcome: ...


class _BufferingSink(ResultSink):
    """Shared buffering: submit() appends, close() marks done, finalize() persists the
    whole buffer via _persist() and reports the outcome. Subclasses implement
    _persist() for their storage target."""

    def __init__(self, repository: DocumentRepository):
        self.repo = repository
        self._buffer: list[Any] = []
        self._closed = False

    def submit(self, results: list[Any]) -> None:
        self._buffer.extend(results)

    def close(self) -> None:
        self._closed = True

    async def finalize(self) -> FinalizationOutcome:
        try:
            await self._persist(self._buffer)
            return FinalizationOutcome(persisted=True)
        except Exception as e:
            return FinalizationOutcome(persisted=False, detail=str(e))

    @abstractmethod
    async def _persist(self, results: list[Any]) -> None: ...
```

### Per-stage sinks (`app/domains/ingestion/result_sink.py`)

```python
class PassthroughSink(_BufferingSink):
    """Pure-transform stages with no storage target (e.g. CleanAndNormalize). Persists
    nothing; produced items still flow downstream via the orchestrator."""

    async def _persist(self, results: list[Any]) -> None:
        return None


class DocumentResultSink(_BufferingSink):
    """After LoadAndParse: persist Document rows (upsert on source_id) and thread the
    stored id forward as metadata['doc_id'] so chunks resolve parent_doc_id."""

    async def _persist(self, results: list[Any]) -> None:
        for item in results:
            doc_id = await self.repo.store_document(
                Document(content=item.content, metadata=item.metadata))
            item.metadata["doc_id"] = doc_id


class ChunkResultSink(_BufferingSink):
    """After Chunk: persist Chunk rows atomically and thread each stored id forward as
    metadata['chunk_id'] for the downstream Enrich/Embed stages."""

    async def _persist(self, results: list[Any]) -> None:
        if not results:
            return
        chunks = [
            Chunk(
                parent_doc_id=item.metadata["doc_id"],
                content=item.content,
                chunk_index=item.metadata["chunk_index"],
                total_chunks=item.metadata["total_chunks"],
                metadata=item.metadata,
            )
            for item in results
        ]
        chunk_ids = await self.repo.store_chunks(chunks)
        for item, chunk_id in zip(results, chunk_ids):
            item.metadata["chunk_id"] = chunk_id


class ChunkMetadataResultSink(_BufferingSink):
    """After EnrichMetadata: merge enrichment into the chunk via update_chunk_metadata."""

    async def _persist(self, results: list[Any]) -> None:
        for item in results:
            await self.repo.update_chunk_metadata(item.metadata["chunk_id"], item.metadata)


class EmbeddingResultSink(_BufferingSink):
    """After Embed: bulk-persist Embedding results in persistence-batch-sized writes
    (independent of the worker's compute batch, D4)."""

    def __init__(self, repository: DocumentRepository, persist_batch_size: int = 128):
        super().__init__(repository)
        self.persist_batch_size = persist_batch_size

    async def _persist(self, results: list[Any]) -> None:
        for i in range(0, len(results), self.persist_batch_size):
            await self.repo.store_embeddings(results[i:i + self.persist_batch_size])
```

### Registry (`config.py`, beside `create_ingestion_pipeline`)

```python

def create_sink_registry() -> dict[str, type[ResultSink]]:
    """Map each stage name to the ResultSink that persists its output. Keys MUST match
    stage .name values; PipelineOrchestrator.make_sink() looks them up."""
    return {
        "LoadAndParse": DocumentResultSink,
        "CleanAndNormalize": PassthroughSink,
        "Chunk": ChunkResultSink,
        "EnrichMetadata": ChunkMetadataResultSink,
        "Embed": EmbeddingResultSink,
    }
```

---

## Job Queue (`app/domains/ingestion/queue.py`)

pgQueuer owns the queue (claiming with `FOR UPDATE SKIP LOCKED`, `LISTEN/NOTIFY`,
retries, dead-lettering, concurrency) — we hand-roll none of that. The domain depends
only on a tiny **`JobQueue` port** (enqueue / set_handler / run); pgQueuer lives behind
one adapter. A second in-memory adapter lets the whole ingestion flow be tested with no
Postgres and no pgQueuer. Per-job runtime state is pgQueuer's; the document-status API
reads the small `job_status` projection in the repository.

> pgQueuer's exact API (imports, handler/enqueue signatures, batch dispatch) is confined
> to `PgQueuerJobQueue` and should be confirmed against the pinned pgQueuer version.

```python
from collections.abc import Awaitable, Callable
from typing import Protocol
import asyncio
import asyncpg
from pgqueuer import QueueManager
from pgqueuer.db import AsyncpgPoolDriver
from pgqueuer.models import Job
from pgqueuer.queries import Queries
from app.domains.ingestion.models import IngestionJob

ENTRYPOINT = "ingest"   # single pgQueuer entrypoint; the handler dispatches by stage_name

# A handler receives a BATCH of jobs (one job, or several under batch dispatch).
JobHandler = Callable[[list[IngestionJob]], Awaitable[None]]


class JobQueue(Protocol):
    """The minimal queue surface the domain uses. Mechanics (claiming, retries,
    NOTIFY, dead-lettering) belong to the implementation — never reimplement them."""

    async def enqueue(self, job: IngestionJob) -> None: ...
    def set_handler(self, handler: JobHandler) -> None: ...
    async def run(self) -> None: ...


class PgQueuerJobQueue:
    """JobQueue backed by pgQueuer — the ONLY module that imports pgQueuer. pgQueuer
    owns claiming/retries/NOTIFY/dead-lettering; a handler that raises propagates so
    pgQueuer requeues the job (= recovery, D5)."""

    def __init__(self, driver: AsyncpgPoolDriver):
        self._qm = QueueManager(driver)
        self._queries = Queries(driver)

    @classmethod
    async def connect(cls, connection_url: str) -> "PgQueuerJobQueue":
        return cls(AsyncpgPoolDriver(await asyncpg.create_pool(connection_url)))

    async def enqueue(self, job: IngestionJob) -> None:
        # payload is the full IngestionJob (incl. the inline item, D1)
        await self._queries.enqueue([ENTRYPOINT], [job.model_dump_json().encode()], [0])

    def set_handler(self, handler: JobHandler) -> None:
        @self._qm.entrypoint(ENTRYPOINT)
        async def _run(pg_job: Job) -> None:
            await handler([IngestionJob.model_validate_json(pg_job.payload)])

    async def run(self) -> None:
        await self._qm.run()


class InMemoryJobQueue:
    """In-process JobQueue for tests — no Postgres, no pgQueuer. Emulates pgQueuer's
    at-least-once + requeue-on-raise semantics so tests reflect reality.

    requeue_on_error=True (default) re-queues a failing job up to max_attempts
    (recovery tests); False re-raises immediately (sharp unit failures). pgQueuer does
    NOT guarantee ordering — tests must not rely on it."""

    def __init__(self, requeue_on_error: bool = True, max_attempts: int = 3):
        self._jobs: asyncio.Queue[tuple[IngestionJob, int]] = asyncio.Queue()
        self._handler: JobHandler | None = None
        self.requeue_on_error = requeue_on_error
        self.max_attempts = max_attempts
        self.dead_letters: list[IngestionJob] = []

    async def enqueue(self, job: IngestionJob) -> None:
        await self._jobs.put((job, 0))

    def set_handler(self, handler: JobHandler) -> None:
        self._handler = handler

    async def run(self) -> None:
        """Drain all jobs (including any enqueued while handling) and stop — so tests
        terminate. Use as the test entrypoint instead of a forever loop."""
        assert self._handler is not None, "set_handler() before run()"
        while not self._jobs.empty():
            job, attempts = await self._jobs.get()
            try:
                await self._handler([job])
            except Exception:
                if not self.requeue_on_error:
                    raise
                if attempts + 1 >= self.max_attempts:
                    self.dead_letters.append(job)
                else:
                    await self._jobs.put((job, attempts + 1))
```

---

## Orchestration & Workers

### 1. PipelineDAG and Orchestrator (`app/domains/ingestion/orchestrator.py`)

```python
from typing import Any
from app.domains.ingestion.pipeline import PipelineStage
from app.domains.ingestion.models import IngestionJob
from app.domains.base.models import PipelineItem
from app.domains.base.repository import DocumentRepository
from app.domains.ingestion.result_sink import ResultSink, FinalizationOutcome
from app.domains.ingestion.queue import JobQueue
from app.core.exceptions import IngestionError
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineDAG:
    """Represents the pipeline as a directed acyclic graph."""

    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages
        self.edges = self._build_edges()

    def _build_edges(self) -> list[tuple[str, str]]:
        return [(self.stages[i].name, self.stages[i + 1].name)
                for i in range(len(self.stages) - 1)]

    def get_downstream_stages(self, stage_name: str) -> list[str]:
        """Stage names that depend on this stage (empty for the terminal stage)."""
        return [target for source, target in self.edges if source == stage_name]

    def get_stage_config(self, stage_name: str) -> dict[str, Any]:
        for stage in self.stages:
            if stage.name == stage_name:
                return stage.config
        raise ValueError(f"Stage {stage_name} not found")


class PipelineOrchestrator:
    """
    Walks the DAG and owns the JOB LIFECYCLE (D3/D5). pgQueuer owns the queue itself
    (claiming, retries, acking); the orchestrator only enqueues jobs through it and,
    after a worker finishes, finalize()s the sink and either enqueues the downstream
    jobs (success) or RAISES so pgQueuer requeues the job (recovery). It also keeps the
    document-keyed `job_status` projection up to date for the status API.
    """

    def __init__(self,
                 dag: PipelineDAG,
                 queue: JobQueue,
                 repository: DocumentRepository,
                 sink_registry: dict[str, type[ResultSink]]):
        self.dag = dag
        self.queue = queue
        self.repository = repository          # storage + job_status projection
        self.sink_registry = sink_registry

    async def ingest_documents(self, items: list[PipelineItem]) -> list[str]:
        """Enqueue one root job per document for the first stage. Returns the public
        document_ids (the source_ids); job ids never leave the system."""
        first_stage = self.dag.stages[0]
        document_ids = []
        for item in items:
            job = self._make_job(item, first_stage.name)
            await self._enqueue(job)
            document_ids.append(job.document_id)
        return document_ids

    # ----- worker ⇄ orchestrator handshake (D4/D5) -----

    def make_sink(self, stage_name: str) -> ResultSink:
        """Build the stage's ResultSink, wired with the repository (D4/D6)."""
        return self.sink_registry[stage_name](self.repository)

    async def mark_processing(self, jobs: list[IngestionJob]) -> None:
        for job in jobs:
            await self.repository.record_job(job.document_id, job.job_id,
                                             job.stage_name, "processing")

    async def mark_failed(self, jobs: list[IngestionJob], error: str) -> None:
        for job in jobs:
            await self.repository.record_job(job.document_id, job.job_id,
                                             job.stage_name, "failed", error)

    async def complete_batch(self, stage_name: str, jobs: list[IngestionJob],
                             produced: list[Any], sink: ResultSink) -> None:
        """Persist the batch and advance the DAG. On persistence failure, record it and
        RAISE so pgQueuer requeues (= recovery, D5). On success pgQueuer acks the job
        when the handler returns; we record status and enqueue the downstream jobs."""
        outcome: FinalizationOutcome = await sink.finalize()
        if not outcome.persisted:
            await self.mark_failed(jobs, f"persistence: {outcome.detail}")
            raise IngestionError(f"persistence failed for {stage_name}: {outcome.detail}")

        for job in jobs:
            await self.repository.record_job(job.document_id, job.job_id,
                                             stage_name, "completed")

        # Fan-out (D1): one downstream job per produced item, per next stage.
        for next_stage_name in self.dag.get_downstream_stages(stage_name):
            for item in produced:
                await self._enqueue(self._make_job(item, next_stage_name))

    # ----- helpers -----

    def _make_job(self, item: PipelineItem, stage_name: str) -> IngestionJob:
        return IngestionJob(
            job_id=str(uuid.uuid4()),
            document_id=item.metadata["source_id"],   # public handle; threads via metadata
            item=item,
            stage_name=stage_name,
            stage_config=self.dag.get_stage_config(stage_name),
            created_at=datetime.utcnow(),
        )

    async def _enqueue(self, job: IngestionJob) -> None:
        await self.repository.record_job(job.document_id, job.job_id, job.stage_name, "queued")
        await self.queue.enqueue(job)
```

### 2. Worker (`app/domains/ingestion/worker.py`)

```python
import logging
import uuid
from app.domains.ingestion.orchestrator import PipelineOrchestrator
from app.domains.ingestion.pipeline import PipelineStage
from app.domains.ingestion.models import IngestionJob
from app.domains.ingestion.queue import JobQueue
from app.core.observability import Observability

logger = logging.getLogger(__name__)


class IngestionWorker:
    """Runs ingestion as the queue's handler — COMPUTE ONLY (D2/D3). The JobQueue
    (pgQueuer in prod, in-memory in tests) claims, dispatches, retries and acks; this
    worker just reconstructs the pure stage, runs it, streams results to a ResultSink,
    and hands the batch to the orchestrator. Raising propagates to the queue, which
    requeues the job (= recovery, D5).

    A dispatch is one job by default, or a list under batch dispatch — which restores
    cross-document model/persistence batching (process_batch over all items).
    """

    def __init__(self,
                 queue: JobQueue,
                 orchestrator: PipelineOrchestrator,
                 stage_registry: dict[str, type[PipelineStage]],
                 observability: Observability | None = None,
                 worker_id: str | None = None):
        self.queue = queue
        self.orchestrator = orchestrator
        self.stage_registry = stage_registry
        self.obs = observability
        self.worker_id = worker_id or str(uuid.uuid4())[:8]
        self.queue.set_handler(self.handle_batch)

    async def handle_batch(self, jobs: list[IngestionJob]) -> None:
        """Process one dispatch (one job, or a batch). Raising → requeue (recovery);
        returning normally → ack."""
        stage_name = jobs[0].stage_name
        await self.orchestrator.mark_processing(jobs)
        sink = self.orchestrator.make_sink(stage_name)
        try:
            stage = self.stage_registry[stage_name](**jobs[0].stage_config)
            produced: list = []
            for result in stage.process_batch([job.item for job in jobs]):  # inline items (D1)
                if getattr(result, "id", None) is None:
                    result.id = str(uuid.uuid4())
                produced.append(result)
            sink.submit(produced)   # hand off for batched persistence (D4)
            sink.close()
        except Exception as e:
            logger.error(f"Worker {self.worker_id} compute failed on {stage_name}: {e}",
                         exc_info=True)
            await self.orchestrator.mark_failed(jobs, str(e))
            raise                   # queue requeues (recovery)
        # Persistence + DAG advance belong to the orchestrator (D5); raises on failure.
        await self.orchestrator.complete_batch(stage_name, jobs, produced, sink)

    async def run(self) -> None:
        """Start consuming — delegates to the queue's run loop."""
        await self.queue.run()
```

**Worker ⇄ stage / orchestrator contract** (what the rewrite above depends on):

- ✅ `IngestionJob` carries the in-flight `PipelineItem` **inline** as `job.item` plus a
  `document_id` (== source_id) (D1). *(In the IngestionJob model.)*
- ✅ The orchestrator exposes `make_sink(stage_name) -> ResultSink` and
  `complete_batch(stage_name, jobs, produced, sink)`, which `finalize()`s the sink,
  then records status and enqueues one downstream job per produced item on success,
  or **raises** so pgQueuer requeues on failure (D5). *(In the `PipelineOrchestrator`
  section.)*
- ✅ `PipelineStage.process_batch(items)` — the batch entrypoint the worker calls;
  default maps `process()` over items, `EmbedStage` overrides it to size its own
  model calls and now produces real `Embedding` results. *(In the Pipeline & Stages
  section.)*
- ✅ The `ResultSink` interface + per-stage sinks (Document / Chunk / Embedding) that
  `make_sink` returns; ids are threaded forward via `metadata['doc_id']` /
  `metadata['chunk_id']` so FKs resolve (D4). *(In the Result Sinks section.)*

---

## Services (Facade Pattern)

### IngestionService (`app/domains/ingestion/service.py`)

```python
from typing import Any
from app.domains.ingestion.orchestrator import PipelineOrchestrator, PipelineDAG
from app.domains.ingestion.pipeline import Pipeline
from app.domains.base.models import PipelineItem, Document, Chunk, Embedding
from app.domains.base.repository import DocumentRepository
from app.core.observability import Observability
import uuid
from datetime import datetime

class IngestionService:
    """High-level facade for the ingestion pipeline; used by the API and CLI. The
    public surface is document-centric — jobs never leak to clients."""

    def __init__(self,
                 pipeline: Pipeline,
                 orchestrator: PipelineOrchestrator,
                 repository: DocumentRepository,
                 observability: Observability | None = None):
        self.pipeline = pipeline
        self.orchestrator = orchestrator
        self.repository = repository          # persistence + document-status reads
        self.obs = observability

    async def ingest_from_paths(self, file_paths: list[str]) -> dict[str, Any]:
        """Queue ingestion of documents loaded from file paths. Returns document_ids."""
        items = []
        for path in file_paths:
            source_id = str(uuid.uuid4())
            items.append(PipelineItem(
                id=source_id,
                content="",  # loaded by LoadAndParseStage
                metadata={
                    "source_id": source_id,
                    "source_path": path,
                    "source_type": self._infer_source_type(path),
                    "created_at": datetime.utcnow().isoformat(),
                },
            ))
        return await self._queue(items)

    async def ingest_from_content(self, documents: list[dict[str, str]]) -> dict[str, Any]:
        """Queue ingestion of pre-loaded content. A client-supplied source_id becomes
        the document_id; otherwise one is assigned."""
        items = []
        for doc in documents:
            content = doc.pop("content")
            source_id = doc.get("source_id", str(uuid.uuid4()))
            items.append(PipelineItem(
                id=source_id,
                content=content,
                metadata={"source_id": source_id,
                          "created_at": datetime.utcnow().isoformat(), **doc},
            ))
        return await self._queue(items)

    async def _queue(self, items: list[PipelineItem]) -> dict[str, Any]:
        document_ids = await self.orchestrator.ingest_documents(items)
        if self.obs:
            self.obs.counter("ingestion.documents_queued", len(items))
        return {
            "documents": [{"document_id": d, "status": "queued"} for d in document_ids],
            "documents_queued": len(document_ids),
        }

    async def get_document_status(self, document_id: str,
                                  verbose: bool = False) -> dict[str, Any] | None:
        """Document-level status derived from persisted data (no job concept): present?
        chunk_count / embedding_count -> pending|in_progress|complete, rolled up to
        'failed' if any of the document's jobs failed. None if unknown. With
        verbose=True, adds a debug-only `jobs` breakdown — the only place per-job state
        is exposed."""
        status = await self.repository.document_status(document_id)
        if status is None:
            return None
        if verbose:
            status["jobs"] = await self.repository.document_jobs(document_id)
        return status

    def _infer_source_type(self, path: str) -> str:
        ext = path.lower().split(".")[-1]
        return {"pdf": "pdf", "txt": "text", "text": "text",
                "html": "html", "htm": "html"}.get(ext, "unknown")
```

---

## API Layer

### Schemas (`app/api/v1/schemas.py`)

```python
from pydantic import BaseModel

class IngestRequest(BaseModel):
    """Request to ingest documents."""
    file_paths: list[str]  # Paths to files to ingest

class IngestFromContentRequest(BaseModel):
    """Request to ingest pre-loaded content."""
    documents: list[dict[str, str]]  # [{"content": "...", "source_id": "..."}]

class DocumentRef(BaseModel):
    """A document handle returned to the client (jobs are never exposed)."""
    document_id: str
    status: str  # "queued"

class IngestResponse(BaseModel):
    """Response from an ingest endpoint — document-centric."""
    documents: list[DocumentRef]
    documents_queued: int

class DocumentStatusResponse(BaseModel):
    """Document-level ingestion status (derived from persisted data)."""
    document_id: str
    status: str  # pending | in_progress | complete | failed
    chunk_count: int
    embedding_count: int
    jobs: dict | None = None  # debug-only, present when ?verbose=true
```

### Endpoints (`app/api/v1/endpoints/ingestion.py`)

```python
from fastapi import APIRouter, Depends, HTTPException
from app.api.v1.schemas import (
    IngestRequest, IngestFromContentRequest, IngestResponse, DocumentStatusResponse,
)
from app.domains.ingestion.service import IngestionService
from app.api.v1.dependencies import get_ingestion_service

router = APIRouter(prefix="/v1/ingest", tags=["ingestion"])

@router.post("/", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Ingest documents from file paths. Returns a document_id per document."""
    return IngestResponse(**await service.ingest_from_paths(req.file_paths))

@router.post("/content", response_model=IngestResponse)
async def ingest_content(
    req: IngestFromContentRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Ingest pre-loaded document content. Returns a document_id per document."""
    return IngestResponse(**await service.ingest_from_content(req.documents))

@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def document_status(
    document_id: str,
    verbose: bool = False,   # debug only: include the per-job breakdown
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentStatusResponse:
    """Document-level ingestion status. Jobs are an internal detail — exposed only
    under ?verbose=true for debugging."""
    status = await service.get_document_status(document_id, verbose=verbose)
    if status is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(**status)
```

### Dependency Injection (`app/api/v1/dependencies.py`)

```python
from fastapi import Depends
from app.core.config import Settings
from app.domains.base.postgres_repository import PostgresRepository
from app.domains.base.sqlite_repository import SqliteRepository
from app.domains.ingestion.service import IngestionService
from app.domains.ingestion.orchestrator import PipelineOrchestrator, PipelineDAG
from app.domains.ingestion.pipeline import Pipeline
from app.domains.ingestion.result_sink import ResultSink
from app.domains.ingestion.queue import JobQueue, PgQueuerJobQueue
from app.core.observability import Observability

async def get_settings() -> Settings:
    return Settings()

async def get_repository(settings: Settings = Depends(get_settings)):
    """Return the configured repository (PostgreSQL or SQLite)."""
    if "postgres" in settings.DOCUMENT_DB_URL:
        repo = PostgresRepository(settings.DOCUMENT_DB_URL,
                                  embedding_dimension=settings.EMBEDDING_DIMENSION)
    else:
        repo = SqliteRepository(settings.DOCUMENT_DB_URL)
    await repo.connect()
    return repo

async def get_job_queue(settings: Settings = Depends(get_settings)) -> JobQueue:
    """The job queue port (pgQueuer in prod; swap an InMemoryJobQueue in tests)."""
    return await PgQueuerJobQueue.connect(settings.QUEUE_DB_URL)

async def get_observability(settings: Settings = Depends(get_settings)) -> Observability | None:
    if settings.OBSERVABILITY_ENABLED:
        pass  # instantiate based on config
    return None

async def get_ingestion_pipeline(settings: Settings = Depends(get_settings)) -> Pipeline:
    from config import create_ingestion_pipeline
    return create_ingestion_pipeline()

async def get_sink_registry() -> dict[str, type[ResultSink]]:
    from config import create_sink_registry
    return create_sink_registry()

async def get_orchestrator(
    pipeline: Pipeline = Depends(get_ingestion_pipeline),
    queue: JobQueue = Depends(get_job_queue),
    repository = Depends(get_repository),
    sink_registry: dict[str, type[ResultSink]] = Depends(get_sink_registry),
) -> PipelineOrchestrator:
    """Return the orchestrator (enqueues via the JobQueue; owns finalize/status/fan-out, D5)."""
    return PipelineOrchestrator(PipelineDAG(pipeline.stages), queue, repository, sink_registry)

async def get_ingestion_service(
    pipeline: Pipeline = Depends(get_ingestion_pipeline),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
    repository = Depends(get_repository),
    observability = Depends(get_observability),
) -> IngestionService:
    return IngestionService(pipeline, orchestrator, repository, observability)
```

---

## Configuration

### Settings (`app/core/config.py`)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application configuration from environment variables."""
    
    # FastAPI
    APP_NAME: str = "RAG Ingestion"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # Databases
    QUEUE_DB_URL: str  # pgQueuer queue database
    DOCUMENT_DB_URL: str  # Document/chunk/embedding storage
    
    # Ingestion
    EMBEDDING_MODEL: str = "sentence-transformers/all-minilm-l6-v2"
    EMBEDDING_DIMENSION: int = 384  # MUST match the model; sets the pgvector column width
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Workers
    WORKER_QUEUE_TIMEOUT_SECONDS: int = 30
    WORKER_CONCURRENCY: int = 4
    
    # Observability
    OBSERVABILITY_ENABLED: bool = False
    OBSERVABILITY_TYPE: str | None = None  # "prometheus", "structured_logging"
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)
```

### .env.example

```bash
DEBUG=true

# Queue database
QUEUE_DB_URL=postgresql://user:password@localhost:5432/rag_queue

# Document/embedding storage
DOCUMENT_DB_URL=postgresql://user:password@localhost:5432/rag_docs

# Embedding
EMBEDDING_MODEL=sentence-transformers/all-minilm-l6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=32

# Workers
WORKER_CONCURRENCY=4

# Observability
OBSERVABILITY_ENABLED=false
```

---

## Observability

### Interface (`app/core/observability.py`)

```python
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any
import time

class Observability(ABC):
    """Abstract observability interface for metrics, logging, tracing."""
    
    @abstractmethod
    async def log(self, level: str, message: str, **context):
        """Log a message with context. Level: debug, info, warning, error."""
        pass
    
    @abstractmethod
    def counter(self, metric_name: str, value: int = 1, tags: dict[str, str] | None = None):
        """Increment a counter metric."""
        pass
    
    @abstractmethod
    def gauge(self, metric_name: str, value: float, tags: dict[str, str] | None = None):
        """Set a gauge metric."""
        pass
    
    @contextmanager
    def timer(self, metric_name: str, tags: dict[str, str] | None = None):
        """Context manager for timing operations."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.gauge(f"{metric_name}.seconds", elapsed, tags=tags)

class NoOpObservability(Observability):
    """No-op implementation. Use for development/testing."""
    
    async def log(self, level, message, **context): pass
    def counter(self, metric_name, value=1, tags=None): pass
    def gauge(self, metric_name, value, tags=None): pass
```

---

## Execution Flow

### Example: Ingesting 2 Documents

```
1. API: POST /v1/ingest with ["doc1.pdf", "doc2.txt"]
   ↓
2. IngestionService.ingest_from_paths()
   - Creates 2 PipelineItems
   ↓
3. PipelineOrchestrator.ingest_documents()
   - Creates 2 IngestionJob objects (stage=LoadAndParse)
   - Enqueues 2 root jobs via pgQueuer
   - Returns the document_ids (source_ids)
   ↓
4. API returns: {"documents": [{"document_id": "src_1", "status": "queued"}, …],
                 "documents_queued": 2}
   ↓
5. pgQueuer dispatches jobs to worker handlers (concurrently). Per D3/D5, workers do
   COMPUTE ONLY; the ORCHESTRATOR records status and enqueues downstream jobs after
   sink.finalize() succeeds, and pgQueuer acks the job (or requeues on a raise). The
   "creates job_N" steps below are driven by the orchestrator, not the worker:
   - Worker A: handles job_1 (LoadAndParse on doc1)
     → Parses PDF, produces a PipelineItem, submit()s it to its ResultSink, close()s
     → Item travels inline in the next job payload (D1); orchestrator finalizes the
       Document write, records status, and enqueues job_3 (CleanAndNormalize on doc1);
       pgQueuer acks job_1 when the handler returns
   
   - Worker B: executes job_2 (LoadAndParse on doc2)
     → (parallel)
   
   - Worker A: pulls job_3 (CleanAndNormalize)
     → Cleans text, creates job_4 (Chunk)
     → Marks job_3 complete
   
   - Worker B: pulls job_4 (Chunk on doc2)
     → (parallel, splits doc2 into 5 chunks)
     → Creates 5 jobs (Enrich on each chunk)
   
   - Workers A,B,C: pull Enrich jobs in parallel
     → Extract metadata, create jobs for Embed
   
   - pgQueuer dispatches Embed jobs (a batch, if batch dispatch is configured) (D2)
     → Worker computes embeddings in its own model-sized batches (compute batch)
     → submit()s embeddings to the ResultSink, then close()s (D3/D4)
     → Orchestrator calls sink.finalize(): the sink persists embeddings to the
       vector DB in its own persistence batches; on success it records status and
       pgQueuer acks the embed jobs (terminal stage → no downstream)
   ↓
6. All chunks and vectors stored in repository
   ↓
7. Client polls GET /v1/ingest/documents/{document_id}/status (document-level; jobs
   are internal, surfaced only via ?verbose=true for debugging)
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Implement `PipelineItem`, storage models (Document, Chunk, Embedding)
- [ ] Implement the SQLAlchemy Core `DocumentRepository` base (shared tables + CRUD)
- [ ] Implement PostgreSQL + SQLite adapters (dialect hooks only)
- [ ] Test transactional guarantees (store_document_with_chunks)
- [ ] Test idempotent re-ingestion (upsert on source_id replaces chunks/embeddings)
- [ ] Add the `job_status` projection table (document-keyed) — pgQueuer owns the queue (D6)

### Phase 2: Pipeline & Stages
- [ ] Implement base stage classes (PipelineStage, MapperStage, ChunkerStage, FilterStage) — stages stay pure (D6)
- [ ] Implement the `ResultSink` interface + per-stage sinks (Document / Chunk / Embedding), with two-tier batching (D4)
- [ ] Implement LoadAndParseStage
- [ ] Implement CleanAndNormalizeStage
- [ ] Implement ChunkStage
- [ ] Implement EnrichMetadataStage
- [ ] Implement EmbedStage (real `Embedding` output; `process_batch` model batching)
- [ ] Wire `EMBEDDING_DIMENSION` through Settings → PostgresRepository DDL
- [ ] Test pipeline composition and data flow

### Phase 3: Orchestration & Workers
- [ ] Implement PipelineDAG and PipelineOrchestrator (owns finalize / ack / downstream-enqueue / recovery — D3/D5)
- [ ] Implement the `JobQueue` port + `PgQueuerJobQueue` and `InMemoryJobQueue` adapters; install pgQueuer's queue tables (D2)
- [ ] Test the full ingestion flow end-to-end on `InMemoryJobQueue` + SQLite (no Postgres/pgQueuer)
- [ ] Implement IngestionWorker as the pgQueuer handler — compute only: run pure stage, submit()/close() to ResultSink (D2/D3)
- [ ] Test parallel execution with multiple workers
- [ ] Test orchestrator finalize()/ack/downstream-enqueue and persistence-failure recovery (D5)

### Phase 4: Services & API
- [ ] Implement IngestionService (facade)
- [ ] Implement API endpoints (POST /v1/ingest, GET /v1/ingest/jobs/{id})
- [ ] Implement dependency injection
- [ ] Test end-to-end API flow

### Phase 5: Observability
- [ ] Implement observability interface and NoOpObservability
- [ ] Add observability calls to worker and stages
- [ ] (Future) Implement Prometheus adapter
- [ ] (Future) Implement structured logging adapter

### Phase 6: Testing & Documentation
- [ ] Unit tests for each stage
- [ ] Integration tests for full pipeline
- [ ] Test repository transactional behavior
- [ ] Write API documentation
- [ ] Write README with setup instructions

---

**End of Specification**
