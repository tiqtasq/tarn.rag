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
JobEnqueuer / JobConsumer ports → pgQueuer (distributed job queue)
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
│   │       │   └── ingestion.py                 # POST /v1/ingest(/content,/file), GET /v1/ingest/documents/{id}/status
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
│   │   │   ├── models.py                        # IngestionJob + Batch (homogeneous dispatch unit)
│   │   │   ├── pipeline.py                      # PipelineStage classes, Pipeline (BatchingWrapper superseded, see D4)
│   │   │   ├── result_sink.py                   # ResultSink interface + per-stage sinks (D4)
│   │   │   ├── stages/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── load_parse.py                # LoadAndParseStage
│   │   │   │   ├── parsers.py                    # Document loaders: PDF registry (pypdf/pdfplumber) + html (bs4)
│   │   │   │   ├── clean_normalize.py           # CleanAndNormalizeStage
│   │   │   │   ├── chunk.py                     # ChunkStage
│   │   │   │   ├── enrich.py                    # EnrichMetadataStage
│   │   │   │   └── embed.py                     # EmbedStage (simple 1-to-1, no batching)
│   │   │   ├── batch.py                         # BatchContext + BatchCoordinator (worker↔orch handshake)
│   │   │   ├── orchestrator.py                  # PipelineOrchestrator (BatchCoordinator), PipelineDAG
│   │   │   ├── queue.py                         # JobEnqueuer + JobConsumer ports + PgQueuer/InMemory adapters (D2/D5)
│   │   │   ├── worker.py                        # IngestionWorker (pure handler; compute only, reports to BatchContext)
│   │   │   └── service.py                       # IngestionService (high-level facade)
│   │   │
│   │   └── retrieval/
│   │       ├── __init__.py
│   │       ├── models.py
│   │       ├── retriever.py                     # Future: retrieval logic
│   │       └── service.py                       # Future: RetrievalService
│   │
│   ├── factories.py                            # Composition factories (create_ingestion_pipeline, create_sink_registry)
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
│   └── init_db.sh                               # Initialize databases (create tables)
│
├── run_worker.py                                # Worker process composition root (consumer side)
├── .env.example                                 # Example environment variables
├── requirements.txt                             # Python dependencies
├── pyproject.toml                               # Project metadata
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

These models are **two forms of the things being ingested** — not four unrelated types,
and not an inheritance hierarchy:

- **In-flight form → `PipelineItem`**: the single uniform type that flows between stages.
  A *document* (at Load/Clean) and a *chunk* (at Chunk/Enrich/Embed) are both just
  `PipelineItem`s — which one it is rides in `metadata`, not the type.
- **At-rest form → `Document`/`Chunk`/`Embedding`**: typed persistence DTOs materialised
  at the ResultSinks. They never flow.

Worked trace of one ingest (watch the form change):

```
after Load    PipelineItem(content="<doc text>", metadata={source_id, doc_id})
after Chunk   PipelineItem(content="<chunk 3>",  metadata={..., chunk_index: 3})
                  |  ChunkResultSink persists it →
                  v
              Chunk(id="c3", parent_doc_id="d1", content="<chunk 3>", chunk_index=3)
after Embed   Embedding(chunk_id="c3", vector=[...])   # terminal: never flows
```

So a chunk-in-flight *is* a `PipelineItem`; `Chunk` is its stored row. `Embedding` is the
exception (Embed is terminal → only ever at-rest). The relations among the four are
**composition** (`Chunk.parent_doc_id`→`Document`, `Embedding.chunk_id`→`Chunk`) and
**transformation** (a sink maps a `PipelineItem` into a DTO) — fields and functions, not
inheritance.

### Model contracts

Pydantic v2 (`arbitrary_types_allowed=True`). → `app/domains/base/models.py`,
`app/domains/ingestion/models.py`.

- **`PipelineItem`** (transport): `id: str | None`, `content: str`, `metadata: dict[str, Any]`.
- **`Document`**: `id`, `content`, `metadata` (carries `source_id`, the idempotency key).
- **`Chunk`**: `id`, `parent_doc_id` (→ Document), `content`, `chunk_index`, `total_chunks`, `metadata`.
- **`Embedding`**: `id`, `chunk_id` (→ Chunk), `vector: list[float]`, `model`, `dimension`, `metadata`.
- **`IngestionJob`** (internal queue payload, never exposed to clients): `job_id`,
  `document_id` (== `source_id`), `item: PipelineItem` (inline, D1), `stage_name`,
  `created_at`. (The worker gets the configured stage from the coordinator's DAG by
  `stage_name`, so no `stage_config` travels in the job.)

**`metadata` conventions** (loose bag; recommended keys): `source_id`, `source_type`,
`source_path`, `doc_id` (set by Load), `chunk_index`, `total_chunks`, `chunk_size`,
`chunk_id` (set by ChunkResultSink), `created_at` (ISO); future NLP: `nlp_entities`,
`nlp_noun_phrases`, `summary`.

**Exceptions** (`app/core/exceptions.py`): `IngestionError` (base — results couldn't be
produced/persisted → the worker propagates so the queue requeues, D5), with subclasses
`DocumentStorageError` and `ChunkNotFoundError`.

---

## Database & Repository Layer

### Repository (`app/domains/base/`)

The repository abstracts the database for both ingestion and retrieval. **SQLAlchemy 2.0
Core (async).** A base `DocumentRepository` holds the shared table definitions and all
dialect-agnostic logic; `PostgresRepository`/`SqliteRepository` override only dialect
hooks. → `repository.py`, `postgres_repository.py`, `sqlite_repository.py`.

**Base (`DocumentRepository`):** owns the `MetaData`/`Table`s (`documents`, `chunks`,
`embeddings`, `job_status`; `metadata` is JSON with a JSONB variant on Postgres) and the
CRUD: `store_document` (upsert on `source_id`, then delete the doc's chunks → idempotent
re-ingest), `store_document_with_chunks`, `store_chunks`, `store_embeddings`,
`update_chunk_metadata`, reads (`get_document` / `get_chunk` / `get_chunks_by_document` /
`query_chunks`), abstract `vector_search`, and the document-status projection
(`record_job`, `document_jobs`, `document_status`).

**Guarantees:** multi-row writes share one `engine.begin()` transaction (atomic);
documents are keyed by `metadata['source_id']` (UNIQUE), so re-ingest upserts the doc and
replaces its chunks/embeddings (cascade) — never duplicates.

**Dialect hooks (adapters):** `_driver_url`, `_vector_type`, `_encode_vector` /
`_decode_vector`, `vector_search`, and `_before_/_after_create_schema`.
- **Postgres** — asyncpg driver + `pgvector` (real `<=>` cosine search); creates the
  `vector` extension before tables and the ivfflat + `source_id`-unique indexes after.
- **SQLite** — aiosqlite driver; vectors stored as JSON, in-memory numpy cosine
  (dev/small-scale); enables `PRAGMA foreign_keys=ON` per connection (for cascade).

### Database Schema

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

### Stage base classes (`app/domains/ingestion/pipeline.py`)

ABCs. `PipelineStage(name, **config)` runs `validate()` in `__init__` — so a subclass
must set any attrs `validate()` reads **before** `super().__init__()`. A stage exposes
`process(item) -> Iterator` and the batch entrypoint `process_batch(items)` (default maps
`process`; override for model batching). Typed helpers: `MapperStage` (1→1, override
`map`), `ChunkerStage` (1→N, sets `chunk_index`/`total_chunks`), `FilterStage` (1→{0,1}).
`Pipeline` is a thin ordered container (`run()` threads items through stages in-process,
for local testing; the distributed engine runs stages individually). Stages stay pure —
no DB/queue access (D6). The spec's `BatchingWrapper` is superseded (D4 → ResultSink).

### Concrete stages (`app/domains/ingestion/stages/`)

Names below are the `stage.name` values (and the sink-registry keys):

- **LoadAndParse** (`MapperStage`) — loads `metadata['source_path']` (txt; html via
  BeautifulSoup; pdf via a **pluggable backend registry**) or uses the given content; assigns a
  provisional `doc_id`. The loaders live in `stages/parsers.py` (`pypdf` default, `pdfplumber`,
  `load_html` (bs4); all permissive, lazily imported). The set of backends is **stage config** (built once); a
  request picks one per call via `metadata['parser']` (**item data**, flows inline — *not* the
  per-job `stage_config` we dropped). The API validates `parser` against the registry (422).
- **CleanAndNormalize** (`MapperStage`) — strips control chars, collapses whitespace.
- **Chunk** (`ChunkerStage`) — recursive character splitting on coarse→fine separators,
  packed into `chunk_size` windows with `overlap`.
- **EnrichMetadata** (`MapperStage`) — adds `char_count`/`word_count` (NLP is future).
- **Embed** (`PipelineStage`, terminal) — vectorizes chunks → `Embedding`s; `process_batch`
  groups items into `model_batch_size` model calls; `chunk_id` from `metadata['chunk_id']`.
  Lazy-loads sentence-transformers (tests inject a fake `encode`).
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

### Interface + sinks (`app/domains/ingestion/result_sink.py`)

`ResultSink` (ABC): `submit(results)` + `close()` are sync (buffer-only, called by the
worker); `async finalize() -> FinalizationOutcome(persisted, detail)` is called by the
orchestration layer (persists, reports the outcome). A `_BufferingSink` base persists its
buffer via `_persist()` and catches failures into the outcome. Per-stage sinks (registered
by stage name):

- **DocumentResultSink** (LoadAndParse) — `store_document` (upsert); threads `metadata['doc_id']`.
- **PassthroughSink** (CleanAndNormalize) — persists nothing.
- **ChunkResultSink** (Chunk) — `store_chunks`; threads `metadata['chunk_id']`.
- **ChunkMetadataResultSink** (EnrichMetadata) — `update_chunk_metadata`.
- **EmbeddingResultSink** (Embed) — bulk `store_embeddings` in persistence-batch-sized writes.

`create_sink_registry()` maps the five stage names → sink classes (lives in `result_sink.py`;
re-exported from `app/factories.py`).

---

## Job Queue (`app/domains/ingestion/queue.py`)

pgQueuer owns the queue (claiming with `FOR UPDATE SKIP LOCKED`, `LISTEN/NOTIFY`,
retries, dead-lettering, concurrency) — we hand-roll none of that. The surface is
**segregated by role** (so each caller depends only on what it uses): **`JobEnqueuer`**
(`enqueue` — the producer, used by the orchestrator) and **`JobConsumer`** (`set_handler`
+ `run` — the consumer runtime; the run loop belongs to pgQueuer). One adapter implements
both. A second in-memory adapter lets the whole ingestion flow be tested with no Postgres
and no pgQueuer. Per-job runtime state is pgQueuer's; the document-status API reads the
small `job_status` projection in the repository.

> pgQueuer's exact API (imports, handler/enqueue signatures, batch dispatch) is confined
> to `PgQueuerJobQueue` and should be confirmed against the pinned pgQueuer version.

`JobEnqueuer` (ABC): `enqueue(job)`. `JobConsumer` (ABC): `set_handler(handler)`, `run()`.
The handler receives a **`Batch`** (`ingestion/models.py`) — a homogeneous unit whose jobs
all share one `stage_name`. Forming the Batch is the **consumer's** responsibility (it's what
makes the worker a dead-simple "one batch → one stage" handler); `Batch`'s constructor
enforces homogeneity (and rejects empties), so a mixed-stage claim fails loudly rather than
silently running the wrong stage. `stage_name` stays on each job (the queue routes by it);
`Batch` derives the batch's stage from its jobs. Today every dispatch is a single job, so
the Batch is trivially homogeneous; batch dispatch must group claims by stage. Two adapters
implement both ports:

- **`PgQueuerJobQueue`** — the only module that imports pgQueuer (lazily). Wraps pgQueuer's
  `QueueManager`/`Queries`, registers the single `ingest` entrypoint, and propagates a
  handler-raise so pgQueuer requeues (recovery). pgQueuer owns claiming/retries/NOTIFY/
  dead-lettering.
- **`InMemoryJobQueue`** — test double: an `asyncio.Queue` emulating at-least-once +
  requeue-on-raise (`requeue_on_error` / `max_attempts` / `dead_letters`); `run()` drains
  and stops so tests terminate. Lets the whole flow run with no Postgres/pgQueuer.

---

## Orchestration & Workers

**Handshake (BatchContext).** The worker↔orchestration handshake is a per-batch **unit of
work** (`app/domains/ingestion/batch.py`), not direct orchestrator calls. The worker
depends only on two ABCs: `BatchCoordinator.begin_batch(batch) -> BatchContext`
(records 'processing', picks the per-stage sink by `batch.stage_name`) and `BatchContext` — `submit(results)`,
`async fail(error)`, and `async complete()` (finalize + persist + record + fan-out;
**raises** on persistence failure so the queue requeues). The worker never holds the
orchestrator, repo, queue, or `ResultSink`.

**`PipelineDAG`** (`orchestrator.py`): sequential edges; `get_downstream_stages` (empty for
the terminal stage) and `get_stage` (the long-lived configured stage instance).

**`PipelineOrchestrator`** (the `BatchCoordinator`): `ingest_documents(items)` enqueues one
root job per document and returns the public document_ids (== source_ids). `begin_batch`
returns an `_OrchestratorBatchContext` that **composes** the per-stage `ResultSink` and owns
status recording (`record_job`), DAG fan-out (one downstream job per produced item, D1), and
recovery (raise on persist-fail). It also owns the `job_status` projection wiring; the
queue owns claiming/retries/acking.

**`IngestionWorker`** (`worker.py`): a **pure job handler** — COMPUTE ONLY (D2/D3). It holds
only the `BatchCoordinator` and exposes `handle_batch`; it has no queue and no run loop, so
the composition root registers `worker.handle_batch` on a `JobConsumer` and runs the
consumer's loop. `handle_batch(batch)` takes a homogeneous `Batch`. Per dispatch:
`begin_batch(batch)` → `coordinator.get_stage(batch.stage_name)` (the DAG's long-lived
instance — so e.g. EmbedStage's model loads once, not per job) → `process_batch` over the
inline items → `ctx.submit(produced)` → `ctx.complete()` (or `ctx.fail(e); raise` on compute
error). The worker holds no stage registry, queue, repo, or `ResultSink`, and never re-checks
the batch's stage (the `Batch` already guarantees one). A raised exception propagates to the
consumer → requeue (recovery, D5).
## Services (Facade Pattern)

**`IngestionService`** (`app/domains/ingestion/service.py`) — the high-level facade used by
the API (and any CLI). The public surface is **document-centric**: jobs never leak.

- `ingest_from_paths(file_paths)` and `ingest_from_content(documents)` shape `PipelineItem`s
  (assigning a `source_id` == `document_id` when the caller doesn't supply one; a
  client-supplied `source_id` is honored), then delegate to `orchestrator.ingest_documents`.
  Both return `{"documents": [{"document_id", "status": "queued"}], "documents_queued": n}`.
- `get_document_status(document_id, verbose=False)` returns the repository's derived status
  (`pending | in_progress | complete | failed`), or `None` if unknown. `verbose=True` adds a
  debug-only `jobs` breakdown (`repository.document_jobs`) — the only place per-job state is
  exposed.

The facade is thin: it holds the pipeline (reference), the orchestrator (queueing), and the
repository (status reads); `observability` is optional (typed `Any`, Phase 5).

---

## API Layer (`app/api/v1/`)

FastAPI, document-centric. Four routes under `APIRouter(prefix="/v1/ingest")`
(`endpoints/ingestion.py`):

- `POST /v1/ingest/` (`IngestRequest{file_paths}`) and `POST /v1/ingest/content`
  (`IngestFromContentRequest{documents}`) → `IngestResponse{documents: [DocumentRef
  {document_id, status}], documents_queued}`. Both take an optional `parser` (PDF backend).
- `POST /v1/ingest/file` (multipart: `files` + optional `parser`) — the API **streams** the
  uploaded bytes to `UPLOAD_DIR` (`copyfileobj`, off-thread, no full-memory read) and ingests
  them by path (parsing runs in the worker, which must share that volume); same
  `IngestResponse`. An object store would replace local staging behind
  `IngestionService._stage_upload`.
- `GET /v1/ingest/documents/{document_id}/status?verbose=` →
  `DocumentStatusResponse{document_id, status, chunk_count, embedding_count, jobs?}`
  (`jobs` is `list[dict] | None`, present only under `?verbose=true`); **404** when unknown.

Schemas live in `schemas.py`; jobs never appear in the contract except that debug `jobs`.

**DI / composition (`dependencies.py`, `app/main.py`, `run_worker.py`).** The wiring is built
**once per process**, not per request: `make_repository(settings)` (Postgres on a `postgres`
URL, else SQLite; heavy backends imported lazily; connects the engine), `make_queue(settings)`
(`PgQueuerJobQueue.connect`), and `build_service` / `build_orchestrator` (which use
`create_ingestion_pipeline` + `create_sink_registry`). The API's `lifespan` (`create_app()` in
`app/main.py`) builds the `IngestionService` once and stores it on `app.state`; the request
dependency `get_ingestion_service(request)` just returns it (tests override it with an
InMemory + SQLite wiring — the real lifespan does not run under httpx's ASGI transport).
`run_worker.py` is the consumer composition root: the same builders, then
`consumer.set_handler(worker.handle_batch)` + `await consumer.run()`. (This supersedes the
earlier per-request DI sketch, which would have reconnected the DB on every call.) The API
process only enqueues root jobs; downstream fan-out happens in the worker process's
orchestrator — both share the same databases.

---

## Configuration

**`Settings`** (`app/core/config.py`, pydantic-settings; cached `get_settings()`):
`QUEUE_DB_URL` and `DOCUMENT_DB_URL` are required (two separate stores). Embedding
(`EMBEDDING_MODEL`; `EMBEDDING_DIMENSION` — must match the model and sets the pgvector column
width; `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_BATCH_SIZE`), worker, and observability fields
have defaults. `model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)`. The
composition **factories** live in **`app/factories.py`** (wiring, not config):
`create_ingestion_pipeline(settings=None)` builds the five configured stages, and
`create_sink_registry` is re-exported from `result_sink.py`. See **`.env.example`** at the
repo root for the environment template.

---

## Observability

**Interface (`app/core/observability.py`).** `Observability` (ABC): abstract `async
log(level, message, **context)`, `counter(name, value=1, tags=None)`, `gauge(name, value,
tags=None)`, plus a **concrete** `timer(name, tags=None)` `@contextmanager` that records
elapsed seconds as `{name}.seconds` via `gauge` (so every adapter — and the no-op — gets it
for free). `NoOpObservability` implements the three abstracts as no-ops.

**Who holds it (and who doesn't).** Observability is **optional and None-able** — core logic
runs with `observability=None`, and every call site guards `self.obs`. It is held by the
**worker** (compute: per-stage `timer`, throughput counters, an error counter + `log` on
compute failure) and the **orchestrator** (lifecycle: `ingest.documents` /
`ingest.jobs_enqueued`, and per-stage `completed` / `failed` / `persist_failed` from the
batch context). **Stages stay pure (D6)** — they take no obs; the worker observes them from
the outside (it knows the stage name, item counts, and timing). The composition roots build it
via `get_observability(settings)` (returns `NoOpObservability` when `OBSERVABILITY_ENABLED`,
else `None`) and thread it through `build_orchestrator` / `build_service` and into the worker.

> Real adapters (Prometheus, structured logging) are future work and plug in behind this ABC.

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
- [x] Implement IngestionService (facade) — document-centric; jobs never leak
- [x] Implement API endpoints (POST /v1/ingest/ and /content, GET /v1/ingest/documents/{document_id}/status)
- [x] Implement configuration (`Settings`, `app/core/config.py`) + composition factories (`app/factories.py`)
- [x] Implement dependency injection + composition roots (`app/main.py` API, `run_worker.py` worker)
- [x] Test end-to-end API flow (httpx ASGITransport over InMemoryJobQueue + SQLite)

### Phase 5: Observability
- [x] Implement observability interface and NoOpObservability
- [x] Add observability calls to the worker (compute) and orchestrator (lifecycle) — stages stay pure (D6)
- [ ] (Future) Implement Prometheus adapter
- [ ] (Future) Implement structured logging adapter

### Phase 6: Testing & Documentation
- [x] Unit tests for each stage (`tests/domains/ingestion/test_stages.py`)
- [x] Integration tests for full pipeline (`tests/integration/test_ingestion_e2e.py` + service/API e2e)
- [x] Test repository transactional behavior (`test_store_document_is_atomic` + idempotency/cascade)
- [x] Write API documentation (`doc/API_SPEC.md`, incl. local startup)
- [ ] Write README with setup instructions (optional — setup lives in `doc/API_SPEC.md`)

---

**End of Specification**
