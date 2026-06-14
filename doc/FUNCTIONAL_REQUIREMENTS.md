# RAG Ingestion & Retrieval — Software Specification

---

## Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Project Structure](#project-structure)
3. [Core Models](#core-models)
4. [Database & Repository Layer](#database--repository-layer)
5. [Pipeline & Stages](#pipeline--stages)
6. [Orchestration & Workers](#orchestration--workers)
7. [Engines (Facade)](#engines-facade)
8. [Configuration](#configuration)
9. [Observability](#observability)
10. [Execution Flow](#execution-flow)

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
Data Sources (files, content, streams)
    ↓
IngestionEngine.create() → ingest_paths / ingest_content / ingest_streams
    ↓
PipelineOrchestrator (walks DAG, enqueues jobs, owns lifecycle)
    ↓
JobEnqueuer / JobConsumer ports → InMemory (embedded) · pgQueuer (distributed)
    ↓
IngestionWorker(s) (registered handlers; compute only, parallel)
    ↓
PipelineStages (load → clean → chunk → enrich → embed)
    ↓
DocumentRepository (§8 index + job_status — one store)
    ↓
RetrievalEngine.create() → await search → ranked, provenance-bearing results
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
tarn.rag/
│
├── tarnrag/
│   ├── __init__.py                              # public API re-exports (IngestionEngine, RetrievalEngine, …)
│   │
│   ├── core/                                    # infra only
│   │   ├── config.py                            # Settings (nested sub-models, GROUP__FIELD env)
│   │   ├── exceptions.py
│   │   └── observability.py                     # Observability ABC + NoOpObservability
│   │
│   ├── embedder.py                              # Embedder ABC + OnnxEmbedder (shared)
│   │
│   ├── storage/                                 # persistence layer
│   │   ├── models.py                            # Document, Chunk, Embedding, PipelineItem
│   │   ├── chunk_store.py                       # ChunkStore ABC (the repository implements it)
│   │   ├── index_meta.py                        # §8 build/identity record (schema + fingerprint)
│   │   ├── retrieval.py                         # Candidate / ChunkRecord (dense_knn + hydrate types)
│   │   ├── status.py                            # DocumentStatusReader (job_status + facts ports)
│   │   └── repository/
│   │       ├── base.py                          # DocumentRepository (SQLAlchemy Core, async; §8 index)
│   │       ├── postgres.py                      # PostgreSQL (pgvector) dialect
│   │       └── sqlite.py                        # SQLite (sqlite-vec/FTS5) dialect
│   │
│   ├── ingestion/
│   │   ├── engine.py                            # IngestionEngine facade + run_worker()
│   │   ├── worker.py                            # IngestionWorker (pure compute handler)
│   │   ├── orchestrator.py                      # PipelineOrchestrator (BatchCoordinator), PipelineDAG
│   │   ├── pipeline.py                          # PipelineStage classes, Pipeline
│   │   ├── queue.py                             # JobEnqueuer/JobConsumer ports + PgQueuer/InMemory
│   │   ├── batch.py                             # BatchContext + BatchCoordinator (worker↔orch handshake)
│   │   ├── result_sink.py                       # ResultSink + per-stage sinks
│   │   ├── models.py                            # IngestionJob + Batch (homogeneous dispatch unit)
│   │   ├── types.py                             # DocumentStatus (public result type)
│   │   ├── factories.py                         # create_ingestion_pipeline / create_sink_registry
│   │   └── stages/                              # load_parse, parsers, clean_normalize, chunk, enrich, embed
│   │
│   └── retrieval/
│       ├── engine.py                            # RetrievalEngine facade
│       └── types.py                             # Query, RetrievalResult, MethodRef
│
├── tests/                                       # mirrors the package; SQLite + InMemory queue
├── scripts/fetch_model.py                       # fetch the ONNX model + tokenizer into the model dir
├── run_worker.py                                # distributed consumer entry: asyncio.run(run_worker())
├── .env.example  pyproject.toml  requirements.txt  README.md
└── doc/                                         # FUNCTIONAL_REQUIREMENTS.md, ModusQ_RetrievalSubsystemSpec.md
```

### Folder Rationale

- **`tarnrag/core/`** — Infrastructure: config, exceptions, observability. No business logic.
- **`tarnrag/embedder.py`** — The shared ONNX embedding pipeline (ingestion passages + retrieval queries).
- **`tarnrag/storage/`** — Persistence: data models, the chunk/index stores, the status read model, and `repository/` (Postgres/SQLite dialects).
- **`tarnrag/ingestion/`** — Ingestion: stages, pipeline, orchestrator, worker, queue, and the `IngestionEngine` facade.
- **`tarnrag/retrieval/`** — Retrieval: the `RetrievalEngine` facade + its types.
- **`tests/`** — Mirrors the package; runs on SQLite + InMemory queue.

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

Pydantic v2 (`arbitrary_types_allowed=True`). → `tarnrag/storage/models.py`,
`tarnrag/ingestion/models.py`.

- **`PipelineItem`** (transport): `id: str | None`, `content: str`, `metadata: dict[str, Any]`.
- **`Document`**: `id`, `content`, `metadata` (carries `source_id`, the idempotency key).
- **`Chunk`**: `id`, `parent_doc_id` (→ Document), `content`, `chunk_index`, `total_chunks`, `metadata`.
- **`Embedding`**: `id`, `chunk_id` (→ Chunk), `vector: list[float]`, `model`, `dimension`, `metadata`.
- **`IngestionJob`** (internal queue payload, never exposed to clients): `job_id`,
  `document_id` (== `source_id`), `item: PipelineItem` (inline, D1), `stage_name`,
  `created_at`. (The worker gets the configured stage from the coordinator's DAG by
  `stage_name`, so no `stage_config` travels in the job.)

**`metadata` conventions** (loose bag; recommended keys): `source_id`, `source_type`,
`source_path`, `content_hash` (sha256 of submitted content; persisted to its own column),
`doc_id` (set by Load), `chunk_index`, `total_chunks`, `chunk_size`,
`chunk_id` (set by ChunkResultSink), `created_at` (ISO); future NLP: `nlp_entities`,
`nlp_noun_phrases`, `summary`.

**Exceptions** (`tarnrag/core/exceptions.py`): `IngestionError` (base — results couldn't be
produced/persisted → the worker propagates so the queue requeues, D5), with subclasses
`DocumentStorageError` and `ChunkNotFoundError`.

---

## Database & Repository Layer

### Repository (`tarnrag/storage/`)

The repository abstracts the database for both ingestion and retrieval. **SQLAlchemy 2.0
Core (async).** A base `DocumentRepository` holds the shared table definitions and all
dialect-agnostic logic; `PostgresRepository`/`SqliteRepository` override only dialect
hooks. → `repository.py`, `postgres_repository.py`, `sqlite_repository.py`.

**Base (`DocumentRepository`):** owns the `MetaData`/`Table`s (`documents`, `chunks`,
`embeddings`, `job_status`; `metadata` is JSON with a JSONB variant on Postgres) and the
CRUD: `store_document` (upsert on `source_id`, then delete the doc's chunks → idempotent
re-ingest), `store_document_with_chunks`, `store_chunks`, `store_embeddings`,
`update_chunk_metadata`, `delete_document` (drops the doc + chunks → embeddings cascade),
reads (`get_document` / `get_chunk` / `get_chunks_by_document` / `query_chunks`),
`list_documents` (inventory + counts), `documents_by_content_hash`, the §8 retrieval reads
(abstract `dense_knn` / `hydrate`), and the document-status projection
(`record_job`, `document_jobs`, `delete_document_jobs`, `document_status`).

**Guarantees:** multi-row writes share one `engine.begin()` transaction (atomic);
documents are keyed by `metadata['source_id']` (UNIQUE, **stable**), so re-ingest upserts the doc
and replaces its chunks/embeddings (cascade) — never duplicates. Each document also stores a
`content_hash` column (sha256 of submitted content); `documents_by_content_hash` looks it up for
content dedup, independent of the source_id identity.

**Dialect hooks (adapters):** `_driver_url`, `_vector_type`, `_encode_vector` /
`_decode_vector`, `dense_knn` / `hydrate`, and `_before_/_after_create_schema`.
- **Postgres** — asyncpg driver + `pgvector` (`<=>` cosine for `dense_knn`); creates the
  `vector` extension before tables and the ivfflat index after.
- **SQLite** — aiosqlite driver; dense vectors in `vec_chunks` (sqlite-vec) + sparse text in
  `fts_chunks` (FTS5), the extension loaded per connection; enables `PRAGMA foreign_keys=ON`
  per connection (for cascade).

### Database Schema

The document/chunk/embedding tables are created from the SQLAlchemy `MetaData` in
the repository base (`metadata.create_all`), and the dialect-only objects (pgvector
extension and the `vector_cosine_ops` ivfflat index on Postgres; the `vec_chunks` /
`fts_chunks` virtual tables on SQLite) by each adapter. The DDL below is the
**equivalent reference** (also handy for `scripts/init_db*.sql`). The `job_status`
table is the API's read-model; pgQueuer owns the real queue tables (its own migrations).

**PostgreSQL** (`scripts/init_db.sql`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    content_hash TEXT,                  -- sha256 of submitted content (content dedup, not identity)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- UNIQUE on source_id makes re-ingestion idempotent (upsert via ON CONFLICT)
CREATE UNIQUE INDEX idx_documents_source_id ON documents ((metadata->>'source_id'));
CREATE INDEX idx_documents_content_hash ON documents(content_hash);

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
    content_hash TEXT,                  -- sha256 of submitted content (content dedup, not identity)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_created ON documents(created_at);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
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

### Stage base classes (`tarnrag/ingestion/pipeline.py`)

ABCs. `PipelineStage(name, **config)` runs `validate()` in `__init__` — so a subclass
must set any attrs `validate()` reads **before** `super().__init__()`. A stage exposes
`process(item) -> Iterator` and the batch entrypoint `process_batch(items)` (default maps
`process`; override for model batching). Typed helpers: `MapperStage` (1→1, override
`map`), `ChunkerStage` (1→N, sets `chunk_index`/`total_chunks`), `FilterStage` (1→{0,1}).
`Pipeline` is a thin ordered container (`run()` threads items through stages in-process,
for local testing; the distributed engine runs stages individually). Stages stay pure —
no DB/queue access (D6). The spec's `BatchingWrapper` is superseded (D4 → ResultSink).

### Concrete stages (`tarnrag/ingestion/stages/`)

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

### Interface + sinks (`tarnrag/ingestion/result_sink.py`)

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
re-exported from `tarnrag/ingestion/factories.py`).

---

## Job Queue (`tarnrag/ingestion/queue.py`)

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
work** (`tarnrag/ingestion/batch.py`), not direct orchestrator calls. The worker
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
## Engines (Facade)

The public surface is two engines, each built from `Settings` via a `create()` factory; jobs
never leak into the contract.

**`IngestionEngine`** (`tarnrag/ingestion/engine.py`) — document-centric producer/query facade.
- `await IngestionEngine.create(settings=None)` wires everything per `Settings.MODE`
  (`embedded` → in-process InMemory queue; `distributed` → pgQueuer + a separate `run_worker()`).
- `ingest_paths(paths)`, `ingest_content(documents)`, `ingest_streams(streams)` shape
  `PipelineItem`s and **return the document IDs (`list[str]`)**. The `document_id` (== `source_id`)
  is **stable** and assigned per `Settings.ID_POLICY`: `caller` requires the caller to supply every
  id (per-doc on `ingest_content`; a parallel `source_ids` list on `ingest_paths`/`ingest_streams`);
  `uuid` assigns one and forbids caller ids. A mismatch fails ingestion (no silent scheme-mixing);
  reuse an id to upsert in place. They delegate to `orchestrator.ingest_documents`; in embedded mode
  the call also drains the pipeline to completion before returning.
- `find_by_content_hash(content_hash)` → the document IDs holding that exact content (content
  dedup, independent of identity). Every document stores a `content_hash` (sha256 of its submitted
  bytes/text); `content_hash(data)` / `content_hash_of_file(path)` compute the key.
- `status(document_id)` → `DocumentStatus` (`pending | in_progress | complete | failed` + chunk /
  embedding counts), or `None` if unknown — the single source of truth for state.
- `list_documents()` → `list[DocumentSummary]` (id, `content_hash`, chunk/embedding counts) — the
  inventory of everything ingested. `delete_document(document_id)` → removes the document and all its derived
  data (chunks, embeddings, retrieval-index rows) plus its job-status records; returns whether it
  existed (idempotent).
- `document_jobs(document_id)` is the **debug-gated** window into per-job state (raises unless
  `APP__DEBUG`). Lifecycle: `aclose()` / `async with`.

**`RetrievalEngine`** (`tarnrag/retrieval/engine.py`) — sync query facade over the §8 index.
- `RetrievalEngine.create(settings=None)` opens the index (read-only) + the shared embedder and
  validates schema + embedding fingerprint (`RetrievalEngine.open(...)` is the lower-level seam).
- `search(Query)` / `search_text(text, *, top_k, dense_k)` → ranked `RetrievalResult`s;
  `asearch` / `asearch_text` are async (thread-offloaded) variants for event-loop callers.
  Lifecycle: `close()` / `with`.

---

## Configuration

**`Settings`** (`tarnrag/core/config.py`, pydantic-settings; cached `get_settings()`). Config is
**grouped into nested sub-models** — `settings.embedding`, `settings.chunking`,
`settings.database`, `settings.worker`, `settings.observability` — read from env via the
`GROUP__FIELD` convention (e.g. `EMBEDDING__MODEL`, `DATABASE__DOCUMENT_URL`). Cross-cutting
`MODE`, `EMBEDDING_DIMENSION`, `UPLOAD_DIR`, `ID_POLICY` (`uuid` | `caller` — how document ids are
assigned) stay top-level/flat; `EMBEDDING_DIMENSION` must match
the model (and sets the pgvector column width). A `model_validator` pins the backend to the mode:
`distributed` requires a Postgres `DATABASE__DOCUMENT_URL` + `DATABASE__QUEUE_URL`; `embedded`
requires SQLite. The composition **factories** live in **`tarnrag/ingestion/factories.py`**
(wiring, not config): `create_ingestion_pipeline(settings=None)` builds the configured stages, and
`create_sink_registry` is re-exported from `result_sink.py`. See **`.env.example`** for the
environment template.

---

## Observability

**Interface (`tarnrag/core/observability.py`).** `Observability` (ABC): abstract `async
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
the outside (it knows the stage name, item counts, and timing). `IngestionEngine.create()` builds
it from `settings.observability` (a `NoOpObservability` when enabled, else `None`) and threads it
through the orchestrator and the worker.

> Real adapters (Prometheus, structured logging) are future work and plug in behind this ABC.

---

## Execution Flow

### Example: Ingesting 2 Documents

```
1. await IngestionEngine.create() → ingest_paths(["doc1.pdf", "doc2.txt"])
   ↓
2. ingest_paths shapes 2 PipelineItems (source_id == document_id)
   ↓
3. PipelineOrchestrator.ingest_documents()
   - Creates 2 IngestionJob objects (stage=LoadAndParse)
   - Enqueues 2 root jobs via pgQueuer
   - Returns the document_ids (source_ids)
   ↓
4. ingest_paths returns the document IDs: ["src_1", "src_2"]
   - (embedded mode also drains the pipeline before returning; distributed returns immediately)
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
7. Caller polls engine.status(document_id) (document-level; jobs are internal,
   surfaced only via the debug-gated engine.document_jobs)
```

**End of Specification**
