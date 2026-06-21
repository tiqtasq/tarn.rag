# RAG Ingestion & Retrieval — Software Specification

> **Status note (2026-06-20):** this spec is the original **ingestion** design and its core architecture
> (the D1–D6 decisions, the queue / worker / orchestrator / ResultSink split, observability) is still
> accurate. Three areas were **rewritten to match the code**: the *Project Structure* (the package was
> reorganized into `core/{components,engine,resources}`, `ingestion/{components,engine,pipeline}`,
> `retrieval/{components,engine,pipeline}`, plus new `generation/` and `eval/` packages), the *Database
> Schema* (now the **§8 typed-column** schema from the ModusQ retrieval spec — `document_id`/`chunk_id`
> keys, denormalized license columns, layout-aware provenance columns, and the `table_cells` /
> `chunk_annotations` / `method_chunks` / `index_meta` tables — **not** the old metadata-bag schema), and
> the *Concrete stages* (LoadAndParse is now structured extraction; Enrich runs doc-phase enrichers; Chunk
> defaults to the structure-aware chunker). The system now spans **ingestion + retrieval + generation**;
> see `doc/ModusQ_RetrievalSubsystemSpec.md`, `doc/retrieval-architecture-design.md`, and
> `doc/generation-architecture-design.md` for the latter two.

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
│   ├── __init__.py            # public API re-exports (TarnRag, IngestionEngine, RetrievalEngine, Query, …)
│   ├── tarnrag.py             # TarnRag — high-level facade over all three engines (composition root)
│   ├── report.py              # Outcome / Report / Issue / Severity (what every facade call returns)
│   ├── console.py             # interactive rich-terminal UI over the facade (optional `console` extra)
│   │
│   ├── core/                                    # infra only
│   │   ├── components/                          # the Component framework: Component, ComponentFactory, Registry
│   │   ├── engine/                              # config.py (Settings), engine.py (Engine base), observability.py
│   │   ├── resources/                           # engine-built injected models: Embedder (+API), CrossEncoder,
│   │   │                                        #   LanguageModel (+API), Resource base
│   │   ├── exceptions.py
│   │   └── hashing.py                           # sha256_hex / sha256_file / compute_content_hash
│   │
│   ├── contracts/                               # cross-boundary shared kernel (leaf package)
│   │   ├── dtos.py                              # Document, Chunk, Embedding, PipelineItem (+document/provenance), MethodRef
│   │   ├── ports.py                             # ChunkStore, RetrievalStore, DocumentFactsSource, JobStatusSource
│   │   ├── results.py                           # Candidate, ChunkRecord, RetrievalResult
│   │   ├── structure.py                         # StructuredDocument / Element / Table / Annotation / ChunkProvenance / Span
│   │   └── index_meta.py                        # §8 build/identity record (schema + fingerprint)
│   │
│   ├── storage/                                 # persistence layer
│   │   ├── status.py                            # DocumentStatusReader (composes the status ports)
│   │   └── repository/
│   │       ├── base.py                          # DocumentRepository (SQLAlchemy Core, async; §8 index)
│   │       ├── chunk_provenance.py              # ChunkProvenance <-> columns/table_cells/chunk_annotations codec
│   │       ├── postgres.py                      # PostgreSQL (pgvector + tsvector) dialect
│   │       └── sqlite.py                        # SQLite (sqlite-vec/FTS5) dialect
│   │
│   ├── ingestion/
│   │   ├── engine/                              # engine.py (IngestionEngine + run_worker), worker, orchestrator,
│   │   │                                        #   queue, batch, result_sink, jobs, types
│   │   ├── pipeline/                            # pipeline.py (PipelineStage/Pipeline), clean_normalize, embed
│   │   └── components/                          # extraction/ (Extractor + plain_text/markdown/html/pdf/docling),
│   │                                            #   chunking/ (Chunker + recursive/structure_aware), enrichment/
│   │
│   ├── retrieval/
│   │   ├── engine/                              # engine.py (RetrievalEngine), retrieval_engine_protocol.py
│   │   ├── pipeline/                            # searcher.py, pipeline.py (RetrievalPipeline), router.py
│   │   └── components/                          # retriever (dense/sparse), fuser (identity/rrf), merger,
│   │                                            #   reranker (cross_encoder), classifier
│   │   └── types.py                             # Query, Purpose, ALL (result types live in contracts/)
│   │
│   ├── generation/                              # the Goal-3 answer layer (generation → retrieval, one-way)
│   │   ├── engine/                              # GenerationEngine facade
│   │   ├── pipeline/                            # GenerationPipeline (reason → ground → assemble → policy)
│   │   ├── components/                          # reasoner (single_hop/iterative/decomposition), grounding, assembler
│   │   ├── context.py  types.py                 # GenerationContext; GenerationResult / ProofStep / Citation
│   │
│   └── eval/                                    # harness.py (retrieval sweep), generation.py, dataset, metrics
│
├── tests/                    # mirrors the package; SQLite + InMemory queue (Postgres/docling tests gated)
├── examples/                 # runnable teaching examples (python -m examples.part_i.…)
├── scripts/fetch_model.py    # fetch the ONNX model + tokenizer into the model dir
├── run_worker.py             # distributed consumer entry: asyncio.run(run_worker())
├── .env.example  pyproject.toml  requirements.txt  README.md  CLAUDE.md
└── doc/                      # this spec, ModusQ_RetrievalSubsystemSpec.md, the design docs, main.pdf
```

### Folder Rationale

- **`tarnrag/core/`** — Infrastructure: the `components/` framework (config-driven Component + factory), `engine/` (Settings, the `Engine` base, observability), `resources/` (engine-built injected models — `Embedder`, `CrossEncoder`, `LanguageModel`), hashing, exceptions. No business logic.
- **`tarnrag/contracts/`** — The dependency-free shared kernel: DTOs, ports, retrieval results, the `StructuredDocument` model, and the index-meta record.
- **`tarnrag/storage/`** — Persistence: the `DocumentRepository` (§8 index + status read model) and its Postgres/SQLite dialects.
- **`tarnrag/ingestion/`** — Ingestion: `components/` (extractors, chunkers, enrichers), `pipeline/` (stage bases + clean/embed), `engine/` (orchestrator, worker, queue, sinks, `IngestionEngine`).
- **`tarnrag/retrieval/`** — Retrieval: `components/` (retriever/fuser/merger/reranker/classifier), `pipeline/` (`RetrievalPipeline` + router), the `RetrievalEngine` facade + `Query`.
- **`tarnrag/generation/`** — The answer layer: reasoners, grounding, proof-tree assembler, and the `GenerationEngine` facade. Depends on retrieval one-way; never the reverse.
- **`tarnrag/eval/`** — Retrieval + generation eval harnesses (IR metrics, sweeps, segmentation).
- **`TarnRag` (`tarnrag.py`)** — the composition root + high-level facade over all three engines (`ingest`/`retrieve`/`ask`), each returning an `Outcome` + `Report`.
- **`tests/`** — Mirrors the package; runs on SQLite + InMemory queue (Postgres + docling tests skip unless their backend is present).

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

Pydantic v2 (`arbitrary_types_allowed=True`). → `tarnrag/contracts/dtos.py`,
`tarnrag/ingestion/engine/jobs.py`.

- **`PipelineItem`** (transport): `id: str | None`, `content: str`, `metadata: dict[str, Any]`, plus the
  two phase-specific layout fields `document: StructuredDocument | None` (document phase) and
  `provenance: ChunkProvenance | None` (chunk phase); `derive()` carries them forward across 1→1 stages.
- **`Document`**: `id`, `content`, `metadata` (carries `source_id`, the idempotency key).
- **`Chunk`**: `id`, `parent_doc_id` (→ Document), `content`, `chunk_index`, `total_chunks`,
  `provenance: ChunkProvenance | None` (the layout-aware trace the sink maps to chunk columns), `metadata`.
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
hooks. → `storage/repository/base.py`, `postgres.py`, `sqlite.py` (+ `chunk_provenance.py`, the
`ChunkProvenance` ⇄ columns/`table_cells`/`chunk_annotations` codec).

**Base (`DocumentRepository`):** implements four narrow ports — `ChunkStore`, `RetrievalStore`,
`JobStatusSource`, `DocumentFactsSource` — and owns the `MetaData`/`Table`s (`documents`, `chunks`,
`table_cells`, `chunk_annotations`, `method_chunks`, `embeddings`, `job_status`, `index_meta`; the schema
is the **§8 typed-column** model, **no JSON metadata bag** on documents/chunks — layout-aware provenance
lives in dedicated columns + child tables). The CRUD: `store_document` / `store_document_with_chunks`
(upsert on the `document_id` PK, then replace the doc's chunks → idempotent re-ingest), `store_chunks`,
`store_embeddings`, `delete_document` / `delete_document_and_jobs` (cascade), reads (`get_document` /
`get_chunk` / `get_chunks_by_document`), `list_documents` (inventory + counts), `documents_by_content_hash`,
the §8 retrieval reads (abstract `dense_knn` / `sparse_search` / `hydrate`), the index identity
(`write_index_meta` / `index_meta`), and the document-status projection (`record_job` / `document_jobs` /
`delete_document_jobs` / `document_status`). (`query_chunks`, `health_check`, and `update_chunk_metadata`
exist but are currently unused / no-ops — retained as documented read-surface / latent capability.)

**Guarantees:** multi-row writes share one `engine.begin()` transaction (atomic); documents are keyed by
`document_id` (== `source_id`, the PK, **stable**), so re-ingest upserts the doc and replaces its
chunks/embeddings (cascade) — never duplicates. Each document also stores a `content_hash` column (sha256
of submitted content); `documents_by_content_hash` looks it up for content dedup, independent of identity.

**Dialect hooks (adapters):** `_driver_url`, `_vector_type`, `_encode_vector` / `_decode_vector`,
`dense_knn` / `sparse_search` / `hydrate`, and `_before_/_after_create_schema`.
- **Postgres** — asyncpg driver + `pgvector` (`<=>` cosine for `dense_knn`) + `to_tsvector`/`ts_rank_cd`
  for `sparse_search`; creates the `vector` extension before tables, and the ivfflat + FTS GIN indexes after.
- **SQLite** — aiosqlite driver; dense vectors in `vec_chunks` (sqlite-vec) + sparse text in `fts_chunks`
  (FTS5), the extension loaded per connection; enables `PRAGMA foreign_keys=ON` per connection (cascade).

### Database Schema (current — the §8 typed-column model)

All tables are created from the SQLAlchemy `MetaData` in `storage/repository/base.py`
(`metadata.create_all`); the dialect-only objects are created by each adapter — on **Postgres** the
`vector` extension + the `ivfflat` ANN index on `embeddings.vector` + the FTS **GIN** index on
`to_tsvector('english', chunks.text)`; on **SQLite** the `vec_chunks` (sqlite-vec) and `fts_chunks`
(FTS5) **virtual tables**. The reference DDL below is dialect-agnostic (SQLite stores `vector` as JSON
text and keeps the `embeddings` table empty — dense vectors live in `vec_chunks` — while Postgres uses a
real `vector(dim)` column). `license_class` is a closed enum: `customer_licensed`, `public_domain`,
`modusq_authored`, `third_party_copyrighted`, `third_party_licensed`. `job_status` is the API's read-model;
pgQueuer owns the real queue tables (its own migrations).

> **What changed from the original design:** the document/chunk identity is now `document_id` / `chunk_id`
> (== `source_id` for documents), **not** a uuid `id` + a `source_id` JSON key; there is **no JSON
> `metadata` bag** on documents or chunks — provenance is typed columns (`license_class`,
> `ai_grounding_allowed`, `available`, `locator`, `header_path`, `level`, `parent_chunk_id`, `geometry`)
> plus the `table_cells` / `chunk_annotations` / `method_chunks` child tables; `chunks` stores `ordinal`
> (position), not `chunk_index`/`total_chunks`.

```sql
CREATE TABLE documents (
    document_id   TEXT PRIMARY KEY,        -- == source_id (the public, stable handle)
    content       TEXT NOT NULL,           -- full doc text (Python-side; not part of the §8 contract)
    title         TEXT,
    source_kind   TEXT NOT NULL DEFAULT 'document',   -- 'standard' | 'sop' | 'method' | 'document' | …
    standard_id   TEXT,
    doc_version   TEXT,
    license_class TEXT NOT NULL DEFAULT 'public_domain',   -- CHECK: closed enum
    content_hash  TEXT                      -- sha256 of submitted content (content dedup, not identity)
);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);

CREATE TABLE chunks (
    chunk_id             TEXT PRIMARY KEY,
    document_id          TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    ordinal              INTEGER NOT NULL,   -- position within the document
    text                 TEXT NOT NULL,      -- canonical chunk text (returned verbatim)
    locator              TEXT,               -- citable locator, e.g. '§6.4.2'
    license_class        TEXT NOT NULL DEFAULT 'public_domain',   -- CHECK enum; denormalized for filtering
    ai_grounding_allowed INTEGER NOT NULL DEFAULT 1,   -- CHECK (0,1)
    available            INTEGER NOT NULL DEFAULT 1,   -- CHECK (0,1)
    content_hash         TEXT NOT NULL,      -- sha256 of the chunk text
    -- layout-aware provenance:
    header_path     TEXT,                    -- JSON list[str] (the section breadcrumb)
    level           INTEGER NOT NULL DEFAULT 0,   -- auto-merge tree: 0 = leaf, >0 = section parent
    parent_chunk_id TEXT,                    -- section parent's chunk_id (soft self-ref, no FK)
    geometry        TEXT                     -- JSON Geometry (char spans + optional PDF page boxes)
);
CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_license  ON chunks(license_class, available);
CREATE INDEX idx_chunks_parent   ON chunks(parent_chunk_id);

-- Cell-level table structure for table chunks (cite/highlight a cell; address by row/col header id).
CREATE TABLE table_cells (
    chunk_id  TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    cell_id   TEXT NOT NULL,
    row INTEGER NOT NULL, col INTEGER NOT NULL,
    row_span INTEGER NOT NULL DEFAULT 1, col_span INTEGER NOT NULL DEFAULT 1,
    is_column_header INTEGER NOT NULL DEFAULT 0, is_row_header INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL DEFAULT '', geometry TEXT,   -- JSON Geometry for the cell
    PRIMARY KEY (chunk_id, cell_id)
);

-- Enricher annotations on a chunk (NER / topic / classification …).
CREATE TABLE chunk_annotations (
    chunk_id      TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    producer      TEXT NOT NULL, type TEXT NOT NULL,
    value         TEXT NOT NULL DEFAULT '{}',   -- JSON payload
    span          TEXT,                          -- JSON Geometry sub-span; null = whole chunk
    deterministic INTEGER NOT NULL DEFAULT 1,    -- FR-5.3 anti-hallucination flag
    PRIMARY KEY (chunk_id, ordinal)
);
CREATE INDEX idx_chunk_annotations_type ON chunk_annotations(type);

-- Resolved reference bundles: which method versions reach which chunks (scope + provenance).
CREATE TABLE method_chunks (
    method_id TEXT NOT NULL, method_version TEXT NOT NULL,
    chunk_id  TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    PRIMARY KEY (method_id, method_version, chunk_id)
);
CREATE INDEX idx_method_chunks_chunk ON method_chunks(chunk_id);

CREATE TABLE embeddings (   -- Postgres: dense ANN here; SQLite: stays empty (vectors live in vec_chunks)
    id        TEXT PRIMARY KEY,
    chunk_id  TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    vector    vector(384),   -- pgvector; dim MUST equal settings.EMBEDDING_DIMENSION. (SQLite: JSON text)
    model     TEXT NOT NULL, dimension INTEGER NOT NULL
);
CREATE INDEX idx_embeddings_chunk ON embeddings(chunk_id);

-- §8 build/compatibility metadata (incl. the embedding fingerprint validated at retrieval open()).
CREATE TABLE index_meta ( key TEXT PRIMARY KEY, value TEXT NOT NULL );

-- Document-status PROJECTION for the API (pgQueuer owns the real queue).
CREATE TABLE job_status (
    job_id      TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,                    -- == document_id; the public handle
    stage_name  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',   -- queued | processing | completed | failed
    error       TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_job_status_document ON job_status (document_id);

-- SQLite only — the §8 search indexes as extension virtual tables:
--   CREATE VIRTUAL TABLE vec_chunks USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[<dim>]);
--   CREATE VIRTUAL TABLE fts_chunks USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61');
```

---

## Pipeline & Stages

### Stage base classes (`tarnrag/ingestion/pipeline/pipeline.py`)

ABCs, and config-driven **`Component`s** (`tarnrag/core/components`): each stage declares a typed
nested `Config` (a pydantic model that pins a `class_name` tag plus the stage's fields) and is
built from it — `ChunkStage(ChunkStage.Config(chunk_size=256))`, or from a dict/JSON spec via
`ComponentFactory.create({"class_name": "Chunk", …})`. Field validation lives on the `Config`
(pydantic constraints + `model_validator`s), not a separate `validate()`. `stage.name` is a
read-only property over `config.name or class_name`, so several instances of one stage class can
coexist in a pipeline (each with its own `name`), defaulting to the tag. A stage exposes
`process(item) -> Iterator` and the batch entrypoint `process_batch(items)` (default maps
`process`; override for model batching). Typed helpers: `MapperStage` (1→1, override `map`),
`ChunkerStage` (1→N, sets `chunk_index`/`total_chunks`), `FilterStage` (1→{0,1}).
`Pipeline` is itself a container **`Component`** (`class_name: "pipeline"`): its Config lists raw
stage specs and `_build_children` instantiates them, so a whole pipeline is a spec —
`{"class_name": "pipeline", "stages": [{"class_name": "Chunk", …}, …]}`. `run()` threads items
through stages in-process for local testing; the distributed engine runs stages individually.
`Pipeline.from_stages([...])` builds one from pre-made stages (tests/advanced wiring). Stages stay
pure — no DB/queue access (D6). The spec's `BatchingWrapper` is superseded (D4 → ResultSink).

### Concrete stages (`tarnrag/ingestion/components/` + `tarnrag/ingestion/pipeline/`)

Names below are each stage's `class_name` tag (its default `stage.name`, and the sink-registry key). The
default pipeline order is **LoadAndParse → Enrich → CleanAndNormalize → Chunk → Embed**.

- **LoadAndParse** (`ingestion/components/extraction/load_parse.py`) — the **structured-extraction** stage
  (replaced the old text-only parser). It infers `source_kind` from the path/`metadata['source_type']`,
  routes to the configured `Extractor` (a child Component; `Config.routes` maps `source_kind → extractor
  spec`, default `pdf → pdf_text`, with `markdown`/`html`/`plain_text`; a per-document
  `metadata['extractor']` override wins — e.g. `"docling"` for high-fidelity PDF), calls `extract(source)`
  → `StructuredDocument`, and sets `item.document` + `item.content = document.text`.
- **Enrich** (`ingestion/components/enrichment/enrich.py`) — the **doc-phase enrichment** driver: runs a
  configured, ordered list of `Enricher` Components over `item.document`, each appending typed
  `Annotation`s (NER / topic / classification …). Default is **none** (passthrough); a stub `acronyms`
  enricher ships. Annotations flow into chunks via the chunker (`ChunkProvenance.annotations`).
- **CleanAndNormalize** (`ingestion/pipeline/clean_normalize.py`) — strips control chars, collapses
  whitespace (operates on the text view).
- **Chunk** (`ingestion/components/chunking/`) — a `ChunkerStage` that delegates to a configured `Chunker`
  child; the default is **`structure_aware`** (splits on the document's heading structure into a multi-level
  leaf + section-parent auto-merge tree, tables atomic, header-path + geometry + element ids on each
  chunk's `ChunkProvenance`). `recursive` (character-window splitting) is the simpler alternative.
- **Embed** (`ingestion/pipeline/embed.py`, terminal) — vectorizes chunks → `Embedding`s; `process_batch`
  groups items into `embedding.batch_size` model calls; `chunk_id` from `metadata['chunk_id']`. Its Config
  holds the `embedding` (`EmbeddingSettings`) identity (kept in sync with the retrieval-side embedder); the
  embedder lazy-loads via `Embedder.create` (local ONNX or an API backend), tests inject a fake via
  `stage._embedder`. With `embedding.inject_header_path` set, each chunk's header path is prepended before
  embedding (an embed-time index variant that rides the fingerprint).
## Result Sinks

Output-side sinks (D4). A worker `submit()`s produced results and `close()`s — both
**synchronous, buffer-only** — and the orchestrator later `await`s `finalize()` to
persist. One sink per stage; the orchestrator's `begin_batch` looks the sink up from
`create_sink_registry()` keyed by the stage `.tag`.

**ID threading.** Storage ids are threaded forward via **metadata**, which stages
propagate (they merge metadata): `DocumentResultSink` writes `metadata['doc_id']`,
`ChunkResultSink` writes `metadata['chunk_id']`. Downstream sinks/stages read those,
so the FK chain (chunk→doc, embedding→chunk) resolves without relying on `item.id`
(stages create fresh items) or on the repository honoring caller-supplied ids.

### Interface + sinks (`tarnrag/ingestion/engine/result_sink.py`)

`ResultSink` (ABC): `submit(results)` + `close()` are sync (buffer-only, called by the
worker); `async finalize() -> FinalizationOutcome(persisted, detail)` is called by the
orchestration layer (persists, reports the outcome). A `_BufferingSink` base persists its
buffer via `_persist()` and catches failures into the outcome. Per-stage sinks (registered
by stage name):

- **DocumentResultSink** (LoadAndParse) — `store_document` (upsert); threads `metadata['doc_id']`.
- **PassthroughSink** (Enrich, CleanAndNormalize) — persists nothing (enrichment annotates
  `item.document`, which rides into chunks via the chunker; clean/normalize only edits the text view).
- **ChunkResultSink** (Chunk) — `store_chunks` (persists chunk rows + `table_cells` + `chunk_annotations`);
  threads `metadata['chunk_id']`.
- **EmbeddingResultSink** (Embed) — bulk `store_embeddings` in persistence-batch-sized writes.

`create_sink_registry()` maps the five stage tags → sink classes (lives in `result_sink.py`).

---

## Job Queue (`tarnrag/ingestion/engine/queue.py`)

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
The handler receives a **`Batch`** (`ingestion/engine/jobs.py`) — a homogeneous unit whose jobs
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
work** (`tarnrag/ingestion/engine/batch.py`), not direct orchestrator calls. The worker
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

**`IngestionEngine`** (`tarnrag/ingestion/engine/engine.py`) — document-centric producer/query facade.
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

**`RetrievalEngine`** (`tarnrag/retrieval/engine/engine.py`) — sync query facade over the §8 index.
- `RetrievalEngine.create(settings=None)` opens the index (read-only) + the shared embedder and
  validates schema + embedding fingerprint (`RetrievalEngine.open(...)` is the lower-level seam).
- `search(Query)` / `search_text(text, *, top_k, dense_k)` → ranked `RetrievalResult`s;
  `asearch` / `asearch_text` are async (thread-offloaded) variants for event-loop callers.
  Lifecycle: `close()` / `with`.

---

## Configuration

**`Settings`** (`tarnrag/core/engine/config.py`, pydantic-settings; cached `get_settings()`). Config is
**grouped into nested sub-models** — `settings.embedding`, `settings.chunking`,
`settings.database`, `settings.worker`, `settings.observability` — read from env via the
`GROUP__FIELD` convention (e.g. `EMBEDDING__MODEL`, `DATABASE__DOCUMENT_URL`). Cross-cutting
`MODE`, `EMBEDDING_DIMENSION`, `UPLOAD_DIR`, `ID_POLICY` (`uuid` | `caller` — how document ids are
assigned) stay top-level/flat; `EMBEDDING_DIMENSION` must match
the model (and sets the pgvector column width). A `model_validator` pins the backend to the mode:
`distributed` requires a Postgres `DATABASE__DOCUMENT_URL` + `DATABASE__QUEUE_URL`; `embedded`
requires SQLite. **`components`** (`dict[str, Any]`) holds named component specs that build pluggable parts
of the system — the **ingestion pipeline** (`"ingestion_pipeline"`), the **retrieval pipeline**
(`"retrieval_pipeline"`), and the **generation pipeline** (`"generation_pipeline"`) — each a raw spec
validated when built via `ComponentFactory`. A `model_validator` fills all three defaults so a Settings is
self-complete (consumers read them directly). Pipeline **composition** is `IngestionEngine.build_pipeline`
(reads `INGESTION_PIPELINE`, injects the Embed stage's embedding identity), `RetrievalEngine.build_searcher`
(reads `RETRIEVAL_PIPELINE` → a `Searcher`), and `GenerationEngine.assemble` (reads `GENERATION_PIPELINE`).
The built-in stages/components **self-register** on import; `create_sink_registry` (output-side wiring)
lives in `result_sink.py`. See **`.env.example`** for the environment template.

---

## Observability

**Interface (`tarnrag/core/engine/observability.py`).** `Observability` (ABC): abstract `async
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
