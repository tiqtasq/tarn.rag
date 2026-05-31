# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

Implementation has **started** (Phase 1 scaffolded). The authoritative design is
`doc/FUNCTIONAL_REQUIREMENTS.md` ("RAG Ingestion Pipeline" spec) — treat it as the
source of truth for structure, models, interfaces, and behavior, and follow its
phased **Implementation Checklist** (Phases 1–6), since later layers depend on
earlier ones (models → repository → stages → orchestration → service/API).

**Done (Phase 1, tested on SQLite):** `pyproject.toml`; `app/domains/base/models.py`
(Pydantic v2 models); `app/core/exceptions.py`; the SQLAlchemy Core repository
(`base/repository.py` + `sqlite_repository.py` tested, `postgres_repository.py`
compiles but needs the `postgres` extra); `app/domains/ingestion/models.py`
(`IngestionJob`); `app/domains/ingestion/queue.py` (`JobQueue` port + `InMemoryJobQueue`
+ lazy `PgQueuerJobQueue`). Tests in `tests/` (8 passing).

**Not yet built:** stages, `ResultSink`s, orchestrator, worker, service, API, DI,
observability. Implementation notes: the `_upsert_document` upsert was made a
**concrete portable** method on the base (not a per-dialect hook — the spec showed
`ON CONFLICT`, but select-then-update/insert is correct on both dialects); schema
creation uses **two hooks** (`_before_create_schema`/`_after_create_schema`) so the
pgvector extension is created before tables and indexes after. Use `datetime.now(UTC)`,
not the spec's `datetime.utcnow()` (deprecated on 3.12).

## What This System Is

A composable, DAG-based RAG ingestion pipeline that turns raw documents into
queryable vector embeddings. The data flow:

```
POST /v1/ingest → IngestionService (facade) → PipelineOrchestrator (walks DAG,
creates jobs) → pgQueuer (distributed queue) → IngestionWorker(s) (parallel) →
PipelineStages (load → clean → chunk → enrich → embed) → DocumentRepository →
PostgreSQL/SQLite
```

## Resolved Architecture Decisions (Authoritative)

These were resolved in design review and **override any conflicting illustrative
code** in `doc/FUNCTIONAL_REQUIREMENTS.md`. That file's "Resolved Architecture
Decisions" section (D1–D6) holds the full rationale; this is the summary.

- **Distributed execution, one job per `(item, stage)` (D1).** Fan-out stages
  enqueue one downstream job per output item (chunker → *m* chunks → *m* embed
  jobs). In-flight items travel **inline in the job payload** — no temp item store.
- **Job granularity (D2):** one chunk per job. The domain depends on a tiny **`JobQueue`
  port** (`enqueue` / `set_handler` / `run`); the worker is a registered handler, not a
  puller. **`PgQueuerJobQueue`** is the only file that imports pgQueuer (pgQueuer owns
  SKIP LOCKED / retries / NOTIFY / dead-lettering); **`InMemoryJobQueue`** is the test
  double — the whole flow runs on it + SQLite with no Postgres/pgQueuer. A dispatch is
  one job (or a list under batch dispatch, restoring cross-document batching). Compute
  batch ⊥ the sink's persistence batch (two-tier batching). Keep the port a delegating
  seam — never reimplement queue mechanics in it.
- **Three-layer split (D3):**
  - **Worker = compute only.** A pgQueuer handler: runs the pure stage, groups work
    into model calls as it sees fit, hands results to a `ResultSink`. Never persists,
    acks, enqueues, or retries. Raising → pgQueuer requeues (recovery).
  - **`ResultSink` = persistence (D4).** Output-side sink; batches and writes
    results. `submit(results)` + `close()` are called by the **worker** (sync,
    buffer-only). `async finalize() -> FinalizationOutcome` is called by the
    **orchestrator only**. Replaces the spec's input-side `BatchingWrapper`.
  - **Orchestrator = lifecycle + DAG walking (D5).** After the worker `close()`s, it
    `await`s `sink.finalize()`; on success it records status + enqueues downstream
    jobs (pgQueuer acks on handler return); on failure it records `failed` and
    **raises** so pgQueuer requeues. Dependencies are implicit (downstream enqueued
    only after upstream persists).
- **Stages stay pure (D6)** — no DB/queue access. **pgQueuer owns the queue tables**;
  the API's document status comes from a small document-keyed **`job_status`
  projection** in the repository (a read-model, not a queue).
- **Public API is document-centric.** Clients get a `document_id` (== `source_id`)
  from `POST /v1/ingest` and poll `GET /v1/ingest/documents/{document_id}/status`
  (derived from persisted data). Jobs are internal — never in the contract, exposed
  only under `?verbose=true` for debugging.

## Architecture & Key Concepts

The spec defines layers that are deliberately decoupled; preserve these seams
when implementing:

- **`app/core/`** — Infrastructure only (config, exceptions, observability). No
  business logic.
- **`app/domains/base/`** — Shared by ingestion and (future) retrieval: data
  models (`PipelineItem`, `Document`, `Chunk`, `Embedding`) and the
  `DocumentRepository` ABC with PostgreSQL + SQLite implementations.
- **`app/domains/ingestion/`** — Stages, `Pipeline`, `PipelineDAG`,
  `PipelineOrchestrator`, `IngestionWorker`, `IngestionService`.
- **`app/domains/retrieval/`** — Future; will reuse `base.repository`. Don't
  build it until ingestion is complete.

Concepts that span multiple files and are easy to get wrong:

- **Stage taxonomy.** All stages subclass `PipelineStage`. Most stages should
  subclass one of the typed base classes rather than `PipelineStage` directly:
  `MapperStage` (1→1, override `map()`), `ChunkerStage` (1→N, override
  `chunk()`), `FilterStage` (1→{0,1}, override `should_keep()`). These base
  classes handle metadata merging automatically — subclasses return *updates* to
  metadata, never mutate the incoming dict.
- **`ResultSink`** (not the spec's `BatchingWrapper`) is how results leave a
  worker. The worker `submit()`s produced results and `close()`s; the sink owns
  persistence and write-batching; the orchestrator calls `finalize()`. See the
  Resolved Architecture Decisions above (D3/D4). The `BatchingWrapper` class in
  the spec is superseded — don't implement it.
- **Database agnosticism.** All storage goes through the `DocumentRepository`,
  now a **SQLAlchemy 2.0 Core (async)** base class: shared table definitions and
  dialect-agnostic CRUD live in the base, and `PostgresRepository`/`SqliteRepository`
  override only the hooks (`_driver_url`, `_vector_type`, `_encode_vector`/`_decode_vector`,
  `_upsert_document`, `_create_dialect_objects`, `vector_search`). Postgres uses
  pgvector (real `<=>` cosine search); SQLite stores vectors as JSON and does
  in-memory cosine (dev/small-scale only). When adding repo behavior, put shared
  logic in the base and dialect specifics in the hooks — don't fork a whole method
  unless the SQL genuinely diverges. Selection is driven by `DOCUMENT_DB_URL` (a
  `postgres` substring picks Postgres, else SQLite).
- **Transactional guarantees.** `store_document_with_chunks`, `store_chunks`,
  and `store_embeddings` must be atomic (all-or-nothing). These are explicitly
  contracted in the repository docstrings and have dedicated integration tests.
- **Idempotency.** Documents are keyed by `metadata['source_id']` (UNIQUE index in
  both backends). Re-ingesting a source **upserts** the document and **replaces**
  its chunks/embeddings (delete chunks → cascade removes embeddings → re-insert);
  re-runs never duplicate.
- **Two databases.** `QUEUE_DB_URL` (pgQueuer job queue) is separate from
  `DOCUMENT_DB_URL` (document/chunk/embedding storage). Don't conflate them.
- **Observability is optional.** Core logic must work with `observability=None`.
  `NoOpObservability` exists for dev/test. Guard every `self.obs` call.
- **Async throughout.** The repository, worker, orchestrator, and service are all
  `async`. The pipeline stages themselves are synchronous generators
  (`process()` yields `PipelineItem`s).

## Intended Stack

From the spec (these tools are prescribed but not yet installed): **Python 3.12** +
FastAPI, **Pydantic v2** / pydantic-settings (use `model_config = ConfigDict(...)` /
`SettingsConfigDict(...)`, not `class Config`), **SQLAlchemy 2.0 Core (async)** for the
repository (asyncpg driver + `pgvector.sqlalchemy` for Postgres, aiosqlite driver +
numpy in-memory cosine for SQLite), **pgQueuer** for the job queue (the worker is a
registered handler; a small `job_status` projection backs the API), sentence-transformers (embeddings), pytest.
Packaging via `pyproject.toml` + `requirements.txt`.

**Type-annotation convention (Python 3.12):** use builtin generics (`list`, `dict`,
`tuple`, `type`) and `X | None` — never `typing.List`/`Dict`/`Optional`. Import
`Iterator`/`Iterable`/`Callable` from `collections.abc`; keep `Any`/`Literal`/
`Protocol` from `typing`. The spec's code blocks already follow this.

## Commands

Dev runs in the **conda env `tarn.rag`** (Python 3.12). Each Bash tool call is a
fresh non-login shell, so prefix commands with `conda run -n tarn.rag` (or the user
activates it interactively with `conda activate tarn.rag`).

```bash
# Run the test suite (config in pyproject.toml: asyncio_mode=auto, pythonpath=["."])
conda run -n tarn.rag python -m pytest -q
conda run -n tarn.rag python -m pytest tests/test_repository.py            # single file
conda run -n tarn.rag python -m pytest tests/test_repository.py::test_name # single test

# Compile-check all modules (postgres_repository needs the `postgres` extra to import)
conda run -n tarn.rag python -m py_compile $(find app -name "*.py")
```

Phase-1 deps (`pydantic`, `pydantic-settings`, `sqlalchemy`, `aiosqlite`, `numpy`,
`pytest`, `pytest-asyncio`) are installed in the env. The Postgres/pgQueuer/API/embed
backends are optional extras in `pyproject.toml` (`postgres`, `queue`, `api`, `embed`)
and are NOT installed yet — keep the SQLite test path free of those imports (the
`postgres_repository` and `PgQueuerJobQueue` modules import their heavy deps lazily).

Tests live flat in `tests/` for now (`test_repository.py`, `test_queue.py`); the
spec's deeper `tests/unit|integration/...` layout can come as the suite grows.
