# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

Implementation is **underway** (Phases 1–3 done). The authoritative design is
`doc/FUNCTIONAL_REQUIREMENTS.md` ("RAG Ingestion Pipeline" spec) — treat it as the
source of truth for structure, models, interfaces, and behavior, and follow its
phased **Implementation Checklist** (Phases 1–6), since later layers depend on
earlier ones (models → repository → stages → orchestration → service/API).

**Done (Phases 1–5, tested on SQLite, all green):**
- Phase 1: `pyproject.toml`; `base/models.py` (Pydantic v2); `core/exceptions.py`; the
  SQLAlchemy Core repository (`base/repository.py` + `sqlite_repository.py` tested,
  `postgres_repository.py` compiles but needs the `postgres` extra); `ingestion/models.py`
  (`IngestionJob`); `ingestion/queue.py` (`JobEnqueuer` + `JobConsumer` ABCs +
  `InMemoryJobQueue` + lazy `PgQueuerJobQueue`).
- Phase 2: `ingestion/pipeline.py` (`PipelineStage`/`MapperStage`/`ChunkerStage`/
  `FilterStage`/`Pipeline`); `ingestion/result_sink.py` (`ResultSink` + 5 sinks +
  `create_sink_registry`); `ingestion/stages/` (Load/Clean/Chunk/Enrich/Embed).
- Phase 3: `ingestion/batch.py` (`BatchContext` + `BatchCoordinator` ABCs — the
  worker↔orchestration handshake); `ingestion/orchestrator.py` (`PipelineDAG` +
  `PipelineOrchestrator`, the `BatchCoordinator`); `ingestion/worker.py`
  (`IngestionWorker`, a pure job handler, depends only on the batch ABCs). End-to-end test
  (`tests/integration/test_ingestion_e2e.py`) runs a doc through the whole pipeline on
  `InMemoryJobQueue` + SQLite, incl. fan-out and idempotent re-ingest.
- Phase 4: `core/config.py` (`Settings` + cached `get_settings`); `app/factories.py`
  (`create_ingestion_pipeline` from Settings + re-export `create_sink_registry`);
  `ingestion/service.py` (`IngestionService` facade — document-centric); `api/v1/`
  (`schemas.py`, `endpoints/ingestion.py`, `dependencies.py`); `app/main.py` (`create_app` +
  lifespan); `run_worker.py` (consumer composition root). API tested via httpx
  ASGITransport over `InMemoryJobQueue` + SQLite (`tests/api/v1/test_ingestion_api.py`,
  `tests/domains/ingestion/test_service.py`). **`fastapi` + `httpx` are now installed** in
  the env (httpx added to the `dev` extra).
- Phase 5: `core/observability.py` (`Observability` ABC — abstract `log`/`counter`/`gauge` +
  concrete `timer` contextmanager — + `NoOpObservability`). Obs is held by the **worker**
  (per-stage `timer`/throughput counters + error counter/`log`) and the **orchestrator**
  (lifecycle: `ingest.documents`/`ingest.jobs_enqueued`, per-stage `completed`/`failed`/
  `persist_failed`); **stages stay pure** (no obs — the worker observes them). Wired via
  `get_observability(settings)` → `build_orchestrator`/`build_service` + the worker. Tests:
  `tests/core/test_observability.py`, `tests/domains/ingestion/test_observability_wiring.py`
  (recording fake asserts emission on success + compute failure).

**Not yet built:** the real `Observability` adapters (Prometheus, structured logging) — future
work behind the ABC; `get_observability` returns `NoOpObservability` when enabled for now.

**Phase-4 notes:** **config vs wiring split** — `Settings` live in `app/core/config.py`; the
composition *factories* live in **`app/factories.py`** (they're wiring, not config). **Wire once,
not per request** — the API `lifespan` and `run_worker.py` build the repo/queue/orchestrator
**once** via shared `make_repository`/`make_queue`/`build_*` helpers in
`api/v1/dependencies.py` and store the service on `app.state`; the request dep
`get_ingestion_service(request)` just returns it (tests override it). This intentionally
replaces the spec's illustrative per-request DI (which would reconnect the DB each call).
The API process only *enqueues* root jobs; downstream fan-out runs in the worker process's
orchestrator. `DocumentStatusResponse.jobs` is `list[dict] | None` (repo's `document_jobs`
returns a list, not the spec's `dict`). The API's `create_app()` needs no env (settings are
read lazily in the lifespan, which httpx's ASGI transport doesn't trigger).

Implementation notes: `_upsert_document` is a **concrete portable** base method (not a
per-dialect hook — `ON CONFLICT` → portable select-then-update/insert); `store_document`
**deletes the doc's chunks on (re)store** so the distributed Load-then-Chunk flow is
idempotent — and `SqliteRepository.__init__` enables `PRAGMA foreign_keys=ON` (per
connection) so the cascade actually removes the chunks' embeddings; schema creation
uses two hooks (`_before_create_schema`/`_after_create_schema`) so pgvector's extension
is made before tables and indexes after; stages run `validate()` inside
`super().__init__()`, so a stage must set any attrs `validate()` reads **before** super;
`EmbedStage` lazy-loads sentence-transformers (tests set `stage._model` to a fake
`encode`; e2e uses a `FakeEmbedStage` in the registry). Use `datetime.now(UTC)`, not the
spec's `datetime.utcnow()` (deprecated on 3.12).

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
- **Job granularity (D2):** one chunk per job. The domain depends on two role-segregated
  ports — **`JobEnqueuer`** (`enqueue`, the producer, used by the orchestrator) and
  **`JobConsumer`** (`set_handler` / `run`, the consumer runtime); the worker is a registered
  handler, not a puller. **`PgQueuerJobQueue`** is the only file that imports pgQueuer (pgQueuer owns
  SKIP LOCKED / retries / NOTIFY / dead-lettering); **`InMemoryJobQueue`** is the test
  double — the whole flow runs on it + SQLite with no Postgres/pgQueuer. The consumer
  hands the worker a **`Batch`** (`ingestion/models.py`) — a **homogeneous** unit (all jobs
  share one `stage_name`; `Batch`'s constructor enforces it, so the worker never re-checks).
  Today every dispatch is one job; batch dispatch must group same-stage claims (restoring
  cross-document batching). Compute batch ⊥ the sink's persistence batch (two-tier
  batching). Keep the port a delegating seam — never reimplement queue mechanics in it.
- **Three-layer split (D3):**
  - **Worker = compute only.** A pure job handler (holds only the coordinator, no queue,
    no run loop — the composition root registers `worker.handle_batch` on a `JobConsumer`
    and runs the consumer's loop): `handle_batch(batch)` gets a homogeneous `Batch`, gets the
    stage via `BatchCoordinator.get_stage(batch.stage_name)` (the DAG's **long-lived**
    instance — so EmbedStage's model loads once, not per job; the worker holds no stage
    registry and the job carries no `stage_config`), runs it, and reports to a
    **`BatchContext`** from `begin_batch(batch)` (a per-batch unit of work —
    `app/domains/ingestion/batch.py`): `ctx.submit()` the results, then `ctx.complete()`
    (or `ctx.fail()` on compute error and re-raise). The worker depends **only** on the
    two batch ABCs — never the orchestrator, repo, queue, or `ResultSink`. Raising →
    queue requeues (recovery).
  - **`ResultSink` = persistence (D4).** Output-side sink; batches and writes results.
    Kept pure and **composed inside the `BatchContext`** — the worker never touches it.
    `async finalize() -> FinalizationOutcome`. Replaces the spec's `BatchingWrapper`.
  - **Orchestrator = `BatchCoordinator` + lifecycle + DAG walking (D5).** `begin_batch`
    records `processing` and builds the context; `ctx.complete()` finalizes the sink,
    records status, and enqueues downstream jobs — or records `failed` and **raises**
    so the queue requeues. (`PipelineOrchestrator` also owns `ingest_documents`.)
    Dependencies are implicit (downstream enqueued only after upstream persists).
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
`Iterator`/`Iterable`/`Callable` from `collections.abc`; keep `Any`/`Literal` from
`typing`. The spec's code blocks already follow this.

**Interfaces convention:** prefer **ABCs** (`abc.ABC` + `@abstractmethod`) over
`typing.Protocol` for interfaces/ports — team preference. Implementations inherit the
ABC explicitly (e.g. `JobEnqueuer`/`JobConsumer`, `DocumentRepository`).

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
`pytest`, `pytest-asyncio`) plus **`fastapi` + `httpx`** (Phase 4) are installed in the env.
The Postgres/pgQueuer/embed backends are optional extras in `pyproject.toml` (`postgres`,
`queue`, `embed`) and are **NOT installed** — keep the SQLite test path free of those imports
(the `postgres_repository` and `PgQueuerJobQueue` modules import their heavy deps lazily, and
`api/v1/dependencies.py` imports the Postgres repo / pgQueuer adapter lazily inside the
`make_*` builders). The API tests run on `InMemoryJobQueue` + SQLite (DI overridden).

Tests mirror `app/` under `tests/` (e.g. `tests/domains/ingestion/`, `tests/api/v1/`); the
suite runs entirely on SQLite + InMemory queue (no Postgres/pgQueuer needed).
