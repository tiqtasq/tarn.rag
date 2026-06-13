# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**tarnrag** — a composable, DAG-based RAG **ingestion + retrieval** library: documents in, a
queryable vector index out. There is no HTTP layer here; a consuming REST API lives in the
separate **tiqtasq.backend** repo. The public surface is two engines:

```python
from tarnrag import IngestionEngine, RetrievalEngine, run_worker, Query, DocumentStatus
```

Data flow:

```
IngestionEngine.create() → ingest_paths/content/streams → PipelineOrchestrator (walks the DAG,
  creates jobs) → queue (InMemory in embedded mode · pgQueuer in distributed) → IngestionWorker(s)
  → stages (load → clean → chunk → enrich → embed) → §8 SqliteIndexStore + DocumentRepository (job_status)

RetrievalEngine.create() → search/search_text → embed query → sqlite-vec KNN → hydrate → ranked results
```

Design specs: `doc/FUNCTIONAL_REQUIREMENTS.md` (ingestion pipeline) and
`doc/ModusQ_RetrievalSubsystemSpec.md` (retrieval subsystem).

## Using the engines

Both facades are built from `Settings` — no hand-wiring:

```python
from tarnrag import IngestionEngine, RetrievalEngine

# Ingestion (MODE='embedded' default → runs the whole pipeline in-process):
engine = await IngestionEngine.create()              # or .create(settings)
ids = await engine.ingest_paths(["/data/spec.pdf"])  # -> list[str]; also ingest_content / ingest_streams
st = await engine.status(ids[0])                     # -> DocumentStatus
await engine.aclose()                                # or: async with await IngestionEngine.create() as engine: ...

# Retrieval (sync; reads the §8 index ingestion built):
with RetrievalEngine.create() as r:                  # validates schema + embedding fingerprint
    hits = r.search_text("how do I inspect a tank?", top_k=8)
```

- **`create()` is the entry point** for both engines (reads `Settings`). Bare constructors and
  `RetrievalEngine.open(...)` are low-level seams for tests / custom wiring.
- **`MODE='embedded'`** (default) runs in-process (InMemory queue) — each ingest call processes to
  completion. **`MODE='distributed'`** enqueues to pgQueuer; run `python run_worker.py` as one or
  more separate consumer processes.
- **Async vs sync is deliberate:** ingestion is **async** (async SQLAlchemy + pgQueuer); retrieval
  is **sync** (sync `sqlite3`/sqlite-vec + ONNX, matching a future C++ port). For event-loop callers
  retrieval also exposes `asearch` / `asearch_text` (thread-offloaded). An `a`-prefix marks the async
  variant of a sync method; lifecycle follows the idiom (`aclose` + `async with` vs `close` + `with`).
- **Jobs are internal** — never in the public surface. The per-job breakdown is available only via
  the debug-gated `IngestionEngine.document_jobs` (raises unless `APP__DEBUG`).

## Package layout

```
tarnrag/
├── core/         # infra only: config, exceptions, observability
├── embedder.py   # Embedder ABC + OnnxEmbedder (shared by both engines)
├── storage/      # persistence layer
│   ├── models.py · chunk_store.py · index_store.py · status.py
│   └── repository/   # base.py (DocumentRepository) · postgres.py · sqlite.py
├── ingestion/    # engine, worker, pipeline, orchestrator, queue, batch,
│                 #   result_sink, models, types, factories, stages/
└── retrieval/    # engine, types
run_worker.py            # distributed consumer entry: asyncio.run(run_worker())
scripts/fetch_model.py   # fetch the ONNX model + tokenizer into the model dir
```

`tarnrag/__init__.py` re-exports the public API; consumers import from `tarnrag`, not the submodules.

## Architecture — the decoupled seams (preserve these)

- **Distributed execution, one job per `(item, stage)`.** Fan-out stages enqueue one downstream job
  per output item (chunker → *m* chunks → *m* embed jobs). In-flight items travel **inline in the
  job payload** — no temp item store.
- **Queue ports.** Two role-segregated ABCs: **`JobEnqueuer`** (`enqueue`, the producer, used by the
  orchestrator) and **`JobConsumer`** (`set_handler` / `run`, the consumer runtime). The worker is a
  registered handler, not a puller. **`PgQueuerJobQueue`** is the only file that imports pgQueuer
  (which owns SKIP LOCKED / retries / NOTIFY / dead-lettering); **`InMemoryJobQueue`** is the
  in-process double (embedded mode + tests, no Postgres). The consumer hands the worker a **`Batch`**
  (`ingestion/models.py`) — homogeneous (all jobs share one `stage_name`, enforced by the
  constructor). Keep the port a delegating seam — never reimplement queue mechanics in it.
- **Three-layer split:**
  - **Worker = compute only** (`ingestion/worker.py`). A pure handler holding only the coordinator
    (no queue, no run loop): `handle_batch(batch)` gets the stage via
    `BatchCoordinator.get_stage(...)` (the DAG's **long-lived** instance, so `EmbedStage`'s model
    loads once), runs it, and reports to a **`BatchContext`** from `begin_batch(batch)`:
    `ctx.submit()` results, then `ctx.complete()` (or `ctx.fail()` + re-raise on compute error).
    Depends **only** on the two batch ABCs — never the orchestrator/repo/queue/`ResultSink`.
    Raising → the queue requeues (recovery).
  - **`ResultSink` = persistence** (`ingestion/result_sink.py`). Output-side; batches and writes
    results. Composed inside the `BatchContext` — the worker never touches it.
    `async finalize() -> FinalizationOutcome`.
  - **Orchestrator = `BatchCoordinator` + lifecycle + DAG walking** (`ingestion/orchestrator.py`).
    `begin_batch` records `processing`; `ctx.complete()` finalizes the sink, records status, and
    enqueues downstream jobs — or records `failed` and **raises** so the queue requeues. Also owns
    `ingest_documents`. Downstream is enqueued only after upstream persists (implicit dependencies).
- **Stages stay pure** — no DB/queue access (the worker observes them). All stages subclass
  `PipelineStage`; prefer a typed base: `MapperStage` (1→1, `map()`), `ChunkerStage` (1→N,
  `chunk()`), `FilterStage` (1→{0,1}, `should_keep()`) — these merge metadata automatically
  (subclasses return *updates*, never mutate the incoming dict).
- **Pluggable PDF parsers (per-request).** `LoadAndParseStage` holds a registry (`stages/parsers.py`:
  `pypdf` default, `pdfplumber`). The available set is stage config; a request picks one via the
  `parser` argument, which the engine writes to `metadata['parser']` (item data flowing inline, not
  per-job config). Unknown parser → rejected at the engine edge. HTML uses `load_html` (BeautifulSoup).
- **The shared embedder** (`embedder.py`). `Embedder` ABC + `OnnxEmbedder` (prefix → tokenize → ONNX
  CPU → mean-pool(mask) → L2; lazy `onnxruntime`/`tokenizers`). The **same** embedder embeds passages
  (ingestion) and queries (retrieval), guaranteeing pipeline identity; `config_fingerprint()` is
  recorded in `index_meta` and retrieval refuses to `open()` on mismatch. Model configurable via
  `settings.embedding` (default all-MiniLM-L6-v2); fetch artifacts with `scripts/fetch_model.py`.
- **The §8 index + status read model** (`storage/`). `SqliteIndexStore(ChunkStore)` is the retrieval
  index — a single SQLite file (`index_meta`, `documents`, `chunks`, `vec_chunks` via sqlite-vec,
  `fts_chunks` via FTS5, `method_chunks`) over **sync `sqlite3`**. `ChunkStore` is the persistence ABC
  both the index and the repository implement. `DocumentStatusReader` (`storage/status.py`) composes
  two **narrow ports** — `JobStatusSource` (the repo's `job_status`) + `DocumentFactsSource` (presence
  + counts, supplied by the index) — and owns the rollup (ISP: only this read model sees both).
- **Database agnosticism** (`storage/repository/`). All document storage goes through
  `DocumentRepository` (SQLAlchemy 2.0 Core, async): shared tables + dialect-agnostic CRUD in
  `base.py`; `postgres.py` / `sqlite.py` override only the hooks (`_driver_url`, `_vector_type`,
  `_encode_vector` / `_decode_vector`, `_upsert_document`, `_create_dialect_objects`,
  `vector_search`). Postgres uses pgvector; SQLite stores vectors as JSON + in-memory cosine
  (dev/small-scale). Selection is driven by `settings.database.document_url` (a `postgres` substring
  → Postgres, else SQLite). Put shared logic in the base, dialect specifics in the hooks.
- **Transactional guarantees & idempotency.** `store_document_with_chunks` / `store_chunks` /
  `store_embeddings` are atomic. Documents are keyed by `metadata['source_id']` (UNIQUE); re-ingesting
  upserts the document and replaces its chunks/embeddings (delete chunks → cascade removes
  embeddings) — re-runs never duplicate.
- **Two databases.** `settings.database.queue_url` (pgQueuer) is separate from
  `settings.database.document_url` (document/chunk/embedding storage). Never conflate them.
- **Observability is optional.** Core logic must work with `observability=None`; `NoOpObservability`
  for dev/test. Guard every `self.obs` call. (Real adapters — Prometheus, structured logging — are
  future work behind the ABC.)
- **Retrieval is dense-only.** `RetrievalEngine.search` = embed → sqlite-vec `dense_knn` → truncate
  `top_k` → `hydrate` → assemble (`score = -distance`). Sparse FTS5/BM25 + RRF fusion + license/scope
  filtering are planned behind the `Retriever` / `Fuser` seams (`dense_knn` already takes a `filter` arg).

## Conventions

- **Config** (`core/config.py`). `Settings` (pydantic-settings) nests per-component sub-models —
  `settings.embedding`, `settings.chunking`, `settings.index`, `settings.database`, `settings.worker`,
  `settings.observability` — read from env via the `GROUP__FIELD` convention (e.g. `EMBEDDING__MODEL`,
  `DATABASE__DOCUMENT_URL`). Cross-cutting `MODE`, `EMBEDDING_DIMENSION`, `UPLOAD_DIR` stay
  top-level/flat. A `model_validator` pins the backend: `distributed` requires Postgres +
  `DATABASE__QUEUE_URL`; `embedded` requires SQLite. See `.env.example`. Each component is built via a
  `create()` classmethod from its config slice (`OnnxEmbedder.create`, `SqliteIndexStore.create`,
  `DocumentRepository.create`).
- **Type annotations (Python 3.12):** builtin generics (`list`, `dict`, `tuple`, `type`) and
  `X | None` — never `typing.List` / `Dict` / `Optional`. Import `Iterator` / `Iterable` / `Callable`
  from `collections.abc`; keep `Any` / `Literal` from `typing`. Use `datetime.now(UTC)`.
- **Interfaces:** prefer **ABCs** (`abc.ABC` + `@abstractmethod`) over `typing.Protocol`;
  implementations inherit explicitly.

## Stack

Python 3.12 · **Pydantic v2** / pydantic-settings (`model_config = ConfigDict(...)` /
`SettingsConfigDict(...)`) · **SQLAlchemy 2.0 Core (async)** (asyncpg + `pgvector` for Postgres,
aiosqlite + numpy for SQLite) · **pgQueuer** (distributed queue) · **sqlite-vec** + FTS5 (the §8
index) · **ONNX** (`onnxruntime` / `tokenizers`) for embeddings · pytest. Heavy backends
(Postgres/pgQueuer, ONNX) import lazily so the SQLite/embedded path stays light — the Postgres repo
and pgQueuer adapter are imported only inside the relevant `create()`.

## Commands

Dev runs in the conda env **`tarn.rag`** (Python 3.12). Each Bash call is a fresh non-login shell, so
prefix with `conda run -n tarn.rag`:

```bash
conda run -n tarn.rag python -m pytest -q                                  # full suite
conda run -n tarn.rag python -m pytest tests/ingestion -q                  # a subset
conda run -n tarn.rag ruff check tarnrag tests                             # lint
conda run -n tarn.rag python -m py_compile $(find tarnrag -name "*.py")    # compile-check
```

The suite runs entirely on SQLite + InMemory queue (no Postgres/pgQueuer needed); the real ONNX
embedder test is gated on the model dir existing.
