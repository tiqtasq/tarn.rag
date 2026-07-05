# Pre-feature fix list — pay down before adding new components

Findings from a full codebase review (2026-07-02, ~11K library lines + 419 tests). The goal of this
list: fix what would **proliferate** if new features are built on top of it — real bugs first, then
latent/dead surface that invites cargo-culting, then code↔doc inconsistencies, then hygiene.

Each item is small enough to be one PR (or part of one themed PR); file:line references are as of
this review.

**Status (2026-07-05):** executed as a PR series — #102 (A1, A4), #103 (A2, A6, A7, A8),
#104 (A9), #105 (A3 + queue wording), #106 (B1 → writer added, A5, B3), #107 (D2),
plus the docs/packaging PR carrying C1–C5 and this status block. Open: B4, B5 (graduated to the
strategic roadmap), B2/D1/D3 (decisions deferred), B6 (real claim-side batching → roadmap P9),
and B7 below.

---

## A. Real bugs / correctness (fix first)

### A1. `LLMSettings.temperature` / `max_tokens` are dead config
`llm_api.py` stores `default_temperature` (lines 43, 147) but **never reads it** — `complete()`
sends `prompt.temperature` directly (the `Prompt` dataclass default `0.0` always wins). Likewise
`prompt.max_tokens or self.default_max_tokens` (lines 53, 162) never falls through, because
`Prompt.max_tokens` defaults to `1024` (truthy). So `LLM__TEMPERATURE` / `LLM__MAX_TOKENS` are
silently ignored for every built-in prompt (all reasoners/checkers construct `Prompt` without
sampling knobs). The `LLMSettings` docstring ("the defaults a `Prompt` may override per call")
describes the intended behavior — the code inverts it.
**Fix:** make `Prompt.max_tokens` / `temperature` default to `None` and have each backend fall back
to its configured defaults. One test per backend.

### A2. `TarnRag.ingest` id collision on file stems
`tarnrag.py:102` uses `p.stem` as the caller `source_id`. Two files with the same stem
(`a.md` + `a.txt`, or `x/readme.md` + `y/readme.md` in one directory tree) silently **upsert over
each other** — the second replaces the first's chunks with no report issue.
**Fix:** derive a collision-free id (relative path, or stem + short content-hash suffix), or at
minimum detect duplicates in `_expand` and emit an `Issue` instead of silently overwriting.

### A3. Stage identity breaks across processes (distributed mode)
`PipelineStage.__init__` auto-names unnamed stages `f"{class_tag}-{N}"` with a **process-global
counter** (`ingestion/pipeline/pipeline.py:33–44`). `job.stage_name` carries that name through the
queue, and the consuming worker resolves it against **its own** DAG's instance names. Any
difference in construction order/count between producer and worker processes (e.g. one process
builds a second engine or an eval pipeline first) shifts the counter → `get_stage()` raises
"Stage not found" for every job. Works today only by convention.
**Fix:** name stages deterministically from their position *within the pipeline*
(`{class_name}-{index}`), not a global counter. Add a test that builds two engines in one process
and round-trips a job name.

### A4. Anthropic backend has no retry/backoff
`OpenAILanguageModel._post_retrying` handles 429/5xx + `Retry-After` (llm_api.py:173–207);
`AnthropicLanguageModel.complete` calls `messages.create` bare (llm_api.py:50–70). A long eval or
batch run on Claude dies on the first sustained 429/529. (The SDK's built-in retries are 2 attempts
— not configured, and not equivalent.)
**Fix:** mirror the retry policy (or configure the SDK's `max_retries` + honor overloaded errors)
so both providers survive batch runs equally.

### A5. `hydrate()` N+1 on `method_chunks`
Both dialects query `method_chunks` **once per chunk id** inside the hydrate loop
(`sqlite.py:290–296`, `postgres.py:172–177`). Every search pays `len(fused)` extra round-trips for
a table that is (today — see B1) always empty.
**Fix:** one `IN (...)` query grouped by `chunk_id` (the `_child_rows_by_chunk` helper already does
exactly this pattern for cells/annotations).

### A6. `TarnRag.close()` leaks distributed resources
`tarnrag.py:80–82` only disconnects the repository. In distributed mode the lazily-built ingestion
engine holds a `PgQueuerJobQueue` with an asyncpg **pool** that is never closed. Related asymmetry:
`GenerationEngine.aclose()` closes the *shared* retrieval engine (generation/engine/engine.py:70–74)
— harmless standalone, wrong under the facade's shared-store composition.
**Fix:** `TarnRag.close()` should close what it built (queue pool included); the generation engine
should only close resources it owns.

### A7. Embedded ingest re-entrancy
`IngestionEngine._submit` calls `queue.run()` per ingest (engine.py:351–358). Two concurrent
`ingest()` calls on one engine interleave two drain loops over one `asyncio.Queue` — jobs get
split arbitrarily between the loops and each `run()` returns when *it* sees an empty queue.
**Fix:** serialize with an `asyncio.Lock` (or document single-flight and assert).

### A8. Blocking hashing on the event loop
`ingest_paths` computes `sha256_file(path)` synchronously per file (engine.py:196); for large
files/directories this stalls the loop (streams already use `asyncio.to_thread` for the copy).
**Fix:** `await asyncio.to_thread(...)` the hash (or hash during the staged copy).

### A9. `documents.source_kind` was never persisted (found post-review, fixed with A2's follow-up)
The ingest path stamps ``metadata["source_type"]`` (the caller's input hint), but the repository's
provenance map reads ``metadata.get("source_kind")`` — a key mismatch, so the ``source_kind`` column
always held its ``"document"`` default. The D3 stringly-typed-metadata hazard, materialized.
**Fixed:** ``LoadAndParseStage`` stamps the extractor-confirmed ``document.source_kind`` into the
outgoing metadata (the persisted key); ``TarnRag.ingest`` uses the now-real stored kind for a
cross-call guard — replacing a document whose stored kind differs from the incoming file's
(``a.pdf`` then ``a.md``) is reported as a WARNING instead of passing silently.

---

## B. Latent / dead surface — decide: implement or remove

Leaving these ambient teaches new code to imitate them.

### B1. `method_chunks` has no writer
The table (base.py:255–268), `MethodRef`, `Query.scope`, the method-scope SQL filters in both
dialects, and `ChunkRecord.methods` all exist — but **nothing ever inserts a row**. The whole
scope-filtering path is untestable end-to-end and `hydrate` pays for it (A5).
**Decide:** add the small writer API (`register_method_bundle(method_id, version, chunk_ids)`) + one
E2E test, **or** move the capability to the spec doc and drop the query-path plumbing until needed.

### B2. `ChunkStore.update_chunk_metadata` is a no-op on the only implementation
base.py:414–418 (deliberate — the §8 chunk metadata bag was dropped; see the deferred-metadata
note). A port method that silently does nothing is a trap for the next enricher author.
**Decide:** either re-add the chunk metadata/position column (it's also wanted for PDF highlighting)
or remove the method from the port until it can be honored.

### B3. `Embedding` DTO fields that are never persisted
On SQLite, `id/model/dimension/metadata` are dropped (documented); on **Postgres `metadata` is
dropped too** — the `embeddings` table has no metadata column (base.py:269–283), contradicting the
docstring in `contracts/dtos.py:122–128`. `EmbedStage` dutifully fills
`metadata={"source_id": ...}` (embed.py:66) that no backend stores.
**Fix:** trim the DTO (or add the column); correct the docstring either way.

### B4. Duplicate embedder instances per process
`EmbedStage` builds its own `Embedder` (embed.py:78–85) while the composition root builds another
for index identity + queries (`TarnRag.open` / both engines' `create`). On ONNX that's **2× model
memory and load time** in every embedded process; on API embedders, two clients.
**Fix:** let the pipeline build accept the shared embedder (inject through
`IngestionEngine.create → build_pipeline`), falling back to self-build for standalone specs.

### B5. Query classifier annotations are recorded but unread
Classifiers append rich `Annotation`s (quoted spans, identifiers — classifier.py:144–155) but the
only consumer of classification is the router's `query_type` string. In particular the **sparse
path ignores exact-match intent**: `_fts_query` ORs bare tokens (sqlite.py:260–264), so a quoted
phrase or identifier the classifier detected is not used for phrase matching.
**Decide:** either wire annotations into `sparse_search` (phrase queries — also a quality win, see
the roadmap) or trim the unused annotation payload to what routing needs.

### B6. Distributed mode never batches
`PgQueuerJobQueue.set_handler` wraps every claimed job in a single-job `Batch`
(queue.py:150–152), so the bulk-ingest win (batched dispatch + bulk persist, ~2.6×) is
**embedded-only**. The stale comment at queue.py:21–23 ("Today every dispatch is a single job")
now misdescribes the InMemory queue as well.
**Fix (min):** correct the comment and document the asymmetry. **Fix (real):** claim-side batching
in the pgQueuer adapter (group claims by stage within a wait window) — also listed in the roadmap.

### B7. Align the Postgres `embeddings` schema with the 1:1 chunk-keyed model *(added 2026-07-04; DONE — schema v2)*
**Done (2026-07-05):** `embeddings(chunk_id PK, vector)` with `ON CONFLICT` upsert; DTO slimmed to
`(chunk_id, vector)`; `SCHEMA_VERSION` bumped to `"2"` (old stores are refused → rebuild by
re-ingest, per the agreed no-migrations stance at this stage). Original item:
The SQLite side is the considered design (`vec_chunks(chunk_id PK, vector)`); the Postgres table
predates the `index_meta` fingerprint gate and kept a surrogate `id` PK plus per-row
`model`/`dimension` copies — redundant (`index_meta` + the `Vector(dim)` type already carry them)
and weaker (`chunk_id` is only indexed, not UNIQUE, though the system semantics are strictly 1:1).
Target: `embeddings(chunk_id PRIMARY KEY, vector)`, and the `Embedding` DTO slims accordingly.
**Blocked on D1** (it's a schema change to a backend that may hold data — needs the migration
stance first).

---

## C. Code ↔ documentation inconsistencies

### C1. CLAUDE.md drift (several)
- **Dialect hooks list is stale** (CLAUDE.md "Database agnosticism"): names `_upsert_document` and
  `_create_dialect_objects` as the subclass hooks. `_create_dialect_objects` doesn't exist and
  `_upsert_document` is shared in the base. Actual hooks: `_driver_url`, `_vector_type`,
  `_encode_vector`/`_decode_vector`, `_before_create_schema`/`_after_create_schema`,
  `_index_chunk_text`, `_clear_chunk_index`, `_count_doc_embeddings`,
  `_embedding_counts_by_document`, plus the `RetrievalStore` port (`dense_knn` / `sparse_search` /
  `hydrate`) — `sparse_search` isn't mentioned at all.
- **Package layout is stale**: `eval/` is described as "metrics · dataset · harness · generation"
  — missing `benchmarks.py`, `benchmark_runner.py`, `layout.py`; `_cli.py`, `core/parsing.py`, and
  `scripts/run_benchmarks.py` / `run_layout_eval.py` are absent; retrieval's
  `retrieval_engine_protocol` is unlisted.
- **`InMemoryJobQueue` is called a "test double"** (CLAUDE.md queue-ports section, and its own
  docstring "In-process job queue for tests", queue.py:52) — it is the **production embedded-mode
  queue**. Describe it as such.

### C2. README extras table incomplete
README lists 8 extras; pyproject defines 10 — `openai` (the OpenAI-compatible LLM backend) and
`benchmarks` (HF `datasets`) are missing from the table, and the README's description of `all`
omits `openai` (pyproject includes it).

### C3. `requirements.txt` duplicates pyproject with different pins
Root `requirements.txt` mixes base deps, extras, and dev tools with pins that pyproject doesn't
have (`pgvector==0.4.2`, `asyncpg==0.31.0`, …). Two sources of truth will drift.
**Fix:** delete it (pyproject + extras are the interface), or generate it and say so in a header.

### C4. Stray tracked artifact + `docs/` vs `doc/` confusion
`docs/bench_hotpotqa_pool.db-journal` is **tracked in git** (an SQLite journal file). The
`.gitignore` covers `*.db` but not `*.db-journal`. Separately, benchmark index DBs live in `docs/`
while design docs live in `doc/` — an unfortunate near-collision.
**Fix:** `git rm --cached` the journal; ignore `*.db-journal` (or the whole `docs/` data dir);
consider renaming the data dir to `bench_data/` or similar.

### C5. Smaller docstring drift
- `core/engine/config.py:10` — "Pipeline composition lives in `IngestionEngine.build_pipeline`";
  composition actually lives in `Settings._fill_default_components` (build_pipeline just reads it).
- `retrieval/types.py:43` — `sparse_k` comment "used in Step B (sparse retriever)" references a
  plan step; the sparse retriever shipped long ago.
- `contracts/dtos.py:122–128` — the `Embedding` Postgres claim (see B3).
- `LLMSettings` docstring vs behavior (see A1).

---

## D. Structural debt to settle before schema-touching features

### D1. No migration story
`connect()` runs `metadata.create_all` only; `IndexMeta.SCHEMA_VERSION` **refuses** an old index
but nothing upgrades one — the only path is delete + re-ingest. Acceptable for eval corpora;
not for a customer store. Before any feature that alters the schema (native table ingest, chunk
metadata bag, annotations changes), pick a stance: lightweight versioned migrations, or an
official `rebuild` command with progress reporting — and write it down.

### D2. `DocumentRepository.base` is becoming a god object
916 lines implementing four ports (`ChunkStore`, `RetrievalStore`, `JobStatusSource`,
`DocumentFactsSource`) plus admin + stats + provenance assembly. It's still coherent, but the next
few features (table-native ingest, annotations queries) will land here by gravity.
**Fix (cheap):** split the provenance/read-assembly half (`_attach_*`, `_chunk_provenance`,
`_create_chunk_record`, `chunk_provenance.py` helpers) into a companion module before adding to it.

### D3. Metadata-bag string conventions
The in-flight `PipelineItem.metadata` threads `doc_id` / `chunk_id` / `source_id` / `content_hash`
/ `parent_ordinal` by bare string key across stages, sinks, and the repository
(`_DOC_PROVENANCE`/`_CHUNK_PROVENANCE` lambda maps centralize *some* of it). One typo = silent
data loss. **Fix (cheap):** a `contracts/keys.py` with named constants (and use them everywhere);
the full typed-payload refactor is not warranted.

---

## Suggested order

1. **PR "config + LLM correctness"** — A1, A4 (+ the LLMSettings docstring).
2. **PR "facade correctness"** — A2, A6, A7, A8.
3. **PR "distributed identity"** — A3 (+ C1 queue wording, B6 minimal).
4. **PR "repository hygiene"** — A5, B1 (decision), B3, D2.
5. **PR "docs + packaging sync"** — C1–C5, C4 git cleanup.
6. **B4 (shared embedder)** and **B5 (phrase-aware sparse)** graduate into the strategic roadmap —
   they are quality/perf features, not just fixes.
