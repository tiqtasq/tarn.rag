# Codebase review — findings

A full read-through of `tarnrag/` (≈6.6k lines, ~70 modules) looking for code smells, duplication,
DRY violations, and dead code. The codebase is, overall, clean, consistently structured, and unusually
well-documented; the items below are refinements, not defects.

This file lists the issues I did **not** fix directly — either because they're judgment calls (removing
documented API, adding to a shared contract), larger refactors that deserve their own review, or things
where "keep as-is" is a defensible choice. The fixes I *did* apply are summarized at the bottom.

Severity legend: **[H]** worth doing soon · **[M]** worth doing · **[L]** cosmetic / optional.

---

## 1. Duplication / DRY

> **Update:** §1.1–1.5 are now **implemented** in this PR (all behavior-preserving; full suite green).
> §1.6 is **won't-fix** (the two helpers serve different consumers and can't disagree in practice). The
> per-item write-ups below are kept as the rationale/record.

### 1.1 [M] `hydrate` assembles `ChunkRecord` twice (sqlite vs postgres)
`SqliteRepository.hydrate` (`storage/repository/sqlite.py:184`) and `PostgresRepository.hydrate`
(`storage/repository/postgres.py:71`) are ~40 near-parallel lines each: select chunk⨝document, fetch
methods per chunk, build a `ChunkRecord` from the same nine fields + provenance, preserving input order.
The provenance fetch (`_chunk_provenance`) is *already* shared in the base; the record assembly is not.

- **Why it persists:** SQLite deliberately uses raw SQL (positional `r[0..8]`) and Postgres uses Core
  (`Mapping` rows) — see the `sqlite.py` module docstring on keeping the vec0/FTS layer "as one block."
- **Recommendation:** extract a base helper `_chunk_record(chunk_id, fields, methods, provenance) ->
  ChunkRecord` (or have both dialects produce a uniform row mapping that a shared assembler consumes),
  so the `ChunkRecord(...)` field list lives in one place. The two queries can stay dialect-specific.
- **Not auto-fixed because:** unifying touches the deliberate raw-SQL/Core split; wants a deliberate call.

### 1.2 [M] 1→1 stages re-thread `PipelineItem` fields by hand (latent field-drop bug)
`MapperStage.process` (`ingestion/pipeline.py:80`), `FilterStage.process` (`:101`), and
`EnrichStage.process` (`ingestion/enrichment/enrich.py:46`) each rebuild the item with
`PipelineItem(content=…, metadata={**item.metadata, …}, document=item.document, provenance=item.provenance)`.

- **Smell:** every flow-through field must be hand-listed in three places. This already bit the codebase
  once — when `document` / `provenance` were added to `PipelineItem` (`contracts/dtos.py:70`), all three
  1→1 stages had to be edited to thread them. The next added field is a silent drop waiting to happen.
- **Recommendation:** add one constructor on the DTO, e.g.
  ```python
  def derive(self, *, content: str | None = None, metadata: dict | None = None) -> PipelineItem:
      """A fresh item (id reset) carrying forward document/provenance; only content/metadata change."""
      return PipelineItem(content=self.content if content is None else content,
                          metadata=self.metadata if metadata is None else metadata,
                          document=self.document, provenance=self.provenance)
  ```
  and have the three stages call `item.derive(content=…, metadata={**item.metadata, **updates})`. New
  flow-through fields then update one method, not three.
- **Note:** `model_copy(update=…)` is *not* a drop-in — it preserves `id`, but these stages intentionally
  emit a fresh (`id=None`) item that the worker re-ids (`worker.py:54`). `derive()` keeps that semantics.
- **Not auto-fixed because:** adds a method to a shared contract type — a design choice worth your sign-off.

### 1.3 [M] sha256 hashing is scattered across three modules
- `core/hashing.py:8` — `content_hash(text)` = `sha256(text.encode()).hexdigest()`.
- `ingestion/engine.py:54,58` — `_sha256_bytes(data)` and `_sha256_file(path)` (streaming).
- `core/embedder.py:108,134` — `_tokenizer_sha256()` (file bytes) and `config_fingerprint()` which
  open-codes `sha256(blob.encode("utf-8")).hexdigest()` — i.e. exactly `content_hash(blob)`.

Four sha256-hexdigest sites, three modules, no shared home. `engine._sha256_bytes(text.encode())` is
literally `content_hash(text)`.
- **Recommendation:** give `core/hashing.py` the full set — `sha256_hex(data: str | bytes)` and
  `sha256_file(path)` — and have `engine.py` and `embedder.config_fingerprint` reuse them. Keep
  `content_hash` as the semantic alias for the dedup key.
- **Not auto-fixed because:** centralizing 1-liners is a mild call, and `embedder`'s hashing is arguably
  "fingerprint, kept local." Low risk if you want it done.

### 1.4 [L] In-process worker is attached in two places
`IngestionEngine.create` (`ingestion/engine.py:134`) and `run_worker` (`:382`) both do
`worker = IngestionWorker(orchestrator, obs); queue.set_handler(worker.handle_batch)`. Minor; a private
`_attach_worker(queue, orchestrator, obs)` would remove the repeat.

### 1.5 [L] `StructureAwareChunker` builds `TempChunk(+ChunkProvenance)` in 5 spots
`_leaf` / `_table_leaf` / `_oversize` / `_parent` / the fallback in `chunk()` (`chunking/structure_aware.py`)
all construct the same shape. A small `_temp_chunk(text, elements, header_path, *, level=0, table=None)`
could absorb the common case. Borderline — the explicit variants currently read well; only do it if it
nets out simpler.

### 1.6 [L] Two "infer kind from extension" helpers
`IngestionEngine._infer_source_type` (`engine.py:359`, maps ext → `pdf/text/html` via `_SOURCE_TYPES`) and
`LoadAndParseStage._infer_kind` (`extraction/load_parse.py:106`, returns the raw extension). Different
outputs for different consumers, so not strictly duplication, but the two extension-parsing paths are
worth a glance to confirm they can't disagree (e.g. on `.htm`).

---

## 2. Dead / unused code

> Note: several "unused" reads below are listed in `doc/FUNCTIONAL_REQUIREMENTS.md` as the intended
> repository read surface, so they're documented API rather than accidental cruft. Flagged so you can
> decide "keep as public API" vs "trim + update the doc," but **not** removed here.

### 2.1 [M] `WorkerSettings` is entirely unused
`core/config.py:90` defines `WorkerSettings(queue_timeout_seconds, concurrency)` and `Settings.worker`
(`:136`), but **nothing reads them** — not the distributed worker (`run_worker`), not the queue. Dead
config. Either wire them into the distributed consume loop (their evident intent) or drop the class +
field until the distributed path needs tuning.

### 2.2 [L] `AppSettings.name` / `AppSettings.version` are unused
Only `settings.app.debug` is read (`engine.py:130`). `name` / `version` (`config.py:40-41`) have no
readers; the class docstring already calls itself "largely vestigial since the FastAPI layer moved out."

### 2.3 [M] `query_chunks` and `health_check` have zero references anywhere
`DocumentRepository.query_chunks` (`storage/repository/base.py`, the `filters`-dict query) and
`health_check` are not called by any source **or test**. `query_chunks` is named in the FR doc; `health_check`
is conventional infra. Recommendation: either cover/use them or remove (and update the FR doc for
`query_chunks`).

### 2.4 [L] Repository reads used only by tests
`get_document` (4 tests), `get_chunk` (1), `get_chunks_by_document` (8), and
`store_document_with_chunks` (20) have no production callers. They're a reasonable repository read/test
surface (and `store_document_with_chunks` is a genuinely handy test fixture), so probably keep — just
confirming they're intended public API, not leftovers.

### 2.5 [L] `ComponentFactory` methods used only by tests
`create_many`, `validate`, and `config_adapter` (`core/components/component_factory.py`) have no callers
outside `tests/core/components`. For a reusable component framework this is defensible public surface
(esp. `config_adapter`'s discriminated-union validation), but if the framework is meant to stay minimal,
they're trim candidates.

### 2.6 [L] `Registry` convenience methods mostly unused
`classes()` is used (by `config_adapter`) and `tags()` by one test; `items()`, `__contains__`,
`__iter__`, and `clear()` (`core/components/registry.py`) appear unused. Collection-completeness API —
harmless, but trimmable if you prefer YAGNI.

---

## 3. Minor smells / observations

### 3.1 [L] `build_index_meta(embedder: Any)` is untyped
`contracts/index_meta.py:22` takes `embedder: Any` to keep `contracts` a dependency-free leaf package
(it can't import `Embedder` from `core`). Defensible, but a local `Protocol` with `embed_meta() ->
dict[str, str]` would document + type the one method it calls without re-introducing the dependency.

### 3.2 [L] `_OrchestratorBatchContext` reaches into orchestrator privates
`ingestion/orchestrator.py:129` calls `self._orch._record/_make_job/_enqueue` and reads `_orch.obs/dag`.
Tight, but the two classes are co-located and the coupling is documented ("backed by the orchestrator's
repo/DAG/queue"). Acceptable; noted for awareness.

### 3.3 [L] Enabling observability yields a no-op
`IngestionEngine.create:108` sets `obs = NoOpObservability() if settings.observability.enabled else None`
— so turning observability *on* installs a do-nothing adapter (real adapters are future work, per the
`observability.py` docstring). Expected given the phase, but a reader may find the inversion surprising.

### 3.4 [L] Columnar whitespace alignment in `worker.py`
`ingestion/worker.py:42-51` aligns assignments with runs of spaces (`stage_name  = …`). Not flagged by
the repo's ruff profile (E2xx off), but inconsistent with the rest of the codebase. Cosmetic.

---

## Fixed in this pass (behavior-preserving; full suite 205 passed / 1 skipped)

1. **DRY: portable-upsert idiom.** Extracted `DocumentRepository._upsert(conn, table, key_column,
   key_value, values, insert_extra=None)`; collapsed the three verbatim "UPDATE-else-INSERT" copies in
   `write_index_meta`, `record_job`, and `_upsert_document` to one-liners. (`storage/repository/base.py`)
2. **DRY: `Pipeline` reuses `ComponentFactory.create_as`.** `Pipeline._build_children` and `from_spec`
   hand-rolled the same "build + isinstance + raise TypeError" that `create_as` already provides (and that
   the retrieval pipeline already uses). Replaced with `create_as`. (`ingestion/pipeline.py`)
3. **Style: merged split `from tarnrag.contracts import` lines** into one import each in `sqlite.py`,
   `result_sink.py`, and `orchestrator.py`.
4. **Doc: fixed a stale docstring** in `create_sink_registry` — it referenced a non-existent
   `PipelineOrchestrator.make_sink` and the wrong key (`.name`); the lookup is in `begin_batch`, keyed by
   the stage `.tag`. (`ingestion/result_sink.py`)
