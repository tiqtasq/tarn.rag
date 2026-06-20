# Codebase review — findings

**Date:** 2026-06-20 · A full read-through of `tarnrag/` (≈3.9k statements, ~80 modules) for code smells,
duplication, dead code, spec deviations, and test-coverage gaps.

Overall the codebase is in very good shape: **303 tests, 297 pass / 6 skip, ruff clean, 91% line
coverage**, consistent Component/Config idioms, narrow ports, and unusually thorough docstrings. The items
below are refinements and a few genuine latent bugs — not a system in trouble. No code was changed; this is
the list.

Severity legend: **[H]** worth doing soon · **[M]** worth doing · **[L]** cosmetic / optional.

> **Resolved since the previous review (2026-06-18):** the `_create_chunk_record` extraction (old §1.1),
> `PipelineItem.derive` (old §1.2), the centralized `core/hashing.py` — `sha256_hex` / `sha256_file` (old
> §1.3), and the shared `_attach_worker` helper (old §1.4) are all **done**. They're dropped from the list.

---

## 1. Correctness / spec deviations (the bugs worth fixing)

### 1.1 [H] RRF / identity fusion has no deterministic tie-break
`RRFFuser.fuse` (`retrieval/components/fuser.py:74`) sorts `sorted(scores, key=lambda cid: scores[cid],
reverse=True)` — score only. `IdentityFuser` doesn't tie-break either. The ModusQ spec makes a secondary
`chunk_id asc` tie-break **mandatory** (`ModusQ_RetrievalSubsystemSpec.md` §5.5, §9) because it is the
contract that makes the future C++ port return byte-identical orderings (R1). Today two chunks with equal
fused score order by dict-insertion, i.e. retriever order — non-deterministic across runs/ports.
- **Fix:** `sorted(scores, key=lambda cid: (-scores[cid], cid))`; same idea for the identity passthrough.
- **No test** asserts ordering on a fused tie — add one.

### 1.2 [H] License/scope filter is post-hydrate, not a pre-filter — scoped queries can under-return
`RetrievalPipeline.search` over-fetches `dense_k`/`sparse_k` (default 50), fuses, hydrates, then drops
disallowed chunks in `_passes` (`retrieval/pipeline/pipeline.py:76`). `dense_knn(query_vec, k)` /
`sparse_search(query_text, k)` take **no filter argument** anywhere (`contracts/ports.py:93,98`,
`sqlite.py`, `postgres.py`). ModusQ §5.4 requires the permitted-chunk filter applied *inside* the
retriever **or** an over-fetch-until-enough fallback (`overfetch_factor`, default 4). Neither exists:
a query scoped to a narrow method set whose in-scope chunks rank past the top-`k` pool returns fewer than
`top_k` even when more in-scope chunks exist deeper in the index. Numerically correct for the common
(mostly-permitted) case; a **recall bug** for tight scopes/licenses.
- **Fix:** add a `filter`/predicate arg to `dense_knn`/`sparse_search` (pushed into the SQL), or an
  over-fetch loop. Either way, update the now-false claim in CLAUDE.md and
  `retrieval-architecture-design.md` §6 that "`dense_knn` already takes a `filter` arg."

### 1.3 [M] Per-purpose license-class policy is not enforced
`_passes` enforces only `available`, `ai_grounding_allowed` (for `GENERATION_GROUNDING`), and method
scope. ModusQ §5.6 specifies a purpose → permitted-`license_class` map and that `third_party_copyrighted`
is **never** returned by any purpose. As written, `EXECUTION`/`AUTHORING` apply no license-class filter and
`third_party_copyrighted` is not categorically excluded. The docstring admits it's deferred — fine to
defer, but it is a stated safety requirement; track it explicitly (a `LicensePolicy` seam).

### 1.4 [M] `RetrievalPipeline` keys retrievers by `class_name` — duplicate-class configs collide
`per_retriever = {r.config.class_name: candidates ...}` (`pipeline.py:61`). Two retrievers of the same
class (e.g. two `dense` with different `dense_k`, or two `sparse` over different fields) collide on the key;
the later silently overwrites the earlier and its candidates vanish from fusion. Key by the unique
`r.config.name or r.config.class_name` instead (the Component framework already supports per-instance
`name`).

---

## 2. Duplication / DRY

### 2.1 [M] Extension → kind is parsed in two layers
`IngestionEngine._infer_source_type` + `_SOURCE_TYPES` (`ingestion/engine/engine.py:51,372`) sets
`metadata['source_type']` from the file extension; `LoadAndParseStage._infer_kind` (`extraction/
load_parse.py:107`) *also* parses the extension, then falls back to `metadata['source_type']`. Two
extension parsers, two vocabularies (`text` vs `txt`), for the same routing decision. Consolidate on one
(let the stage own kind detection, or have the engine pass a canonical kind the stage trusts).

### 2.2 [L] Evidence-accumulation dedup repeated across reasoners
`IterativeReasoner.reason` and `DecompositionReasoner.reason` (`generation/components/reasoner.py:175,217`)
both run the same `for r in results: if r.chunk_id not in seen: seen.add(...); evidence.append(...)`. A
small shared `_accumulate(evidence, seen, results)` helper on `Reasoner` removes the repeat.

### 2.3 [L] Container `_build_children` + nullable-field + `_ensure_children` boilerplate
~7 containers (`RetrievalPipeline`, `RoutingRetrievalPipeline`, `GenerationPipeline`,
`CascadingGroundingChecker`, `LoadAndParseStage`, `EnrichStage`, `Pipeline`) repeat the same shape:
declare `self._x: T | None = None`, build in `_build_children`, call `_ensure_children()` first thing in
the run method. It's a consistent, readable idiom — flagged only as a candidate for a tiny mixin if it
grows further. Not worth churning now.

---

## 3. Dead / unused code

### 3.1 [M] `WorkerSettings` is entirely unused
`WorkerSettings(queue_timeout_seconds, concurrency)` + `Settings.worker` (`core/engine/config.py:125,172`)
have **no readers** — not `run_worker`, not the queue. Either wire them into the distributed consume loop
(their evident intent) or drop the class + field until the distributed path needs tuning. (Carried over
from the previous review — still dead.)

### 3.2 [M] `query_chunks` and `health_check` have zero references
`DocumentRepository.query_chunks` and `health_check` (`storage/repository/base.py:492,313`) are called by
no source **and no test**. Cover/use or remove (and drop `query_chunks` from any doc that lists it as the
read surface).

### 3.3 [L] `AppSettings.name` / `AppSettings.version` unused; `api` extra vestigial
Only `app.debug` is read. And the `api = [fastapi, uvicorn, python-multipart]` extra (`pyproject.toml:19`)
has no consumer — the HTTP layer lives in **tiqtasq.backend**, and CLAUDE.md states "no HTTP layer here."
Drop the extra (or document why it's kept).

---

## 4. Test-coverage gaps

### 4.1 [H] Postgres adapter: 0% coverage
`storage/repository/postgres.py` (50/50 statements missed). `tests/storage/repository/test_postgres.py`
exists but **skips** without `TARNRAG_TEST_POSTGRES_URL`, so CI never exercises `dense_knn` (pgvector
`<=>`), `sparse_search` (`to_tsvector`/`ts_rank_cd`), `hydrate`, or the asyncpg path. The entire second
dialect — and the dense/sparse parity the design promises — is unverified. A dockerized `pgvector/pgvector`
service in CI closes this; `retrieval-architecture-design.md` §9 already flags it as the known gap.

### 4.2 [M] pgQueuer adapter path untested
`ingestion/engine/queue.py` 74% — the missed lines (98–131) are `PgQueuerJobQueue.enqueue`/`run`/
`set_handler`, i.e. the only real distributed-queue mechanics. The whole suite runs on `InMemoryJobQueue`;
the pgQueuer adapter is written + reviewed but never run. Hard to unit-test without pgQueuer, but worth a
gated integration test alongside 4.1.

### 4.3 [L] docling / html extractors
`extraction/docling_pdf.py` 22% (heavy converter gated on the optional `docling` install — only the
`DoclingDocument → StructuredDocument` `_map` is partially covered); `extraction/html.py` 86%. Acceptable
given the gating, but the docling `_map` deserves a few more constructed-document cases.

---

## 5. Minor smells / observations

- **[L]** `RetrievalPipeline._passes` / `_in_scope` carry `# type: ignore[union-attr]` for the
  `scope: list[MethodRef] | str` union — a typing smell a small `_scoped(query) -> list[MethodRef] | None`
  helper would erase.
- **[L]** `RetrievalContext` carries `cross_encoder` (a reranker-only resource); the `CrossEncoderReranker`
  raises a clear error if it's `None`. Injecting an optional resource for one consumer is acceptable but
  slightly leaky — fine as is.
- **[L]** Observability "enabled ⇒ `NoOpObservability`" inversion (`ingestion/engine/engine.py`): turning
  observability *on* installs a do-nothing adapter (real adapters are future work). Expected for the phase,
  surprising to a reader. (Carried over.)
- **[L]** Doc/code drift is itself a smell: CLAUDE.md + `FUNCTIONAL_REQUIREMENTS.md` describe the pre-reorg
  layout (`ingestion/stages/`, `core/embedder.py`, the metadata-bag chunk schema) and a non-existent
  `dense_knn` filter arg. Addressed in the doc-cleanup pass; noted here for completeness.

---

## Summary table

| # | Sev | Area | One-line |
|---|-----|------|----------|
| 1.1 | H | retrieval | RRF/identity fusion lacks the mandatory `(score desc, chunk_id asc)` tie-break (C++ parity) |
| 1.2 | H | retrieval | license/scope filter is post-hydrate with no over-fetch — scoped queries under-return |
| 1.3 | M | retrieval | per-purpose license-class policy (§5.6) not enforced; `third_party_copyrighted` not excluded |
| 1.4 | M | retrieval | `RetrievalPipeline` keys retrievers by `class_name` — duplicate-class configs collide |
| 2.1 | M | ingestion | extension→kind parsed in both the engine and `LoadAndParse` |
| 2.2 | L | generation | evidence-accumulation dedup duplicated across two reasoners |
| 2.3 | L | framework | container build/`_ensure_children` boilerplate repeated (idiom, optional) |
| 3.1 | M | config | `WorkerSettings` entirely unused |
| 3.2 | M | storage | `query_chunks` / `health_check` have zero references |
| 3.3 | L | config/pkg | `AppSettings.name`/`version` unused; `api` extra vestigial |
| 4.1 | H | tests | Postgres adapter 0% coverage (skipped without a live DB) |
| 4.2 | M | tests | pgQueuer adapter path untested (in-memory only) |
| 4.3 | L | tests | docling/html extractor coverage thin |
