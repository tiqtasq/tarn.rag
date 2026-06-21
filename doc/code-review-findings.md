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

### 1.1 [H] RRF / identity fusion has no deterministic tie-break — ✅ RESOLVED
> **✅ Resolved** (`feature/code-review-fixes`): both fusers now route through a shared `_ranked` helper
> that sorts `(score desc, chunk_id asc)`, so equal-score hits break by id rather than retriever/insertion
> order. Covered by `test_rrf_fusion_tie_breaks_by_chunk_id` + `test_identity_fusion_orders_best_first`.

`RRFFuser.fuse` (`retrieval/components/fuser.py:74`) sorted `sorted(scores, key=lambda cid: scores[cid],
reverse=True)` — score only. `IdentityFuser` didn't tie-break either. The ModusQ spec makes a secondary
`chunk_id asc` tie-break **mandatory** (`ModusQ_RetrievalSubsystemSpec.md` §5.5, §9) because it is the
contract that makes the future C++ port return byte-identical orderings (R1). Two chunks with equal fused
score used to order by dict-insertion, i.e. retriever order — non-deterministic across runs/ports.
- **Fix applied:** `sorted(hits, key=lambda h: (-h.score, h.chunk_id))` via the shared `_ranked` helper.

### 1.2 [H] License/scope filter is post-hydrate, not a pre-filter — scoped queries can under-return — ✅ RESOLVED
> **✅ Resolved** (`feature/code-review-fixes`): the permitted-chunk filter moved **into the retrievers**.
> `Query.permitted_filter()` builds a `ChunkFilter` (available / grounding / method-scope) that the
> retrievers pass to `dense_knn` / `sparse_search` (now `(…, k, filter=None)`); each dialect applies it as
> a SQL predicate and **over-fetches** (`DocumentRepository._overfetch`, ×4 window) until ≥ `k` permitted
> hits or the index is exhausted, so a tight scope no longer under-returns. The post-hydrate `_passes` was
> removed. On Postgres the filtered dense path also sets `ivfflat.probes = lists` (the ANN otherwise probes
> one list and under-returns regardless of `LIMIT`). Tested on SQLite (`test_dense_knn_filter_backfills_*`,
> scope, sparse) **and** live pgvector (`test_filter_drops_disallowed_and_scopes`). Per-purpose
> `license_class` policy stays deferred → finding 1.3.

`RetrievalPipeline.search` used to over-fetch `dense_k`/`sparse_k`, fuse, hydrate, then drop disallowed
chunks in `_passes`; `dense_knn` / `sparse_search` took **no filter argument**, so a query scoped to a
narrow method set whose in-scope chunks ranked past the top-`k` pool returned fewer than `top_k` even when
more in-scope chunks existed deeper in the index — numerically fine for the common (mostly-permitted) case,
a **recall bug** for tight scopes/licenses (ModusQ §5.4 mandates the in-retriever pre-filter + over-fetch).
- **Fix applied:** `ChunkFilter` + `dense_knn`/`sparse_search(…, filter)` + the shared `_overfetch` loop;
  the false "`dense_knn` already takes a `filter` arg" claim in CLAUDE.md is now actually true.

### 1.3 [M] Per-purpose license-class policy is not enforced — ✅ RESOLVED
> **✅ Resolved** (`feature/code-review-fixes`): added a config-driven **`LicensePolicy`** seam
> (`retrieval/components/license_policy.py`). `DefaultLicensePolicy` ships the ModusQ §5.6 map — every
> purpose may see the four shippable classes and **`third_party_copyrighted` is never listed**, so it can
> never be returned. The engine builds it from the `LICENSE_POLICY` spec in `Settings.components` (default
> filled) and injects it via `RetrievalContext.filter_for`, which adds `license_classes` to the
> `ChunkFilter`; both dialects filter `license_class IN (…)`. Deployments tune the per-purpose map (or swap
> the policy) without touching the retrievers. Covered by a policy unit test, a store license-class test,
> an engine end-to-end test (copyrighted chunk never returned), and the gated PG test.

The old `_passes` enforced only `available`, `ai_grounding_allowed` (for `GENERATION_GROUNDING`), and method
scope — `EXECUTION`/`AUTHORING` applied no license-class filter and `third_party_copyrighted` was not
categorically excluded, despite ModusQ §5.6 requiring a purpose → permitted-`license_class` map with
`third_party_copyrighted` never permitted. Now enforced via the `LicensePolicy` seam.

### 1.4 [M] `RetrievalPipeline` keys retrievers by `class_name` — duplicate-class configs collide — ✅ RESOLVED
> **✅ Resolved** (`feature/code-review-fixes-2`): `RetrievalPipeline._retriever_keys()` keys each retriever
> by its configured `name` (else `class_name`), disambiguating duplicates with a `#n` suffix, so two
> same-class retrievers no longer collide on one key (which dropped one's candidates from fusion). Covered
> by `test_retriever_keys_disambiguate_duplicates`. The common single-class / uniquely-named case is
> unchanged.

`per_retriever = {r.config.class_name: candidates ...}` used to let two retrievers of the same class (e.g.
two `dense`, or two `sparse` over different fields) collide on the key — the later silently overwrote the
earlier and its candidates vanished from fusion.

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
| 1.1 | H | retrieval | ✅ **Resolved** — fusion now applies the `(score desc, chunk_id asc)` tie-break (shared `_ranked`) + regression tests |
| 1.2 | H | retrieval | ✅ **Resolved** — filter moved into the retrievers (`ChunkFilter` + `dense_knn`/`sparse_search` filter arg + `_overfetch` backfill; PG `ivfflat.probes`) |
| 1.3 | M | retrieval | ✅ **Resolved** — `LicensePolicy` seam (§5.6 default map; `third_party_copyrighted` never permitted) → `ChunkFilter.license_classes` |
| 1.4 | M | retrieval | ✅ **Resolved** — retrievers keyed by `name`/`class_name` with `#n` disambiguation (`_retriever_keys`) |
| 2.1 | M | ingestion | extension→kind parsed in both the engine and `LoadAndParse` |
| 2.2 | L | generation | evidence-accumulation dedup duplicated across two reasoners |
| 2.3 | L | framework | container build/`_ensure_children` boilerplate repeated (idiom, optional) |
| 3.1 | M | config | `WorkerSettings` entirely unused |
| 3.2 | M | storage | `query_chunks` / `health_check` have zero references |
| 3.3 | L | config/pkg | `AppSettings.name`/`version` unused; `api` extra vestigial |
| 4.1 | H | tests | Postgres adapter 0% coverage (skipped without a live DB) |
| 4.2 | M | tests | pgQueuer adapter path untested (in-memory only) |
| 4.3 | L | tests | docling/html extractor coverage thin |
