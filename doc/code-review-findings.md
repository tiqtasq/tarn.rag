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

### 2.1 [M] Extension → kind is parsed in two layers — ✅ RESOLVED
> **✅ Resolved** (`feature/code-review-fixes-2`): one shared `infer_source_kind(path)`
> (`extraction/load_parse.py`) is now the single extension→kind parser, used by both `LoadAndParseStage`
> (routing) and `IngestionEngine` (stamping `metadata['source_type']`). The engine's separate
> `_infer_source_type` / `_SOURCE_TYPES` (a coarser, markdown-blind vocabulary that mapped `.md` →
> `"unknown"`) is gone, so the two paths can't drift or disagree (e.g. on `.htm` / markdown).

Previously `IngestionEngine._infer_source_type` + `_SOURCE_TYPES` and `LoadAndParseStage._infer_kind` each
parsed the extension with different vocabularies (`text` vs `txt`; markdown unrecognized by the engine) for
the same routing decision.

### 2.2 [L] Evidence-accumulation dedup repeated across reasoners — ✅ RESOLVED
> **✅ Resolved** (`feature/code-review-fixes-l`): extracted `Reasoner._accumulate(evidence, seen, results)`;
> `IterativeReasoner` and `DecompositionReasoner` both call it instead of repeating the dedup loop.

`IterativeReasoner.reason` and `DecompositionReasoner.reason` both ran the same `for r in results: if
r.chunk_id not in seen: seen.add(...); evidence.append(...)` pooling loop.

### 2.3 [L] Container `_build_children` + nullable-field + `_ensure_children` boilerplate — ⏸️ WON'T-FIX (reviewed)
> **Decision: keep** (2026-06-21). Reviewed all 8 containers (`Pipeline`, `RetrievalPipeline`,
> `RoutingRetrievalPipeline`, `GenerationPipeline`, `CascadingGroundingChecker`, `ChunkStage`,
> `EnrichStage`, `LoadAndParseStage`): their children are **heterogeneous** (lists / single / optional /
> spec-keyed caches with different types + build logic), and the only shared mechanic (build-once + guard)
> already lives in `Component._ensure_children` / `_build_children`. Any factoring (declarative
> `{field: (key, type)}` map or a mixin) would lose the typed attributes and add reflection/indirection —
> a net-negative abstraction, with no type-checker pain to relieve (CI runs ruff, not mypy). Left as the
> consistent, readable idiom the finding itself recommended keeping.

The repeated shape is a 2–4 line `__init__` that null-inits typed child fields + one `self._ensure_children()`
call at the top of the run method — irreducible without harming readability.

---

## 3. Dead / unused code

### 3.1 [M] `WorkerSettings` is entirely unused — ⏸️ KEEP (decision)
> **Decision: keep** (2026-06-20). Left in place as the config home for the distributed consume loop's
> future tuning (`queue_timeout_seconds` / `concurrency`), to be wired when that path is built out.

`WorkerSettings(queue_timeout_seconds, concurrency)` + `Settings.worker` (`core/engine/config.py:125,172`)
have **no readers** — not `run_worker`, not the queue. (Retained rather than removed.)

### 3.2 [M] `query_chunks` and `health_check` have zero references — ⏸️ KEEP (decision)
> **Decision: keep** (2026-06-20). `query_chunks` stays as the documented repository read-surface (FR doc);
> `health_check` stays as conventional infra for a future readiness probe. Retained rather than removed.

`DocumentRepository.query_chunks` and `health_check` (`storage/repository/base.py:492,313`) are called by
no source **and no test** today.

### 3.3 [L] `AppSettings.name` / `AppSettings.version` unused; `api` extra vestigial — ✅ RESOLVED
> **✅ Resolved** (`feature/code-review-fixes-l`): removed the unused `AppSettings.name` / `version` (only
> `app.debug` is read), and dropped the vestigial `api` (`fastapi` / `uvicorn` / `python-multipart`) extra
> from `pyproject.toml` — the HTTP layer lives in tiqtasq.backend (CLAUDE.md: "no HTTP layer here").

---

## 4. Test-coverage gaps

### 4.1 [H] Postgres adapter: 0% coverage — ✅ RESOLVED
> **✅ Resolved** (`feature/ci-postgres-coverage`): the coverage workflow (`.github/workflows/codecov.yml`)
> now runs a `pgvector/pgvector:pg16` **service** and sets `TARNRAG_TEST_POSTGRES_URL`, so the gated
> `test_postgres.py` (dense `<=>` KNN, `ts_rank_cd` sparse, Core `hydrate` + provenance, ivfflat/GIN
> indexes, FK-cascade) runs in CI and `postgres.py` contributes coverage instead of showing 0%.

`storage/repository/postgres.py` (50/50 statements missed) was skipped in CI without
`TARNRAG_TEST_POSTGRES_URL`, leaving the entire second dialect unverified.

### 4.2 [M] pgQueuer adapter path untested — ✅ RESOLVED (and caught a real bug)
> **✅ Resolved** (`feature/ci-postgres-coverage`): added `tests/ingestion/test_pgqueuer.py` — a gated
> integration test that installs the pgQueuer schema, enqueues a job, runs the real `PgQueuerJobQueue.run`
> loop, and asserts the entrypoint decodes the payload into a homogeneous `Batch`. It runs in CI on the new
> Postgres service (4.1).
>
> **It immediately caught a production bug:** `PgQueuerJobQueue.__init__` built `QueueManager(driver)`, but
> pinned pgQueuer (1.0.2) expects a `Queries` (`QueueManager.run` reads `self.queries.qbe`), so the
> distributed `run()` path raised `AttributeError` — i.e. the distributed worker was broken and nothing
> exercised it. Fixed to `QueueManager(self._queries)`. This is exactly the untested-path risk the finding
> flagged.

`ingestion/engine/queue.py` was 74% — the missed lines were `PgQueuerJobQueue.enqueue`/`run`/`set_handler`,
the only real distributed-queue mechanics, since the whole suite ran on `InMemoryJobQueue`.

### 4.3 [L] docling / html extractors — ✅ RESOLVED
> **✅ Resolved** (`feature/code-review-fixes-l`): **html.py → 100%** (added tests for loose text nodes,
> skipped `script`/`style` subtrees, `<pre>` code, container recursion, and the empty-table path). **docling
> `_map` → 89%** (added a constructed-document case for list items / code / captions / empty-item skip; the
> remaining gap is the heavy-converter `extract` path, still gated on the full `docling` package). CI now
> installs **docling-core** (test-only) in `codecov.yml` so the `_map` tests run and contribute coverage;
> the heavy `docling` converter test stays skipped.

`extraction/docling_pdf.py` was 22% (only the `_map` partially covered, gated on `docling-core` not being
installed in CI) and `extraction/html.py` 86%.

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
- **[M] Stale `test-and-build.yml` workflow — ✅ removed.** It built a Docker image from a **non-existent
  `Dockerfile`** and ran `pytest tests/unit` (**no such directory**) on `main` push/PR, so it could never
  pass (leftover service template; this is a library with no Docker image). Deleted — the real test/coverage
  job is `codecov.yml`.
- **[L] Node-20 action deprecation — ✅ bumped.** CI logged that `actions/checkout@v4` / `setup-python@v4`
  (and an older `checkout@v3` + deprecated `::set-output` in `branch-name-validation.yml`) run on the
  deprecated Node 20. Bumped to `checkout@v5` / `setup-python@v6` and migrated `set-output` to
  `$GITHUB_OUTPUT`.

---

## Summary table

| # | Sev | Area | One-line |
|---|-----|------|----------|
| 1.1 | H | retrieval | ✅ **Resolved** — fusion now applies the `(score desc, chunk_id asc)` tie-break (shared `_ranked`) + regression tests |
| 1.2 | H | retrieval | ✅ **Resolved** — filter moved into the retrievers (`ChunkFilter` + `dense_knn`/`sparse_search` filter arg + `_overfetch` backfill; PG `ivfflat.probes`) |
| 1.3 | M | retrieval | ✅ **Resolved** — `LicensePolicy` seam (§5.6 default map; `third_party_copyrighted` never permitted) → `ChunkFilter.license_classes` |
| 1.4 | M | retrieval | ✅ **Resolved** — retrievers keyed by `name`/`class_name` with `#n` disambiguation (`_retriever_keys`) |
| 2.1 | M | ingestion | ✅ **Resolved** — one shared `infer_source_kind`; the engine's markdown-blind `_SOURCE_TYPES` removed |
| 2.2 | L | generation | ✅ **Resolved** — shared `Reasoner._accumulate` |
| 2.3 | L | framework | ⏸️ **Won't-fix** — heterogeneous children; abstraction would be net-negative (reviewed) |
| 3.1 | M | config | ⏸️ **Keep** — retained as the distributed-worker tuning config (decision) |
| 3.2 | M | storage | ⏸️ **Keep** — `query_chunks` (documented read-surface) + `health_check` (infra) retained (decision) |
| 3.3 | L | config/pkg | ✅ **Resolved** — removed `AppSettings.name`/`version` + the vestigial `api` extra |
| 4.1 | H | tests | ✅ **Resolved** — CI runs a `pgvector` service; `test_postgres.py` now covers `postgres.py` |
| 4.2 | M | tests | ✅ **Resolved** — gated `test_pgqueuer.py` added; **caught + fixed** a real `QueueManager(driver)` bug |
| 4.3 | L | tests | ✅ **Resolved** — html.py → 100%; docling `_map` → 89% (CI installs `docling-core`) |
