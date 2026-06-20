# Retrieval architecture — comparing methods (design)

**Status:** ✅ **Implemented** (2026-06-20) — slices 1–6 below are all built. Code:
`tarnrag/retrieval/components/` (`retriever.py` = `RetrievalContext` + dense/sparse `Retriever`s;
`fuser.py` = identity/RRF; `merger.py` = `AutoMerger`; `reranker.py` = `CrossEncoderReranker`;
`classifier.py`), `retrieval/pipeline/` (`searcher.py`, `pipeline.py` = `RetrievalPipeline`, `router.py` =
`RoutingRetrievalPipeline`), and `tarnrag/eval/`. The two earlier deviations are now **fixed** (see
`doc/code-review-findings.md` 1.1 / 1.2): the RRF/identity fusers apply the mandatory `(score desc,
chunk_id asc)` tie-break, and the license/scope filter is a **pre-filter inside the retrievers** —
`dense_knn`/`sparse_search` take a `ChunkFilter` and over-fetch to backfill, so scoped queries no longer
under-return.

The second original ask: **compare retrieval methods**. Retrieval used to be dense-only (`embed →
dense_knn → hydrate → assemble`) in one monolithic `RetrievalEngine.search`. This designed the seams so
methods became **config-driven, swappable Components** (mirroring the ingestion pipeline), threaded the
persisted layout-aware provenance into the read path, and laid out the build order. The evaluation
harness (which *measures* the comparison) followed once ≥2 methods existed.

**Locked decisions (this design):** (1) retrievers / fusers / rerankers / mergers are **Components**,
composed by spec; (2) the read path carries the **full `ChunkProvenance`** (geometry / header_path /
level / parent_chunk_id / annotations / table); (3) the method set is dense, sparse (BM25), hybrid
(RRF), **cross-encoder reranking, auto-merging, header-path injection**.

## 1. The retrieval flow

```
Query
  ├─ Retrieve   each configured Retriever → list[Candidate]   (parallel; over-fetched: dense_k / sparse_k)
  ├─ Fuse       Fuser merges the per-retriever lists → fused hits (chunk_id, score, component_scores)
  ├─ Hydrate    store.hydrate(ids) → ChunkRecord[]  (now WITH provenance — §5)
  ├─ Filter     drop records failing license / scope / purpose   (query-driven; over-fetch covers it)
  ├─ Auto-merge (optional Merger) collapse sibling leaves → their section parent (parent_chunk_id)
  ├─ Rerank     (optional Reranker) cross-encoder re-scores the surviving set
  ├─ Truncate   top_k
  └─ Assemble   RetrievalResult[]  (text + fused/rerank score + component scores + provenance)
```

Ordering notes (all configurable, but the defaults):
- **Over-fetch then truncate.** Retrievers pull `dense_k`/`sparse_k` (≫ `top_k`); the final `top_k`
  truncation happens *after* filter/merge/rerank, so filtering/merging never starves the result.
- **Filter before merge/rerank**, so disallowed chunks can't be merged into a parent or waste a
  rerank slot.
- **Auto-merge before rerank**: merging shrinks the candidate set (cheaper rerank) and rerank then
  scores exactly the units that will be returned.

## 2. The Component seams

All four are `Component`s (registered `class_name`, built by `ComponentFactory`, composed by spec).
They are **pure strategies** — config only; the store + embedder are injected at call time via a
`RetrievalContext`, exactly as stages don't hold the repository.

```python
@dataclass
class RetrievalContext:
    store: RetrievalStore     # dense_knn / sparse_search / hydrate (a new READ port — §6)
    embedder: Embedder        # query embedding (dense + header-path injection)

class Retriever(Component):
    async def retrieve(self, query: Query, ctx: RetrievalContext) -> list[Candidate]: ...
    # dense   → ctx.embedder.embed_query → ctx.store.dense_knn(vec, dense_k)
    # sparse  → ctx.store.sparse_search(query.text, sparse_k)   (BM25 over fts_chunks — §6)

class Fuser(Component):
    def fuse(self, per_retriever: dict[str, list[Candidate]]) -> list[FusedHit]: ...
    # rrf     → reciprocal rank fusion (1/(k+rank)); FusedHit carries component_scores per retriever
    # single  → identity passthrough when there's one retriever

class Reranker(Component):                       # optional
    async def rerank(self, query: Query, records: list[ChunkRecord]) -> list[ChunkRecord]: ...
    # cross_encoder → score query×text with a cross-encoder model, re-order

class Merger(Component):                          # optional — auto-merging
    async def merge(self, records: list[ChunkRecord], ctx: RetrievalContext) -> list[ChunkRecord]: ...
    # auto_merge → when ≥N sibling leaves share a parent_chunk_id, replace them with the parent
    #              (fetched via ctx.store.hydrate([parent_id])); threshold N is config
```

`FusedHit` is a small carrier: `(chunk_id, score: float, component_scores: dict[str, float])` — it maps
straight onto `RetrievalResult.score` + `.component_scores` at assembly.

## 3. Composition: a `RetrievalPipeline` container Component

The composition is its own **container Component** — `RetrievalPipeline` — a *sibling* of the ingestion
`Pipeline` (both are config-driven containers; **not** a subclass, because retrieval's flow is
heterogeneous — parallel retrievers + fan-in + fixed steps — not `Pipeline`'s linear item-threading). Its
spec lives in `Settings.components` under `RETRIEVAL_PIPELINE` (the analog of `INGESTION_PIPELINE`):

```json
{
  "class_name": "retrieval_pipeline",
  "retrievers": [{"class_name": "dense", "dense_k": 50}, {"class_name": "sparse", "sparse_k": 50}],
  "fuser":      {"class_name": "rrf", "k": 60},   // explicit "identity" fuser when there's one retriever
  "reranker":   null,
  "merger":     null
}
```

`RetrievalPipeline` builds its children in `_build_children` (factory `create_as`, like every container)
and **owns the flow**: `async search(query, ctx: RetrievalContext) -> list[RetrievalResult]` runs §1. The
`RetrievalEngine` stays a thin facade — `open()` / `create()` do the compat check + store/embedder
construction (the `Engine` base), then build the `RetrievalPipeline` from the spec and delegate:
`search(query)` → `pipeline.search(query, RetrievalContext(repository, embedder))`.

**Why a container Component, not engine-direct (revised from the first draft):** it separates the flow
from the engine's lifecycle (engine = facade, pipeline = algorithm — and the pipeline is unit-testable
against a fake store, no real engine/repo), and makes the composition a first-class spec/object the eval
harness builds many of and runs. **Comparing methods = different `RetrievalPipeline`s.** The honest
caveat: it's a *sibling* of `Pipeline`, not a subclass — retrieval's heterogeneous flow doesn't reuse
`Pipeline.run`'s linear item-threading; it's a "pipeline" in the composition-of-Components sense.

The query-time knobs stay on `Query` (`top_k`, `dense_k`, `sparse_k`, `purpose`, `scope`): the spec picks
*which* methods; the `Query` tunes *this* request.

## 4. License / scope filter

Driven by the **`Query`** (`purpose`, `scope`), not by config, so it's a fixed step rather than a swappable
Component. `Query.permitted_filter()` builds a `ChunkFilter` (`available` / `ai_grounding_allowed` columns
on `chunks` + method `scope` via `method_chunks`) that the retrievers pass into `dense_knn`/`sparse_search`;
**the predicate is pushed into the retriever SQL** and the search over-fetches to backfill past disallowed
chunks (the first-draft post-hydrate filter was replaced — finding 1.2). *Deferred:* the per-purpose
`license_class` policy (finding 1.3).

## 5. Provenance in the read path (the contract extension)

`ChunkRecord` and `RetrievalResult` each gain a **`provenance: ChunkProvenance | None`**, and `hydrate`
fills it — reusing the `chunk_provenance` module already built for ingestion:

- extend `hydrate`'s SELECT with the chunk provenance columns (`header_path`, `level`,
  `parent_chunk_id`, `geometry`) → `chunk_provenance.row_to_provenance`;
- fetch `table_cells` / `chunk_annotations` for the hit set and attach `table` / `annotations`
  (the same grouped-fetch the repository's `_attach_*` use).

This is what makes the advanced methods reachable: **auto-merging** needs `parent_chunk_id`,
**citation/highlighting** needs `geometry` + `header_path`, **entity/cell** retrieval needs
`annotations` / `table`.

## 6. Sparse retrieval (new store method) + the read port

- **`sparse_search(query_text, k) -> list[Candidate]`** — a new repository method (sibling of
  `dense_knn`). SQLite: raw FTS5 `bm25(fts_chunks)` over `fts_chunks` (the index ingestion already
  writes). Postgres: `tsvector`/`tsquery`. `raw_score` = BM25.
- **`RetrievalStore` port** (`dense_knn`, `sparse_search`, `hydrate`) — the narrow READ seam retrievers
  depend on (ISP), parallel to the write-side `ChunkStore`. Implemented by `DocumentRepository`.

## 7. Header-path injection — an embed-time variant, not a query-time toggle

Injection prepends the section breadcrumb to the chunk text **before embedding** — so it changes what's
in the vector index. It therefore lives on the **Embed stage** (ingestion) *and* the query embedder
(retrieval), configured consistently, and is compared by building a **separate index** (injected) vs the
plain one. It is *not* a retriever you flip per query. The design note: keep injection a property of the
embedding identity (it already participates in the fingerprint), so an injected index won't `open` for a
plain query embedder — the compatibility check protects against mismatched comparison.

## 8. Build slices (all delivered)

1. ✅ **Read-path provenance + `RetrievalStore` port + `RetrievalPipeline` + Retriever/Fuser + sparse + RRF.**
   `ChunkRecord`/`RetrievalResult` += `provenance`; `hydrate` fills it; `sparse_search`
   (SQLite FTS5 **and** Postgres tsvector/GIN); `Retriever`(dense/sparse) + `Fuser`(identity/rrf);
   `RetrievalPipeline` composed from `RETRIEVAL_PIPELINE`; the engine delegates to it. Delivers the
   **dense / sparse / hybrid** trio.
2. ✅ **License/scope filter** on the flow (§4) — a pre-filter **inside the retrievers** (`ChunkFilter` from
   `Query.permitted_filter()` → `dense_knn`/`sparse_search` + `_overfetch` backfill; `available`,
   `ai_grounding_allowed` for `GENERATION_GROUNDING`, and method scope). *Per-purpose license-class policy
   (§5.6 of the ModusQ spec) is still deferred — tracked in `code-review-findings.md` (1.3).*
3. ✅ **Auto-merging** `Merger` (`AutoMerger`) — uses slice 1's `parent_chunk_id`.
4. ✅ **Cross-encoder reranking** `Reranker` (`CrossEncoderReranker`, `OnnxCrossEncoder` resource).
5. ✅ **Header-path injection** — the Embed-stage variant (`EmbeddingSettings.inject_header_path`); part of
   the embedding fingerprint, so an injected index won't `open()` with a non-injecting embedder.
6. ✅ **Evaluation harness** (`tarnrag/eval/`) — `sweep` across `RETRIEVAL_PIPELINE` specs with
   `hit@k` / MRR / nDCG, segmented by `query_type` (`by_query_type` / `format_segmented`).

## 9. Decisions (resolved)

- **Single-retriever path:** explicit **identity `Fuser`** — the flow stays uniform (always retrieve →
  fuse → …). ✓
- **Composition:** a **`RetrievalPipeline` container Component** (sibling of `Pipeline`); the engine is a
  thin facade over it — revised from the first draft's engine-direct lean (see §3 for the why). ✓
- **`component_scores` on a reranked hit:** **keep both** — the fused per-retriever scores AND a
  `rerank` score, so the breakdown stays transparent. ✓
- **Postgres sparse:** implement **both dialects in slice 1** (SQLite FTS5 + Postgres tsvector/GIN) — full
  parity from the start. *Test note:* the suite runs SQLite, so the Postgres path is written + reviewed
  but needs a live Postgres to exercise. ✓
