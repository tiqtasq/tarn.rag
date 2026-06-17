# Retrieval architecture — comparing methods (design)

The second original ask: **compare retrieval methods**. Today retrieval is dense-only (`embed →
dense_knn → hydrate → assemble`) in one monolithic `RetrievalEngine.search`. This designs the seams so
methods become **config-driven, swappable Components** (mirroring the ingestion pipeline), threads the
persisted layout-aware provenance into the read path, and lays out the build order. The evaluation
harness (which *measures* the comparison) is the follow-on once ≥2 methods exist.

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

## 3. Engine composition (config-driven)

A `RETRIEVAL_PIPELINE` spec lives in `Settings.components` (the analog of `INGESTION_PIPELINE`):

```json
{
  "retrievers": [{"class_name": "dense", "dense_k": 50}, {"class_name": "sparse", "sparse_k": 50}],
  "fuser":      {"class_name": "rrf", "k": 60},
  "reranker":   null,
  "merger":     null
}
```

`RetrievalEngine.create` builds these via `ComponentFactory` (reusing `create_as`) and `search`
orchestrates the flow (§1), passing the `RetrievalContext`. **Comparing methods = varying this spec** —
which is exactly what the eval harness will sweep. (A thin `RetrievalPipeline` container Component is an
option, but the engine already owns the async orchestration, so it composes the children directly.)

The query-time knobs that aren't config (per-request) stay on `Query`: `top_k`, `dense_k`, `sparse_k`,
`purpose`, `scope`. The config picks *which* methods; the `Query` tunes *this* request.

## 4. License / scope filter

Driven by the **`Query`** (`purpose`, `scope`), not by config, so it's a fixed engine step, not a
swappable Component. It drops hydrated records by `license_class` / `ai_grounding_allowed` / `available`
(already columns on `chunks`) and by method `scope` (via `method_chunks`). Applied post-hydrate on the
over-fetched set. *Optimization (later):* push the predicate into the retriever SQL to avoid fetching
disallowed chunks at all.

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

## 8. Build slices

1. **Read-path provenance + the `RetrievalStore` port + Retriever/Fuser seams + sparse + RRF hybrid.**
   The keystone: `ChunkRecord`/`RetrievalResult` += `provenance`; `hydrate` fills it; `sparse_search`;
   `Retriever`(dense/sparse) + `Fuser`(rrf); engine composes from `RETRIEVAL_PIPELINE`. Delivers the
   **dense / sparse / hybrid** trio — the first real comparison.
2. **License/scope filter** wired onto the flow (§4).
3. **Auto-merging** `Merger` (§2) — depends on slice 1's `parent_chunk_id` in the read path.
4. **Cross-encoder reranking** `Reranker` — adds the model dependency behind the optional seam.
5. **Header-path injection** — the Embed-stage + query-embedder variant (§7) for an injected index.
6. **Evaluation harness** — runs the engine across `RETRIEVAL_PIPELINE` specs over a query set with
   metrics (recall@k / MRR / nDCG); this is where the *comparison* becomes quantitative.

## 9. Open decisions (to confirm before/while building slice 1)

- **Single-retriever path:** identity `Fuser`, or let the engine skip fusion when one retriever? (Lean:
  an explicit identity fuser, so the flow is uniform.)
- **`RetrievalPipeline` container** vs the engine holding the children directly. (Lean: engine-direct;
  it already orchestrates.)
- **Where `component_scores` come from for a reranked result** — keep the pre-rerank fused components
  and add a `rerank` score, or replace? (Lean: keep both — `component_scores` stays transparent.)
- **Postgres sparse** (tsvector) parity — design it in slice 1 or stub until a Postgres target exists?
  (Lean: define the abstract method; implement SQLite now, Postgres when needed — same pattern as the
  rest of the dual-dialect store.)
