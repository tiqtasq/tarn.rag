# Strategic roadmap — performance & closing the SOTA gap

Companion to `doc/phases.md` (the measured history: Phases 0–3 + the post-MOTHRAG plan) and
`doc/pre-feature-fixes.md` (debt to clear first). Two goals, as requested:

1. **Increase the performance of the RAG system** — answer quality *and* systems throughput/latency.
2. **Close the gap with existing SOTA RAG systems** — capabilities strong open/commercial stacks
   ship that tarn.rag doesn't, prioritized by fit with tarn.rag's differentiators (offline-first,
   layout-grade provenance, license filtering, one-store portability).

Ground rules carried over from the measured phases: **one lever per PR, measured before/after with
the in-repo harnesses** (`eval/harness` sweep, `scripts/run_benchmarks.py`,
`scripts/run_layout_eval.py`); anything that changes what's indexed rides the embedding
fingerprint and is compared as a separate index.

Evidence base (see `doc/phases.md` for full tables):

- Hybrid (dense+BM25, RRF): +0.015 F1 on Wikipedia QA, but **+0.057 source-hit on TAT-QA overall,
  +0.074 on tables** — hybrid's real home is layout/table corpora.
- Attribution precision 0.94 overall with a **consistent table penalty** (table vs text: F1
  0.51/0.60, attribution 0.88/0.99, citation 0.73/0.88) — the linearized-table representation is
  the measured bottleneck.
- Multi-hop pool retrieval ceiling (~0.51 hit at pool scale) unbroken by reader/embedder/bridge/γ
  levers — further wiki-QA benchmark chasing is low-leverage; differentiated settings are not.
- Bulk-ingest 3.6 → 9.4 docs/s (2.6×) — embedded only; distributed path unbatched.

---

## Goal 1 — increase RAG performance

### Tier 1: highest confidence (measured lever or direct fix of a measured deficit)

**P1. Native table representation at ingest** *(the single most evidence-backed quality lever)*
Tables are currently embedded/read as linearized markdown; every TAT-QA metric shows the penalty.
The schema is already ahead of the code: `table_cells` persists cell grid + headers + geometry, and
`Table.column_headers_for`/`row_headers_for` exist. Work: (a) embed a table leaf as a
*header-contextualized rendering* (per-row "col-header: value" lines rather than raw markdown
grid); (b) give the reader a structured view of retrieved table chunks (render from cells, cite
cells). Measure: TAT-QA source-hit + attribution segmentation (table vs text) — target closing
most of the 0.09–0.11 gaps.

**P2. Make hybrid retrieval the shipped default**
`RETRIEVAL_PIPELINE` still defaults to dense-only. Hybrid is free at query time (FTS index is
built at ingest anyway), never lost on any segment measured, and is the default posture of every
SOTA stack. Flip the default spec to `retrievers: [dense, sparse] + rrf`; keep dense-only one
config edit away. Measure: the standing harness suites (regression gate already exists in
`tests/retrieval/test_hybrid_regression.py`).

**P3. Phrase- and identifier-aware sparse queries**
`_fts_query` ORs bare tokens; the structural classifier already detects quoted spans and
identifiers but nothing consumes them (pre-feature list B5). Use them: quoted span → FTS5 phrase
query; identifier → required term (AND). Directly strengthens the lexical arm that P2 promotes.
Measure: TAT-QA (identifier-heavy) + an eval-set slice of lexical-classified queries.

**P4. Reranker on by default for quality profiles**
The cross-encoder reranker exists but is opt-in and unmeasured on the differentiated setting.
Sweep `cross_encoder` (local, offline-capable — fits the offline differentiator) vs `llm_judge`
on TAT-QA + the pool benchmark; ship the winner in a documented "quality" config profile (default
stays lean). If the local model wins, upgrade the ONNX cross-encoder to a current one
(bge-reranker class) behind the same `RerankSettings`.

### Tier 2: strong candidates (mechanism exists, needs the feature + measurement)

**P5. Contextual chunk augmentation (embed-time)**
Anthropic-style contextual retrieval: prepend a short generated context ("this chunk is from §X
of doc Y discussing Z") to each chunk before embedding. tarn.rag already has the cheap
deterministic version (header-path injection) and the exact seam for the full version
(`EmbedStage._embed_text`; rides the fingerprint so indexes can't mix). Costs one LLM call per
chunk at ingest — offer it as an opt-in enricher. Published gains elsewhere are large
(~35–49% retrieval-failure reduction); measure on the pool benchmark, where the ceiling lives.

**P6. Section-summary pseudo-chunks (RAPTOR-lite)**
The auto-merge tree already persists section parents (`level > 0`, `parent_chunk_id`). Add an
opt-in enrich step that embeds an LLM *summary* of each section parent as a retrievable
pseudo-chunk (provenance = the section). This is hierarchical retrieval for long documents at a
fraction of RAPTOR's machinery, and it reuses the existing tree + merger. Measure on a
long-document corpus (not per-question distractor sets).

**P7. Retrieval-aware multi-hop: rerank the pooled evidence**
Phase 0–3 showed decomposition wins but pool retrieval caps it. Two cheap composition changes,
measured independently: (a) run each sub-question through the *hybrid* pipeline (today the
benchmark default is dense unless flagged); (b) rerank the pooled evidence against the *original*
question before the synthesis read (the pool is assembled in retrieval order today —
`reasoner.py::_accumulate`). Both are config/composition, no new components.

**P8. Embedder upgrade path as a supported workflow**
The fingerprint system makes embedder swaps safe but a swap means silent full re-ingest by hand.
Add a `reindex` operation (read stored documents → re-chunk/re-embed → new store) so measuring
bge/e5/API embedders — and shipping upgrades — is one command. Then actually sweep 2–3 modern
small embedders vs gte-small on TAT-QA + pool; ship the best offline default.

**P8b. Generic branching (`Router`) component family** *(added 2026-07-05, from review discussion)*
The N+2 pattern — a selector spec + `routes: dict[value → component spec]` + a `default` spec —
already exists twice, specialized (`RoutingRetrievalPipeline`, `LoadAndParse`'s extractor routes).
Generalize it as: a small shared base in `core/components` owning the config shape, child
construction, and lookup; **thin per-seam subclasses** (`routing_chunker` first) so each stays a
typed member of the family it branches over (one generic duck-typed class would erase the typed
seams and can't bridge sync/async); seam-specific selectors, starting with a deterministic
`metadata_key` selector. Scope is **component-level branching inside one stage** — the ingestion
DAG stays linear (no conditional edges, no fan-in). Route decisions are recorded (annotation /
trace) like the two existing instances. First consumer: `routing_chunker` (different chunker
configs per document kind); retrofit of the two precedents only if it fits without contortion.

### Tier 3: systems performance

**P9. Distributed batching parity** — claim-side batching in the pgQueuer adapter (group claimed
jobs by stage in a short window) so the measured 2.6× embedded ingest win applies to the
scale-out path that needs it most. (Pre-feature list B6 is the minimal doc fix; this is the real one.)

**P10. Query latency budget + instrumentation** — the deferred half of post-MOTHRAG Option 3.
Time retrieve/fuse/hydrate/merge/rerank per query (the `SearchTrace` seam is the natural carrier),
publish a budget, then fix what it exposes — starting with the hydrate N+1 (pre-feature A5) and a
small query-embedding LRU (repeat queries pay a full ONNX forward today).

**P11. Postgres ANN modernization** — `ivfflat` with a full-probe workaround for filtered search
(postgres.py:68–75) is correctness-first but slow at scale. Move to **HNSW** (pgvector ≥ 0.5) with
iterative scan for filtered queries; keep the over-fetch contract. Only matters once a real
distributed deployment exists — schedule accordingly.

---

## Goal 2 — close the gap with SOTA RAG systems

What strong contemporary stacks (LlamaIndex/LangChain pipelines, Cohere/Voyage APIs, GraphRAG,
Self-RAG-style systems, contextual retrieval) have that tarn.rag lacks, filtered by fit. Items
P1–P8 above are themselves gap-closers; these are the additional ones.

**S1. Query understanding as the shipped default**
SOTA stacks rewrite/expand queries by default; tarn.rag's `multi_query` retriever, structural
classifier, and router all exist but nothing is wired in the default spec. Ship a documented
"quality" profile: `routing_retrieval_pipeline` with the structural classifier — lexical →
sparse-weighted hybrid, semantic → hybrid + multi-query (when an LLM is configured). Add a HyDE
variant retriever (hypothetical-answer embedding — trivially another `Retriever` component) to the
sweep before deciding what the profile contains. Selling point: the routing decision is visible in
`explain` — most stacks can't show *why*.

**S2. Entity-link expansion (GraphRAG-lite)**
The enrichment seam, `chunk_annotations` table, and the type index already exist. Add: an entity
enricher (start deterministic: the acronym enricher pattern + capitalized-span heuristics; LLM
optional and flagged non-deterministic per FR-5.3), then a post-retrieval expansion step (a
`Merger`-like component) that pulls the top co-entity chunks. That is the useful 20% of GraphRAG
without a graph store, and it strengthens exactly where the pool ceiling lives (bridge entities).
Measure on 2Wiki/MuSiQue pool.

**S3. Self-assessing generation (Self-RAG-lite: retrieve-on-demand + verified citations)**
The pieces exist separately: `grounded_retrieval` (γ re-retrieval), grounding cascade, abstention
policy. Compose and calibrate them into a shipped "verified answers" profile: heuristic→LLM
cascading checker + `min_grounded` threshold + abstain, with the γ loop only where the checker
flags gaps. Publish calibration numbers (abstention precision/recall on MuSiQue-unanswerable —
`should_abstain` labels are already loaded). Verified, provenance-linked answers are the
differentiator SOTA marketing claims and rarely proves.

**S4. Public scorecard + comparable eval settings**
tarn.rag's eval story is strong but self-referential. Add a BEIR subset (2–3 tasks) and one
RAG-specific public set to `eval/benchmarks.py`, and publish a `SCORECARD.md` (dataset × config ×
metric, regenerated by a script). Needed both to *know* the SOTA gap and to credibly claim
closures. Cheap: the loaders + runner pattern already exist.

**S5. Streaming answers**
Every SOTA system streams; `LanguageModel.complete` is request/response. Add
`complete_streaming` to the port (both backends support it natively), stream the final synthesis
read, and let `ask` surface an async iterator (the REST layer in tiqtasq.backend then gets SSE
for free). Reasoning/decomposition calls stay non-streamed.

**S6. Figures & multimodal provenance (later)**
`ElementKind.FIGURE` and page-box geometry already exist; docling extracts figures. The increment:
persist figure elements with geometry, add an optional captioning enricher (flagged
non-deterministic), and let citations point at figure regions. Defer until P1/S1–S3 land — table
QA is the nearer, measured win in the same "layout" lane.

**S7. Real observability adapter**
`Observability` is a seam with only a no-op behind it. One structured-logging adapter (JSON lines:
per-stage timings, per-query latency + component breakdown, token usage) makes every roadmap item
above measurable in production, not just in the harness. Prometheus can follow; don't start with it.

---

## Sequencing (one PR each unless noted)

| order | item | why now |
|---|---|---|
| 0 | `pre-feature-fixes.md` PRs 1–5 | stop debt proliferating under everything below |
| 1 | **P2 + P3** hybrid default + phrase-aware sparse | cheapest measured win; P3 makes P2 stronger |
| 2 | **P1** native table representation (2 PRs: embed-side, reader-side) | biggest evidence-backed quality lever |
| 3 | **P4** reranker sweep → quality profile | completes the retrieval stack |
| 4 | **S1** query-understanding profile (+ HyDE sweep) | turns existing components into shipped value |
| 5 | **P7** multi-hop composition (2 PRs) | attacks the pool ceiling with what exists |
| 6 | **P5** contextual augmentation | the big published lever, now measurable against 1–5 |
| 7 | **S4** scorecard + public sets | lock in gains; quantify the remaining SOTA gap |
| 8 | **S3** verified-answers profile | differentiation |
| 9 | **P8** reindex workflow + embedder sweep | unlocks safe model upgrades |
| 10 | **S2** entity expansion · **P6** section summaries | bigger bets, informed by the scorecard |
| 11 | **P9–P11** systems track · **S5** streaming · **S7** observability | as deployment needs mature |

Re-evaluate the ordering after step 7 — the scorecard is the checkpoint that says which remaining
gap is real.
