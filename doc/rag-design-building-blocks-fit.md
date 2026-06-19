# RAG Building Blocks — Architecture-Fit Assessment

**What this is:** for each building block in [`rag-design-building-blocks.md`](./rag-design-building-blocks.md),
a verdict on whether the requirement is **compatible with the current code (extend through an
existing seam)** or **requires redesign (an existing base class / contract must change shape)**.

**Date:** 2026-06-16. Grounded against the contracts as they exist today (see the *Appendix* for
the exact files/classes checked), not against memory.

## Legend (mapped to the question: "extend vs redesign")

| Tag | Meaning |
|---|---|
| ✅ **Extend** | Compatible. Add via a seam that already exists — no base-class / contract change. |
| 🆕 **New module** | Also compatible (no contract change), but a *net-new* subsystem rather than an extension of existing code. |
| ⚠️ **Redesign** | An existing base class / contract must change shape (the only category that breaks compatibility). |

## Headline

The core was clearly built **anticipating hybrid retrieval, full provenance, one store, and the
embedding fingerprint** — so most of Tables 1 & 4 are extensions through seams that already exist.
The genuine redesigns are few and predictable. Tables 2–3 and the guard/eval blocks are net-new
subsystems, because the system today **stops at retrieval** (no organize / classify / answer layers).

- The earlier ask — *compare retrieval methods* — is **entirely in the ✅ column** (hybrid BM25+RRF →
  cross-encoder rerank). The `Candidate` / `RetrievalResult.component_scores` / `Query.sparse_k`
  contracts already anticipate it. Cheapest high-value next step.
- The redesigns collapse into essentially **one decision**: *do we adopt structured parsing
  (Docling)?* — because that gateway is what makes structure-aware chunking, header-path provenance,
  and full parent-expansion real.

---

## Table 1 — Ingestion engine

| Block | Verdict | Touches / why |
|---|---|---|
| Format detection & routing | ✅ Extend | Add content-sniff (python-magic/libmagic) + backends to the `DEFAULT_PDF_PARSERS`-style registry; `LoadAndParseStage` already routes on `metadata['source_type']` / `['parser']`. |
| Layout-aware extraction (Docling) | ⚠️ **Redesign** | The parser seam is `Callable[[str], str]` → plain text (`stages/parsers.py`). Structure / tables / bbox / reading-order cannot pass through it; `LoadAndParse` + `Chunk` must consume a *structured document*. Plain-text bulk (pymupdf4llm) is ✅ extend; **structured** extraction is the redesign. |
| OCR (scanned pages only) | ✅ Extend | New parser backend (Surya / Tesseract / RapidOCR) with an internal "has text layer?" check; returns text. (Per-page provenance pulls in the structured-parser redesign.) |
| Chunking — hierarchy + **late chunking** | ⚠️ **Redesign** | Header-aware chunking rides on the structured-parser redesign. **Late chunking** *additionally* breaks `Embedder.embed_passages` (it embeds independent texts) and the chunk→embed-each ordering — late chunking embeds the document once and pools per chunk-span. |
| Deduplication | ✅ Extend | New pre-embed stage (MinHash/LSH). Corpus-wide near-dup check needs **one new repo lookup method** (stages are DB-free by rule D6), not a contract change. |
| Provenance & metadata enrichment | ✅ Extend | **Already built.** `chunks` has typed provenance columns (`locator`, `license_class`, `content_hash`, `ai_grounding_allowed`, `available`) plus the `_CHUNK_PROVENANCE` seam — its comment: *"adding a provenance field is one entry here, not edits in four methods."* Add `page` / `bbox` / `char_offset` / `header_path` (and the `tenant_id` slot) as columns + one entry each. |
| Embedding (Qwen3 / BGE-M3 / Matryoshka) | ✅ / ⚠️ | Dense model swap + prefixes + Matryoshka dim = ✅ config via `OnnxEmbedder` (dim is already part of the frozen fingerprint identity). **BGE-M3 learned sparse in one pass** = ⚠️ the `Embedder` port returns dense only — *or* take sparse from BM25/FTS and avoid touching the port. |
| Index & store (one store + sparse, bridge-ready) | ✅ Extend | One store already (`DocumentRepository` = vectors + metadata + FTS + `index_meta`). Add `sparse_search` — the FTS **write** path (`_index_chunk_text` → `fts_chunks`) already exists. pgvector path is already there. **Qdrant/Milvus** = ⚠️ the SQLAlchemy-Core repository base assumes a relational store. |

---

## Table 2 — Imposing structure ("organize")

All 🆕 **New module.** These are batch analytics over the corpus: read embeddings/chunks from the
repository, write labels / clusters / tree nodes back as metadata (✅ via the provenance seam). None
modify an existing contract.

| Block | Verdict | Touches / why |
|---|---|---|
| Taxonomy / ontology | 🆕 New module | Config + a seed label set; feeds classification. |
| Topic modeling (BERTopic) | 🆕 New module | Offline job over existing embeddings; extractive labels → chunk/doc metadata (✅). |
| Hierarchy / summary tree | 🆕 New module | Extractive tree built offline; nodes want a new table. |
| Knowledge graph (GLiNER) | 🆕 New module (+ 2nd store) | Net-new extraction module **plus a second store** (Kuzu/Neo4j) alongside → dual-store sync, no shared transaction. Doesn't change existing contracts, but is its own subsystem with its own consistency story. |

---

## Table 3 — Document classification

| Block | Verdict | Touches / why |
|---|---|---|
| Approach (SetFit / GLiNER / ModernBERT) | 🆕 New module (or ✅ stage) | A classifier subsystem. If run **at ingest** it's a ✅ new `EnrichMetadata`-style stage. Predicted labels → provenance/metadata (✅). |
| Multi-label / hierarchy mapping | ✅ Extend | Thresholds + map to taxonomy nodes; labels as metadata columns. |
| LLM classification (fallback) | 🆕 New module | New local-LLM + guided-decoding (outlines/xgrammar) dep; output → label metadata. |

---

## Table 4 — Retrieval engine

| Block | Verdict | Touches / why |
|---|---|---|
| Query processing (routing, filters, HyDE) | ✅ Extend | `Query` already carries `purpose` / `scope`; add metadata-filter routing in the repo query + an optional pre-step. HyDE adds an LLM dep. |
| Lexical / sparse (BM25) | ✅ Extend | Add `sparse_search(q, k) → Candidate`; FTS write already exists; `Candidate`'s docstring already says *"bm25 for sparse."* |
| Dense (HNSW) | ✅ Done | `dense_knn` present (sqlite-vec / pgvector). |
| Hybrid fusion (RRF) | ✅ Extend | Define the `Retriever` / `Fuser` seams (today named **only in prose** in the engine docstring) + refactor `search`. `RetrievalResult.component_scores` + `Query.sparse_k` were built for exactly this. An engine-flow refactor, **not** a contract change. |
| Reranking (cross-encoder) | ✅ Extend | New reranker component + model; post-retrieval step over candidates; the rerank score fits `RetrievalResult`. |
| Structure-aware (parent expansion / auto-merge) | ✅ Extend | Query-time logic over `ordinal` / `document_id` / `locator` (+ `get_chunks_by_document`). Full power needs header-paths from ingestion → parser redesign. |
| Visual retrieval (ColPali / ColQwen) | ⚠️ **Redesign (deep)** | Multi-vector per page breaks `Embedder` (image→multi-vector), the `Embedding` DTO (single `vector: list[float]`), and `dense_knn` (single-vector ANN vs MaxSim late-interaction); also skips parsing. A parallel modality. |
| Grounding / verification (HHEM / Lynx) | 🆕 New module | Operates on a generated **answer** — that answer-boundary layer does not exist yet (the system stops at retrieval). Net-new; no existing-contract change. |
| Evaluation harness (RAGAS / Phoenix) | 🆕 New module | The part_ii eval plan. Pure **retrieval-method** metrics (recall@k / MRR / nDCG over qrels) = light and dependency-thin; faithfulness/groundedness needs the answer layer. |

---

## Multi-tenant readiness — ✅ Extend (but do it early)

The hard requirement — **"all data access through one resolution seam"** — is **already met**:
`DocumentRepository` (implementing `ChunkStore` + `DocumentFactsSource` + `JobStatusSource`)
centralizes every read/write. The rest is additive:

- a `tenant_id` column on `documents` / `chunks` / `job_status` (+ a provenance slot),
- tenant **bound at repository construction** so query `WHERE tenant_id=…` is applied centrally and
  **the port ABCs do not change** (no DTO or port redesign),
- tenant carried in the job payload and in cache keys.

The only real cost is that retrofitting the per-query tenant filter *later* is painful — so add the
column + filter **now**, exactly as the doc argues. Embedding model + reranker are already shared
(one `Embedder`), satisfying "never duplicate per tenant."

---

## The redesign shortlist (the only items that touch base classes / contracts)

1. **Structured / layout-aware parsing** → the `Callable[[str], str]` parser seam + `LoadAndParse` /
   `Chunk`. *Gateway:* unlocks structure-aware chunking, header-path provenance, and full
   parent-expansion.
2. **Late chunking** → `Embedder.embed_passages` + the chunk→embed ordering.
3. **Learned sparse from the embedder (BGE-M3)** → the `Embedder` port. **Avoidable** if sparse comes
   from BM25/FTS instead.
4. **Visual retrieval (ColPali)** → `Embedder` + the `Embedding` DTO + `dense_knn` (a parallel modality).
5. **Non-SQL vector store (Qdrant/Milvus)** → the SQLAlchemy-Core repository base.

Everything else is an extension through an existing seam or a net-new module bolted alongside.

---

## Strategic reads

- **Cheapest high-value path (all ✅):** hybrid (BM25 + dense) → RRF → cross-encoder rerank, behind
  new `Retriever` / `Fuser` seams. The result/candidate/query contracts already anticipate it, and it
  *is* the "compare retrieval methods" need — pair it with a small qrels + IR-metrics harness.
- **The one big fork:** "adopt structured parsing (Docling)?" Redesign #1 is the gateway for several
  downstream blocks; treat it as a deliberate decision, not an incremental extension.
- **The durable asset** (per the doc's own caveat) is the **evaluation harness** — wire it first so
  every model/method choice above is swappable behind a measured number.

---

## Appendix — contracts checked (for traceability)

| Area | File / symbol | What it constrains |
|---|---|---|
| In-flight form | `contracts/dtos.py` — `PipelineItem` (`content` + `metadata` bag) | Loose, uniform; new fields ride in `metadata` → most stage-level work is extension. |
| At-rest DTOs | `contracts/dtos.py` — `Document` / `Chunk` / `Embedding` | `Embedding.vector` is dense `list[float]` (no sparse / multi-vector field). |
| Persistence ports | `contracts/ports.py` — `ChunkStore`, `DocumentFactsSource`, `JobStatusSource` | The single data-access seam; all implemented by `DocumentRepository`. |
| Repository + schema | `storage/repository/base.py` — `DocumentRepository`, `_CHUNK_PROVENANCE`, `dense_knn`, `hydrate`, `_index_chunk_text` | Typed provenance columns + provenance seam; dense ANN present; FTS **write** present, no sparse **read** yet; no `tenant_id`. |
| Embedder port | `core/resources/embedder.py` — `Embedder.embed_passages/embed_query`, `config_fingerprint`, `_identity` | Dense-only, independent-text embedding; frozen identity → fingerprint → `index_meta`; retrieval refuses on mismatch. |
| Retrieval | `retrieval/engine/engine.py` (`search` = dense-only), `contracts/results.py` (`Candidate`, `RetrievalResult.component_scores`), `retrieval/types.py` (`Query.sparse_k`, `purpose`, `scope`) | Fusion/sparse anticipated in the data contracts; `Retriever`/`Fuser` seams named in prose but **not yet defined**. |
