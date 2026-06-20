# RAG Design: Building Blocks

**Scope:** Single company, ~100,000 documents. Ingestion and retrieval engines only (no API/serving layer). Built single-tenant now, designed to convert to multi-tenant via the bridge pattern (namespace per tenant) later. All choices favor open-source, self-hosted (offline-capable), and the prevention of hallucinations.

> **This is a SOTA-options menu, not a status report** — it deliberately surveys the landscape so choices
> stay swappable behind the eval harness. **What tarnrag has adopted so far (2026-06-20):** structure-aware
> layout extraction (pdfplumber fast tier + Docling high-fidelity tier behind the `Extractor` seam),
> deterministic structure-aware chunking with header-path injection and full provenance; an ONNX embedder
> (default `gte-small`, with OpenAI/Voyage/Gemini API backends); one store (SQLite + sqlite-vec/FTS5, or
> Postgres + pgvector); hybrid BM25+dense → RRF → cross-encoder rerank → auto-merge; a generation layer with
> grounding + proof trees; and retrieval + generation eval harnesses. **Not yet adopted** (still menu items):
> dedup (MinHash/LSH), late chunking, learned-sparse / Matryoshka / visual (ColPali) embedding, the organize
> layer (BERTopic / knowledge graph / classification), and answer-boundary groundedness scorers (HHEM/Lynx).
> See [`rag-design-building-blocks-fit.md`](./rag-design-building-blocks-fit.md) for the per-block
> extend-vs-redesign verdicts and their current build status.

## Design principles (the through-line)

The anti-hallucination strategy is not one component, it is a bias applied across every block:

1. **Extractive or deterministic over generative.** Wherever a step has a generative and a non-generative option (chunk labels, structure summaries, classification labels), prefer the non-generative one. Nothing that never generated text can hallucinate.
2. **Provenance to the source span on every chunk.** Source id, page, bbox or character offset, content hash, and header path travel with each chunk. This is what makes every downstream claim verifiable by exact match rather than by a probabilistic judge.
3. **Maximize "right context present."** A large share of RAG hallucination is missing or wrong context, not bad generation. Hybrid retrieval plus reranking is therefore an anti-hallucination measure, not just a quality one.
4. **Extraction fidelity reduces hallucination at the source.** Garbled OCR or shredded tables become ungrounded context. High-fidelity parsing (Docling) or bypassing parsing entirely (visual retrieval) removes a whole class of errors.
5. **A verification and abstention layer at the answer boundary.** Score groundedness, enforce citations, and abstain when support is insufficient.

**Multi-tenant readiness constraint (applied now, even at one tenant):** keep one store holding vectors plus metadata plus sparse fields, thread a `tenant_id` field through the schema from day one (always "default" for now), and route all data access through one resolution seam. Conversion to bridge then becomes a routing and provisioning change, not a rewrite. Share the embedding model across all future tenants; never duplicate it per tenant.

---

## Table 1: Ingestion engine

| Building block | Design decision | Open-source options (SOTA) | Pros | Cons |
|---|---|---|---|---|
| Format detection and routing | Route by detected type to the right extractor | Apache Tika, python-magic, libmagic | Robust, simple, handles the heterogeneity of a real corporate corpus | Edge cases (corrupt files, mislabeled types) need fallbacks |
| Layout-aware extraction | Parse to a structured document that preserves semantic hierarchy, route tables to a precise extractor | Docling (SOTA for structured RAG: DocLayNet layout, TableFormer tables, DoclingDocument hierarchy, air-gapped), Marker (fast, Surya OCR), MinerU (CJK/scientific/financial), pymupdf4llm (fast bulk), pdfplumber and Camelot (precise tables) | Structure feeds chunking and provenance; fewer extraction errors means fewer downstream hallucinations; Docling output maps directly onto the structure tree | Model weights and GPU for the strong parsers; cold-start latency; multi-level heading detection still imperfect; PyMuPDF is AGPL-3.0 |
| OCR (scanned pages only) | OCR only when a page has no text layer; skip born-digital | Surya, PaddleOCR, Tesseract, RapidOCR, docTR | Unlocks scans and image-only PDFs | OCR errors propagate as ungrounded context; for scan-heavy corpora prefer visual retrieval (see Table 4) over an OCR-to-text pipeline |
| Chunking | Split on the Docling hierarchy (heading then paragraph then sentence), keep paragraphs atomic, add late chunking for cross-chunk context | Docling chunker, semchunk, custom recursive splitter, late-chunking recipe | Verbatim, traceable units; structure-aware boundaries; late chunking adds context with no generation | Propositional and contextual chunking introduce an LLM generation step, so they carry hallucination risk; prefer deterministic header-path injection |
| Deduplication | Detect near-duplicates before embedding | datasketch (MinHash/LSH), simhash, semhash, embedding-based clustering | Cuts corporate duplicate sprawl; fewer conflicting versions in context, a known confusion and hallucination source | Threshold tuning; must keep the canonical version with correct provenance |
| Provenance and metadata enrichment | Attach source id, page, bbox or char offset, content hash, header path, and a `tenant_id` slot to every chunk | Own schema in the store | The anti-hallucination backbone: every chunk is verifiable against its exact source location | Requires discipline to thread the context through every pipeline stage and background job |
| Embedding generation | One shared open model, Matryoshka dimensions for storage control | Qwen3-Embedding 0.6B/4B/8B (Apache-2.0, instruction-aware, dims 32 to 1024), BGE-M3 (dense plus sparse plus multi-vector in one model), EmbeddingGemma-300M (small), Nomic Embed | Self-hosted, no data egress (suits regulated use); BGE-M3 gives you sparse and dense from one pass | Leaderboards churn fast, so validate on your own corpus; plan embedding versioning, since a model upgrade forces re-index |
| Index and store (write side) | One store for vectors, metadata, and sparse vectors, designed bridge-ready | pgvector (with Row-Level Security, plus ParadeDB/pg_search for BM25), Qdrant (payload partitioning, native sparse vectors), Milvus | Single system for vector, metadata, filtering, and access control; clean bridge path | pgvector schema-per-tenant is comfortable only into the low hundreds of tenants; Qdrant collection-per-tenant has the same ceiling, so plan payload partitioning for high tenant counts |

**Anti-hallucination pick:** Docling parsing, deterministic structure-aware chunking plus late chunking, full provenance on every chunk, Qwen3-Embedding or BGE-M3.

---

## Table 2: Imposing structure (the "organize" goal)

| Building block | Design decision | Open-source options (SOTA) | Pros | Cons |
|---|---|---|---|---|
| Taxonomy / ontology | Seed a taxonomy with domain SMEs, auto-extend by clustering | SKOS tooling, plain config; no heavy library required | Anchors classification and navigation; human-validated labels do not drift | Ongoing maintenance as the corpus grows |
| Topic modeling / clustering | Unsupervised clustering with extractive cluster labels | BERTopic (HDBSCAN plus UMAP plus c-TF-IDF), scikit-learn clustering | Labels are extracted keywords, not generated text, so they cannot hallucinate; reuses your embeddings | Cluster count and granularity need tuning; clusters are not human-named without review |
| Hierarchy / summary tree | Extractive aggregation, not abstractive RAPTOR | TextRank/LexRank (sumy), centroid sentence selection, RAPTOR repo (abstractive, for comparison only) | Extractive nodes are verbatim and validatable; keeps cross-section grouping without summaries that hallucinate | Less compression and fluency than abstractive; rebuild cost when the corpus changes |
| Knowledge graph | Grounded entity and relation triples, each citing a source span | GLiNER (zero-shot NER), GLiREL or a relation-extraction model, Kuzu or Neo4j | Enables multi-hop and cross-reference retrieval; every node and edge traces to source, so no summary hallucination | Extraction precision varies; dual-store sync with the vector store (no shared transaction) |

**Anti-hallucination pick:** BERTopic with extractive labels for navigation; a GLiNER-grounded graph for multi-hop, skipping any generated community summaries.

---

## Table 3: Document classification

| Building block | Design decision | Open-source options (SOTA) | Pros | Cons |
|---|---|---|---|---|
| Approach selection | Prefer a discriminative encoder over a generative classifier | SetFit (few-shot, label-efficient), GLiNER (zero-shot span and label), zero-shot NLI (deberta-v3-mnli), ModernBERT fine-tune, fastText (fast baseline) | Deterministic, calibratable, cheap, and cannot invent a label outside the set; SetFit needs only ~8 to 16 examples per class | Needs some labeled examples; a new class means retraining or a zero-shot fallback |
| Multi-label and hierarchy mapping | Map predictions to taxonomy nodes with a per-label threshold | scikit-multilearn, a classifier head over embeddings | Aligns classification with the taxonomy from Table 2; thresholds give you an abstain option | Maintaining hierarchical consistency (parent implies child) takes rules |
| LLM-based classification (only if encoders fall short) | Constrain output to the fixed label set, force abstain when unsure | Local LLM with structured or guided decoding (outlines, xgrammar) | Flexible, no training data needed | Generative, so constrain decoding to the label enum and verify, or it will invent labels |

**Anti-hallucination pick:** SetFit or a fine-tuned ModernBERT head against a fixed taxonomy. Discriminative classifiers cannot hallucinate a label; reserve any LLM use for grammar-constrained decoding.

---

## Table 4: Retrieval engine

| Building block | Design decision | Open-source options (SOTA) | Pros | Cons |
|---|---|---|---|---|
| Query processing | Light query analysis and metadata-filter routing; limit generative expansion | Own routing logic, GLiNER for entity filters, HyDE (optional) | Routing into the right section or class lifts precision and cuts cross-section noise | HyDE and query expansion generate text, but they affect only retrieval, not the answer, so risk is low; still flag and measure |
| Lexical / sparse retrieval | Run BM25 alongside dense | Tantivy, OpenSearch, ParadeDB/pg_search, bm25s, Qdrant sparse, BGE-M3 sparse | Exact-term and rare-token matches that dense embeddings miss; keeps the right context present | A second index to keep in sync (unless one store does both) |
| Dense retrieval | ANN over embeddings (HNSW) | pgvector HNSW, Qdrant, FAISS (as an engine) | Semantic recall, paraphrase tolerance | Index parameter tuning; filtered ANN degrades on a shared index unless partitioned per tenant |
| Hybrid fusion | Combine sparse and dense with Reciprocal Rank Fusion | Native in Qdrant and OpenSearch, or a short RRF implementation | Best recall, fewer missing-context hallucinations | Fusion weights need tuning, especially across modalities |
| Reranking | Cross-encoder rerank of the top candidates | bge-reranker-v2-m3 (fast, Apache-2.0 baseline), Qwen3-Reranker-4B (strong, slow, 32k context), gte-reranker-modernbert-base (small, fast, accurate), ColBERT (late interaction) | Precision lift tightens the context window, which directly lowers hallucination | Latency and GPU; raw scores need calibration before thresholding |
| Structure-aware retrieval | Small-to-big parent expansion, auto-merging, metadata plus ACL plus tenant filter | Own logic over the store, parent pointers and header paths from ingestion | Coherent verbatim context with provenance; reuses the structure built at ingest | Added query-time logic and failure modes; measure per query type |
| Visual retrieval (frontier alternative) | For figure, table, or scan-heavy corpora, retrieve over page images and skip OCR | ColQwen2.5, ColPali, ColNomic, byaldi, Qdrant or Vespa multi-vector | SOTA on visually-rich and financial documents (ViDoRe NDCG@5 ~81 vs ~67 for text plus OCR); removes OCR error as a hallucination source | Heavy storage (~700 to 1024 vectors per page); GPU indexing; plan embedding versioning |
| Grounding and verification (anti-hallucination core) | Score answer groundedness on every response, enforce span-level citations, abstain when support is weak | Vectara HHEM-2.1-open (fast per-request groundedness), Patronus Lynx 8B (regulated-domain faithfulness, reference-free PASS/FAIL), RAGAS faithfulness, NeMo Guardrails (grounding rails), MiniCheck | Catches fluent-but-unsupported answers before the user sees them; citations make each claim checkable; abstention beats a confident wrong answer | Added latency and compute; the detector is itself imperfect, so layer a fast detector on every request with a sampled LLM-judge |
| Evaluation harness | A golden set and the RAG triad, structured per tenant from the start | RAGAS, Arize Phoenix, DeepEval, Promptfoo | Measures hallucination rate (faithfulness, context precision and recall) and gates regressions | Golden-set curation is manual; one set will not generalize across future tenants |

**Anti-hallucination pick:** hybrid (BM25 plus dense) into a cross-encoder reranker, structure-aware parent expansion for coherent verbatim context, then HHEM or Lynx groundedness scoring with citation enforcement and an abstention gate. Add ColPali only if the corpus is visually dense or scan-heavy.

---

## Recommended default stack

A pragmatic, fully open, self-hosted starting point to measure against, not a final answer:

- **Parse:** Docling (pymupdf4llm for simple born-digital bulk, pdfplumber for precise tables)
- **Chunk:** structure-aware on the Docling hierarchy plus late chunking, deterministic header-path injection, full provenance
- **Embed:** BGE-M3 (gives dense and sparse in one pass) or Qwen3-Embedding
- **Store:** pgvector with Row-Level Security and pg_search, or Qdrant with payload partitioning; one store, `tenant_id` from day one
- **Retrieve:** hybrid (BM25 plus dense) to RRF to bge-reranker-v2-m3, then structure-aware parent expansion
- **Organize:** BERTopic (extractive labels) plus a GLiNER-grounded graph; SetFit or ModernBERT for classification against a fixed taxonomy
- **Guard:** HHEM-2.1-open or Lynx groundedness scoring, citation enforcement, abstention gate
- **Measure:** RAGAS plus Phoenix, golden set kept per-tenant-ready

Caveat consistent with a measurement-first approach: every "SOTA" model name above will shift within months. The durable asset is the evaluation harness, so wire that first and treat model choices as swappable behind it.

## Multi-tenant readiness checklist (so bridge conversion is a routing change, not a rewrite)

- One store holding vectors, metadata, and sparse fields, with a `tenant_id` field present now (value "default")
- All data access routed through a single resolution seam (a repository layer), not scattered queries
- Provenance schema includes the tenant slot
- Embedding model and reranker shared across tenants, never duplicated
- Background and ingestion jobs carry tenant context explicitly in their payload
- Caches (embedding, rerank) keyed by tenant from the start
- Evaluation golden sets structured per tenant
