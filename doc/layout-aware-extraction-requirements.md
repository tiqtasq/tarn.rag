# Layout-Aware Extraction — Requirements

**Status:** Draft v0.2 (incorporates kickoff round-2 feedback) · **Date:** 2026-06-16 · **No code changes yet.**
**Relates to:** [`rag-design-building-blocks.md`](./rag-design-building-blocks.md) (Table 1 "Layout-aware
extraction") and [`rag-design-building-blocks-fit.md`](./rag-design-building-blocks-fit.md) (redesign #1, the gateway).

This document specifies **what** the layout-aware extraction layer must do and **which contracts must
change**. It deliberately stops short of the design (class shapes, algorithms).

## Decisions captured

| # | Topic | Decision |
|---|---|---|
| 1 | Formats in scope | **Markdown + PDF** now; architecture must be **plugin-style** so other formats add later without contract changes. |
| 2 | Structure model | **Tool-neutral internal `StructuredDocument`**, shaped as a **superset of DoclingDocument**, with **user-supplied enrichment components running alongside Docling** (NER, topics, classification, …) on the existing `Component` framework. |
| 3 | Provenance fidelity | **Highlight-grade** — geometry per element/chunk (page + bbox for PDF; char-offset spans for Markdown/text). |
| 4 | Parse strategy | **Tiered routing** — fast path for born-digital; high-fidelity only for complex/scanned/table-heavy docs. |
| 5 | Tables | **Cell-level geometry**, plus row/column structure with **header & id addressability** (query a cell by its row/column header or id). |
| 6 | In-flight carriage | **Extend `PipelineItem`** to carry the structured document (preferred; exact shape settled in design). |
| 7 | Deployment | **GPU optionally available** on ingestion workers; **throughput is not a current priority**; offline-capable. |
| 8 | Backward-compat / migration | **None required** — no users yet; "tests pass ⇒ merge". |
| 9 | License filtering | **Not a requirement** — the operator configuring the system is aware of component licenses. |

## Scope

**In scope:** the extraction contract — a tool-neutral structured-document model, the format-extractor
and enrichment-component seams, and how structure + highlight-grade provenance thread from extraction
through the in-flight contract to the chunk schema.

**Out of scope (named consumers, separate follow-on docs):** the structure-aware **chunking algorithm**
and **late chunking** (the *other* Table-1 redesign) — though this doc anticipates the chunker's needs
(FR-7); retrieval changes; the choice of specific NER/topic/classifier models; the answer/generation layer.

---

## 1. Context & problem

Today extraction is a single `MapperStage` over plain strings:

- `LoadAndParseStage.map(text, metadata) -> (content: str, updates)` (`ingestion/stages/load_parse.py`).
- PDF backends are a `name -> Callable[[str], str]` registry (`stages/parsers.py`: `pypdf`, `pdfplumber`); `.md`/`.txt` are read as **raw text**, `.html` via BeautifulSoup.
- Output is a `str` placed in `PipelineItem.content`; the `Chunk` stage splits that string.

A `str`-only contract structurally cannot carry: document **hierarchy** (headings → paragraphs →
sentences), **tables** as structured units, per-element **geometry** (page/bbox/char-offset), or
**enricher annotations**. Highlight-grade provenance and structure-aware chunking are therefore
impossible without changing the contract — this is "redesign #1".

---

## 2. Goals / Non-goals

**Goals**
- A tool-neutral `StructuredDocument` contract (superset of DoclingDocument) produced by **pluggable
  format extractors** and refined by **pluggable enrichment components**.
- Highlight-grade provenance threaded onto every chunk (geometry + header-path + content-hash).
- Tiered routing so high-fidelity parsing is used where it's worth it.
- Reuse the existing `Component`/`ComponentFactory` framework so users add extractors and enrichers by
  config, with no base-class edits.

**Non-goals (here)**
- The chunking algorithm / late chunking; retrieval; specific ML model selection; answer generation.

---

## 3. Functional requirements

### FR-1 — Formats & routing
- **FR-1.1** Produce structured output for **Markdown** and **PDF** at minimum.
- **FR-1.2** Format → extractor is a **registry/plugin** (mirrors today's parser registry). Adding a
  format (HTML, DOCX, …) is a new registered extractor, **no contract change**.
- **FR-1.3** Detect type robustly (extension + content sniff) with a fallback for corrupt/mislabeled
  files; on failure, fall back to a trivial text extractor rather than dropping the document.
- **FR-1.4** **Tiered routing**: a fast path (born-digital text) and a high-fidelity path
  (Docling-grade). Routing is configurable (per-source-type default + per-document override, like
  today's `metadata['parser']`) and chooses the right extractor per document — including using the
  **GPU when present** and staying viable on CPU-only workers. The extractor that ran is recorded in
  provenance.

### FR-2 — `StructuredDocument` model (the core new contract)
- **FR-2.1** Tool-neutral internal model; **superset of DoclingDocument** so nothing Docling emits is
  lost and other backends map onto it.
- **FR-2.2** Represent: a **document** with ordered **elements** in reading order; a **hierarchy**
  (section/heading levels → paragraphs → inline/sentence granularity as available); element **kinds**
  at least: heading, paragraph, list/list-item, **table**, figure/image, caption, code.
- **FR-2.3** Every element carries **provenance** (FR-4) and a **header-path** (the chain of enclosing
  headings, e.g. `["3 Safety", "3.2 Lockout"]`).
- **FR-2.4** **Extensible annotation slots**: elements and the document can hold arbitrary, typed
  **annotations** added by enrichers (FR-5) without changing the core model.
- **FR-2.5** A trivial/plain-text source maps to a valid `StructuredDocument` (one document, one or
  more text blocks) — graceful degradation, so a `.txt` never needs a heavy parser.

### FR-3 — Tables
- **FR-3.1** Tables are preserved as **structured** elements (rows × cells) **and** carry a rendered
  text/markdown form for embedding/display.
- **FR-3.2** **Cell-level geometry**: every cell carries its own page + bbox (PDF) / char-span (text).
- **FR-3.3** Preserve the **row/column structure**, including **header cells** (column headers and, where
  present, row headers) and **stable addressing** (row index/id, column index/id), so a cell is
  **queryable by its header or id** — e.g. "the cell under column 'Torque', row '6.4.2'".
- **FR-3.4** A table is an **atomic unit** for downstream chunking (never split mid-table) while keeping
  its internal cell structure intact.

### FR-4 — Provenance (highlight-grade)
- **FR-4.1** Every element — and every chunk derived from it — carries: `source_id`, **header-path**,
  **content-hash**, the **extractor used**, and **geometry**.
- **FR-4.2** **Char offsets are universal; visual boxes are additive.** Every element/chunk carries a
  **character-offset span** `(start, end)` into the document's normalized text — works for *all* source
  kinds, so offset-based retrieval/processing is uniform across PDF and text. PDFs **additionally** carry
  **page + bbox(es)** for visual highlighting (a span may cross lines/pages → multiple boxes);
  Markdown/text carry offsets only (which satisfies "highlight-grade" for Markdown — confirmed).
- **FR-4.3** Geometry must be **sufficient to highlight the exact source region** in a viewer
  (PDF page+bbox; text char-span; table → cell geometry per FR-3.2).
- **FR-4.4** Provenance threads end-to-end: extractor → `StructuredDocument` element → in-flight item →
  **chunk schema columns** (added via the existing `_CHUNK_PROVENANCE` seam in `storage/repository/base.py`).

### FR-5 — Enrichment components (run alongside the base extractor)
- **FR-5.1** After base extraction, a **configurable, ordered pipeline of enrichers** may annotate the
  `StructuredDocument`. Enrichers are **`Component`s** (registered, JSON-configurable) — same framework
  as the ingestion stages — so a user adds their own (NER, topic, classification, custom) **without
  touching base classes**.
- **FR-5.2** Enrichers must operate at **both document level** (e.g. topic/classification) **and
  element/span level** (e.g. named-entity spans), attaching results as annotations (FR-2.4).
- **FR-5.3** Each annotation records its **producer** and a **deterministic-vs-generative** flag
  (anti-hallucination: generative enrichers are explicitly marked and constrained).
- **FR-5.4** Enricher annotations carry their own provenance (which element/span) so they remain
  traceable downstream.
- **FR-5.5** Docling's own enrichments map in as annotations like any other producer (superset model).

### FR-6 — Pipeline / in-flight integration
- **FR-6.1** The in-flight contract carries the structured document by **extending `PipelineItem`**
  (preferred) with a typed structured payload, threaded through extract → enrich → chunk. The exact
  shape (typed field vs typed object in the existing bag, optionality for chunk-stage items) is settled
  in design; the requirement is that it **flows and is validated**.
- **FR-6.2** Extraction + enrichment compose under the existing `Settings.components` /
  `Pipeline`/`Component` model, so the whole document-processing stage is **configurable as data**.

### FR-7 — Anticipating the chunker (forward-compatibility)
The chunking algorithm is a separate doc, but the `StructuredDocument` contract must already serve it:
- **FR-7.1** Expose **hierarchy + reading order** so the chunker can split on structure (heading →
  paragraph → sentence).
- **FR-7.2** Flag **atomic units** the chunker must not split (paragraph, list-item, table, code block,
  figure+caption).
- **FR-7.3** Make **header-path** (FR-2.3) available per element for deterministic injection into each
  chunk.
- **FR-7.4** Provide **sentence boundaries** within text blocks where the extractor supplies them
  (best-effort) for sub-paragraph splitting.
- **FR-7.5** Give every element a **stable id** and a **contiguous source span**, so a chunk can
  reference its source element(s) and reconstruct a contiguous source region — this is what **late
  chunking** needs to embed whole-document context and map vectors back to chunks.
- **FR-7.6** **Geometry composes**: a chunk spanning several elements aggregates their geometry (list of
  page+bboxes / merged char-span), so highlight provenance survives chunk-boundary merges.

---

## 4. Non-functional requirements

- **NFR-1 Offline-capable.** Extractors/enrichers and their model weights run from local/pre-fetched
  assets (no runtime network dependency), consistent with the project's self-hosted posture.
- **NFR-2 Performance.** **Throughput is not a current priority** (explicitly deprioritized at kickoff):
  the design should stay reasonable at ~100k docs but need not hit a throughput target now. Per-worker
  model load is amortized (load once, not per document).
- **NFR-3 Hardware.** Must run **CPU-only**; a **GPU is optionally available** on ingestion workers and
  high-fidelity extractors/enrichers should use it when present.
- **NFR-4 Determinism / anti-hallucination.** Prefer deterministic/extractive extraction + enrichment;
  any generative step is flagged (FR-5.3) and must not silently enter provenance as fact.
- **NFR-5 Swappability.** Extractors and enrichers are swappable behind the contract (the "model
  choices are swappable behind the eval harness" principle).

---

## 5. Required contract changes (requirements-level)

| Existing contract | Required change |
|---|---|
| Parser seam `Callable[[str], str]` (`stages/parsers.py`) | Replace with an **extractor port** returning a `StructuredDocument` (registry stays plugin-style). |
| — (new) | A **`StructuredDocument`** model (superset of DoclingDocument), incl. a **structured table model** with cell geometry + header/id addressing — the central new contract. |
| — (new) | An **enricher port** (`Component`-based) that annotates a `StructuredDocument`. |
| In-flight `PipelineItem` (`content: str`) | **Extend** to carry the structured document through extract→enrich→chunk (FR-6.1). |
| Chunk schema + `_CHUNK_PROVENANCE` | Add **geometry** (char-span always; + page/bbox for PDF), **header-path**, element id + contiguous-span references (FR-7.5/7.6). |
| `LoadAndParseStage` (`MapperStage`, returns text) | Becomes the **structured-extraction stage**: route → extract → enrich → emit `StructuredDocument`. |

---

## 6. Acceptance criteria

- **AC-1** A PDF and a Markdown document each parse to a `StructuredDocument` with a heading hierarchy
  and per-element geometry (PDF: page+bbox; MD: char-span).
- **AC-2** A custom enricher (e.g. a stub NER) registers and runs **by config**, attaching element-level
  annotations, **without editing any base class**.
- **AC-3** A parsed table exposes **cell-level geometry** and lets you address a cell **by row/column
  header or id**.
- **AC-4** Each resulting chunk carries geometry + header-path + content-hash sufficient to **highlight
  the source region** (single element and multi-element chunks both, per FR-7.6).
- **AC-5** Swapping the PDF extractor (Docling ↔ fast born-digital) requires **no contract change**;
  adding a new format (e.g. HTML) is a new registered extractor only.
- **AC-6** Plain-text ingestion still succeeds end-to-end (graceful degradation, FR-2.5).
- **AC-7** A high-fidelity extractor uses the **GPU when present** and falls back to CPU when not.

---

## 7. Deferred to the design phase

- Exact shape of the `PipelineItem` extension (typed field vs typed object in the bag; how chunk-stage
  items relate to the document-level structure).
- Sentence-boundary granularity (FR-7.4) is best-effort and depends on the chosen extractor's output.
- The cell-addressing **query surface** (FR-3.3) — how callers query by row/column header downstream —
  is a retrieval/chunker concern detailed in those follow-on docs; here we only require the structure
  be captured and addressable.
