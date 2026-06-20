# StructuredDocument & PipelineItem Extension — Design

**Status:** ✅ **Implemented** (2026-06-20). Drafted 2026-06-16; built since.
**Implements (from):** [`layout-aware-extraction-requirements.md`](./layout-aware-extraction-requirements.md).
**Scope:** the two gating contracts only — the `StructuredDocument`/table model and the `PipelineItem`
extension.

**Where the code lives now:** the model is in `tarnrag/contracts/structure.py` (`PageBox`, `Span`,
`Geometry`, `ElementKind`, `Annotation`, `TableCell`, `Table`, `Element`, `StructuredDocument`,
`ChunkProvenance`); the `PipelineItem` extension (`document` / `provenance` + `derive()`) is in
`tarnrag/contracts/dtos.py`. The consumers this doc named as deferred are also built: the enricher stage
(`ingestion/components/enrichment/`), the structure-aware chunker (`ingestion/components/chunking/
structure_aware.py`), and the geometry/table/annotation **schema** (chunk columns `header_path` / `level` /
`parent_chunk_id` / `geometry` + the `table_cells` and `chunk_annotations` tables in
`storage/repository/base.py`).

**Deltas from this draft** (the model as built): `Element` gained a `sentences: list[Span] | None` field
(FR-7.4) and a `subspan()` helper; `ChunkProvenance` gained `level` + `parent_chunk_id` (the auto-merging
tree) and `table` (so a table chunk carries its cells). The sketches below are otherwise accurate.

Code blocks below are the original **illustrative pydantic sketches** — see `contracts/structure.py` for
the final shapes.

## Design decisions (quick scan)

| # | Decision | Choice & why |
|---|---|---|
| D1 | Two in-flight phases | *document phase* (`item.document` set) → *chunk phase* (`item.provenance` set). One uniform `PipelineItem`, the engine stays generic. |
| D2 | Geometry | `list[Span]`; **every span has char offsets** (`start`/`end` into the doc's normalized text — universal) plus optional **`boxes`** (page+bbox) for PDF visual highlighting. Offsets are the universal substrate; bbox is an additive visual layer, so PDFs get *both*. |
| D3 | Hierarchy | **Flat ordered `elements` (reading order) + `parent_id` + `level`**, not a nested tree — easier for chunker/enrichers to iterate; header-path precomputed per element. |
| D4 | PipelineItem extension | Add **two optional typed fields** `document` and `provenance` (phase-specific, explicit) rather than a union or loose metadata. Both default `None` ⇒ backward-compatible. |
| D5 | Uniform extraction | **Everything** (even plain `.txt`/`.md`) yields a `StructuredDocument` (trivial = one text element), so the document phase is uniform (FR-2.5). |
| D6 | Tables | Structured `cells` (row/col/spans, header flags, **cell-level geometry**) **+** a rendered `markdown` form; in-memory header/id addressing via helpers. |
| D7 | Mutability | The `StructuredDocument` is **mutable** (pydantic, not frozen): enrichers append annotations to elements/doc. (In-place vs copy is the enricher-stage's call — out of scope here.) |

---

## 1. The two phases (how it flows)

```
LoadAndParse(route + extract)        item.document = StructuredDocument ; item.content = doc.text()
        │                            item.provenance = None
        ▼
Enricher*  (FR-5)                     annotate item.document (elements / doc-level); item unchanged otherwise
        │
        ▼
Chunk  (FR-7)                         read item.document → fan out N chunk items, each:
        │                              item.content    = chunk text
        │                              item.provenance = ChunkProvenance(source_element_ids, geometry, header_path, …)
        │                              item.document   = None
        ▼
Embed → ResultSink                    sink maps item.provenance → chunk schema columns (via _CHUNK_PROVENANCE)
```

`content: str` stays meaningful in **both** phases (document plain-text rendering, then chunk text), so
any text-only stage (e.g. clean/normalise, embed) keeps working unchanged.

---

## 2. Geometry (D2 / FR-4)

**Char offsets are the universal substrate; page+bbox is an additive visual layer.** Every span carries
`start`/`end` (char offsets into the document's *normalized text* — the substrate that gets chunked and
embedded), so all offset-based functionality (span-overlap math, char-based highlighting in a text view,
late-chunking offset→chunk mapping, sub-element annotation spans, contiguous-span reconstruction) works
**identically on PDF and text**. A span *additionally* carries `boxes` (page + bbox) when the source is
paged/visual (PDF); text/markdown leave it empty.

```python
class PageBox(BaseModel):
    page: int                                   # 1-based
    bbox: tuple[float, float, float, float]     # (x0, y0, x1, y1) in PDF points

class Span(BaseModel):
    start: int                                  # char offset into the document's normalized text — UNIVERSAL
    end: int
    boxes: list[PageBox] = []                   # visual boxes covering this span (PDF); [] for text/markdown

Geometry = list[Span]                           # usually one contiguous span; a list when a chunk merges
                                                # non-adjacent source (e.g. injected header + body)
```

This is exactly what Docling's `ProvenanceItem` does (it pairs a `charspan` **with** `bbox` + `page_no`),
so carrying both is *more* faithful to "superset of DoclingDocument," not less. (An earlier draft modelled
PDF and text as mutually-exclusive `Region`s — that was a modelling error: it would have excluded PDFs
from every offset-based retrieval/processing path.)

---

## 3. `StructuredDocument` model (FR-2)

Superset of DoclingDocument: a flat, reading-ordered element list with hierarchy by reference.

```python
class ElementKind(str, Enum):
    HEADING = "heading"; PARAGRAPH = "paragraph"; LIST = "list"; LIST_ITEM = "list_item"
    TABLE = "table"; FIGURE = "figure"; CAPTION = "caption"; CODE = "code"

class Annotation(BaseModel):                    # FR-2.4 / FR-5
    producer: str                               # enricher Component name (or "docling")
    type: str                                   # "entity" | "topic" | "classification" | …
    value: dict[str, Any]                       # typed payload, e.g. {"label": "PERSON", "text": "…"}
    span: Geometry | None = None                # span-level (None = whole element / document)
    deterministic: bool = True                  # FR-5.3 anti-hallucination flag

class Element(BaseModel):
    id: str                                     # stable (FR-7.5)
    kind: ElementKind
    text: str                                   # canonical extracted text of this element
    geometry: Geometry = []                     # FR-4
    header_path: list[str] = []                 # FR-2.3 / FR-7.3, e.g. ["3 Safety", "3.2 Lockout"]
    parent_id: str | None = None                # enclosing section/heading element (D3)
    level: int | None = None                    # heading depth, when kind == HEADING
    atomic: bool = True                         # FR-7.2: chunker must not split this element
    annotations: list[Annotation] = []
    table: "Table | None" = None                # set iff kind == TABLE

class StructuredDocument(BaseModel):
    source_id: str
    source_kind: str                            # "pdf" | "markdown" | …
    extractor: str                              # which extractor ran (FR-1.4 provenance)
    elements: list[Element]                     # reading order (D3)
    annotations: list[Annotation] = []          # document-level (e.g. topic/classification)
    content_hash: str
    # helpers (not fields): text() -> reading-order concat; element_by_id(id); elements_in(span)
```

**Hierarchy (D3):** reading order = list order; nesting = `parent_id` (+ `level` for headings).
`header_path` is precomputed on each element so chunker/sinks never re-walk the tree. A nested tree was
rejected — it complicates linear iteration (the chunker's primary access pattern) for no gain we need.

---

## 4. Table model (D6 / FR-3)

```python
class TableCell(BaseModel):
    id: str
    row: int; col: int
    row_span: int = 1; col_span: int = 1
    is_column_header: bool = False              # FR-3.3
    is_row_header: bool = False
    text: str
    geometry: Geometry = []                     # FR-3.2 cell-level

class Table(BaseModel):
    n_rows: int; n_cols: int
    cells: list[TableCell]
    markdown: str                               # rendered form for embedding/display
    # helpers: cell_at(row, col); column_headers_for(cell); row_headers_for(cell)
```

- **Cell-level geometry** (FR-3.2) lives on each `TableCell`.
- **Header & id addressing** (FR-3.3): header cells are flagged; `column_headers_for`/`row_headers_for`
  resolve a data cell to its header texts by position (handling spans). This satisfies the model
  requirement; **persisting** tables so they're *queryable* by header at retrieval time is a
  schema/retrieval follow-on (deferred per requirements §7).
- The table is **one atomic element** for chunking (`Element.atomic = True`) while keeping `cells` intact.

This maps cleanly onto Docling's `TableData`/`TableCell` (row/col offsets, span, `column_header`/`row_header`),
so the Docling backend is a near-direct mapping; other backends fill what they can.

---

## 5. `PipelineItem` extension (D4 / FR-6.1)

Today: `id`, `content: str`, `metadata`. The extension adds two **optional, phase-specific** typed fields:

```python
class PipelineItem(BaseModel):
    id: str | None = None
    content: str                                # text view (doc plain-text | chunk text)
    metadata: dict[str, Any] = Field(default_factory=dict)
    document: StructuredDocument | None = None      # set in the document phase
    provenance: "ChunkProvenance | None" = None     # set in the chunk phase
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ChunkProvenance(BaseModel):               # the chunk-phase target the Chunk stage fills
    source_element_ids: list[str]               # contiguous element span this chunk covers (FR-7.5)
    geometry: Geometry = []                      # aggregated across merged elements (FR-7.6)
    header_path: list[str] = []
    content_hash: str
    annotations: list[Annotation] = []           # annotations overlapping this chunk
```

**Why two fields, not a union or metadata:**
- *Explicit phase intent* — `document` is the rich doc; `provenance` is the per-chunk structured trace.
  A union (`structure: StructuredDocument | ChunkProvenance`) obscures which phase an item is in.
- *Typed + validated* (the requirement) — geometry/header-path are structured; stuffing bbox lists into
  the loose `metadata` bag would be fragile and unqueryable.
- *Backward-compatible* — both default `None`, so every existing `PipelineItem(content=…)` and test still
  constructs and flows unchanged; the loose-metadata philosophy is preserved for everything else.

**Note on weight:** the chunk-phase item carries `provenance` (element-**id** references), **not** a copy
of the whole `StructuredDocument` — so fan-out doesn't duplicate the document N times. When **late
chunking** is designed it can thread the document (or re-fetch by id); we leave a clean seam, not the
mechanism.

---

## 6. How it threads to storage (the sink, briefly)

The chunk `ResultSink` maps `item.provenance` → chunk columns via the existing `_CHUNK_PROVENANCE` seam in
`storage/repository/base.py`. New chunk columns (exact types are the **schema design's** job, deferred):
`header_path`, `geometry` (serialized region list), `source_element_ids`, alongside the existing
`content_hash`/`locator`. Cell-level table structure persists with the table chunk (likely serialized);
making it *queryable by header* is the retrieval follow-on.

---

## 7. How it serves the consumers (validation of the contract)

- **Enrichers (FR-5):** receive the document-phase item, append `Annotation`s to `item.document`
  (element-level via `Element.annotations`, doc-level via `StructuredDocument.annotations`), each tagged
  with `producer` + `deterministic`. *(The enricher stage's process signature is its own design.)*
- **Chunker (FR-7):** iterates `elements` in reading order, respects `atomic`, reads `header_path`, splits
  text blocks on best-effort sentence boundaries, and emits chunk items whose `provenance` references the
  source element ids and aggregates their geometry. Everything it needs (D3, FR-7.1–7.6) is present.
- **Highlighting:** a chunk's `provenance.geometry` (PDF page+bbox / text char-span, table → cell geometry)
  is sufficient to highlight the exact source region (AC-3/AC-4).

---

## 8. Open forks (resolved as built)

1. **Geometry persistence** — ✅ stored as a **JSON column** (`chunks.geometry`, `table_cells.geometry`,
   `chunk_annotations.span`), cross-dialect; not yet normalised (no retrieval filter needs it).
2. **`atomic` default** — ✅ `Element.atomic = True`; the chunker splits *within* an oversize text element
   (`StructureAwareChunker._oversize`) and keeps tables/list-items whole.
3. **Sentence boundaries (FR-7.4)** — ✅ carried as `Element.sentences: list[Span] | None` when the
   extractor provides them (`None` ⇒ not provided); the chunker's text packing is element-level today.
4. **`content` in the document phase** — ✅ `LoadAndParseStage` sets `item.content = document.text`, so the
   text-only middle stages (clean/normalize, embed) keep working unchanged.
