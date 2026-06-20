# Extraction Seam — Design

**Status:** ✅ **Implemented** (2026-06-20). Drafted 2026-06-16.
**Implements (from):** [`layout-aware-extraction-requirements.md`](./layout-aware-extraction-requirements.md)
(FR-1) and [`structured-document-design.md`](./structured-document-design.md) (the model it produces).

How a `Source` becomes a `StructuredDocument` — the contract that **replaced** the `Callable[[str], str]`
parser registry. Code: `tarnrag/ingestion/components/extraction/` (`extractor.py` = `Source` + `Extractor`
+ `_create_document`/`_read_text`; `plain_text.py`, `markdown.py`, `html.py`, `pdf.py` (`pdf_text`),
`docling_pdf.py`; `load_parse.py` = the `LoadAndParseStage` router). The routing lives on
`LoadAndParseStage.Config.routes` (a `source_kind → extractor spec` map) + `default_extractor`, not a
standalone `extract()` function.

## Decisions

| # | Decision | Why |
|---|---|---|
| E1 | **Extractors are `Component`s** (registered by `class_name`, like pipeline stages) | Consistency with the whole config-driven architecture; extractor options (e.g. Docling settings) become `Config` fields; tiered routing becomes config. The same framework the enrichers (FR-5) will use. |
| E2 | A small `Source` input (id, kind, path-or-content, hints) | Decouples extractors from "how the bytes arrived"; carries routing hints (`metadata['extractor']`). |
| E3 | `extract(source)` **router**: explicit override → `source_kind` map → plain-text fallback | FR-1.2/1.4 (plugin registry + routing) and FR-1.3 (graceful fallback). Tiered fast/high-fidelity slots in as the route values grow to per-format maps. |
| E4 | Offsets index the **normalized text** the extractor emits (`StructuredDocument.text`) | Matches the model: one substrate for chunking/embedding/overlap; per extractor (plain text = the text itself; markdown = cleaned block text). |

## Port

```python
class Source(BaseModel):
    source_id: str
    source_kind: str = ""          # "pdf" | "markdown" | "text" | … (detected/declared by the caller)
    path: str | None = None
    content: str | None = None     # already-loaded text, if not from disk
    metadata: dict[str, Any] = {}  # hints, e.g. {"extractor": "plain_text"}

class Extractor(Component):
    class Config(Component.Config): ...          # concrete extractors pin class_name + add options
    @abstractmethod
    def extract(self, source: Source) -> StructuredDocument: ...

def extract(source: Source) -> StructuredDocument:   # the router
    name = source.metadata.get("extractor") or DEFAULT_ROUTES.get(source.source_kind, "plain_text")
    return ComponentFactory.get().create({"class_name": name}).extract(source)
```

## Built-in extractors

- **`plain_text`** — graceful default (FR-2.5): any text → one paragraph element spanning the whole
  document. No dependencies; the fallback for unknown kinds.
- **`markdown`** — structure-aware first cut: ATX headings, paragraphs, fenced code, with char-offset
  geometry into the normalized text and precomputed `header_path` / `parent_id`. **Follow-on:** lists
  and pipe-tables (→ `Table`/`TableCell` with cell geometry; the model is already in place and tested).
- **`pdf_text`** — fast born-digital tier (FR-1.4): pdfplumber text lines → paragraph elements with
  **page + bbox** geometry (highlight-grade) *and* char offsets. No semantic structure (headings/tables)
  or OCR — that is Docling. `pdfplumber` is the optional `parsers` extra (lazily imported). Route:
  `pdf → pdf_text`.
- **`docling`** — high-fidelity tier: maps a `DoclingDocument` → heading hierarchy, paragraphs/lists/code,
  and **structured tables (cell-level geometry + header addressing)**, with page+bbox provenance. The
  `DoclingDocument → StructuredDocument` mapping (`_map`) depends only on `docling-core` and is
  unit-tested against a *constructed* document; the heavy converter (`docling`, the optional `[docling]`
  extra) is imported lazily and exercised by a converter-gated e2e test. Opt-in via
  `metadata['extractor'] = "docling"`. *(pymupdf4llm intentionally skipped: its Markdown output loses
  highlight-grade bbox geometry, and it is AGPL.)*

## Status of the follow-ons

- **`LoadAndParse` rewiring** — ✅ done. `LoadAndParseStage` is the structured-extraction stage: it infers
  `source_kind`, selects the routed extractor (per-document `metadata['extractor']` override wins), calls
  `extract(source)`, and sets `item.document` + `item.content = document.text`. The old string-`content`
  path still works (`content` is the text view).
- **Enricher stage contract (FR-5)** — ✅ done (`ingestion/components/enrichment/` — `EnrichStage` runs a
  configured list of `Enricher`s over `item.document`; default is none).
- **Structure-aware chunker (FR-7)** — ✅ done (`StructureAwareChunker`, the default chunker).
- **Tiered routing config** — ◑ partial. The route map *is* config (`routes` keys `pdf → pdf_text` by
  default; point a kind at `{"class_name": "docling"}` for high-fidelity, or use the per-document
  `metadata['extractor']` override). A first-class per-format `{fast, high}` selector is the small piece
  still open. Markdown lists/pipe-tables are the remaining markdown-extractor follow-on (the `Table` model
  is in place and persisted).
