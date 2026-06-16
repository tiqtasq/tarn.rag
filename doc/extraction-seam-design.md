# Extraction Seam — Design

**Status:** Draft v0.1 · **Date:** 2026-06-16 · **Branch:** `feature/ingestion-redesign-layout-aware-extraction`.
**Implements (from):** [`layout-aware-extraction-requirements.md`](./layout-aware-extraction-requirements.md)
(FR-1) and [`structured-document-design.md`](./structured-document-design.md) (the model it produces).

How a `Source` becomes a `StructuredDocument` — the contract that replaces the `Callable[[str], str]`
parser registry.

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

## Deferred (next slices on this branch)

- **Tiered routing config** — today `pdf → pdf_text` (fast) with `docling` opt-in per document; a
  per-format `{fast, high}` map (chosen via Settings/config) is the small remaining piece. Markdown
  lists/pipe-tables are the other extractor follow-on (the `Table` model is already in place).
- **`LoadAndParse` rewiring** — `LoadAndParseStage` becomes the structured-extraction stage: detect
  `source_kind`, call `extract(source)`, set `item.document` + `item.content = document.text`. Done as
  its own slice because it changes the live pipeline (the old string `content` path stays working —
  `content` is still the text view).
- **Enricher stage contract (FR-5)** and the **structure-aware chunker (FR-7)**.
