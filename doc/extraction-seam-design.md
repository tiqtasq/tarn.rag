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

## Built-in extractors (this slice)

- **`plain_text`** — graceful default (FR-2.5): any text → one paragraph element spanning the whole
  document. No dependencies; the fallback for unknown kinds.
- **`markdown`** — structure-aware first cut: ATX headings, paragraphs, fenced code, with char-offset
  geometry into the normalized text and precomputed `header_path` / `parent_id`. **Follow-on:** lists
  and pipe-tables (→ `Table`/`TableCell` with cell geometry; the model is already in place and tested).

## Deferred (next slices on this branch)

- **`docling` PDF extractor** — maps `DoclingDocument` → `StructuredDocument` (page+bbox geometry,
  structured tables). Heavy dep (model weights, optional GPU); lazily imported; tiered behind the fast
  born-digital path.
- **`LoadAndParse` rewiring** — `LoadAndParseStage` becomes the structured-extraction stage: detect
  `source_kind`, call `extract(source)`, set `item.document` + `item.content = document.text`. Done as
  its own slice because it changes the live pipeline (the old string `content` path stays working —
  `content` is still the text view).
- **Enricher stage contract (FR-5)** and the **structure-aware chunker (FR-7)**.
