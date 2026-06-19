"""Born-digital PDF extractor (the fast tier) — pdfplumber text lines with page+bbox geometry.

The fast path for born-digital PDFs (FR-1.4): each extracted text line becomes a paragraph element
carrying its page + bounding box, so chunks are highlight-grade (page+bbox) AND offset-addressable
(char spans into the normalized text). It does NOT recover semantic structure (headings, tables) or
OCR scanned pages — that is the high-fidelity Docling extractor (a follow-on). ``pdfplumber`` is an
optional dep (the ``parsers`` extra); it is imported lazily, so importing this module never needs it.
"""

from __future__ import annotations

from typing import Literal

from tarnrag.contracts import Element, ElementKind, PageBox, Span, StructuredDocument
from tarnrag.ingestion.components.extraction.extractor import Extractor, Source

_SEP = "\n"  # joins line texts into StructuredDocument.text


class PdfTextExtractor(Extractor):
    """Fast born-digital PDF extraction: one paragraph element per text line, with page+bbox geometry."""

    class Config(Extractor.Config):
        class_name: Literal["pdf_text"] = "pdf_text"

    config: PdfTextExtractor.Config

    def extract(self, source: Source) -> StructuredDocument:
        if not source.path:
            raise ValueError("the 'pdf_text' extractor needs a PDF file path (source.path)")
        elements, text = self._extract(source.path)
        return self._create_document(source, text, elements, extractor=self.config.class_name, default_kind="pdf")

    @staticmethod
    def _extract(path: str) -> tuple[list[Element], str]:
        import pdfplumber

        elements: list[Element] = []
        parts: list[str] = []
        offset = 0
        with pdfplumber.open(path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                for line in page.extract_text_lines():
                    text = line["text"].strip()
                    if not text:
                        continue
                    start = offset
                    end = start + len(text)
                    offset = end + len(_SEP)
                    parts.append(text)
                    bbox = (
                        float(line["x0"]), float(line["top"]),
                        float(line["x1"]), float(line["bottom"]),
                    )
                    elements.append(
                        Element(
                            id=f"e{len(elements)}",
                            kind=ElementKind.PARAGRAPH,
                            text=text,
                            geometry=[Span(start=start, end=end,
                                           boxes=[PageBox(page=page_no, bbox=bbox)])],
                        )
                    )
        return elements, _SEP.join(parts)
