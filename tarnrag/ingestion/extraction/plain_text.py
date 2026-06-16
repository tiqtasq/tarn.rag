"""Plain-text extractor — the graceful-degradation default (FR-2.5).

Any readable text becomes a one-paragraph ``StructuredDocument`` spanning the whole document. No
dependencies; this is the fallback for unknown/unsupported source kinds.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from tarnrag.contracts import Element, ElementKind, Span, StructuredDocument
from tarnrag.ingestion.extraction.extractor import Extractor, Source


class PlainTextExtractor(Extractor):
    """Wrap raw text as a single paragraph element covering the whole document."""

    class Config(Extractor.Config):
        class_name: Literal["plain_text"] = "plain_text"

    config: PlainTextExtractor.Config

    def extract(self, source: Source) -> StructuredDocument:
        text = self._read_text(source)
        elements = (
            [Element(id="e0", kind=ElementKind.PARAGRAPH, text=text,
                     geometry=[Span(start=0, end=len(text))])]
            if text
            else []
        )
        return StructuredDocument(
            source_id=source.source_id,
            source_kind=source.source_kind or "text",
            extractor=self.config.class_name,
            text=text,
            elements=elements,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
