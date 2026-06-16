"""HTML extractor — BeautifulSoup visible text → a single-block StructuredDocument.

Like ``plain_text`` but strips markup (preserves the previous loader's behaviour). Structure-aware
HTML (heading hierarchy from ``<h1..6>``, tables) is a follow-on; ``beautifulsoup4`` is the optional
``parsers`` extra, imported lazily.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from tarnrag.contracts import Element, ElementKind, Span, StructuredDocument
from tarnrag.ingestion.extraction.extractor import Extractor, Source


class HtmlExtractor(Extractor):
    """Extract visible text from HTML (markup stripped) as one paragraph element."""

    class Config(Extractor.Config):
        class_name: Literal["html"] = "html"

    config: HtmlExtractor.Config

    def extract(self, source: Source) -> StructuredDocument:
        text = self._text(source)
        elements = (
            [Element(id="e0", kind=ElementKind.PARAGRAPH, text=text,
                     geometry=[Span(start=0, end=len(text))])]
            if text
            else []
        )
        return StructuredDocument(
            source_id=source.source_id,
            source_kind=source.source_kind or "html",
            extractor=self.config.class_name,
            text=text,
            elements=elements,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def _text(source: Source) -> str:
        from bs4 import BeautifulSoup

        raw = Extractor._read_text(source)
        return BeautifulSoup(raw, "html.parser").get_text(separator="\n").strip()
