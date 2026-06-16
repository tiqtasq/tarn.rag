"""Markdown extractor — parse Markdown structure into a ``StructuredDocument``.

A focused first cut: ATX headings, paragraphs, and fenced code blocks, with char-offset geometry into
the normalized document text and precomputed header paths. Lists and pipe-tables map onto the model
(``Table``/``TableCell`` etc.) too but are a follow-on; for now non-heading/code lines coalesce into
paragraphs (graceful, lossless as text).
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from tarnrag.contracts import Element, ElementKind, Span, StructuredDocument
from tarnrag.ingestion.extraction.extractor import Extractor, Source

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = "```"
_SEP = "\n\n"  # joins normalized block texts into StructuredDocument.text


class MarkdownExtractor(Extractor):
    """Structure-aware Markdown extraction (headings / paragraphs / code) with offsets + header paths."""

    class Config(Extractor.Config):
        class_name: Literal["markdown"] = "markdown"

    config: MarkdownExtractor.Config

    def extract(self, source: Source) -> StructuredDocument:
        elements, text = self._assemble(self._blocks(self._read_text(source)))
        return StructuredDocument(
            source_id=source.source_id,
            source_kind=source.source_kind or "markdown",
            extractor=self.config.class_name,
            text=text,
            elements=elements,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def _blocks(md: str) -> list[tuple[str, str, int | None]]:
        """Split Markdown into ``(kind, normalized_text, level)`` blocks; ``level`` set for headings."""
        lines = md.splitlines()
        blocks: list[tuple[str, str, int | None]] = []
        i, n = 0, len(lines)
        while i < n:
            line = lines[i]
            if line.strip().startswith(_FENCE):
                code: list[str] = []
                i += 1
                while i < n and not lines[i].strip().startswith(_FENCE):
                    code.append(lines[i])
                    i += 1
                i += 1  # consume the closing fence (if present)
                blocks.append(("code", "\n".join(code), None))
                continue
            heading = _HEADING.match(line)
            if heading:
                blocks.append(("heading", heading.group(2).strip(), len(heading.group(1))))
                i += 1
                continue
            if not line.strip():
                i += 1
                continue
            para: list[str] = []
            while i < n and lines[i].strip() and not MarkdownExtractor._is_special(lines[i]):
                para.append(lines[i].strip())
                i += 1
            blocks.append(("paragraph", " ".join(para), None))
        return blocks

    @staticmethod
    def _is_special(line: str) -> bool:
        return bool(_HEADING.match(line)) or line.strip().startswith(_FENCE)

    @staticmethod
    def _assemble(blocks: list[tuple[str, str, int | None]]) -> tuple[list[Element], str]:
        """Build elements + the normalized text, assigning each element a span into that text and the
        header path of its enclosing headings (``parent_id`` = the deepest enclosing heading)."""
        elements: list[Element] = []
        stack: list[tuple[int, str, str]] = []  # (level, text, element_id) of open headings
        parts: list[str] = []
        offset = 0
        for idx, (kind, text, level) in enumerate(blocks):
            start = offset
            end = start + len(text)
            offset = end + len(_SEP)
            parts.append(text)
            eid = f"e{idx}"
            if kind == "heading" and level is not None:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                elements.append(
                    Element(
                        id=eid, kind=ElementKind.HEADING, text=text, level=level,
                        header_path=[t for _, t, _ in stack],
                        parent_id=stack[-1][2] if stack else None,
                        geometry=[Span(start=start, end=end)],
                    )
                )
                stack.append((level, text, eid))
            else:
                elements.append(
                    Element(
                        id=eid,
                        kind=ElementKind.CODE if kind == "code" else ElementKind.PARAGRAPH,
                        text=text,
                        header_path=[t for _, t, _ in stack],
                        parent_id=stack[-1][2] if stack else None,
                        geometry=[Span(start=start, end=end)],
                    )
                )
        return elements, _SEP.join(parts)
