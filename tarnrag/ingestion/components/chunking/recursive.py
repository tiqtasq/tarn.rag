"""Recursive character chunker — the structure-agnostic fallback.

Splits the document's normalized text on coarse→fine separators and packs the pieces into
``chunk_size`` windows with ``overlap`` carried between neighbours (the original chunking behaviour,
now a ``Chunker`` component). Each window is mapped back to the source elements its char-span overlaps,
so even a structure-blind chunk still carries a header path and (for citation) its char geometry. It is
flat — every chunk is a leaf (``level`` 0, no parents).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tarnrag.contracts import ChunkProvenance, Element, Span, StructuredDocument
from tarnrag.ingestion.components.chunking.chunker import Chunker

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveCharacterChunker(Chunker):
    """Recursive character chunking: split on coarse→fine separators, then pack the pieces into
    ``chunk_size`` windows with ``overlap`` characters carried between adjacent chunks."""

    class Config(Chunker.Config):
        class_name: Literal["recursive"] = "recursive"
        chunk_size: int = Field(default=512, gt=0)
        overlap: int = Field(default=50, ge=0)

        @model_validator(mode="after")
        def _overlap_below_chunk_size(self) -> RecursiveCharacterChunker.Config:
            if self.overlap >= self.chunk_size:
                raise ValueError("overlap must be smaller than chunk_size")
            return self

    config: RecursiveCharacterChunker.Config

    def chunk(self, document: StructuredDocument) -> list[Chunker.TempChunk]:
        text = document.text
        chunks: list[Chunker.TempChunk] = []
        cursor = 0
        for piece in self._split_recursive(text):
            if not piece.strip():
                continue
            start = text.find(piece, max(0, cursor - self.config.overlap))
            if start < 0:
                start = cursor
            end = start + len(piece)
            cursor = end
            chunks.append(self._make_chunk(piece, document, start, end))
        if not chunks and text.strip():
            chunks.append(self._make_chunk(text, document, 0, len(text)))
        return chunks

    def _make_chunk(self, text: str, document: StructuredDocument, start: int, end: int) -> Chunker.TempChunk:
        """A leaf chunk over ``text``, with provenance mapped from the elements its span overlaps."""
        covered = self._elements_overlapping(document, start, end)
        return Chunker.TempChunk(
            text=text,
            provenance=ChunkProvenance(
                source_element_ids=[e.id for e in covered],
                geometry=[Span(start=start, end=end)],  # char span only — the recursive cut may straddle boxes
                header_path=covered[0].header_path if covered else [],
                content_hash=self._hash(text),
                annotations=self._annotations(covered),
            ),
        )

    @staticmethod
    def _elements_overlapping(document: StructuredDocument, start: int, end: int) -> list[Element]:
        """The elements (reading order) any of whose spans overlap the half-open range ``[start, end)``."""
        out: list[Element] = []
        for element in document.elements:
            if any(span.start < end and start < span.end for span in element.geometry):
                out.append(element)
        return out

    def _split_recursive(self, text: str) -> list[str]:
        """Split into separator-bounded pieces, then greedily pack them into ``chunk_size`` windows,
        carrying ``overlap`` trailing characters into the next chunk."""
        size, overlap = self.config.chunk_size, self.config.overlap
        pieces = self._split_to_pieces(text, _SEPARATORS)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) > size:
                chunks.append(current)
                current = (current[-overlap:] if overlap else "") + piece
            else:
                current += piece
        if current.strip():
            chunks.append(current)
        return chunks or [text]

    def _split_to_pieces(self, text: str, separators: list[str]) -> list[str]:
        """Break text into pieces each <= chunk_size, recursing to finer separators when a piece is
        still too large."""
        if len(text) <= self.config.chunk_size or not separators:
            return [text]
        sep, rest = separators[0], separators[1:]
        parts = text.split(sep) if sep else list(text)
        pieces: list[str] = []
        for part in parts:
            unit = part + sep if sep else part
            if len(unit) <= self.config.chunk_size:
                pieces.append(unit)
            else:
                pieces.extend(self._split_to_pieces(unit, rest))
        return pieces
