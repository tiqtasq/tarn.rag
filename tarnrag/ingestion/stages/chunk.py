"""ChunkStage — split a document into retrieval-sized chunks."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from tarnrag.ingestion.pipeline import ChunkerStage

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class ChunkStage(ChunkerStage):
    """
    Recursive character chunking: split on coarse→fine separators, then pack the
    pieces into ``chunk_size`` windows with ``overlap`` characters carried between
    adjacent chunks.
    """

    class Config(ChunkerStage.Config):
        class_name: Literal["Chunk"] = "Chunk"
        chunk_size: int = Field(default=512, gt=0)
        overlap: int = Field(default=50, ge=0)

        @model_validator(mode="after")
        def _overlap_below_chunk_size(self) -> ChunkStage.Config:
            if self.overlap >= self.chunk_size:
                raise ValueError("overlap must be smaller than chunk_size")
            return self

    config: ChunkStage.Config

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        return [(c, {"chunk_size": len(c)}) for c in self._split_recursive(text)]

    def _split_recursive(self, text: str) -> list[str]:
        """
        Split into separator-bounded pieces, then greedily pack them into ``chunk_size`` windows,
        carrying ``overlap`` trailing characters into the next chunk.
        """
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
        """Break text into pieces each <= chunk_size, recursing to finer separators
        when a piece is still too large."""
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
