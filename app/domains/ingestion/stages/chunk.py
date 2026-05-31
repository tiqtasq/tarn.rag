"""ChunkStage — split a document into retrieval-sized chunks."""

from __future__ import annotations

from typing import Any

from app.domains.ingestion.pipeline import ChunkerStage

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class ChunkStage(ChunkerStage):
    """Recursive character chunking: split on coarse→fine separators, then pack the
    pieces into ``chunk_size`` windows with ``overlap`` characters carried between
    adjacent chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50, **config: Any):
        # Set before super().__init__(), which runs validate().
        self.chunk_size = chunk_size
        self.overlap = overlap
        super().__init__(name="Chunk", chunk_size=chunk_size, overlap=overlap, **config)

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        return [(c, {"chunk_size": len(c)}) for c in self._split_recursive(text)]

    def _split_recursive(self, text: str) -> list[str]:
        pieces = self._split_to_pieces(text, _SEPARATORS)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) > self.chunk_size:
                chunks.append(current)
                current = (current[-self.overlap :] if self.overlap else "") + piece
            else:
                current += piece
        if current.strip():
            chunks.append(current)
        return chunks or [text]

    def _split_to_pieces(self, text: str, separators: list[str]) -> list[str]:
        """Break text into pieces each <= chunk_size, recursing to finer separators
        when a piece is still too large."""
        if len(text) <= self.chunk_size or not separators:
            return [text]
        sep, rest = separators[0], separators[1:]
        parts = text.split(sep) if sep else list(text)
        pieces: list[str] = []
        for part in parts:
            unit = part + sep if sep else part
            if len(unit) <= self.chunk_size:
                pieces.append(unit)
            else:
                pieces.extend(self._split_to_pieces(unit, rest))
        return pieces

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
