"""Chunking: split a loaded document into retrievable chunks.

Importing this package registers the built-in chunk stage with the global ``ComponentFactory`` (it
self-registers under its ``class_name``), so ``Pipeline.from_spec`` / ``ComponentFactory`` can build it.
The structure-aware chunkers (a ``Chunker`` component family) will land alongside ``chunk.py`` here.
"""

from tarnrag.ingestion.chunking.chunk import ChunkStage

__all__ = ["ChunkStage"]
