"""Chunking: split a loaded document into retrieval chunks + the auto-merging tree.

Importing this package registers the ``Chunk`` stage and the built-in chunkers (``structure_aware``,
``recursive``) with the global ``ComponentFactory``, so ``Pipeline.from_spec`` / ``ComponentFactory``
can build them. The chunkers are imported before ``ChunkStage`` (the driver builds a chunker child).
"""

from tarnrag.ingestion.chunking.chunker import Chunker
from tarnrag.ingestion.chunking.recursive import RecursiveCharacterChunker
from tarnrag.ingestion.chunking.structure_aware import StructureAwareChunker
from tarnrag.ingestion.chunking.chunk import ChunkStage

__all__ = [
    "Chunker",  # TempChunk is reached as Chunker.TempChunk
    "RecursiveCharacterChunker",
    "StructureAwareChunker",
    "ChunkStage",
]
