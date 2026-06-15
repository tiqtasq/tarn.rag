"""Shared composition — the wiring common to both engine facades.

``IngestionEngine.create`` (the index **producer**) and ``RetrievalEngine.create`` (the
**reader**) both open from ``Settings`` and need the same two pieces: the connected
``DocumentRepository`` and the shared ``OnnxEmbedder``, each built with the same
``EMBEDDING_DIMENSION``. Building them here, once, keeps the producer and reader from drifting
on that dimension — they MUST agree, or an index the writer produces won't ``open`` for the
reader (the embedding fingerprint check fails).
"""

from __future__ import annotations

from tarnrag.core.config import Settings
from tarnrag.embedder import Embedder, OnnxEmbedder
from tarnrag.storage.repository import DocumentRepository


async def build_repository_and_embedder(
    settings: Settings,
) -> tuple[DocumentRepository, Embedder]:
    """Connect the repository and build the shared embedder from one ``Settings`` — the common
    prefix of both engines' ``create()``. Both receive ``EMBEDDING_DIMENSION`` so the index and
    the query embedder agree on vector size."""
    repository = await DocumentRepository.create(
        settings.database, settings.EMBEDDING_DIMENSION
    )
    embedder = OnnxEmbedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
    return repository, embedder
