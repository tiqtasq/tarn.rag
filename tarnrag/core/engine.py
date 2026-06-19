"""Base class for the engine facades — shared construction + async lifecycle.

``IngestionEngine`` (the index producer) and ``RetrievalEngine`` (the reader) both open from
``Settings`` and both run over a single ``DocumentRepository``. ``Engine`` holds what they share:
the construction of the backing store + embedder (built with the same ``EMBEDDING_DIMENSION`` so
a producer-built index opens for the reader), and the async-context lifecycle over the repository.
"""

from __future__ import annotations

from typing import Self

from tarnrag.core.config import Settings
from tarnrag.core.resources.embedder import Embedder, build_embedder
from tarnrag.storage.repository import DocumentRepository


class Engine:
    """
    Shared base for the two engine facades. Subclasses construct via their own ``create`` /
    ``open`` (which set ``repository``); the base supplies the store+embedder construction and
    the repository lifecycle (``aclose`` + async-context manager).
    """

    repository: DocumentRepository  # set by each subclass's constructor

    @staticmethod
    async def _build_repository_and_embedder(
        settings: Settings,
    ) -> tuple[DocumentRepository, Embedder]:
        """Connect the repository and build the shared embedder from one ``Settings`` — the
        common prefix of both engines' construction. Both receive ``EMBEDDING_DIMENSION`` so the
        index and the query embedder agree on vector size (else a producer-built index won't
        ``open`` for the reader — the embedding fingerprint check fails)."""
        repository = await DocumentRepository.create(
            settings.database, settings.EMBEDDING_DIMENSION
        )
        embedder = build_embedder(settings.embedding, settings.EMBEDDING_DIMENSION)
        return repository, embedder

    async def aclose(self) -> None:
        """Release the repository (its async engine pool)."""
        await self.repository.disconnect()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
