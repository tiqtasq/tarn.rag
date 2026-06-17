"""RetrievalEngine — turns a query into ranked, provenance-bearing results (ModusQ §5).

A thin async facade over the §8 repository: it does the compatibility check + store/embedder
construction (the ``Engine`` base), builds the ``RetrievalPipeline`` from ``Settings`` (the
``RETRIEVAL_PIPELINE`` spec — dense by default; configure ``retrievers`` + a ``fuser`` for hybrid), and
delegates ``search`` to it. ``open()`` is the one place compatibility is checked — it refuses an index
built with a different embedding pipeline (fingerprint) or schema. **Comparing retrieval methods = varying
the ``RETRIEVAL_PIPELINE`` spec.**
"""

from __future__ import annotations

from typing import Any

from tarnrag.core.components import ComponentFactory
from tarnrag.core.config import RETRIEVAL_PIPELINE, Settings, get_settings
from tarnrag.core.embedder import Embedder
from tarnrag.core.engine import Engine
from tarnrag.core.exceptions import RetrievalError
from tarnrag.contracts import SCHEMA_VERSION, RetrievalResult
from tarnrag.storage.repository import DocumentRepository
from tarnrag.retrieval.pipeline import RetrievalPipeline
from tarnrag.retrieval.retriever import RetrievalContext
from tarnrag.retrieval.types import Query

_DEFAULT_PIPELINE: dict[str, Any] = {"class_name": "retrieval_pipeline"}  # dense + identity fuser


class RetrievalEngine(Engine):
    """
    Async query facade over the §8 repository. ``create`` / ``open`` validate index compatibility; the
    engine then delegates ``search`` to a config-driven ``RetrievalPipeline`` (retrieve → fuse → hydrate
    → assemble). ``search`` / ``search_text`` are async — they await the store.
    """

    def __init__(self, repository: DocumentRepository, embedder: Embedder, config: Any = None):
        self.repository = repository
        self.embedder = embedder
        self.config = config
        spec = getattr(config, "components", {}).get(RETRIEVAL_PIPELINE) if config is not None else None
        self._pipeline = ComponentFactory.get().create_as(spec or _DEFAULT_PIPELINE, RetrievalPipeline)

    @classmethod
    async def open(
        cls, repository: DocumentRepository, embedder: Embedder, config: Any = None
    ) -> RetrievalEngine:
        """Validate compatibility, then return a query-ready engine. Refuses on mismatch."""
        meta = await repository.index_meta()
        if not meta.get("schema_version"):
            raise RetrievalError("retrieval index has not been built yet (no index_meta)")
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise RetrievalError(
                f"schema_version mismatch: index {meta.get('schema_version')!r} "
                f"!= engine {SCHEMA_VERSION!r}"
            )
        index_fp = meta.get("embedding_config_fingerprint")
        engine_fp = embedder.config_fingerprint()
        if index_fp != engine_fp:
            raise RetrievalError(
                "embedding_config_fingerprint mismatch — the index was built with a different "
                f"embedding pipeline (index {index_fp!r} != engine {engine_fp!r})"
            )
        return cls(repository, embedder, config)

    @classmethod
    async def create(cls, settings: Settings | None = None) -> RetrievalEngine:
        """Open a query-ready engine straight from ``Settings`` — connects the repository (the same
        store ingestion writes) and the shared embedder, then validates compatibility via ``open``."""
        settings = settings or get_settings()
        repository, embedder = await cls._build_repository_and_embedder(settings)
        return await cls.open(repository, embedder, config=settings)

    async def search(self, query: Query) -> list[RetrievalResult]:
        """Run the configured retrieval pipeline (retrieve → fuse → top_k → hydrate → assemble)."""
        return await self._pipeline.search(query, RetrievalContext(self.repository, self.embedder))

    async def search_text(
        self, text: str, *, top_k: int = 8, dense_k: int = 50
    ) -> list[RetrievalResult]:
        """Convenience over :meth:`search`: build a :class:`Query` from a raw string."""
        return await self.search(Query(text=text, top_k=top_k, dense_k=dense_k))
