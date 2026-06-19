"""RetrievalEngine — turns a query into ranked, provenance-bearing results (ModusQ §5).

A thin async facade over the §8 repository: it does the compatibility check + store/embedder
construction (the ``Engine`` base), builds the ``Searcher`` from ``Settings`` (the ``RETRIEVAL_PIPELINE``
spec — a ``RetrievalPipeline`` (dense by default; configure ``retrievers`` + a ``fuser`` for hybrid) or a
``RoutingRetrievalPipeline``), and delegates ``search`` to it. ``open()`` is the one place compatibility is checked — it refuses an index
built with a different embedding pipeline (fingerprint) or schema. **Comparing retrieval methods = varying
the ``RETRIEVAL_PIPELINE`` spec.**
"""

from __future__ import annotations

from typing import Any

from tarnrag.core.components import ComponentFactory
from tarnrag.core.engine.config import RETRIEVAL_PIPELINE, Settings, get_settings
from tarnrag.core.resources.cross_encoder import CrossEncoder, OnnxCrossEncoder
from tarnrag.core.resources.embedder import Embedder
from tarnrag.core.engine.engine import Engine
from tarnrag.core.exceptions import RetrievalError
from tarnrag.contracts import SCHEMA_VERSION, RetrievalResult
from tarnrag.storage.repository import DocumentRepository
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.components.searcher import Searcher
from tarnrag.retrieval.types import Query

_DEFAULT_PIPELINE: dict[str, Any] = {"class_name": "retrieval_pipeline"}  # dense + identity fuser


class RetrievalEngine(Engine):
    """
    Async query facade over the §8 repository. ``create`` / ``open`` validate index compatibility; the
    engine then delegates ``search`` to a config-driven ``RetrievalPipeline`` (retrieve → fuse → hydrate
    → assemble). ``search`` / ``search_text`` are async — they await the store.
    """

    def __init__(self, repository: DocumentRepository, embedder: Embedder, settings: Settings | None = None):
        self.repository = repository
        self.embedder = embedder
        self.settings = settings
        spec = settings.components.get(RETRIEVAL_PIPELINE) if settings is not None else None
        self._pipeline: Searcher = ComponentFactory.get().create_as(spec or _DEFAULT_PIPELINE, Searcher)
        # The cross-encoder is built (cheaply — the model loads lazily) only when settings are present;
        # it's unused unless the pipeline has a reranker. A direct ``open()`` without settings has none.
        self._cross_encoder: CrossEncoder | None = (
            OnnxCrossEncoder.create(settings.rerank) if settings is not None else None
        )

    @classmethod
    async def open(
        cls, repository: DocumentRepository, embedder: Embedder, settings: Settings | None = None
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
        return cls(repository, embedder, settings)

    @classmethod
    async def create(cls, settings: Settings | None = None) -> RetrievalEngine:
        """Open a query-ready engine straight from ``Settings`` — connects the repository (the same
        store ingestion writes) and the shared embedder, then validates compatibility via ``open``."""
        settings = settings or get_settings()
        repository, embedder = await cls._build_repository_and_embedder(settings)
        return await cls.open(repository, embedder, settings)

    async def search(self, query: Query) -> list[RetrievalResult]:
        """Run the configured retrieval pipeline (retrieve → fuse → hydrate → filter → merge → rerank →
        top_k → assemble)."""
        ctx = RetrievalContext(self.repository, self.embedder, self._cross_encoder)
        return await self._pipeline.search(query, ctx)

    async def search_text(
        self, text: str, *, top_k: int = 8, dense_k: int = 50
    ) -> list[RetrievalResult]:
        """Convenience over :meth:`search`: build a :class:`Query` from a raw string."""
        return await self.search(Query(text=text, top_k=top_k, dense_k=dense_k))
