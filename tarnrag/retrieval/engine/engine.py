"""RetrievalEngine — turns a query into ranked, provenance-bearing results (ModusQ §5).

A thin async facade over the §8 repository: it validates index compatibility (``open()`` — the one place
checked, refusing an index built with a different embedding pipeline (fingerprint) or schema), builds the
``Searcher`` from ``Settings`` (the ``RETRIEVAL_PIPELINE`` spec — a ``RetrievalPipeline`` (dense by default;
configure ``retrievers`` + a ``fuser`` for hybrid) or a ``RoutingRetrievalPipeline``), and delegates
``search`` to it. **Comparing retrieval methods = varying the ``RETRIEVAL_PIPELINE`` spec.**
"""

from __future__ import annotations

from typing import Any

from tarnrag.core.components import ComponentFactory
from tarnrag.core.engine.config import RETRIEVAL_PIPELINE, Settings, get_settings
from tarnrag.core.resources.cross_encoder import CrossEncoder, OnnxCrossEncoder
from tarnrag.core.resources.embedder import Embedder
from tarnrag.core.engine.engine import Engine
from tarnrag.core.exceptions import RetrievalError
from tarnrag.contracts import IndexMeta, RetrievalResult
from tarnrag.storage.repository import DocumentRepository
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.pipeline.searcher import Searcher
from tarnrag.retrieval.types import Query

_DEFAULT_PIPELINE: dict[str, Any] = {"class_name": "retrieval_pipeline"}  # dense + identity fuser


class RetrievalEngine(Engine):
    """
    Async query facade over the §8 repository. Built from ``Settings`` via ``create`` / ``open`` (which
    validate index compatibility); the bare constructor is the data-holder seam — inject a pre-built
    ``Searcher`` (tests / advanced wiring). Delegates ``search`` to a config-driven ``RetrievalPipeline``
    (retrieve → fuse → hydrate → assemble); ``search`` / ``search_text`` are async — they await the store.
    """

    def __init__(
        self,
        repository: DocumentRepository,
        embedder: Embedder,
        searcher: Searcher,
        cross_encoder: CrossEncoder | None = None,
    ):
        """Pure data-holder: inject the resources (the read store + query embedder + an optional reranker)
        and the pre-built ``Searcher``. Build from ``Settings`` via :meth:`open` / :meth:`create`."""
        self.repository = repository
        self.embedder = embedder
        self._pipeline = searcher
        self._cross_encoder = cross_encoder

    @staticmethod
    def build_searcher(settings: Settings | None = None) -> Searcher:
        """Build the retrieval ``Searcher`` from the ``RETRIEVAL_PIPELINE`` spec (default: dense + identity
        fuser) — the retrieval analog of ``IngestionEngine.build_pipeline``."""
        spec = settings.components.get(RETRIEVAL_PIPELINE) if settings is not None else None
        return ComponentFactory.get().create_as(spec or _DEFAULT_PIPELINE, Searcher)

    @classmethod
    async def open(
        cls, repository: DocumentRepository, embedder: Embedder, settings: Settings | None = None
    ) -> RetrievalEngine:
        """Validate index compatibility, then return a query-ready engine — the ``Searcher`` + (only when
        a reranker is configured) the cross-encoder, built from ``Settings``. Refuses on a schema or
        embedding-fingerprint mismatch."""
        meta = await repository.index_meta()
        if not meta.get("schema_version"):
            raise RetrievalError("retrieval index has not been built yet (no index_meta)")
        conflict = IndexMeta.conflict(meta, embedder)
        if conflict:
            raise RetrievalError(conflict)
        # The cross-encoder's model loads lazily, so building it here is cheap; it stays unused unless the
        # configured pipeline has a reranker (a settings-less ``open`` has none).
        cross_encoder = OnnxCrossEncoder.create(settings.rerank) if settings is not None else None
        return cls(repository, embedder, cls.build_searcher(settings), cross_encoder)

    @classmethod
    async def create(
        cls,
        settings: Settings | None = None,
        *,
        repository: DocumentRepository | None = None,
        embedder: Embedder | None = None,
    ) -> RetrievalEngine:
        """Open a query-ready engine from ``Settings`` — connecting the repository (the same store
        ingestion writes) + the shared embedder, then validating via :meth:`open`. A composition root
        (``TarnRag``) may inject a pre-built ``repository`` + ``embedder`` to share one store/embedder
        across engines; pass both or neither (mirrors ``IngestionEngine.create``)."""
        settings = settings or get_settings()
        if repository is None and embedder is None:
            repository = await DocumentRepository.create(settings.database, settings.EMBEDDING_DIMENSION)
            embedder = Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
        elif repository is None or embedder is None:
            raise ValueError("inject both repository and embedder, or neither")
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
