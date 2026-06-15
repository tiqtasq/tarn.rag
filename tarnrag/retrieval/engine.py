"""RetrievalEngine — turns a query into ranked, provenance-bearing results (ModusQ §5).

Step A is **dense-only**: embed the query → dense KNN → hydrate → assemble. Retrieval runs on the
``DocumentRepository`` (the same store ingestion writes — sqlite-vec on SQLite, pgvector on
Postgres), so the engine is **async**: ``create`` / ``open`` / ``search`` await the repository.
``open()`` is the one place compatibility is checked — it refuses to run against an index built
with a different embedding pipeline (fingerprint) or schema. Step B adds the sparse retriever,
RRF fusion, and the license/scope filter behind the ``Retriever``/``Fuser`` seams.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tarnrag.core.config import Settings, get_settings
from tarnrag.composition import build_repository_and_embedder
from tarnrag.embedder import Embedder
from tarnrag.contracts import SCHEMA_VERSION, MethodRef, RetrievalResult
from tarnrag.storage.repository import DocumentRepository
from tarnrag.retrieval.types import Query


class RetrievalError(Exception):
    """
    The engine cannot serve queries against this index (incompatibility at ``open()``).
    """


class RetrievalEngine:
    """
    Async query facade over the §8 repository (ModusQ §5): embed → KNN → hydrate → assemble. Built
    via ``create`` (or the ``open`` seam), which refuses an index whose embedding fingerprint or
    schema differs. ``search`` / ``search_text`` are async — they await the repository.
    """

    def __init__(self, repository: DocumentRepository, embedder: Embedder, config: Any = None):
        self.repository = repository
        self.embedder = embedder
        self.config = config

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
        """Open a query-ready engine straight from ``Settings`` — connects the repository (the
        same store ingestion writes) and the shared embedder, then validates compatibility via
        ``open``. The easy entry point; ``open`` is the lower-level seam for injecting your own
        repository/embedder."""
        settings = settings or get_settings()
        repository, embedder = await build_repository_and_embedder(settings)
        return await cls.open(repository, embedder, config=settings)

    async def search(self, query: Query) -> list[RetrievalResult]:
        """Dense-only (Step A): embed → KNN → truncate top_k → hydrate → assemble. The embedding
        is offloaded to a thread (ONNX is CPU-bound and releases the GIL) so it doesn't block the
        event loop; the KNN + hydrate are the repository's async calls."""
        query_vec = await asyncio.to_thread(self.embedder.embed_query, query.text)
        candidates = (await self.repository.dense_knn(query_vec, query.dense_k))[: query.top_k]
        records = {
            r.chunk_id: r
            for r in await self.repository.hydrate([c.chunk_id for c in candidates])
        }
        results: list[RetrievalResult] = []
        for c in candidates:
            rec = records.get(c.chunk_id)
            if rec is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=rec.chunk_id,
                    text=rec.text,
                    score=-c.raw_score,  # distance → higher is better
                    component_scores={"dense": c.raw_score},
                    document_id=rec.document_id,
                    source_kind=rec.source_kind,
                    standard_id=rec.standard_id,
                    locator=rec.locator,
                    license_class=rec.license_class,
                    methods=[MethodRef(m, v) for m, v in rec.methods],
                )
            )
        return results

    async def search_text(
        self, text: str, *, top_k: int = 8, dense_k: int = 50
    ) -> list[RetrievalResult]:
        """Convenience over :meth:`search`: build a :class:`Query` from a raw string."""
        return await self.search(Query(text=text, top_k=top_k, dense_k=dense_k))

    async def aclose(self) -> None:
        """Release the repository (its async engine pool)."""
        await self.repository.disconnect()

    async def __aenter__(self) -> RetrievalEngine:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
