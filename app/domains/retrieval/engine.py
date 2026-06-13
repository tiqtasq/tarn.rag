"""RetrievalEngine — turns a query into ranked, provenance-bearing results (ModusQ §5).

Step A is **dense-only**: embed the query → sqlite-vec KNN → hydrate → assemble. The engine is
**sync** (matches the index store and the future C++ port); the async API bridges via a thread.
``open()`` is the one place compatibility is checked — it refuses to run against an index built
with a different embedding pipeline (fingerprint) or schema. Step B adds the sparse retriever,
RRF fusion, and the license/scope filter behind the ``Retriever``/``Fuser`` seams.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.domains.base.embedder import Embedder, OnnxEmbedder
from app.domains.base.index_store import SCHEMA_VERSION, SqliteIndexStore
from app.domains.retrieval.types import MethodRef, Query, RetrievalResult


class RetrievalError(Exception):
    """The engine cannot serve queries against this index (incompatibility at ``open()``)."""


class RetrievalEngine:
    def __init__(self, store: SqliteIndexStore, embedder: Embedder, config: Any = None):
        self.store = store
        self.embedder = embedder
        self.config = config

    @classmethod
    def open(
        cls, store: SqliteIndexStore, embedder: Embedder, config: Any = None
    ) -> RetrievalEngine:
        """Validate compatibility, then return a query-ready engine. Refuses on mismatch."""
        meta = store.index_meta()
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
        return cls(store, embedder, config)

    @classmethod
    def create(cls, settings: Settings | None = None) -> RetrievalEngine:
        """Open a query-ready engine straight from ``Settings`` — builds the index store
        (read-only) and the shared embedder, then validates compatibility via ``open``. The
        easy entry point; ``open`` is the lower-level seam for injecting your own store/embedder."""
        settings = settings or get_settings()
        store = SqliteIndexStore.from_settings(settings)
        embedder = OnnxEmbedder.from_settings(settings)
        return cls.open(store, embedder, config=settings)

    def search(self, query: Query) -> list[RetrievalResult]:
        """Dense-only (Step A): embed → KNN → truncate top_k → hydrate → assemble."""
        query_vec = self.embedder.embed_query(query.text)
        candidates = self.store.dense_knn(query_vec, query.dense_k)[: query.top_k]
        records = {r.chunk_id: r for r in self.store.hydrate([c.chunk_id for c in candidates])}
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

    def search_text(self, text: str, *, top_k: int = 8, dense_k: int = 50) -> list[RetrievalResult]:
        """Convenience over :meth:`search`: build a :class:`Query` from a raw string."""
        return self.search(Query(text=text, top_k=top_k, dense_k=dense_k))

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> RetrievalEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
