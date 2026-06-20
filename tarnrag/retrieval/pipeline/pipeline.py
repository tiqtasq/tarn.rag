"""RetrievalPipeline — the retrieval composition as a container Component.

A *sibling* of the ingestion ``Pipeline`` (both are config-driven container ``Component``s; not a
subclass — retrieval's flow is heterogeneous: parallel retrievers + fan-in + fixed steps). It builds the
configured retrievers + fuser (+ optional merger / reranker) as children and owns the flow:

    retrieve (parallel, over-fetch) → fuse → hydrate → filter → auto-merge → rerank → top_k → assemble.

The ``RetrievalEngine`` is a thin facade over it: it does the compatibility check + store/embedder/
cross-encoder construction, then delegates.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import Field

from tarnrag.contracts import RetrievalResult
from tarnrag.core.components import ComponentFactory
from tarnrag.retrieval.components.fuser import Fuser
from tarnrag.retrieval.components.merger import Merger
from tarnrag.retrieval.components.reranker import Reranker
from tarnrag.retrieval.components.retriever import RetrievalContext, Retriever
from tarnrag.retrieval.pipeline.searcher import Searcher
from tarnrag.retrieval.types import Query


class RetrievalPipeline(Searcher):
    """Compose the configured retrievers + fuser and run the retrieval flow over a ``RetrievalContext``."""

    class Config(Searcher.Config):
        class_name: Literal["retrieval_pipeline"] = "retrieval_pipeline"
        retrievers: list[dict[str, Any]] = Field(default_factory=lambda: [{"class_name": "dense"}])
        fuser: dict[str, Any] = Field(default_factory=lambda: {"class_name": "identity"})
        merger: dict[str, Any] | None = None  # optional auto-merging step (none ⇒ skipped)
        reranker: dict[str, Any] | None = None  # optional second-pass rerank (none ⇒ skipped)

    config: RetrievalPipeline.Config

    def __init__(self, config: RetrievalPipeline.Config) -> None:
        super().__init__(config)
        self._retrievers: list[Retriever] = []
        self._fuser: Fuser | None = None
        self._merger: Merger | None = None
        self._reranker: Reranker | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the retriever + fuser (+ optional merger / reranker) children through the factory."""
        self._retrievers = [factory.create_as(spec, Retriever) for spec in self.config.retrievers]
        self._fuser = factory.create_as(self.config.fuser, Fuser)
        self._merger = factory.create_as(self.config.merger, Merger) if self.config.merger else None
        self._reranker = (
            factory.create_as(self.config.reranker, Reranker) if self.config.reranker else None
        )

    async def search(self, query: Query, ctx: RetrievalContext) -> list[RetrievalResult]:
        self._ensure_children()
        lists = await asyncio.gather(*(r.retrieve(query, ctx) for r in self._retrievers))
        per_retriever = {r.config.class_name: candidates for r, candidates in zip(self._retrievers, lists)}
        # Candidates are already license/scope-filtered inside each retriever (over-fetched), so the
        # fused set is permitted; merge / rerank, then top_k.
        fused = self._fuser.fuse(per_retriever)
        records = {rec.chunk_id: rec for rec in await ctx.store.hydrate([h.chunk_id for h in fused])}
        results = [
            RetrievalResult.from_record(rec, score=h.score, component_scores=h.component_scores)
            for h in fused
            if (rec := records.get(h.chunk_id)) is not None
        ]
        if self._merger is not None:
            results = await self._merger.merge(results, ctx)
        if self._reranker is not None:
            results = await self._reranker.rerank(query, results, ctx)
        return results[: query.top_k]
