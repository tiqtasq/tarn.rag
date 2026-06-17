"""RetrievalPipeline — the retrieval composition as a container Component.

A *sibling* of the ingestion ``Pipeline`` (both are config-driven container ``Component``s; not a
subclass — retrieval's flow is heterogeneous: parallel retrievers + fan-in + fixed steps). It builds the
configured retrievers + fuser as children and owns the flow:

    retrieve (parallel, over-fetch) → fuse → top_k → hydrate → assemble.

(The license/scope filter, auto-merging, and reranking land in later slices.) The ``RetrievalEngine`` is
a thin facade over it: it does the compatibility check + store/embedder construction, then delegates.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import Field

from tarnrag.contracts import MethodRef, RetrievalResult
from tarnrag.core.components import Component, ComponentFactory
from tarnrag.retrieval.fuser import Fuser
from tarnrag.retrieval.retriever import RetrievalContext, Retriever
from tarnrag.retrieval.types import Query


class RetrievalPipeline(Component):
    """Compose the configured retrievers + fuser and run the retrieval flow over a ``RetrievalContext``."""

    class Config(Component.Config):
        class_name: Literal["retrieval_pipeline"] = "retrieval_pipeline"
        retrievers: list[dict[str, Any]] = Field(default_factory=lambda: [{"class_name": "dense"}])
        fuser: dict[str, Any] = Field(default_factory=lambda: {"class_name": "identity"})

    config: RetrievalPipeline.Config

    def __init__(self, config: RetrievalPipeline.Config) -> None:
        super().__init__(config)
        self._retrievers: list[Retriever] = []
        self._fuser: Fuser | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the retriever + fuser children through the framework factory (the container hook)."""
        self._retrievers = [factory.create_as(spec, Retriever) for spec in self.config.retrievers]
        self._fuser = factory.create_as(self.config.fuser, Fuser)

    async def search(self, query: Query, ctx: RetrievalContext) -> list[RetrievalResult]:
        self._ensure_children()
        lists = await asyncio.gather(*(r.retrieve(query, ctx) for r in self._retrievers))
        per_retriever = {r.config.class_name: candidates for r, candidates in zip(self._retrievers, lists)}
        fused = self._fuser.fuse(per_retriever)[: query.top_k]
        records = {rec.chunk_id: rec for rec in await ctx.store.hydrate([h.chunk_id for h in fused])}
        results: list[RetrievalResult] = []
        for h in fused:
            rec = records.get(h.chunk_id)
            if rec is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=rec.chunk_id,
                    text=rec.text,
                    score=h.score,
                    component_scores=h.component_scores,
                    document_id=rec.document_id,
                    source_kind=rec.source_kind,
                    standard_id=rec.standard_id,
                    locator=rec.locator,
                    license_class=rec.license_class,
                    methods=[MethodRef(m, v) for m, v in rec.methods],
                    provenance=rec.provenance,
                )
            )
        return results
