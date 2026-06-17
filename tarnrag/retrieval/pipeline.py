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

from tarnrag.contracts import ChunkRecord, MethodRef, RetrievalResult
from tarnrag.core.components import Component, ComponentFactory
from tarnrag.retrieval.fuser import Fuser
from tarnrag.retrieval.retriever import RetrievalContext, Retriever
from tarnrag.retrieval.types import ALL, Purpose, Query


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
        fused = self._fuser.fuse(per_retriever)  # ranked, over-fetched; filter then take top_k
        records = {rec.chunk_id: rec for rec in await ctx.store.hydrate([h.chunk_id for h in fused])}
        results: list[RetrievalResult] = []
        for h in fused:
            rec = records.get(h.chunk_id)
            if rec is None or not self._passes(rec, query):
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
            if len(results) >= query.top_k:
                break
        return results

    @staticmethod
    def _passes(record: ChunkRecord, query: Query) -> bool:
        """The license/scope filter (query-driven): drop unavailable chunks; drop non-grounding chunks
        for a GENERATION_GROUNDING query; restrict to the query's method scope. (License-class-by-purpose
        policy is a deferred extension — it needs a purpose → allowed-classes mapping.)"""
        if not record.available:
            return False
        if query.purpose == Purpose.GENERATION_GROUNDING and not record.ai_grounding_allowed:
            return False
        return RetrievalPipeline._in_scope(record, query.scope)

    @staticmethod
    def _in_scope(record: ChunkRecord, scope: object) -> bool:
        """Whether ``record`` is in the query's method ``scope`` (``ALL`` ⇒ unrestricted). A scoped
        ``MethodRef`` with no version matches any version of its method id."""
        if scope == ALL:
            return True
        return any(
            ref.method_id == mid and (ref.method_version is None or ref.method_version == ver)
            for ref in scope  # type: ignore[union-attr]
            for mid, ver in record.methods
        )
