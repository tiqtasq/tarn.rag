"""The Retriever component family: a query → ranked candidates from the store.

A ``Retriever`` is a config-driven ``Component`` (registered by ``class_name``, swappable by spec) — a
pure strategy. The store + query embedder are injected at call time via a ``RetrievalContext`` (the
retriever holds no repository), mirroring how ingestion stages don't hold the store. ``RetrievalPipeline``
runs the configured retrievers and fuses their candidate lists.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

from tarnrag.contracts import Candidate, RetrievalStore
from tarnrag.core.components import Component
from tarnrag.core.embedder import Embedder
from tarnrag.retrieval.types import Query


@dataclass
class RetrievalContext:
    """What a retriever needs at query time: the read store + the query embedder."""

    store: RetrievalStore
    embedder: Embedder


class Retriever(Component):
    """Port: produce ranked ``Candidate``s (best first) for a query, from the store."""

    class Config(Component.Config):
        """Base retriever config; concrete retrievers pin ``class_name``."""

    @abstractmethod
    async def retrieve(self, query: Query, ctx: RetrievalContext) -> list[Candidate]:
        """The query's ranked candidates (best first); ``k`` comes from the ``Query``."""


class DenseRetriever(Retriever):
    """Embed the query (off-thread; ONNX is CPU-bound) and KNN the dense vector index."""

    class Config(Retriever.Config):
        class_name: Literal["dense"] = "dense"

    config: DenseRetriever.Config

    async def retrieve(self, query: Query, ctx: RetrievalContext) -> list[Candidate]:
        vec = await asyncio.to_thread(ctx.embedder.embed_query, query.text)
        return await ctx.store.dense_knn(vec, query.dense_k)


class SparseRetriever(Retriever):
    """Lexical (BM25 / tsvector) search over the sparse index."""

    class Config(Retriever.Config):
        class_name: Literal["sparse"] = "sparse"

    config: SparseRetriever.Config

    async def retrieve(self, query: Query, ctx: RetrievalContext) -> list[Candidate]:
        return await ctx.store.sparse_search(query.text, query.sparse_k)
