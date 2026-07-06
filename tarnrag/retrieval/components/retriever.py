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
from typing import TYPE_CHECKING, Literal

from tarnrag.contracts import Candidate, ChunkFilter, RetrievalStore
from bausatz import Component
from tarnrag.core.parsing import extract_json
from tarnrag.core.resources.cross_encoder import CrossEncoder
from tarnrag.core.resources.embedder import Embedder
from tarnrag.core.resources.llm import LanguageModel, Prompt
from tarnrag.retrieval.types import Query

if TYPE_CHECKING:
    from tarnrag.retrieval.components.license_policy import LicensePolicy


@dataclass
class RetrievalContext:
    """The runtime resources retrieval components need at query time: the read store, the query embedder,
    the optional cross-encoder model (only when a reranker is configured), the optional license policy
    (the engine injects it when built from ``Settings``; ``None`` ⇒ no license-class filtering), and the
    optional ``llm`` — injected for the bridge components (multi-query expansion, the LLM relevance judge);
    ``None`` on the lean default path, which makes no LLM call."""

    store: RetrievalStore
    embedder: Embedder
    cross_encoder: CrossEncoder | None = None
    license_policy: LicensePolicy | None = None
    llm: LanguageModel | None = None

    def filter_for(self, query: Query) -> ChunkFilter:
        """The permitted-chunk filter for ``query`` — via the injected ``LicensePolicy`` (purpose →
        license classes + availability / grounding / scope), or the policy-free baseline
        (``Query.permitted_filter``) when no policy is configured."""
        if self.license_policy is not None:
            return self.license_policy.filter_for(query)
        return query.permitted_filter()


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
        return await ctx.store.dense_knn(vec, query.dense_k, ctx.filter_for(query))


class SparseRetriever(Retriever):
    """Lexical (BM25 / tsvector) search over the sparse index."""

    class Config(Retriever.Config):
        class_name: Literal["sparse"] = "sparse"

    config: SparseRetriever.Config

    async def retrieve(self, query: Query, ctx: RetrievalContext) -> list[Candidate]:
        return await ctx.store.sparse_search(query.text, query.sparse_k, ctx.filter_for(query))


# The bridge query-expansion prompt — rewrite into several focused search queries (paraphrases + per-sub-fact
# queries for multi-hop), so dense retrieval over the union lifts recall. ``N`` is filled with num_variants.
_EXPAND_SYSTEM = (
    "Rewrite the user's question as N alternative search queries that together retrieve the facts needed "
    "to answer it — close paraphrases, and for multi-hop questions a query for each sub-fact. Reply with a "
    'SINGLE JSON object and nothing else: {"queries": ["...", "..."]}. Keep each query short and focused.'
)


class MultiQueryRetriever(Retriever):
    """Bridge retriever: expand the query into several variants via ``ctx.llm``, dense-retrieve each, and
    RRF-fuse the candidate lists — lifts recall on under-specified / multi-hop questions (Phase-2 bridge
    substrate). The original query is always included; with no ``ctx.llm`` it degrades gracefully to a
    single dense retrieval over the original. An LLM call per query (the bridge is opt-in, off the lean
    default)."""

    class Config(Retriever.Config):
        class_name: Literal["multi_query"] = "multi_query"
        num_variants: int = 3  # LLM rewrites (the original query is always included as well)
        rrf_k: int = 60  # RRF constant fusing the per-variant rankings

    config: MultiQueryRetriever.Config

    async def retrieve(self, query: Query, ctx: RetrievalContext) -> list[Candidate]:
        variants = await self._expand(query, ctx)
        vectors = await asyncio.to_thread(lambda: [ctx.embedder.embed_query(v) for v in variants])
        chunk_filter = ctx.filter_for(query)
        lists = await asyncio.gather(
            *(ctx.store.dense_knn(vec, query.dense_k, chunk_filter) for vec in vectors)
        )
        return self._fuse(lists)

    async def _expand(self, query: Query, ctx: RetrievalContext) -> list[str]:
        """``[original, *rewrites]`` — the original plus up to ``num_variants`` LLM rewrites (just the
        original if no LLM, or the reply doesn't parse)."""
        if ctx.llm is None:
            return [query.text]
        system = _EXPAND_SYSTEM.replace("N", str(self.config.num_variants))
        completion = await ctx.llm.complete(Prompt(system=system, user=f"Question: {query.text}"))
        data = extract_json(completion.text)
        raw = data.get("queries") if isinstance(data, dict) else None
        rewrites = [str(q).strip() for q in raw if str(q).strip()] if isinstance(raw, list) else []
        return [query.text, *rewrites[: self.config.num_variants]]

    def _fuse(self, lists: list[list[Candidate]]) -> list[Candidate]:
        """RRF the per-variant candidate lists into one ranked ``Candidate`` list (deterministic tie-break:
        fused score desc, then chunk_id asc — the ModusQ §5.5 contract)."""
        scores: dict[str, float] = {}
        raw: dict[str, float] = {}
        for candidates in lists:
            for c in candidates:
                scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (self.config.rrf_k + c.rank)
                raw.setdefault(c.chunk_id, c.raw_score)
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))
        return [Candidate(chunk_id=cid, rank=i + 1, raw_score=raw[cid]) for i, cid in enumerate(ranked)]
