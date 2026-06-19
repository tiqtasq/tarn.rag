"""The Reranker component family: a second-pass re-scoring of the retrieved shortlist.

A ``Reranker`` is an optional, config-driven ``Component`` that re-scores and re-orders the surviving
results just before truncation. ``cross_encoder`` scores each (query, passage) pair with a cross-encoder
model — more accurate than the first-pass retrievers, but too costly to run over the whole index, so it
runs last (after filter + auto-merge, over the shortlist). The model is injected at call time via the
``RetrievalContext`` (``ctx.cross_encoder``), like the embedder — the component itself stays a pure
strategy. The §9 decision is to **keep** the first-pass component scores: the rerank score is added under
its own key, the originals stay visible.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import replace
from typing import Literal

from tarnrag.contracts import RetrievalResult
from tarnrag.core.components import Component
from tarnrag.core.exceptions import RetrievalError
from tarnrag.retrieval.components.retriever import RetrievalContext
from tarnrag.retrieval.types import Query


class Reranker(Component):
    """Port: re-score + re-order the result shortlist (best first)."""

    class Config(Component.Config):
        """Base reranker config; concrete rerankers pin ``class_name``."""

    @abstractmethod
    async def rerank(
        self, query: Query, results: list[RetrievalResult], ctx: RetrievalContext
    ) -> list[RetrievalResult]:
        """Return the results re-scored and re-ordered (best first)."""


class CrossEncoderReranker(Reranker):
    """Re-score each result with a cross-encoder (query × text), set it as the new ``score``, and
    re-order. The original first-pass scores stay in ``component_scores``; the rerank score is added
    under ``score_key``. The model comes from ``ctx.cross_encoder`` (configured in ``RerankSettings``)."""

    class Config(Reranker.Config):
        class_name: Literal["cross_encoder"] = "cross_encoder"
        score_key: str = "cross_encoder"  # key the rerank score is surfaced under in component_scores

    config: CrossEncoderReranker.Config

    async def rerank(
        self, query: Query, results: list[RetrievalResult], ctx: RetrievalContext
    ) -> list[RetrievalResult]:
        if not results:
            return results
        if ctx.cross_encoder is None:
            raise RetrievalError(
                "a cross_encoder reranker is configured but the RetrievalContext has no cross-encoder "
                "model — set the `rerank` settings (model dir) so the engine can build it"
            )
        # Cross-encoders are CPU-bound (ONNX) — score off the event loop, like query embedding.
        scores = await asyncio.to_thread(ctx.cross_encoder.score, query.text, [r.text for r in results])
        rescored = [
            replace(r, score=s, component_scores={**r.component_scores, self.config.score_key: s})
            for r, s in zip(results, scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored
