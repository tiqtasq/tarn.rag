"""The Fuser component family: merge per-retriever candidate lists into one ranked list.

A ``Fuser`` is a config-driven ``Component``. ``identity`` is the single-retriever passthrough (the flow
stays uniform — always retrieve → fuse); ``rrf`` is Reciprocal Rank Fusion for hybrid retrieval. Each
``FusedHit`` keeps the per-retriever scores (``component_scores``) so the breakdown stays transparent.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from tarnrag.contracts import Candidate
from tarnrag.core.components import Component


@dataclass
class FusedHit:
    """A fused hit: a chunk id, its fused ``score`` (higher = better), and the per-retriever scores."""

    chunk_id: str
    score: float
    component_scores: dict[str, float] = field(default_factory=dict)


class Fuser(Component):
    """Port: merge ``{retriever_name: candidates}`` into one ranked list of ``FusedHit``s (best first)."""

    class Config(Component.Config):
        """Base fuser config; concrete fusers pin ``class_name``."""

    @abstractmethod
    def fuse(self, per_retriever: dict[str, list[Candidate]]) -> list[FusedHit]:
        """Fuse the per-retriever candidate lists into one ranked list."""


class IdentityFuser(Fuser):
    """Single-retriever passthrough: keep the retriever's ranking (``score = -rank``, so higher is
    better and ``top_k`` truncation by score works), surfacing its raw score as the component score."""

    class Config(Fuser.Config):
        class_name: Literal["identity"] = "identity"

    config: IdentityFuser.Config

    def fuse(self, per_retriever: dict[str, list[Candidate]]) -> list[FusedHit]:
        return [
            FusedHit(c.chunk_id, score=-float(c.rank), component_scores={name: c.raw_score})
            for name, candidates in per_retriever.items()
            for c in candidates
        ]


class RRFFuser(Fuser):
    """Reciprocal Rank Fusion: ``score(chunk) = Σ_retrievers 1 / (k + rank)`` — rank-based, so it fuses
    retrievers with incomparable raw scores (cosine distance vs BM25)."""

    class Config(Fuser.Config):
        class_name: Literal["rrf"] = "rrf"
        k: int = Field(default=60, gt=0)

    config: RRFFuser.Config

    def fuse(self, per_retriever: dict[str, list[Candidate]]) -> list[FusedHit]:
        scores: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        for name, candidates in per_retriever.items():
            for c in candidates:
                scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (self.config.k + c.rank)
                components.setdefault(c.chunk_id, {})[name] = c.raw_score
        ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [FusedHit(cid, score=scores[cid], component_scores=components[cid]) for cid in ranked]
