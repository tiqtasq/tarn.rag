"""Retrieval query types (ModusQ spec §5.1). Result types live in ``tarnrag.contracts``."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tarnrag.contracts import Annotation, ChunkFilter, MethodRef


class Purpose(str, Enum):
    """
    Retrieval intent (ModusQ §5.1) — drives scope/licensing policy in later steps.
    """

    EXECUTION = "EXECUTION"
    AUTHORING = "AUTHORING"
    GENERATION_GROUNDING = "GENERATION_GROUNDING"


# Sentinel for "no scope restriction" (whole index).
ALL = "ALL"


@dataclass
class Query:
    """
    A retrieval request: the query text plus knobs (purpose, method scope, top_k / dense_k /
    sparse_k), and a classification (``query_type`` + ``annotations``) a ``QueryClassifier`` may fill.

    ``query_type`` is the cheap route key the ``RoutingRetrievalPipeline`` dispatches on; ``annotations``
    is the rich, extensible channel a classifier writes its findings to — the same ``Annotation`` type
    enrichment uses on chunks (so an LLM classifier's findings carry the ``deterministic`` flag, and a
    span can mark which substring is an identifier). Both default empty: an unclassified query routes to
    the pipeline's default. Set them via a classifier or supply them directly.
    """

    text: str
    purpose: Purpose = Purpose.EXECUTION
    scope: list[MethodRef] | str = ALL  # MethodRef[] or ALL
    top_k: int = 8
    dense_k: int = 50
    sparse_k: int = 50  # used in Step B (sparse retriever)
    query_type: str = ""  # headline classification label; the router's route key
    annotations: list[Annotation] = field(default_factory=list)  # the classifier's rich findings

    def permitted_filter(self) -> ChunkFilter:
        """The permitted-chunk filter for this query (ModusQ §5.6) — available-only by default, grounding
        required for ``GENERATION_GROUNDING``, restricted to ``scope`` unless it is ``ALL``. The retrievers
        pass it into the store so disallowed chunks are dropped before ``top_k`` (with over-fetch), instead
        of truncating first and filtering after (which could under-return for a tight scope)."""
        return ChunkFilter(
            require_grounding=self.purpose == Purpose.GENERATION_GROUNDING,
            method_scope=None if self.scope == ALL else tuple(self.scope),
        )
