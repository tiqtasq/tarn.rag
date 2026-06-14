"""
Retrieval result types — the ranked ``Candidate`` and the hydrated ``ChunkRecord`` produced by
the ``dense_knn`` / ``hydrate`` retrieval interface. Shared by the §8 index store and the
repository (which both implement that interface), so neither has to import from the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """
    A ranked candidate from a retriever (rank is 1-based; raw_score is engine-specific:
    distance for dense KNN, bm25 for sparse).
    """

    chunk_id: str
    rank: int
    raw_score: float


@dataclass(frozen=True)
class ChunkRecord:
    """
    A hydrated chunk: canonical text + provenance + license, for result assembly.
    """

    chunk_id: str
    text: str
    document_id: str
    source_kind: str
    standard_id: str | None
    locator: str | None
    license_class: str
    methods: list[tuple[str, str]] = field(default_factory=list)  # (method_id, method_version)
