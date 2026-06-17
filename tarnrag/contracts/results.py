"""Retrieval result contracts — the types produced along the retrieval path.

``Candidate`` (a ranked hit from a retriever) and ``ChunkRecord`` (a hydrated chunk) are the
repository's ``dense_knn`` / ``hydrate`` return types; ``RetrievalResult`` is the engine's final
assembled, provenance-bearing hit. Kept here (not in ``retrieval/``) so the storage layer can
produce ``Candidate`` / ``ChunkRecord`` without depending on the retrieval package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tarnrag.contracts.dtos import MethodRef
from tarnrag.contracts.structure import ChunkProvenance


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
    ai_grounding_allowed: bool = True  # may this chunk ground generated output? (purpose-driven filter)
    available: bool = True  # is this chunk currently retrievable?
    methods: list[tuple[str, str]] = field(default_factory=list)  # (method_id, method_version)
    provenance: ChunkProvenance | None = None  # layout-aware: geometry / header_path / tree / annotations / table


@dataclass(frozen=True)
class RetrievalResult:
    """
    One ranked hit: chunk text + fused score + per-component scores + provenance/licensing.
    """

    chunk_id: str
    text: str
    score: float
    component_scores: dict[str, float]
    document_id: str
    source_kind: str
    standard_id: str | None
    locator: str | None
    license_class: str
    methods: list[MethodRef] = field(default_factory=list)
    provenance: ChunkProvenance | None = None  # layout-aware provenance (citation / highlight / merge / entity)

    @classmethod
    def from_record(
        cls, record: ChunkRecord, *, score: float, component_scores: dict[str, float]
    ) -> RetrievalResult:
        """Assemble a result from a hydrated chunk + its score — the shared chunk→result mapping used
        by the pipeline's assembly and the auto-merger's section-parent construction."""
        return cls(
            chunk_id=record.chunk_id,
            text=record.text,
            score=score,
            component_scores=component_scores,
            document_id=record.document_id,
            source_kind=record.source_kind,
            standard_id=record.standard_id,
            locator=record.locator,
            license_class=record.license_class,
            methods=[MethodRef(m, v) for m, v in record.methods],
            provenance=record.provenance,
        )
