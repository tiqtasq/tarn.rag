"""Generation result contracts — the proof tree the ``GenerationEngine`` produces.

A ``GenerationResult`` is an *answer* plus an inspectable *proof*: a list of ``ProofStep``s (one reasoning
step / intermediate claim) each citing the evidence it rests on. A ``Citation`` carries our layout-grade
provenance (char spans + page boxes + header path) pulled straight from the retrieved hit, so every step
is highlightable back to the source — the differentiator over plain span citation.

These are consumed by tiqtasq's REST layer, so they are serializable + versioned; they do *not* cross the
C++ retrieval binding (only the retrieval contracts in ``tarnrag.contracts`` do). Dataclasses, mirroring
``RetrievalResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tarnrag.contracts import Geometry, RetrievalResult


@dataclass(frozen=True)
class Citation:
    """One cited span of evidence: a chunk + the geometry/header-path that makes it highlightable."""

    chunk_id: str
    document_id: str
    locator: str | None = None
    header_path: list[str] = field(default_factory=list)
    geometry: Geometry = field(default_factory=list)  # char spans (+ page boxes), from the chunk provenance

    @classmethod
    def from_result(cls, result: RetrievalResult) -> Citation:
        """Build a citation from a retrieved hit, pulling geometry + header path from its provenance."""
        prov = result.provenance
        return cls(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            locator=result.locator,
            header_path=list(prov.header_path) if prov else [],
            geometry=list(prov.geometry) if prov else [],
        )


@dataclass(frozen=True)
class ProofStep:
    """One reasoning step: a claim, the evidence it rests on, and whether the grounding check found that
    evidence to support it (``grounded`` is ``True`` when no checker ran — see the pipeline)."""

    claim: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True


@dataclass(frozen=True)
class GenerationResult:
    """The ``GenerationEngine``'s output: the answer + an inspectable proof tree + the evidence drawn on.

    ``grounded`` / ``abstained`` are set by the grounding check (slice 3); the slice-2 MVP returns
    ``grounded=True``, ``abstained=False`` (no verification yet)."""

    answer: str
    proof: list[ProofStep] = field(default_factory=list)
    evidence: list[RetrievalResult] = field(default_factory=list)
    grounded: bool = True  # did the grounding check pass within budget? (slice 3)
    abstained: bool = False  # the refusal path (slice 3)
