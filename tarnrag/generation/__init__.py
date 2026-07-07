"""Generation domain — the multi-hop generation layer (Goal 3), built *on top of* retrieval.

A new top-level package next to ``ingestion/`` and ``retrieval/``. It consumes a retrieval *port*
(``RetrievalEngineProtocol``) and a ``LanguageModel`` (a ``core`` ``Resource``), and is itself consumed by
tiqtasq's REST layer. The dependency is strictly one-way (``generation → retrieval``), which keeps the
retrieval core self-contained and C++-portable. See ``doc/generation-architecture-design.md``.

``GenerationEngine.create()`` is the entry point (question → ``GenerationResult``); ``GenerationPipeline``
composes the ``Reasoner`` + ``EvidenceAssembler`` (+ optional ``GroundingChecker``) seams. Slice 3 adds
grounding + abstention; multi-hop is slice 4.
"""

from tarnrag.generation.components.assembler import EvidenceAssembler, ProvenanceAssembler
from tarnrag.generation.components.grounding import (
    CascadingGroundingChecker,
    GroundingChecker,
    HeuristicGroundingChecker,
    LLMGroundingChecker,
    Verdict,
)
from tarnrag.generation.context import GenerationContext
from tarnrag.generation.engine.engine import GenerationEngine
from tarnrag.generation.pipeline.pipeline import GenerationPipeline
from tarnrag.generation.components.answerability import AnswerabilityGateReasoner
from tarnrag.generation.components.grounded_retrieval import GroundedRetrievalReasoner
from tarnrag.generation.components.table_lookup import TableLookupReasoner
from tarnrag.generation.components.reasoner import (
    DecompositionReasoner,
    IterativeReasoner,
    ReasonedAnswer,
    ReasonedStep,
    Reasoner,
    SingleHopReasoner,
)
from tarnrag.generation.types import Citation, GenerationResult, ProofStep

__all__ = [
    "GenerationEngine",
    "GenerationPipeline",
    "GenerationContext",
    "Reasoner",
    "SingleHopReasoner",
    "IterativeReasoner",
    "DecompositionReasoner",
    "GroundedRetrievalReasoner",
    "AnswerabilityGateReasoner",
    "TableLookupReasoner",
    "ReasonedAnswer",
    "ReasonedStep",
    "EvidenceAssembler",
    "ProvenanceAssembler",
    "GroundingChecker",
    "HeuristicGroundingChecker",
    "LLMGroundingChecker",
    "CascadingGroundingChecker",
    "Verdict",
    "Citation",
    "ProofStep",
    "GenerationResult",
]
