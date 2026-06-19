"""Generation domain — the multi-hop generation layer (Goal 3), built *on top of* retrieval.

A new top-level package next to ``ingestion/`` and ``retrieval/``. It consumes a retrieval *port*
(``RetrievalEngineProtocol``) and a ``LanguageModel`` (a ``core`` ``Resource``), and is itself consumed by
tiqtasq's REST layer. The dependency is strictly one-way (``generation → retrieval``), which keeps the
retrieval core self-contained and C++-portable. See ``doc/generation-architecture-design.md``.

``GenerationEngine.create()`` is the entry point (question → ``GenerationResult``); ``GenerationPipeline``
composes the ``Reasoner`` + ``EvidenceAssembler`` seams. Slice 2 ships the single-hop MVP (no grounding
check / multi-hop yet — slices 3 / 4).
"""

from tarnrag.generation.components.assembler import EvidenceAssembler, ProvenanceAssembler
from tarnrag.generation.context import GenerationContext
from tarnrag.generation.engine.engine import GenerationEngine
from tarnrag.generation.pipeline.pipeline import GenerationPipeline
from tarnrag.generation.components.reasoner import ReasonedAnswer, ReasonedStep, Reasoner, SingleHopReasoner
from tarnrag.generation.types import Citation, GenerationResult, ProofStep

__all__ = [
    "GenerationEngine",
    "GenerationPipeline",
    "GenerationContext",
    "Reasoner",
    "SingleHopReasoner",
    "ReasonedAnswer",
    "ReasonedStep",
    "EvidenceAssembler",
    "ProvenanceAssembler",
    "Citation",
    "ProofStep",
    "GenerationResult",
]
