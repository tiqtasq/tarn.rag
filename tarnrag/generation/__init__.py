"""Generation domain — the multi-hop generation layer (Goal 3), built *on top of* retrieval.

A new top-level package next to ``ingestion/`` and ``retrieval/``. It consumes a retrieval *port*
(``RetrievalEngineProtocol``) and a ``LanguageModel`` (a ``core`` ``Resource``), and is itself consumed by
tiqtasq's REST layer. The dependency is strictly one-way (``generation → retrieval``), which keeps the
retrieval core self-contained and C++-portable. See ``doc/generation-architecture-design.md``.

Slice 2 (this MVP) ships the result contracts (the proof tree) and — once the LLM provider + reasoner
strategy are settled — the ``GenerationPipeline`` / ``Reasoner`` / ``EvidenceAssembler`` / ``GenerationEngine``.
"""

from tarnrag.generation.types import Citation, GenerationResult, ProofStep

__all__ = ["Citation", "ProofStep", "GenerationResult"]
