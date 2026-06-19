"""GenerationPipeline — the generation composition as a container Component (sibling of ``RetrievalPipeline``).

Built from a ``GENERATION_PIPELINE`` spec, it composes a ``Reasoner`` + an ``EvidenceAssembler`` (the
``GroundingChecker`` arrives in slice 3) and owns the flow:

    reason (retrieve + read) → assemble proof tree → ``GenerationResult``.

``GenerationEngine`` is a thin facade over it: it builds the retrieval port + the LLM, then delegates
``answer``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from tarnrag.core.components import Component, ComponentFactory
from tarnrag.generation.assembler import EvidenceAssembler
from tarnrag.generation.context import GenerationContext
from tarnrag.generation.reasoner import Reasoner
from tarnrag.generation.types import GenerationResult
from tarnrag.retrieval.types import Query


class GenerationPipeline(Component):
    """Compose a ``Reasoner`` + ``EvidenceAssembler`` and run the generation flow over a ``GenerationContext``."""

    class Config(Component.Config):
        class_name: Literal["generation_pipeline"] = "generation_pipeline"
        reasoner: dict[str, Any] = Field(default_factory=lambda: {"class_name": "single_hop"})
        assembler: dict[str, Any] = Field(default_factory=lambda: {"class_name": "provenance"})

    config: GenerationPipeline.Config

    def __init__(self, config: GenerationPipeline.Config) -> None:
        super().__init__(config)
        self._reasoner: Reasoner | None = None
        self._assembler: EvidenceAssembler | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the reasoner + assembler children through the factory."""
        self._reasoner = factory.create_as(self.config.reasoner, Reasoner)
        self._assembler = factory.create_as(self.config.assembler, EvidenceAssembler)

    async def answer(self, query: Query, ctx: GenerationContext) -> GenerationResult:
        self._ensure_children()
        reasoned = await self._reasoner.reason(query, ctx)
        proof = self._assembler.assemble(reasoned)
        return GenerationResult(answer=reasoned.answer, proof=proof, evidence=reasoned.evidence)
