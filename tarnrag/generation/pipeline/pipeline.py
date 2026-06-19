"""GenerationPipeline — the generation composition as a container Component (sibling of ``RetrievalPipeline``).

Built from a ``GENERATION_PIPELINE`` spec, it composes a ``Reasoner`` + an ``EvidenceAssembler`` + an
optional ``GroundingChecker`` and owns the flow:

    reason (retrieve + read) → ground-check → assemble proof tree → apply abstention policy → ``GenerationResult``.

``GenerationEngine`` is a thin facade over it: it builds the retrieval port + the LLM, then delegates
``answer``. With no ``grounding_checker`` configured, verification is skipped and ``grounded`` stays
``True`` (the slice-2 behavior).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from tarnrag.core.components import Component, ComponentFactory
from tarnrag.generation.components.assembler import EvidenceAssembler
from tarnrag.generation.components.grounding import GroundingChecker, Verdict
from tarnrag.generation.components.reasoner import Reasoner
from tarnrag.generation.context import GenerationContext
from tarnrag.generation.types import GenerationResult
from tarnrag.retrieval.types import Query


class GenerationPipeline(Component):
    """Compose a ``Reasoner`` + ``EvidenceAssembler`` (+ optional ``GroundingChecker``) and run the
    generation flow over a ``GenerationContext``."""

    class Config(Component.Config):
        class_name: Literal["generation_pipeline"] = "generation_pipeline"
        reasoner: dict[str, Any] = Field(default_factory=lambda: {"class_name": "single_hop"})
        assembler: dict[str, Any] = Field(default_factory=lambda: {"class_name": "provenance"})
        grounding_checker: dict[str, Any] | None = None  # None ⇒ no verification (grounded stays True)
        uncertain_is_grounded: bool = True  # how to collapse an UNCERTAIN verdict (lenient by default)
        # Abstention policy (γ-cap): an answer counts as grounded iff at least ``min_grounded`` of its
        # steps pass verification; below that, ``abstain`` decides refusal vs. a best-grounded flag.
        min_grounded: float = 1.0
        abstain: bool = False
        refusal: str = "I don't have enough grounded evidence to answer that confidently."

    config: GenerationPipeline.Config

    def __init__(self, config: GenerationPipeline.Config) -> None:
        super().__init__(config)
        self._reasoner: Reasoner | None = None
        self._assembler: EvidenceAssembler | None = None
        self._grounding: GroundingChecker | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the reasoner + assembler (+ optional grounding checker) children through the factory."""
        self._reasoner = factory.create_as(self.config.reasoner, Reasoner)
        self._assembler = factory.create_as(self.config.assembler, EvidenceAssembler)
        self._grounding = (
            factory.create_as(self.config.grounding_checker, GroundingChecker)
            if self.config.grounding_checker
            else None
        )

    async def answer(self, query: Query, ctx: GenerationContext) -> GenerationResult:
        self._ensure_children()
        reasoned = await self._reasoner.reason(query, ctx)
        flags = None
        if self._grounding is not None:
            verdicts = await self._grounding.check(reasoned, ctx)
            flags = [self._is_grounded(v) for v in verdicts]  # collapse each Verdict to a grounded flag
        proof = self._assembler.assemble(reasoned, flags)
        answer, grounded, abstained = self._apply_policy(reasoned.answer, flags)
        return GenerationResult(
            answer=answer, proof=proof, evidence=reasoned.evidence, grounded=grounded, abstained=abstained
        )

    def _is_grounded(self, verdict: Verdict) -> bool:
        """Collapse a three-valued ``Verdict`` to the grounded flag the proof tree + policy consume —
        ``UNCERTAIN`` resolves per the ``uncertain_is_grounded`` config (lenient by default)."""
        if verdict is Verdict.GROUNDED:
            return True
        if verdict is Verdict.UNGROUNDED:
            return False
        return self.config.uncertain_is_grounded

    def _apply_policy(self, answer: str, verdicts: list[bool] | None) -> tuple[str, bool, bool]:
        """Map the per-step verdicts to ``(answer, grounded, abstained)`` via the abstention policy.
        No checker ⇒ unverified, treated as grounded (slice-2 behavior)."""
        if verdicts is None:
            return answer, True, False
        fraction = sum(verdicts) / len(verdicts) if verdicts else 1.0
        grounded = fraction >= self.config.min_grounded
        if grounded or not self.config.abstain:
            return answer, grounded, False  # pass, or return the best-grounded answer flagged
        return self.config.refusal, False, True  # below the bar + abstain ⇒ refuse
