"""The GroundingChecker seam — verify each reasoning step's claim against its cited evidence (slice 3).

A pure ``Component``: given a ``ReasonedAnswer`` (steps citing evidence by index + the evidence), it returns
a per-step grounded verdict — is each claim actually supported by *its own* cited passages? Two backends,
both config-driven:

- ``HeuristicGroundingChecker`` — content-word overlap (LLM-free, no extra resources, C++-portable);
- ``LLMGroundingChecker`` — a single batched verdict from the ``LanguageModel`` (catches paraphrase /
  contradiction the overlap misses).

The ``GenerationPipeline`` runs the configured checker (optional), stamps the verdicts onto the proof tree,
and applies the abstention policy. None configured ⇒ no verification (``grounded`` stays ``True``),
preserving the slice-2 behavior.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from typing import Literal

from tarnrag.contracts import RetrievalResult
from tarnrag.core.components import Component
from tarnrag.core.resources.llm import Prompt
from tarnrag.generation.components._parsing import extract_json
from tarnrag.generation.components.reasoner import ReasonedAnswer, ReasonedStep
from tarnrag.generation.context import GenerationContext


class GroundingChecker(Component):
    """Port: a per-step grounded verdict for a ``ReasonedAnswer`` — is each claim supported by its cited
    evidence? Returns a ``list[bool]`` aligned to ``reasoned.steps``."""

    class Config(Component.Config):
        """Base grounding-checker config; concrete checkers pin ``class_name``."""

    @abstractmethod
    async def check(self, reasoned: ReasonedAnswer, ctx: GenerationContext) -> list[bool]:
        """One grounded verdict per step (aligned to ``reasoned.steps``)."""

    @staticmethod
    def _cited_passages(step: ReasonedStep, reasoned: ReasonedAnswer) -> list[RetrievalResult]:
        """The evidence a step cites, by index, in order."""
        return [reasoned.evidence[i] for i in step.cited]


# A compact stop set for the overlap heuristic (content words only — small + self-contained).
_STOPWORDS = frozenset(
    "a an the of to in on for and or but with at by from as into about than that this these those is are "
    "was were be been being it its do does did can could should would will may might has have had not no "
    "you your we our they their he she his her i".split()
)
_TOKEN = re.compile(r"\w+")


class HeuristicGroundingChecker(GroundingChecker):
    """LLM-free: a claim is grounded when enough of its content words appear in its cited passages — the
    overlap ratio is ``>= min_overlap``. Cheap, deterministic, C++-portable; it misses paraphrase and
    contradiction (that's what the LLM checker is for)."""

    class Config(GroundingChecker.Config):
        class_name: Literal["heuristic_grounding"] = "heuristic_grounding"
        min_overlap: float = 0.6  # fraction of the claim's content words that must appear in cited text
        stopwords: list[str] | None = None  # None ⇒ built-in set

    config: HeuristicGroundingChecker.Config

    async def check(self, reasoned: ReasonedAnswer, ctx: GenerationContext) -> list[bool]:
        stop = frozenset(self.config.stopwords) if self.config.stopwords is not None else _STOPWORDS
        verdicts: list[bool] = []
        for step in reasoned.steps:
            claim_words = self._content_words(step.claim, stop)
            if not claim_words:
                verdicts.append(False)  # a claim with no content words can't be shown grounded
                continue
            cited_text = " ".join(p.text for p in self._cited_passages(step, reasoned))
            covered = len(claim_words & self._content_words(cited_text, stop))
            verdicts.append(covered / len(claim_words) >= self.config.min_overlap)
        return verdicts

    @staticmethod
    def _content_words(text: str, stop: frozenset[str]) -> set[str]:
        return {w for w in _TOKEN.findall(text.lower()) if len(w) > 1 and w not in stop}


_SYSTEM = (
    "You are a strict fact-checker. For each numbered claim, decide whether it is FULLY supported by its "
    "cited passages ALONE (no outside knowledge). Reply with a SINGLE JSON object and nothing else:\n"
    '{"verdicts": [true, false, ...]} — one boolean per claim, in order. true = supported.'
)


class LLMGroundingChecker(GroundingChecker):
    """Verify each claim against its cited passages with the ``LanguageModel`` — one batched call returning
    a boolean per step. Falls back to ``grounded=True`` if the reply isn't parseable, so a verification
    hiccup never spuriously refuses an otherwise-good answer."""

    class Config(GroundingChecker.Config):
        class_name: Literal["llm_grounding"] = "llm_grounding"

    config: LLMGroundingChecker.Config

    async def check(self, reasoned: ReasonedAnswer, ctx: GenerationContext) -> list[bool]:
        if not reasoned.steps:
            return []
        completion = await ctx.llm.complete(Prompt(system=_SYSTEM, user=self._format(reasoned)))
        return self._parse(completion.text, len(reasoned.steps))

    @staticmethod
    def _format(reasoned: ReasonedAnswer) -> str:
        """The user message: each numbered claim + the passages it cites."""
        blocks = []
        for n, step in enumerate(reasoned.steps, 1):
            cited = GroundingChecker._cited_passages(step, reasoned)
            passages = "\n".join(f"  - {p.text}" for p in cited) if cited else "  (no passages cited)"
            blocks.append(f"Claim {n}: {step.claim}\nCited passages:\n{passages}")
        return "\n\n".join(blocks)

    @staticmethod
    def _parse(text: str, n: int) -> list[bool]:
        """Parse ``{\"verdicts\": [bool, ...]}``; pad/truncate to ``n``; fall back to all-grounded."""
        data = extract_json(text)
        verdicts = data.get("verdicts") if isinstance(data, dict) else None
        if not isinstance(verdicts, list):
            return [True] * n  # unparseable ⇒ don't penalize (no spurious refusal)
        out = [bool(v) for v in verdicts[:n]]
        return out + [True] * (n - len(out))  # pad any missing verdicts as grounded
