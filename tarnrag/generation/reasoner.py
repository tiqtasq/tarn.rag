"""The Reasoner seam — retrieve, read, and produce an answer with supporting reasoning steps.

A ``Reasoner`` is a config-driven ``Component`` (a pure strategy; resources injected via
``GenerationContext``). The slice-2 ``SingleHopReasoner`` does **one** retrieval round: it searches the
port, lays the hits out as numbered passages, asks the ``LanguageModel`` to answer using ONLY those
passages and to cite — per claim — the passage numbers it relied on (a strict-JSON reply), and parses that
into reasoning steps. Multi-hop (a retrieve↔read loop + question decomposition) is slice 4, behind the
same seam.

It returns a ``ReasonedAnswer`` (answer + steps citing evidence *by index* + the evidence list); the
``EvidenceAssembler`` turns those cited indices into provenance-bearing ``Citation``s.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from tarnrag.contracts import RetrievalResult
from tarnrag.core.components import Component
from tarnrag.core.llm import Prompt
from tarnrag.generation.context import GenerationContext
from tarnrag.retrieval.types import Query


@dataclass(frozen=True)
class ReasonedStep:
    """One reasoning step before provenance is attached: a claim + the evidence indices it cites."""

    claim: str
    cited: list[int] = field(default_factory=list)  # indices into ``ReasonedAnswer.evidence``


@dataclass
class ReasonedAnswer:
    """A ``Reasoner``'s output: the answer, the steps (citing evidence by index), and the evidence."""

    answer: str
    steps: list[ReasonedStep]
    evidence: list[RetrievalResult]


class Reasoner(Component):
    """Port: produce a ``ReasonedAnswer`` for a query over the ``GenerationContext`` (retrieve + read)."""

    class Config(Component.Config):
        """Base reasoner config; concrete reasoners pin ``class_name``."""

    @abstractmethod
    async def reason(self, query: Query, ctx: GenerationContext) -> ReasonedAnswer:
        """Retrieve, read, and return the answer + supporting steps + the evidence drawn on."""


_SYSTEM = (
    "You are a careful technical assistant. Answer the question using ONLY the numbered passages "
    "provided; do not use outside knowledge. If the passages do not contain the answer, say so plainly.\n\n"
    "Reply with a SINGLE JSON object and nothing else:\n"
    '{"answer": "<concise answer>", "steps": [{"claim": "<one supported statement>", '
    '"cited": [<passage numbers this claim relies on>]}]}\n'
    'Each claim\'s "cited" lists the 1-based passage numbers that support it; use only numbers shown.'
)


class SingleHopReasoner(Reasoner):
    """One retrieval round + one read — the MVP reasoner. Search → numbered passages → strict-JSON answer
    with per-claim citations. Falls back to citing all retrieved passages if the reply isn't parseable
    JSON, so an answer always carries its evidence."""

    class Config(Reasoner.Config):
        class_name: Literal["single_hop"] = "single_hop"
        top_k: int = 8  # passages to retrieve and lay before the model

    config: SingleHopReasoner.Config

    async def reason(self, query: Query, ctx: GenerationContext) -> ReasonedAnswer:
        results = await ctx.retrieval.search(replace(query, top_k=self.config.top_k))
        completion = await ctx.llm.complete(Prompt(system=_SYSTEM, user=self._format(query.text, results)))
        answer, steps = self._parse(completion.text, len(results))
        return ReasonedAnswer(answer=answer, steps=steps, evidence=results)

    @staticmethod
    def _format(question: str, results: list[RetrievalResult]) -> str:
        """The user message: the question + the retrieved passages, numbered 1-based for citation."""
        if not results:
            return f"Question: {question}\n\nPassages: (none retrieved)"
        passages = "\n".join(
            f"[{i + 1}] {SingleHopReasoner._label(r)}{r.text}" for i, r in enumerate(results)
        )
        return f"Question: {question}\n\nPassages:\n{passages}"

    @staticmethod
    def _label(result: RetrievalResult) -> str:
        """A light source hint for the model — the chunk's header path or locator, if any."""
        prov = result.provenance
        if prov and prov.header_path:
            return " > ".join(prov.header_path) + " — "
        return f"{result.locator} — " if result.locator else ""

    @staticmethod
    def _parse(text: str, n: int) -> tuple[str, list[ReasonedStep]]:
        """Parse ``{answer, steps:[{claim, cited}]}`` from the reply; fall back to ``(text, cite-all)``."""
        data = SingleHopReasoner._extract_json(text)
        if not isinstance(data, dict) or "answer" not in data:
            answer = text.strip()
            return answer, [ReasonedStep(claim=answer, cited=list(range(n)))]
        answer = str(data.get("answer", "")).strip()
        steps: list[ReasonedStep] = []
        for s in data.get("steps") or []:
            if not isinstance(s, dict):
                continue
            claim = str(s.get("claim", "")).strip()
            if claim:
                steps.append(ReasonedStep(claim=claim, cited=SingleHopReasoner._valid_indices(s.get("cited"), n)))
        if not steps:  # well-formed answer but no usable steps -> one step over all evidence
            steps = [ReasonedStep(claim=answer, cited=list(range(n)))]
        return answer, steps

    @staticmethod
    def _extract_json(text: str) -> Any:
        """The first parseable JSON object in ``text`` (whole string, else the outermost ``{...}``)."""
        text = text.strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(text[start : end + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
            return None

    @staticmethod
    def _valid_indices(cited: Any, n: int) -> list[int]:
        """Map 1-based passage numbers to in-range 0-based evidence indices — dropped if out of range,
        deduped, order preserved (bools rejected, since ``True``/``False`` are ints in Python)."""
        if not isinstance(cited, list):
            return []
        out: list[int] = []
        for c in cited:
            if isinstance(c, bool) or not isinstance(c, int):
                continue
            i = c - 1
            if 0 <= i < n and i not in out:
                out.append(i)
        return out
