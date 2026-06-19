"""GenerationEngine — question → answer + proof tree + evidence (the Goal-3 facade).

The library entry point, sibling of ``IngestionEngine`` / ``RetrievalEngine``. ``create()`` wires the
composition root: it builds the retrieval engine (consumed through the ``RetrievalEngineProtocol`` port),
the ``LanguageModel`` resource (``LanguageModel.create`` over ``Settings.llm``), and the
``GenerationPipeline`` from the ``GENERATION_PIPELINE`` spec — then delegates ``answer`` to the pipeline
over a ``GenerationContext``. tiqtasq.backend wraps this in REST. Building a concrete ``RetrievalEngine``
here is the allowed direction (``generation → retrieval``); the pipeline + seams only ever see the port.
"""

from __future__ import annotations

from typing import Any, Self

from tarnrag.core.components import ComponentFactory
from tarnrag.core.engine.config import GENERATION_PIPELINE, Settings, get_settings
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.generation.context import GenerationContext
from tarnrag.generation.pipeline import GenerationPipeline
from tarnrag.generation.types import GenerationResult
from tarnrag.retrieval.engine.engine import RetrievalEngine
from tarnrag.retrieval.protocol import RetrievalEngineProtocol
from tarnrag.retrieval.types import Query

_DEFAULT_PIPELINE: dict[str, Any] = {"class_name": "generation_pipeline"}  # single-hop + provenance assembler


class GenerationEngine:
    """Async question facade: retrieve → read → proof tree. Built via ``create()`` from ``Settings``, or
    constructed directly with an injected retrieval port + LLM (tests / custom wiring)."""

    def __init__(
        self,
        retrieval: RetrievalEngineProtocol,
        llm: LanguageModel,
        pipeline: GenerationPipeline,
    ) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self._pipeline = pipeline

    @classmethod
    async def create(cls, settings: Settings | None = None) -> GenerationEngine:
        """Wire the composition: the retrieval engine (as the port), the LLM, and the pipeline spec."""
        settings = settings or get_settings()
        retrieval = await RetrievalEngine.create(settings)
        llm = LanguageModel.create(settings.llm)
        spec = settings.components.get(GENERATION_PIPELINE) or _DEFAULT_PIPELINE
        pipeline = ComponentFactory.get().create_as(spec, GenerationPipeline)
        return cls(retrieval, llm, pipeline)

    async def answer(self, query: Query) -> GenerationResult:
        """Answer a question: retrieve, read, and return the answer + proof tree + evidence."""
        return await self._pipeline.answer(query, GenerationContext(self.retrieval, self.llm))

    async def answer_text(self, text: str) -> GenerationResult:
        """Convenience over :meth:`answer`: build a :class:`Query` from a raw question string."""
        return await self.answer(Query(text=text))

    async def aclose(self) -> None:
        """Release the underlying retrieval engine's resources (best-effort — the port may be a fake)."""
        closer = getattr(self.retrieval, "aclose", None)
        if closer is not None:
            await closer()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
