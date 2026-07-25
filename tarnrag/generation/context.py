"""GenerationContext — the runtime resources the generation pipeline needs, injected at call time.

Mirrors ``RetrievalContext``: the pipeline and its seams are pure strategies; the retrieval port + the LLM
are passed in via this context (so tests fake them by injection, not registration). ``retrieval`` is the
**port** (``RetrievalEngineProtocol``), so the concrete engine — Python today, a C++ adapter later — is
interchangeable, and the ``generation → retrieval`` dependency stays one-way.
"""

from __future__ import annotations

from dataclasses import dataclass

from tarnrag.core.resources.cross_encoder import CrossEncoder
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.retrieval.engine.retrieval_engine_protocol import RetrievalEngineProtocol


@dataclass
class GenerationContext:
    """The resources a generation run consumes: the retrieval port + the reader LLM, plus the optional
    ``cross_encoder`` for evidence reranking (P7: re-order the pooled passages against the original
    question before the synthesis read). The model is lazy (loads on first use), so carrying it costs
    nothing when no reasoner is configured to rerank; ``None`` on injected/test wiring without one."""

    retrieval: RetrievalEngineProtocol
    llm: LanguageModel
    cross_encoder: CrossEncoder | None = None
