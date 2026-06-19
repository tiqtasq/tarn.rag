"""GenerationContext — the runtime resources the generation pipeline needs, injected at call time.

Mirrors ``RetrievalContext``: the pipeline and its seams are pure strategies; the retrieval port + the LLM
are passed in via this context (so tests fake them by injection, not registration). ``retrieval`` is the
**port** (``RetrievalEngineProtocol``), so the concrete engine — Python today, a C++ adapter later — is
interchangeable, and the ``generation → retrieval`` dependency stays one-way.
"""

from __future__ import annotations

from dataclasses import dataclass

from tarnrag.core.llm import LanguageModel
from tarnrag.retrieval.protocol import RetrievalEngineProtocol


@dataclass
class GenerationContext:
    """The resources a generation run consumes: the retrieval port + the reader LLM."""

    retrieval: RetrievalEngineProtocol
    llm: LanguageModel
