"""Resources — the engine-built, injected runtime models (siblings of ``Component``).

A ``Resource`` is the *model a strategy consumes*: long-lived, built by an engine from its ``Settings``
slice and injected at call time (not registered + composed like a ``Component``). This subpackage groups
them — the ``Resource`` base, the embedder (local ONNX + HTTP-API backends), the cross-encoder reranker,
and the generation ``LanguageModel`` (+ its Anthropic backend). Import the concrete classes from their
modules (e.g. ``from tarnrag.core.resources.embedder import Embedder``).
"""
