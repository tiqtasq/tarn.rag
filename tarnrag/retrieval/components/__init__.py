"""Retrieval components — the config-driven Components built from the ``RETRIEVAL_PIPELINE`` spec.

The ``Searcher`` family (the ``Searcher`` base + the ``RetrievalPipeline`` and ``RoutingRetrievalPipeline``
containers) and the inner seams a pipeline composes — ``Retriever`` (dense / sparse), ``Fuser`` (identity /
rrf), the optional ``Merger`` (auto-merge) and ``Reranker`` (cross-encoder) — plus the routing
``QueryClassifier``. Each self-registers with the global ``ComponentFactory`` on import. Import from the
modules (e.g. ``from tarnrag.retrieval.components.pipeline import RetrievalPipeline``).
"""
