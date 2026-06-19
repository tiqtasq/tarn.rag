"""Retrieval components — the inner seams a pipeline composes (config-driven Components).

The swappable parts a ``Searcher`` composes: ``Retriever`` (dense / sparse), ``Fuser`` (identity / rrf),
the optional ``Merger`` (auto-merge) and ``Reranker`` (cross-encoder), plus the routing ``QueryClassifier``.
Each self-registers with the global ``ComponentFactory`` on import. The ``Searcher`` base + the pipeline
containers that compose these live in ``retrieval/pipeline/``. Import from the modules
(e.g. ``from tarnrag.retrieval.components.retriever import DenseRetriever``).
"""
