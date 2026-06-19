"""Retrieval pipeline — the Searcher orchestrators.

The ``Searcher`` base + the two pipeline containers the engine builds from the ``RETRIEVAL_PIPELINE`` spec:
``RetrievalPipeline`` (retrieve → fuse → hydrate → filter → merge → rerank → top_k) and
``RoutingRetrievalPipeline`` (classify the query, dispatch to a per-type sub-pipeline). The inner seams
they compose live in ``retrieval/components/``. Import from the modules
(e.g. ``from tarnrag.retrieval.pipeline.pipeline import RetrievalPipeline``).
"""
