"""Retrieval engine — the query facade.

``RetrievalEngine`` (``RetrievalEngine.create()``) opens an index (validating the embedding fingerprint +
schema) and delegates ``search`` to the configured ``Searcher`` built from the ``RETRIEVAL_PIPELINE`` spec.
Import from the module (``from tarnrag.retrieval.engine.engine import RetrievalEngine``).
"""
