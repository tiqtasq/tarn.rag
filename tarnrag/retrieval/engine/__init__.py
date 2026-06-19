"""Retrieval engine — the query facade + its port.

``RetrievalEngine`` (``RetrievalEngine.create()``) opens an index (validating the embedding fingerprint +
schema) and delegates ``search`` to the configured ``Searcher`` built from the ``RETRIEVAL_PIPELINE`` spec.
``retrieval_engine_protocol`` holds ``RetrievalEngineProtocol`` — the port the engine satisfies
structurally and that the generation layer depends on. Import from the modules
(e.g. ``from tarnrag.retrieval.engine.engine import RetrievalEngine``).
"""
