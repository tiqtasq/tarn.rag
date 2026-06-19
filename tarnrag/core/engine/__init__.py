"""Engine runtime — the operational core shared across the layers.

Groups the cross-cutting runtime concerns: ``config`` (the ``Settings`` tree the engines read), the
``Engine`` base (store + embedder construction + the async lifecycle, shared by ``IngestionEngine`` /
``RetrievalEngine``), and ``observability``. Import from the modules
(e.g. ``from tarnrag.core.engine.config import Settings``).
"""
