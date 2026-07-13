"""Custom exception classes — the domain errors raised across the system."""


class IngestionError(Exception):
    """
    A stage's results could not be produced or persisted.

    The worker propagates it so the queue requeues the job (recovery, D5).
    """


class RetrievalError(Exception):
    """
    The retrieval engine cannot serve queries against this index (incompatibility at ``open()``).
    """


class GenerationError(Exception):
    """
    The generation pipeline cannot produce an answer as configured (e.g. a component needs a
    resource the ``GenerationContext`` doesn't carry).
    """
