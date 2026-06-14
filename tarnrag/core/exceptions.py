"""Custom exception classes."""


class IngestionError(Exception):
    """
    A stage's results could not be produced or persisted.

    The worker propagates it so the queue requeues the job (recovery, D5).
    """


class DocumentStorageError(IngestionError):
    """
    A document / chunk / embedding write failed.
    """


class ChunkNotFoundError(IngestionError):
    """
    ``update_chunk_metadata`` targeted a chunk id that does not exist.
    """

    def __init__(self, chunk_id: str):
        super().__init__(f"chunk not found: {chunk_id}")
        self.chunk_id = chunk_id
