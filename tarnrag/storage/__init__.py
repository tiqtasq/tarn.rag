"""
Storage — the shared persistence layer (data models, the chunk/index stores, the
document-status read model, and the document repository with Postgres / SQLite dialects).

Three distinct stores back the system, and the **retrieval index is always SQLite** — only the
document/queue backends change with ``Settings.MODE`` (``embedded`` | ``distributed``):

* **Retrieval index** — ``SqliteIndexStore`` at ``INDEX__DB_PATH`` (e.g. ``index.db``). The §8
  index: ``documents`` (metadata), ``chunks`` (text + provenance), ``vec_chunks`` (vectors),
  ``fts_chunks``, ``method_chunks``, ``index_meta``. Everything ``RetrievalEngine`` reads, and
  where the ingestion sinks write — in **both** modes.
* **Document store** — ``DocumentRepository`` at ``DATABASE__DOCUMENT_URL``. Holds the
  ``job_status`` projection only: SQLite ``rag_docs.db`` (embedded) or Postgres (distributed). It
  also defines ``documents``/``chunks``/``embeddings`` (pgvector) tables, but the engine routes
  ingestion data to the index store, so those tables are created-but-unused.
* **Job queue** — at ``DATABASE__QUEUE_URL``. In-process ``InMemoryJobQueue`` (embedded; nothing
  persisted) or pgQueuer-managed tables on Postgres (distributed).

So Postgres only ever holds ``job_status`` (+ the queue); the retrieval index never leaves SQLite.

What lives where, by mode:

| Data                                          | Embedded               | Distributed                         |
|-----------------------------------------------|------------------------|-------------------------------------|
| Retrieval index (chunks, vectors, FTS, meta)  | SQLite `index.db`      | SQLite `index.db` (shared volume)   |
| `job_status` projection                       | SQLite `rag_docs.db`   | Postgres (`DATABASE__DOCUMENT_URL`) |
| Job queue                                     | in-process (not saved) | Postgres (`DATABASE__QUEUE_URL`)    |
| Staged uploads (`UPLOAD_DIR`)                 | local dir              | shared-volume dir                   |
| Repo `documents`/`chunks`/`embeddings` tables | created, unused        | created, unused                     |
"""
