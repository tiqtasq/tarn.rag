"""
Storage — the persistence layer: the document repository (Postgres / SQLite dialects) and the
document-status read model. The cross-boundary contracts it implements (DTOs, the chunk-store /
status ports, retrieval results, index_meta) live in ``tarnrag.contracts``.

One store backs the system: the **repository** (``DocumentRepository``) at
``DATABASE__DOCUMENT_URL``. It holds the documents/chunks, the §8 retrieval index, the
``index_meta`` build record, and the ``job_status`` projection — everything the ingestion sinks
write and everything ``RetrievalEngine`` reads. Only the backend changes with ``Settings.MODE``
(``embedded`` | ``distributed``); the job queue is the one separate piece:

* **Repository** — ``DocumentRepository``. Embedded: a single SQLite file ``rag_docs.db`` whose
  §8 tables (``documents`` / ``chunks`` / ``method_chunks`` / ``index_meta`` + the ``vec_chunks``
  (sqlite-vec) and ``fts_chunks`` (FTS5) virtual tables) are the **portable retrieval artifact**
  a future C++ port consumes — it reads only those tables and ignores the rest. Distributed:
  Postgres, with dense retrieval on the ``embeddings`` table (pgvector). job_status lives here too.
* **Job queue** — at ``DATABASE__QUEUE_URL``. In-process ``InMemoryJobQueue`` (embedded; nothing
  persisted) or pgQueuer-managed tables on Postgres (distributed).

The retrieval index is no longer a separate SQLite file — it lives wherever the repository does
(sqlite-vec on SQLite, pgvector on Postgres); only the SQLite form is the C++-consumable artifact.

What lives where, by mode:

| Data                                          | Embedded                  | Distributed                         |
|-----------------------------------------------|---------------------------|-------------------------------------|
| Repository: docs/chunks/§8 index + job_status | SQLite `rag_docs.db`      | Postgres (`DATABASE__DOCUMENT_URL`) |
| Dense vectors                                 | `vec_chunks` (sqlite-vec) | `embeddings` (pgvector)             |
| Job queue                                     | in-process (not saved)    | Postgres (`DATABASE__QUEUE_URL`)    |
| Staged uploads (`UPLOAD_DIR`)                 | local dir                 | shared-volume dir                   |
"""
