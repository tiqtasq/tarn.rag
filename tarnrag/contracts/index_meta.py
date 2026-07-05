"""The §8 ``index_meta`` build/identity record — schema version + embedding-pipeline identity.

The small handshake between the two sides of retrieval, grouped under :class:`IndexMeta`: the ingestion
**producer** stamps ``IndexMeta.build(embedder)`` onto the index (``DocumentRepository.write_index_meta``),
and both engines check ``IndexMeta.conflict(meta, embedder)`` — refusing an index built with a different
schema or embedding pipeline (the ``embedding_config_fingerprint``). It lives in its own module so both
sides depend on it rather than on each other or on any one store.
"""

from __future__ import annotations

import time
from typing import Any


class IndexMeta:
    """The §8 index build/identity record — pure static helpers over the record dict (no instances).
    ``build`` produces it from an embedder; ``conflict`` checks an existing record against an embedder.
    Both reference the same ``SCHEMA_VERSION`` + field names, so they stay in sync here in one place."""

    # Bumped when the §8 schema changes incompatibly; conflict() refuses a mismatch (the store must be
    # rebuilt by re-ingesting — the deliberate, blunt migration stance while stores are rebuildable).
    # v2 (2026-07-05): embeddings became strictly chunk-keyed — Postgres ``embeddings`` is
    # ``(chunk_id PK, vector)``; the surrogate id + per-row model/dimension columns are gone.
    SCHEMA_VERSION = "2"
    INGESTION_VERSION = "0.1.0"
    FTS_TOKENIZER = "unicode61"  # the fts_chunks tokenizer (recorded so a reader tokenizes the same)

    @staticmethod
    def build(embedder: Any, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build the §8 record from the embedder: schema/ingestion version + build time, plus the
        embedding-pipeline identity (model / tokenizer / pooling / normalize / prefixes / dim and the
        ``embedding_config_fingerprint``) that :meth:`conflict` validates. All values stringified."""
        meta: dict[str, Any] = {
            "schema_version": IndexMeta.SCHEMA_VERSION,
            "ingestion_version": IndexMeta.INGESTION_VERSION,
            "built_at": str(int(time.time())),
            "fts_tokenizer": IndexMeta.FTS_TOKENIZER,
            **embedder.embed_meta(),
            **(extra or {}),
        }
        return {k: str(v) for k, v in meta.items()}

    @staticmethod
    def conflict(meta: dict[str, str], embedder: Any) -> str | None:
        """Why an existing index's ``meta`` is incompatible with ``embedder`` — a human-readable reason,
        or ``None`` if they agree. Both sides of the §8 handshake use it: the ingestion producer (to
        refuse re-stamping an index built with a different embedder) and ``RetrievalEngine.open`` (to
        refuse serving one). Assumes ``meta`` is non-empty — the caller decides what an absent
        (never-built) index means."""
        if meta.get("schema_version") != IndexMeta.SCHEMA_VERSION:
            return (
                f"schema_version mismatch: index {meta.get('schema_version')!r} "
                f"!= engine {IndexMeta.SCHEMA_VERSION!r}"
            )
        index_fp = meta.get("embedding_config_fingerprint")
        engine_fp = embedder.config_fingerprint()
        if index_fp != engine_fp:
            return (
                "embedding_config_fingerprint mismatch — the index was built with a different "
                f"embedding pipeline (index {index_fp!r} != engine {engine_fp!r})"
            )
        return None
