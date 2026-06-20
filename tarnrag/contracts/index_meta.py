"""
The §8 ``index_meta`` build/identity record — schema version + embedding-pipeline identity.

This is the small handshake between the two sides of retrieval: the ingestion **producer**
stamps it onto the index (``DocumentRepository.write_index_meta``), and ``RetrievalEngine.open``
**validates** it — refusing to serve an index built with a different schema or embedding
pipeline (the ``embedding_config_fingerprint``). It lives in its own module so both sides depend
on it rather than on each other or on any one store.
"""

from __future__ import annotations

import time
from typing import Any

# Bumped when the §8 schema changes incompatibly; RetrievalEngine.open refuses a mismatch.
SCHEMA_VERSION = "1"
INGESTION_VERSION = "0.1.0"
FTS_TOKENIZER = "unicode61"  # the fts_chunks tokenizer (recorded so a reader tokenizes the same)


def build_index_meta(embedder: Any, extra: dict[str, str] | None = None) -> dict[str, str]:
    """
    Build the §8 ``index_meta`` record from the embedder: schema/ingestion version + build time,
    plus the embedding-pipeline identity (model / tokenizer / pooling / normalize / prefixes /
    dim and the ``embedding_config_fingerprint``) that retrieval validates. All values stringified.
    """
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ingestion_version": INGESTION_VERSION,
        "built_at": str(int(time.time())),
        "fts_tokenizer": FTS_TOKENIZER,
        **embedder.embed_meta(),
        **(extra or {}),
    }
    return {k: str(v) for k, v in meta.items()}


def index_meta_conflict(meta: dict[str, str], embedder: Any) -> str | None:
    """Why an existing index's ``meta`` is incompatible with ``embedder`` — a human-readable reason, or
    ``None`` if they agree. Both sides of the §8 handshake use it: the ingestion producer (to refuse
    re-stamping an index built with a different embedder) and ``RetrievalEngine.open`` (to refuse serving
    one). Assumes ``meta`` is non-empty — the caller decides what an absent (never-built) index means."""
    if meta.get("schema_version") != SCHEMA_VERSION:
        return f"schema_version mismatch: index {meta.get('schema_version')!r} != engine {SCHEMA_VERSION!r}"
    index_fp = meta.get("embedding_config_fingerprint")
    engine_fp = embedder.config_fingerprint()
    if index_fp != engine_fp:
        return (
            "embedding_config_fingerprint mismatch — the index was built with a different "
            f"embedding pipeline (index {index_fp!r} != engine {engine_fp!r})"
        )
    return None
