"""Content hashing — the one text-content hash used across the pipeline (the dedup / identity key)."""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """The sha256 hex digest of ``text`` (utf-8) — a document's / chunk's content-dedup key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
