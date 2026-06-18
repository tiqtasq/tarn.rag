"""sha256 hashing — the one home for the project's content-dedup keys, embedding fingerprints, and
file hashes, so the same digest is computed the same way everywhere (text/bytes content, config blobs,
tokenizer/model files). ``compute_content_hash`` is the named alias for the document/chunk dedup key."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_hex(data: str | bytes) -> str:
    """sha256 hex digest of ``data`` — a ``str`` is utf-8 encoded, ``bytes`` hashed as-is."""
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def sha256_file(path: str | Path, *, block_size: int = 1 << 16) -> str:
    """Streaming sha256 hex of a file's bytes (constant memory — 64 KiB blocks by default)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def compute_content_hash(text: str) -> str:
    """The content-dedup / identity key for a document or chunk — sha256 of its text."""
    return sha256_hex(text)
