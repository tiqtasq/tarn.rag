"""Shared helpers for the example suite — imported by every example as ``examples.common``.

Examples are run as modules from the repo root, e.g.::

    python -m examples.part_i.example_01.ingestion

so this module imports cleanly as ``examples.common`` (no ``sys.path`` juggling). It holds the
incidental setup every example repeats — the model + docs locations, the model-present check, a
default ``Settings`` factory, and a per-example SQLite path. Each example keeps the *interesting*
part — the config knob or API it demonstrates — explicit and inline.
"""

from __future__ import annotations

from pathlib import Path

from tarnrag.core.config import DatabaseSettings, EmbeddingSettings, Settings

EXAMPLES_DIR = Path(__file__).resolve().parent          # the examples/ package
REPO_ROOT = EXAMPLES_DIR.parent
DOCS_DIR = EXAMPLES_DIR / "docs"                        # holds named corpora (docs/corpus-1, ...)
MODEL_DIR = REPO_ROOT / "models" / "all-MiniLM-L6-v2"   # local ONNX model + tokenizer


def corpus(name: str) -> Path:
    """Path to a named document corpus under ``examples/docs/`` (e.g. ``corpus("corpus-1")``).
    Add a corpus by dropping files in a new ``examples/docs/<name>/`` directory."""
    return DOCS_DIR / name


def require_model() -> None:
    """Exit with a friendly hint if the embedding model hasn't been fetched yet — every example
    needs it (to embed passages and queries)."""
    if not (MODEL_DIR / "model.onnx").exists():
        raise SystemExit(
            f"Embedding model not found under {MODEL_DIR}.\n"
            f"Fetch it once (needs network):  python scripts/fetch_model.py"
        )


def example_db(example_file: str) -> Path:
    """The SQLite store for ONE example: a ``rag_docs.db`` next to the calling script, so each
    example keeps its own isolated, re-runnable store. Pass ``__file__`` from the example."""
    return Path(example_file).resolve().parent / "rag_docs.db"


def base_settings(db_path: Path, **overrides: object) -> Settings:
    """Default embedded/SQLite settings shared by every example.

    Pass ``db_path`` (usually ``example_db(__file__)``) and override only the knob an example is
    demonstrating, e.g. ``base_settings(db, chunking=ChunkingSettings(size=128))``. The default
    ``ID_POLICY='caller'`` lets examples use readable, stable document ids. An ingestion and a
    retrieval built from the same ``db_path`` + ``embedding`` stay compatible (retrieval validates
    an embedding *fingerprint*), so reuse this for both sides of an example.
    """
    defaults: dict[str, object] = {
        "_env_file": None,           # explicit config only — ignore any ambient .env, for reproducibility
        "MODE": "embedded",          # whole pipeline in-process (no Postgres / job queue)
        "ID_POLICY": "caller",       # examples supply their own stable document ids
        "EMBEDDING_DIMENSION": 384,  # output dimension of all-MiniLM-L6-v2
        "database": DatabaseSettings(document_url=f"sqlite:///{db_path}"),
        "embedding": EmbeddingSettings(model_dir=str(MODEL_DIR)),
    }
    return Settings(**{**defaults, **overrides})
