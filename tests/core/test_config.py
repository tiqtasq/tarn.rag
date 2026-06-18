"""Settings: grouped sub-models, env loading, and distributed-config validation."""

import pytest
from pydantic import ValidationError

from tarnrag.core.config import Settings


def test_defaults_and_groups():
    s = Settings(_env_file=None)  # ignore the local .env, exercise defaults
    assert s.MODE == "embedded"
    assert s.EMBEDDING_DIMENSION == 384
    assert s.embedding.provider == "onnx"  # local ONNX by default
    assert s.embedding.model == "thenlper/gte-small"
    assert s.database.document_url.startswith("sqlite")  # zero-config embedded default


def test_nested_env_loading(monkeypatch):
    monkeypatch.setenv("EMBEDDING__MODEL", "my/model")
    monkeypatch.setenv("EMBEDDING__BATCH_SIZE", "8")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
    s = Settings(_env_file=None)
    assert s.embedding.model == "my/model"
    assert s.embedding.batch_size == 8
    assert s.EMBEDDING_DIMENSION == 768  # cross-cutting field stays top-level/flat


def test_kwargs_override_nested():
    s = Settings(_env_file=None, embedding={"model": "k"}, EMBEDDING_DIMENSION=128)
    assert s.embedding.model == "k"
    assert s.EMBEDDING_DIMENSION == 128


def test_distributed_requires_postgres_and_queue():
    # distributed + the SQLite document default -> rejected
    with pytest.raises(ValidationError, match="Postgres DATABASE__DOCUMENT_URL"):
        Settings(_env_file=None, MODE="distributed", database={"queue_url": "postgresql://q"})
    # distributed + Postgres docs but no queue -> rejected
    with pytest.raises(ValidationError, match="DATABASE__QUEUE_URL"):
        Settings(_env_file=None, MODE="distributed", database={"document_url": "postgresql://docs"})
    # fully configured distributed -> ok
    s = Settings(
        _env_file=None, MODE="distributed",
        database={"document_url": "postgresql://docs", "queue_url": "postgresql://q"},
    )
    assert s.MODE == "distributed"
    # embedded with the SQLite default -> ok (no validation trip)
    assert Settings(_env_file=None).MODE == "embedded"


def test_embedded_rejects_postgres_document_url():
    # embedded + Postgres document store -> rejected (embedded is SQLite / single-process)
    with pytest.raises(ValidationError, match="must not use a Postgres"):
        Settings(_env_file=None, MODE="embedded", database={"document_url": "postgresql://docs"})
    # embedded + the SQLite default -> ok
    assert Settings(_env_file=None, MODE="embedded").database.document_url.startswith("sqlite")
