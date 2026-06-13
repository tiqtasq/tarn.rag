import pytest_asyncio

from tarnrag.storage.repository.sqlite import SqliteRepository


@pytest_asyncio.fixture
async def repo(tmp_path):
    """A connected SQLite repository on a temp file db (embedding_dimension=3)."""
    repository = SqliteRepository(
        f"sqlite:///{tmp_path}/test.db", embedding_dimension=3
    )
    await repository.connect()
    yield repository
    await repository.disconnect()
