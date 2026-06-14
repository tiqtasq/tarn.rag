"""
Regression test for loading the sqlite-vec extension on the async ``SqliteRepository``.

This locks in the de-risking finding that vec0 (dense KNN) + FTS5 work through aiosqlite, and —
crucially — guards the **private ``._conn`` access** in ``SqliteRepository``'s connect hook (see
``_load_sqlite_vec``). If a future aiosqlite version renames that attribute, extension loading
breaks and this test fails loudly rather than silently disabling dense retrieval.
"""

import sqlite_vec


def _vec(values: list[float]) -> bytes:
    return sqlite_vec.serialize_float32(values)


async def test_sqlite_vec_and_fts5_available_through_repository(repo):
    """The connect hook loads sqlite-vec on every connection, so vec0 + FTS5 are usable
    through the repository's own async engine."""
    async with repo.engine.begin() as conn:
        # The extension is present on this connection.
        assert (await conn.exec_driver_sql("select vec_version()")).scalar_one()

        # Dense KNN via a vec0 virtual table.
        await conn.exec_driver_sql(
            "CREATE VIRTUAL TABLE t_vec USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[3])"
        )
        await conn.exec_driver_sql(
            "INSERT INTO t_vec(chunk_id, embedding) VALUES (?, ?)", ("c1", _vec([1.0, 0.0, 0.0]))
        )
        await conn.exec_driver_sql(
            "INSERT INTO t_vec(chunk_id, embedding) VALUES (?, ?)", ("c2", _vec([0.0, 1.0, 0.0]))
        )
        knn = (
            await conn.exec_driver_sql(
                "SELECT chunk_id FROM t_vec WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
                (_vec([0.9, 0.1, 0.0]),),
            )
        ).fetchall()
        assert knn == [("c1",)]  # nearest to [0.9, 0.1, 0.0] is c1

        # Sparse BM25 via FTS5.
        await conn.exec_driver_sql(
            "CREATE VIRTUAL TABLE t_fts USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61')"
        )
        await conn.exec_driver_sql(
            "INSERT INTO t_fts(chunk_id, text) VALUES (?, ?)", ("c1", "storage tank inspection")
        )
        fts = (
            await conn.exec_driver_sql("SELECT chunk_id FROM t_fts WHERE t_fts MATCH 'tank'")
        ).fetchall()
        assert fts == [("c1",)]
