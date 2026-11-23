from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from scripts import migrate_mysql as migrations


@pytest.mark.parametrize("scenario", [
    "empty", "old", "interrupted", "drift", "manual", "missing_baseline", "tracked_missing_baseline",
])
async def test_mysql_upgrade_preflight_and_tracking(_test_engine, scenario, capsys):
    name = f"mewhelp_migration_test_{uuid4().hex}"
    async with _test_engine.connect() as raw:
        conn = await raw.execution_options(isolation_level="AUTOCOMMIT")
        await conn.exec_driver_sql(f"CREATE DATABASE {name} CHARACTER SET utf8mb4")
        try:
            await conn.exec_driver_sql(f"USE {name}")
            if scenario != "empty":
                for file in migrations.BASELINE:
                    for statement in migrations.sql_statements((migrations.SQL_DIR / file).read_text(encoding="utf-8")):
                        await conn.exec_driver_sql(statement)
                await conn.exec_driver_sql("INSERT INTO conversations (id, user_id) VALUES (1, 'legacy')")
                await conn.exec_driver_sql(
                    "INSERT INTO tickets (ticket_no, conversation_id, description, ticket_type) "
                    "VALUES ('legacy-ticket', 1, 'original content', '咨询')"
                )
            if scenario == "interrupted":
                await conn.exec_driver_sql("ALTER TABLE review_queue ADD COLUMN publication_chunk_id BIGINT NULL")
            if scenario == "drift":
                await conn.exec_driver_sql("ALTER TABLE tickets ADD COLUMN submission_key VARCHAR(255) NULL")
                with pytest.raises(migrations.PreflightError, match=r"tickets\.submission_key"):
                    await migrations.migrate(conn, apply=True)
                catalog, tables, versions = await migrations.inspect_schema(conn)
                assert "review_queue.publication_chunk_id" not in catalog
                assert migrations.HISTORY not in tables and not versions
                return
            if scenario == "manual":
                for file in migrations.VERSIONS:
                    for statement in migrations.sql_statements((migrations.SQL_DIR / file).read_text(encoding="utf-8")):
                        await conn.exec_driver_sql(statement)
            if scenario in {"missing_baseline", "tracked_missing_baseline"}:
                if scenario == "tracked_missing_baseline":
                    assert await migrations.migrate(conn, apply=True) == 0
                await conn.exec_driver_sql("ALTER TABLE messages DROP COLUMN content")
                with pytest.raises(DBAPIError, match="Unknown column"):
                    await conn.exec_driver_sql("SELECT content FROM messages LIMIT 1")
                before = await migrations.inspect_schema(conn)
                for apply in (False, True):
                    with pytest.raises(migrations.PreflightError, match=r"messages\.content"):
                        await migrations.migrate(conn, apply=apply)
                    assert await migrations.inspect_schema(conn) == before
                return

            status = await migrations.migrate(conn)
            output = capsys.readouterr().out
            assert status == (2 if scenario == "interrupted" else 0)
            if scenario == "interrupted":
                assert "部分完成" in output and "review_queue.publication_chunk_id" in output
            if scenario == "manual":
                assert "待补版本记录" in output
            _, tables, _ = await migrations.inspect_schema(conn)
            assert migrations.HISTORY not in tables

            assert await migrations.migrate(conn, apply=True) == 0
            assert await migrations.migrate(conn, apply=True) == 0
            _, _, versions = await migrations.inspect_schema(conn)
            assert set(versions) == set(migrations.VERSIONS)
            assert all(len(checksum) == 64 for checksum in versions.values())
            if scenario != "empty":
                assert await conn.scalar(text("SELECT description FROM tickets WHERE ticket_no = 'legacy-ticket'")) == "original content"
        finally:
            await conn.exec_driver_sql(f"USE {_test_engine.url.database}")
            await conn.exec_driver_sql(f"DROP DATABASE {name}")
