"""Inspect or apply the MewHelp MySQL schema from the repository SQL files."""

import argparse
import asyncio
import hashlib
import re
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
HISTORY = "mewhelp_schema_migrations"
TICKET_SCOPE_VERSION = "ch13-ticket-submission-scope.sql"
TICKET_SCOPE_KEYS = "(LEFT(t.submission_key, 7) IN ('ticket:', 'refund:'))"
BASELINE = (
    "ch02-ddl.sql", "ch03-ddl.sql", "ch03-staging-review.sql", "ch04-ddl.sql",
    "ch06-ticket-type.sql", "ch07-ddl.sql", "ch08-ddl.sql", "ch09-ddl.sql", "ch10-ddl.sql",
)
# Each tuple describes the catalog objects created by one atomic ALTER in the SQL file.
VERSIONS = {
    "ch11-reliability.sql": (
        (("review_queue.publication_chunk_id", ("bigint", "YES")),),
        (("qa_extraction_staging.publication_chunk_id", ("bigint", "YES")),),
        (("tickets.submission_key", ("varchar(255)", "YES", "ascii_bin")),
         ("tickets.uq_ticket_submission", ((0, "conversation_id", None), (0, "submission_key", None)))),
    ),
    "ch12-handoff.sql": (
        (("messages.handoff_context", ("json", "YES")),),
        (("tickets.handoff_context", ("json", "YES")), ("tickets.assigned_to", ("varchar(64)", "YES")),
         ("tickets.resolution", ("text", "YES"))),
    ),
    TICKET_SCOPE_VERSION: (
        (("tickets.submission_owner", ("varbinary(256)", "YES")),
         ("tickets.uq_ticket_account_submission", ((0, "submission_owner", None), (0, "submission_key", None)))),
    ),
}


class PreflightError(Exception):
    pass


def _required_baseline_columns() -> set[str]:
    from app.db.models import Base

    # Objects added by tracked migrations are checked separately so old schemas can upgrade.
    tracked_objects = {
        name for version in VERSIONS.values() for step in version for name, _ in step
    }
    return {
        f"{table.name}.{column.name}"
        for table in Base.metadata.tables.values()
        for column in table.columns
    } - tracked_objects


def sql_statements(source: str) -> list[str]:
    source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("--"))
    # The checked-in DDL uses single-quoted literals, some with semicolons in comments.
    return [part.strip() for part in re.split(r";(?=(?:[^']*'[^']*')*[^']*$)", source) if part.strip()]


async def inspect_schema(conn):
    columns = (await conn.execute(text(
        "SELECT table_name, column_name, column_type, is_nullable, collation_name "
        "FROM information_schema.columns WHERE table_schema = DATABASE()"
    ))).all()
    indexes = (await conn.execute(text(
        "SELECT index_name, non_unique, column_name, sub_part FROM information_schema.statistics "
        "WHERE table_schema = DATABASE() AND table_name = 'tickets' "
        "AND index_name IN ('uq_ticket_submission', 'uq_ticket_account_submission') "
        "ORDER BY index_name, seq_in_index"
    ))).all()
    catalog = {f"{table}.{name}": (kind, nullable, collation) for table, name, kind, nullable, collation in columns}
    index_shapes = {}
    for name, non_unique, column, prefix in indexes:
        index_shapes.setdefault(f"tickets.{name}", []).append((non_unique, column, prefix))
    catalog.update({name: tuple(shape) for name, shape in index_shapes.items()})
    tables = {row[0] for row in columns}
    history = {}
    if HISTORY in tables:
        history = dict((await conn.exec_driver_sql(f"SELECT version, sha256 FROM {HISTORY}")).all())
    return catalog, tables, history


def step_status(catalog, expected):
    present = []
    missing = []
    mismatched = []
    for key, shape in expected:
        actual = catalog.get(key)
        if actual is None:
            missing.append(key)
        elif (actual if isinstance(shape[0], tuple) else actual[:len(shape)]) == shape:
            present.append(key)
        else:
            mismatched.append(f"{key}: expected {shape}, found {actual}")
    return present, missing, mismatched


async def _submission_owner_pending(conn) -> bool:
    return bool(await conn.scalar(text(
        f"SELECT EXISTS (SELECT 1 FROM tickets t WHERE {TICKET_SCOPE_KEYS} AND t.submission_owner IS NULL)"
    )))


async def _preflight_ticket_scope(conn, *, has_owner, recorded) -> bool:
    conflicts = (await conn.exec_driver_sql(
        "SELECT MIN(t.ticket_no), MAX(t.ticket_no), COUNT(*) "
        "FROM tickets t JOIN conversations c ON c.id = t.conversation_id "
        f"WHERE {TICKET_SCOPE_KEYS} GROUP BY CAST(c.user_id AS BINARY), t.submission_key "
        "HAVING COUNT(*) > 1 ORDER BY MIN(t.ticket_no) LIMIT 5"
    )).all()
    if conflicts:
        examples = "、".join(f"{first} / {last}（{count} 条）" for first, last, count in conflicts)
        raise PreflightError(f"同账号提交标识存在历史冲突，先核对原工单：{examples}")
    if not has_owner:
        return False

    wrong_owner = await conn.scalar(text(
        "SELECT t.ticket_no FROM tickets t JOIN conversations c ON c.id = t.conversation_id "
        "WHERE t.submission_owner IS NOT NULL AND NOT (t.submission_owner <=> "
        f"CASE WHEN {TICKET_SCOPE_KEYS} THEN CAST(c.user_id AS BINARY) END) LIMIT 1"
    ))
    if wrong_owner is not None:
        raise PreflightError(f"工单提交账号与会话归属不一致：{wrong_owner}")
    pending = await _submission_owner_pending(conn)
    if recorded and pending:
        raise PreflightError(f"{TICKET_SCOPE_VERSION} 已记录，但仍有工单提交账号未回填")
    return pending


async def migrate(conn, *, apply=False):
    catalog, tables, history = await inspect_schema(conn)
    empty = not tables
    missing_base = sorted(_required_baseline_columns() - catalog.keys())
    if empty:
        print("ch02–ch10: 初始化待执行（仅建表和列）")
    elif missing_base:
        raise PreflightError("ch02–ch10 基线缺失：" + ", ".join(missing_base))
    else:
        for key, required in (("tickets.ticket_type", "'退款'"),
                              ("qa_extraction_staging.status", "'approved','rejected'")):
            if required not in catalog[key][0]:
                raise PreflightError(f"ch02–ch10 基线定义不匹配：{key} = {catalog[key][0]}")
        print("ch02–ch10: 基线已就绪")

    errors = []
    plan = []
    prior_incomplete = False
    interrupted = False
    for version, shapes in VERSIONS.items():
        source = (SQL_DIR / version).read_text(encoding="utf-8")
        digest = hashlib.sha256(source.encode()).hexdigest()
        statements = sql_statements(source)
        backfill = None
        if version == TICKET_SCOPE_VERSION:
            if not statements or not statements[-1].startswith("UPDATE tickets "):
                raise PreflightError(f"{version} 缺少工单提交账号回填")
            backfill = statements.pop()
        if len(statements) != len(shapes) or any(not stmt.startswith("ALTER TABLE ") for stmt in statements):
            raise PreflightError(f"{version} 的 SQL 与预检定义不一致")
        states = [step_status(catalog, shape) for shape in shapes]
        present = [item for found, _, _ in states for item in found]
        missing = [item for _, needed, _ in states for item in needed]
        mismatched = [item for _, _, drift in states for item in drift]
        mixed_step = any(found and needed for found, needed, _ in states)
        if version in history and (history[version] != digest or missing or mismatched):
            errors.append(f"{version} 的记录与当前 SQL 或数据库结构不一致")
        if prior_incomplete and (present or version in history):
            errors.append(f"{version} 已存在部分或全部变更，但前置版本未完成")
        if mismatched:
            errors.extend(mismatched)
        if mixed_step:
            errors.append(f"{version} 的同一条 ALTER 只有部分变更存在")
        if mismatched or mixed_step:
            state = "结构不一致"
        elif not missing:
            state = "已记录" if version in history else "已完成，待补版本记录"
        elif present:
            state = "部分完成"
            interrupted = True
        else:
            state = "待执行"
        print(f"{version}: {state}")
        if missing:
            print("  待新增：" + ", ".join(missing))
        if present and missing:
            print("  已存在：" + ", ".join(present))
        prior_incomplete = prior_incomplete or bool(missing or mismatched)
        pending = [stmt for stmt, (_, needed, _) in zip(statements, states, strict=True) if needed]
        plan.append((version, digest, shapes, pending, backfill))

    if errors:
        raise PreflightError("\n".join(errors))
    if "tickets.submission_key" in catalog and await _preflight_ticket_scope(
        conn, has_owner="tickets.submission_owner" in catalog, recorded=TICKET_SCOPE_VERSION in history,
    ):
        print(f"{TICKET_SCOPE_VERSION}: 部分完成，待回填工单提交账号")
        interrupted = True
    if not apply:
        return 2 if interrupted else 0

    if empty:
        for file in BASELINE:
            for stmt in sql_statements((SQL_DIR / file).read_text(encoding="utf-8")):
                if stmt.startswith(("CREATE TABLE ", "ALTER TABLE ")):
                    await conn.exec_driver_sql(stmt)
    if HISTORY not in tables:
        await conn.exec_driver_sql(
            f"CREATE TABLE {HISTORY} (version VARCHAR(128) PRIMARY KEY, sha256 CHAR(64) NOT NULL, "
            "recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP) ENGINE=InnoDB"
        )

    for version, digest, shapes, pending, backfill in plan:
        if version in history:
            continue
        for stmt in pending:
            await conn.exec_driver_sql(stmt)
        current, _, _ = await inspect_schema(conn)
        if any(missing or drift for _, missing, drift in (step_status(current, shape) for shape in shapes)):
            raise PreflightError(f"{version} 执行后结构与预期不一致")
        if backfill is not None:
            await conn.exec_driver_sql(backfill)
            if await _submission_owner_pending(conn):
                raise PreflightError(f"{version} 未完成工单提交账号回填")
        await conn.execute(text(f"INSERT INTO {HISTORY} (version, sha256) VALUES (:version, :digest)"),
                           {"version": version, "digest": digest})
        print(f"{version}: 已写入版本记录")
    return 0


async def run(database_url, *, apply):
    engine = create_async_engine(database_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            lock = await conn.scalar(text("SELECT GET_LOCK(CONCAT('mewhelp:', MD5(DATABASE())), 0)"))
            if lock != 1:
                raise PreflightError("无法取得迁移锁，请确认目标库名称以及是否有另一条命令在运行")
            try:
                return await migrate(conn, apply=apply)
            finally:
                await conn.execute(text("SELECT RELEASE_LOCK(CONCAT('mewhelp:', MD5(DATABASE())))"))
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="apply verified pending SQL and record each version")
    args = parser.parse_args()
    from app.config import settings

    try:
        return asyncio.run(run(settings.database_url, apply=args.apply))
    except PreflightError as error:
        parser.exit(2, f"预检阻止升级：{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
