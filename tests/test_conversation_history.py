from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, insert

from app.api.conversations import router
from app.db import repository
from app.db.models import Message


async def test_long_history_pagination_and_sidebar_query_budget(db_session_factory, _test_engine):
    cid = await repository.create_conversation("history-user")
    other = await repository.create_conversation("other-user")
    for _ in range(49):
        await repository.create_conversation("history-user")
    async with db_session_factory.begin() as session:
        await session.execute(insert(Message), [
            {"conversation_id": cid, "role": "user" if i % 2 == 0 else "assistant",
             "content": f"消息{i:03d}" + "猫" * 60}
            for i in range(125)
        ])
        await session.execute(insert(Message), [
            {"conversation_id": other, "role": "user", "content": "other conversation"},
            {"conversation_id": cid, "role": "tool", "content": "hidden tool"},
        ])
    original = await repository.list_dialog_messages(cid)
    statements = []

    def count_select(conn, cursor, statement, *args):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    api = FastAPI()
    api.include_router(router)
    event.listen(_test_engine.sync_engine, "before_cursor_execute", count_select)
    try:
        async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test") as client:
            sidebar = (await client.get("/api/conversations", params={"user_id": "history-user"})).json()["items"]
            assert len(sidebar) == 50 and len(statements) == 1
            assert sidebar[-1]["preview"] == original[0].content[:40]
            url = f"/api/conversations/{cid}/messages"
            statements.clear()
            first = await client.get(url)
            page = first.json()
            assert len(page["items"]) == 50 and len(statements) == 2
            assert page["previous_question"] == original[74].content
            ids = [row["id"] for row in page["items"]]
            while page["next_before_id"]:
                response = await client.get(url, params={"before_id": page["next_before_id"]})
                page = response.json()
                ids = [row["id"] for row in page["items"]] + ids
            assert ids == [row.id for row in original]
            assert (await client.get(url, params={"limit": 101})).status_code == 422
            print(f"History first page: {len(first.content)} bytes, 50/125 rows, 2 SELECTs. Sidebar: 1 SELECT.")
    finally:
        event.remove(_test_engine.sync_engine, "before_cursor_execute", count_select)
