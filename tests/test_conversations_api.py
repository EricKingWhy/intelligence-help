# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch07 验收配套:会话列表 / 历史消息两个只读接口(路由层,repository 打桩;
真库行为在 test_repository.py::test_list_conversations_* 覆盖)。"""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def conv_client():
    from app.api.conversations import router
    test_app = FastAPI()
    test_app.include_router(router)
    return TestClient(test_app)


def test_list_conversations_passes_user_id(conv_client, monkeypatch):
    from app.api import conversations as api
    seen = {}

    async def fake_list(user_id):
        seen["user_id"] = user_id
        return [{"id": 2, "status": "进行中", "preview": "猫粮怎么选",
                 "has_summary": False, "updated_at": None},
                {"id": 1, "status": "进行中", "preview": "订单1001到哪了",
                 "has_summary": True, "updated_at": None}]
    monkeypatch.setattr(api.repository, "list_conversations", fake_list)

    r = conv_client.get("/api/conversations", params={"user_id": "u1"})
    assert r.status_code == 200 and seen["user_id"] == "u1"
    items = r.json()["items"]
    assert [it["id"] for it in items] == [2, 1]
    assert items[1]["has_summary"] is True


def test_list_messages_serializes_dialog(conv_client, monkeypatch):
    from app.api import conversations as api

    async def fake_get(cid):
        return object()

    async def fake_msgs(cid, *, before_id, limit):
        assert (cid, before_id, limit) == (7, None, 51)
        return [SimpleNamespace(id=1, role="user", content="订单1001到哪了",
                                created_at=None, handoff_context=None),
                SimpleNamespace(id=2, role="assistant", content="在路上",
                                created_at=None, handoff_context=None)]
    monkeypatch.setattr(api.repository, "get_conversation", fake_get)
    monkeypatch.setattr(api.repository, "list_dialog_messages", fake_msgs)

    r = conv_client.get("/api/conversations/7/messages")
    assert r.status_code == 200
    assert r.json()["next_before_id"] is None
    assert [(m["role"], m["content"]) for m in r.json()["items"]] == [
        ("user", "订单1001到哪了"), ("assistant", "在路上")]


def test_list_messages_404_when_missing(conv_client, monkeypatch):
    from app.api import conversations as api

    async def fake_get(cid):
        return None
    monkeypatch.setattr(api.repository, "get_conversation", fake_get)
    assert conv_client.get("/api/conversations/999999/messages").status_code == 404


async def test_history_pages_replay_persisted_ticket_offer(db_session_factory, monkeypatch):
    from httpx import ASGITransport, AsyncClient
    from langchain_core.messages import HumanMessage

    from app.db import repository
    from app.graph import nodes
    from app.main import app

    cid = await repository.create_conversation("u1")
    user_id = await repository.append_message(cid, "user", "我要投诉")
    state = {"conversation_id": cid, "user_id": "u1",
             "messages": [HumanMessage("我要投诉", id=f"db-{user_id}")]}
    old_wording = nodes.COMPLAINT_REPLY
    monkeypatch.setattr(nodes, "COMPLAINT_REPLY", "我会核实并跟进。")
    await nodes.log_node({**state, **await nodes.complaint_reply(state)})
    await nodes.log_node({**state, "answer": old_wording, "suggested_actions": []})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        url = f"/api/conversations/{cid}/messages"
        page = (await client.get(url, params={"limit": 1})).json()
        assert page["items"][0]["content"] == old_wording
        assert page["items"][0]["can_create_ticket"] is False
        older = (await client.get(url, params={"before_id": page["next_before_id"]})).json()["items"]
        assert older[-1]["content"] == "我会核实并跟进。"
        assert older[-1]["can_create_ticket"] is True
        assert older[0]["handoff_message_id"] == user_id
