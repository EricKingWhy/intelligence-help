# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import asyncio

from app.db import repository


def test_create_ticket_action_writes(client, monkeypatch):
    async def fake_create(cid, desc, ttype):
        assert cid == 5 and ttype == "投诉"
        return "T20260715001"

    from app.api import actions
    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    r = client.post("/api/actions/create-ticket", json={
        "conversation_id": 5, "description": "东西坏了", "ticket_type": "投诉"})
    assert r.status_code == 200
    assert r.json()["ticket_no"] == "T20260715001"


def test_create_ticket_action_rejects_bad_type(client):
    r = client.post("/api/actions/create-ticket", json={
        "conversation_id": 5, "description": "x", "ticket_type": "乱填"})
    assert r.status_code == 422


async def test_ticket_retry_reuses_the_durable_result(db_session_factory):
    conversation_id = await repository.create_conversation("customer-a")

    async def submit():
        return await repository.create_ticket(conversation_id, "包裹破损", "售后",
                                              submission_key="ticket:request-1")

    first, second = await asyncio.gather(submit(), submit())
    assert first == second == await submit()
    assert first != await repository.create_ticket(conversation_id, "包裹破损", "售后",
                                                   submission_key="ticket:request-2")


def test_completed_agent_confirmation_returns_original_ticket(client, monkeypatch):
    async def result(conversation_id, submission_key):
        assert conversation_id == 5
        assert submission_key == "agent:" + "a" * 64
        return "T-existing"

    monkeypatch.setattr(repository, "get_ticket_result", result)
    response = client.post("/api/actions/resume", json={
        "conversation_id": 5, "confirmed": True, "confirmation_id": "a" * 64,
    })
    assert response.is_success
    assert "T-existing" in response.text
