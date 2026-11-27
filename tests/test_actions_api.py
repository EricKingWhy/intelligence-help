# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import asyncio
import hashlib

import httpx
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import actions, auth
from app.db import base, repository
from app.db.models import Ticket


def test_create_ticket_action_writes(client, monkeypatch):
    async def fake_create(cid, desc, ttype):
        assert cid == 5 and ttype == "投诉"
        return "T20260715001"

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



async def test_refund_idempotency_across_owned_conversations(db_session_factory, _test_engine, monkeypatch):
    accounts = {"customer-a": "fixture-a", "Customer-a": "fixture-b"}
    monkeypatch.setattr(auth, "credentials", lambda: {
        name: {"role": "customer", "sha256": hashlib.sha256(secret.encode()).hexdigest()}
        for name, secret in accounts.items()
    })
    first = await repository.create_conversation("customer-a")
    second = await repository.create_conversation("customer-a")
    other = await repository.create_conversation("Customer-a")

    app = FastAPI()
    app.include_router(actions.router)
    transport = httpx.ASGITransport(app=app)
    headers = {"Idempotency-Key": "refund-retry"}
    user = ("customer-a", accounts["customer-a"])
    second_user = ("Customer-a", accounts["Customer-a"])

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async def refund(cid, *, credentials=user, order="1001", reason="质量问题"):
            return await client.post(
                "/api/actions/create-refund", auth=credentials, headers=headers,
                json={"conversation_id": cid, "order_id": order, "reason": reason},
            )

        a, b = await asyncio.gather(refund(first), refund(second))
        assert a.status_code == b.status_code == 200
        ticket_no = a.json()["ticket_no"]
        assert b.json()["ticket_no"] == ticket_no

        # Reconnect through an independent engine to rule out process-local replay state.
        new_engine = create_async_engine(_test_engine.url)
        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(base, "async_session", async_sessionmaker(new_engine, expire_on_commit=False))
                replay = await refund(second)
        finally:
            await new_engine.dispose()
        assert replay.status_code == 200 and replay.json()["ticket_no"] == ticket_no

        assert (await refund(second, reason="发错货")).status_code == 409
        assert (await refund(second, order="2002")).status_code == 409
        assert (await refund(second, order="not-owned")).status_code == 404
        assert (await refund(other)).status_code == 404
        separate_account = await refund(other, credentials=second_user)
        assert separate_account.status_code == 200
        assert separate_account.json()["ticket_no"] != ticket_no

        async def create_ticket(cid):
            return await client.post(
                "/api/actions/create-ticket", auth=user, headers=headers,
                json={"conversation_id": cid, "description": "普通工单", "ticket_type": "咨询"},
            )

        manual = await create_ticket(second)
        assert manual.status_code == 200 and manual.json()["ticket_no"] != ticket_no
        manual_replay = await create_ticket(first)
        assert manual_replay.status_code == 200 and manual_replay.json() == manual.json()
        async with db_session_factory() as db:
            rows = (await db.scalars(select(Ticket))).all()
            assert len(rows) == 3
            assert next(row.conversation_id for row in rows if row.ticket_no == ticket_no) in (first, second)



async def test_agent_ticket_replay_remains_bound_to_its_conversation(db_session_factory):
    first = await repository.create_conversation("customer-a")
    second = await repository.create_conversation("customer-a")
    confirmation = "a" * 64
    key = "agent:" + confirmation
    ticket_no = await repository.create_ticket(first, "已确认工单", "咨询", submission_key=key)
    assert await repository.get_ticket_result(second, key) is None

    app = FastAPI()
    app.include_router(actions.router)
    app.dependency_overrides[auth.identity] = lambda: ("customer-a", "customer")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/actions/resume", json={
            "conversation_id": first, "confirmed": True, "confirmation_id": confirmation,
        })
    assert response.is_success
    assert ticket_no in response.text
