# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch09 审核 API:列表/详情/通过(写回知识库)/驳回;只有待审可流转;写回失败状态不动。"""
import asyncio

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import review as review_api
from app.db import repository


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(review_api.router)
    return a


@pytest.fixture
async def client(app, db_session_factory):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


@pytest.fixture
def stub_kb(monkeypatch):
    async def vectorize(*, chunk_ids):
        for chunk_id in chunk_ids:
            await repository.mark_chunk_vectorized(chunk_id, str(chunk_id))

    monkeypatch.setattr(review_api.dualwrite, "vectorize_pending", vectorize)


async def _seed():
    rid = await repository.insert_review_item("猫窝是否支持水洗", "可以,以平台售后规则为准")
    lcq = await repository.insert_low_confidence(
        None, "猫窝可以洗吗", "user_feedback", None,
        retrieved_chunks=[{"question": "q", "answer": "a", "rerank_score": 0.3, "section_path": "s"}])
    await repository.set_matched_review(lcq, rid)
    return rid


async def test_queue_lists_pending(client):
    rid = await _seed()
    r = await client.get("/api/review/queue", params={"status": "待审"})
    assert r.status_code == 200
    items = r.json()["items"]
    assert items[0]["id"] == rid and items[0]["occurrence_count"] == 1


async def test_detail_carries_raws_and_snapshot(client):
    rid = await _seed()
    r = await client.get(f"/api/review/{rid}")
    assert r.status_code == 200
    d = r.json()
    assert d["normalized_question"] == "猫窝是否支持水洗"
    assert d["raws"][0]["source"] == "user_feedback"
    assert d["raws"][0]["retrieved_chunks"][0]["rerank_score"] == 0.3
    assert (await client.get("/api/review/99999")).status_code == 404


async def test_concurrent_approve_writes_once(client, stub_kb):
    rid = await _seed()
    url = f"/api/review/{rid}/approve"
    data = {"approved_answer": "可以水洗,注意阴干"}
    a, b = await asyncio.gather(client.post(url, json=data), client.post(url, json=data))
    assert a.status_code == b.status_code == 200
    assert a.json()["chunk_ids"] == b.json()["chunk_ids"]
    assert await repository.list_chunk_pairs() == [("猫窝是否支持水洗", data["approved_answer"])]
    assert (await client.get(f"/api/review/{rid}")).json()["review_status"] == "通过"
    assert (await client.post(f"/api/review/{rid}/reject")).status_code == 409


async def test_approve_retries_existing_publication(client, monkeypatch, stub_kb):
    rid = await _seed()
    url = f"/api/review/{rid}/approve"
    data = {"approved_answer": "可以水洗"}
    vectorize = review_api.dualwrite.vectorize_pending

    async def fail_vectorize(*, chunk_ids):
        raise RuntimeError("Milvus 不在线")

    monkeypatch.setattr(review_api.dualwrite, "vectorize_pending", fail_vectorize)
    assert (await client.post(url, json=data)).status_code == 502
    assert (await client.get(f"/api/review/{rid}")).json()["publishing"] is True
    assert (await client.post(f"/api/review/{rid}/reject")).status_code == 409
    assert (await client.post(url, json={"approved_answer": "不同答案"})).status_code == 409
    monkeypatch.setattr(review_api.dualwrite, "vectorize_pending", vectorize)

    assert (await client.post(url, json=data)).status_code == 200
    assert await repository.list_chunk_pairs() == [("猫窝是否支持水洗", data["approved_answer"])]


async def test_reject_flips_status(client):
    rid = await _seed()
    assert (await client.post(f"/api/review/{rid}/reject")).status_code == 200
    assert (await client.get(f"/api/review/{rid}")).json()["review_status"] == "驳回"
    assert (await client.post(f"/api/review/{rid}/approve",
                              json={"approved_answer": "x"})).status_code == 409
