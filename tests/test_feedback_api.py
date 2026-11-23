# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""Feedback route behavior for persisted replies and legacy callers."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import feedback as feedback_api


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(feedback_api.router)
    return TestClient(app)


@pytest.fixture
def spy_insert(monkeypatch):
    calls = []

    async def fake_insert(cid, q, source, reason, retrieved_chunks=None):
        calls.append({"cid": cid, "q": q, "source": source, "chunks": retrieved_chunks})
        return 1
    monkeypatch.setattr(feedback_api.repository, "insert_low_confidence", fake_insert)
    return calls


def test_down_uses_question_and_evidence_from_selected_reply(client, spy_insert, monkeypatch):
    question = "退货运费谁承担"
    evidence = [{"id": 101, "question": "退货运费"}]

    async def selected(cid, message_id):
        assert (cid, message_id) == (1, 10)
        return SimpleNamespace(handoff_context={"feedback": {"question": question, "evidence": evidence}})

    monkeypatch.setattr(feedback_api.repository, "get_feedback_reply", selected)
    response = client.post("/api/feedback", json={
        "conversation_id": 1, "message_id": 10, "question": "客户端气泡文本", "rating": "down",
    })
    assert response.status_code == 200 and response.json()["pooled"] is True
    assert spy_insert[0]["q"] == question
    assert spy_insert[0]["source"] == "user_feedback"
    assert spy_insert[0]["chunks"] == evidence


@pytest.mark.parametrize("message_id", [None, 10])
def test_legacy_feedback_pools_without_evidence(client, spy_insert, monkeypatch, message_id):
    async def selected(cid, reply_id):
        assert (cid, reply_id) == (1, 10)
        return SimpleNamespace(handoff_context=None)

    monkeypatch.setattr(feedback_api.repository, "get_feedback_reply", selected)
    response = client.post("/api/feedback", json={
        "conversation_id": 1, "message_id": message_id, "question": "历史问题", "rating": "down",
    })
    assert response.status_code == 200 and response.json()["pooled"] is True
    assert spy_insert[0]["q"] == "历史问题" and spy_insert[0]["chunks"] is None


def test_up_logs_only(client, spy_insert):
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "up", "question": "q"})
    assert r.status_code == 200 and r.json()["pooled"] is False
    assert spy_insert == []                       # 👍 不落库


def test_rejects_bad_rating(client, spy_insert):
    r = client.post("/api/feedback", json={"conversation_id": 1, "rating": "meh", "question": "q"})
    assert r.status_code == 422
