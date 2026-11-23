import json

import httpx
import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from pymilvus.exceptions import MilvusException
from sqlalchemy import func, select

from app.db.models import LowConfidenceQuestion, Message
from app.graph import nodes, runtime
from app.graph.build import build_graph
from app.main import app


@pytest.mark.parametrize(
    ("scenario", "answer", "diagnosis"),
    [
        ("limit", "繁忙", {"stage": "intent", "kind": "rate_limited"}),
        ("timeout", "超时", {"stage": "generation", "kind": "timeout"}),
        ("milvus", "知识库暂时无法查询", {"stage": "knowledge_retrieval", "kind": "knowledge_unavailable"}),
        ("check", "超时", {"stage": "evidence_check", "kind": "timeout"}),
    ],
)
async def test_service_failure_recovers_without_polluting_knowledge_queue(
    scenario, answer, diagnosis, db_session_factory, monkeypatch,
):
    async def same_question(query, history):
        return query

    async def classify(query, history):
        return {"intent": "物流" if scenario == "timeout" else "商品咨询", "confidence": 0.95}

    async def reject_model(_):
        response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"))
        response.raise_for_status()

    async def timeout(_):
        raise TimeoutError()

    class Model:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, config):
            raise TimeoutError()

    async def hits(query, **kwargs):
        if scenario == "milvus":
            raise MilvusException(message="fixture Milvus unavailable")
        return [{"id": 17, "question": "猫砂盆保修", "answer": "保修一年",
                 "rerank_score": 0.95, "section_path": "政策 / 保修", "content_type": "policy"}]

    async def rewrite(query):
        return {"standard": query, "expanded": []}

    async def tool_specs():
        return []

    monkeypatch.setattr(nodes.coref, "resolve", same_question)
    if scenario == "limit":
        monkeypatch.setattr(nodes.intent_mod.llm, "structured", lambda *a, **k: RunnableLambda(reject_model))
    else:
        monkeypatch.setattr(nodes.intent_mod, "classify", classify)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kwargs: Model())
    monkeypatch.setattr(nodes.registry, "get_all_specs", tool_specs)
    monkeypatch.setattr(nodes.query_understanding, "understand", rewrite)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", hits)
    monkeypatch.setattr(nodes.selfcheck, "_chain", lambda: RunnableLambda(timeout))
    graph = build_graph(checkpointer=InMemorySaver())
    monkeypatch.setattr(runtime, "get_graph", lambda: graph)

    async def skip_summary(cid):
        return None

    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", skip_summary)
    question = "猫砂盆保修多久"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/chat", json={"user_id": "u1", "message": question})
        assert response.is_success
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
        assert any(event.get("replace") and answer in event["delta"] for event in events)
        actions = next(event["items"] for event in events if event.get("event") == "actions")
        assert [action["type"] for action in actions] == ["retry_turn", "transfer_human"]
        assert actions[0]["draft"]["message"] == question
        done = next(event for event in events if event.get("event") == "done")
        cid = done["conversation_id"]

        feedback = await client.post("/api/feedback", json={
            "conversation_id": cid, "message_id": done["message_id"], "question": question, "rating": "down",
        })
        assert feedback.is_success and feedback.json()["pooled"] is False
        async with db_session_factory() as session:
            assert await session.scalar(select(func.count()).select_from(LowConfidenceQuestion)) == 0
            message = await session.get(Message, actions[1]["message_id"])
            assert message.handoff_context["service_failure"] == diagnosis
            reply = await session.get(Message, done["message_id"])
            assert reply.handoff_context["service_failure"] == diagnosis
        messages = await client.get(f"/api/conversations/{cid}/messages")
        assert messages.json()["items"][-1]["service_failed"] is True

        async def knowledge_intent(query, history):
            return {"intent": "商品咨询", "confidence": 0.95}

        async def no_hits(query, **kwargs):
            return []

        monkeypatch.setattr(nodes.intent_mod, "classify", knowledge_intent)
        monkeypatch.setattr(nodes.retrieval, "search_knowledge", no_hits)
        again = await client.post("/api/chat", json={"user_id": "u1", "message": question, "conversation_id": cid})
        assert again.is_success
        assert "retry_turn" not in again.text
        messages = await client.get(f"/api/conversations/{cid}/messages")
        assert "service_failed" not in messages.json()["items"][-1]
        async with db_session_factory() as session:
            item = (await session.scalars(select(LowConfidenceQuestion))).one()
            assert item.source == "retrieval_low_conf"


async def test_generation_failure_after_ticket_attempt_offers_handoff_only():
    from langchain_core.messages import HumanMessage, ToolMessage

    result = await nodes.fallback_reply({
        "messages": [
            HumanMessage("我要建工单", id="db-12"),
            ToolMessage("工单系统响应超时", name="create_ticket", tool_call_id="confirmed"),
        ],
        "service_failure": {"stage": "generation", "kind": "timeout"},
    })
    assert result["suggested_actions"] == [{"type": "transfer_human", "message_id": 12}]
    assert "确认原工单状态" in result["answer"]
