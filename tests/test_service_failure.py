import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from pymilvus.exceptions import MilvusException
from sqlalchemy import func, select

from app.db.models import LowConfidenceQuestion, Message, Ticket, ToolAuditLog
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
        ("faq_check", "超时", {"stage": "evidence_check", "kind": "timeout"}),
        ("faq_deadline", "超时", {"stage": "faq_tool", "kind": "timeout"}),
        ("faq_after_write", "确认原工单状态", {"stage": "faq_tool", "kind": "timeout"}),
        ("faq_milvus", "知识库暂时无法查询", {"stage": "faq_retrieval", "kind": "knowledge_unavailable"}),
        ("faq_limit", "繁忙", {"stage": "faq_retrieval", "kind": "rate_limited"}),
        ("faq_fallback_milvus", "知识库暂时无法查询", {"stage": "faq_retrieval", "kind": "knowledge_unavailable"}),
        ("faq_fallback_limit", "繁忙", {"stage": "faq_retrieval", "kind": "rate_limited"}),
        ("faq_retrieval_after_write", "确认原工单状态", {"stage": "faq_retrieval", "kind": "knowledge_unavailable"}),
    ],
)
async def test_service_failure_recovers_without_polluting_knowledge_queue(
    scenario, answer, diagnosis, db_session_factory, monkeypatch,
):
    after_write = scenario.endswith("_after_write")
    faq_deadline = diagnosis["stage"] == "faq_tool"
    faq_retrieval = diagnosis["stage"] == "faq_retrieval"
    uses_faq = scenario.startswith("faq_")
    faq_fallback = scenario.startswith("faq_fallback_")
    category = "商品规格" if faq_fallback else None
    searches = []

    async def same_question(query, history):
        return query

    async def classify(query, history):
        return {"intent": "物流" if uses_faq or scenario == "timeout" else "商品咨询", "confidence": 0.95}

    async def raise_rate_limit(_):
        response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"))
        response.raise_for_status()

    async def timeout(_):
        if faq_deadline:
            await asyncio.Future()
        raise TimeoutError()

    model_calls = 0

    class Model:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, config):
            nonlocal model_calls
            model_calls += 1
            if uses_faq:
                if model_calls > 1:
                    return AIMessage(content="没有查到政策。")
                args = {"keyword": "猫砂盆保修多久"}
                if category:
                    args["category"] = category
                calls = [{"name": "query_faq", "id": "faq", "args": args}]
                if after_write:
                    calls.insert(0, {"name": "create_ticket", "id": "create", "args": {
                        "ticket_type": "咨询", "description": "请帮我核对保修期",
                    }})
                    calls.insert(1, {"name": "submit_refund", "id": "refund", "args": {
                        "order_id": "1001", "reason": "质量问题",
                    }})
                return AIMessage(content="我先查一下", tool_calls=calls)
            raise TimeoutError()

    async def hits(query, *, category=None, **kwargs):
        searches.append(category)
        if faq_fallback and category:
            return []
        if faq_retrieval and diagnosis["kind"] == "rate_limited":
            await raise_rate_limit(query)
        if scenario == "milvus" or faq_retrieval:
            raise MilvusException(message="fixture Milvus unavailable")
        return [{"id": 17, "question": "猫砂盆保修", "answer": "保修一年",
                 "rerank_score": 0.95, "section_path": "政策 / 保修", "content_type": "policy"}]

    async def rewrite(query):
        return {"standard": query, "expanded": []}

    async def tool_specs():
        if not uses_faq:
            return []
        faq = nodes.registry.get_builtin_spec("query_faq")
        specs = [replace(faq, timeout=0.05)] if faq_deadline else [faq]
        if after_write:
            specs.append(nodes.registry.get_builtin_spec("create_ticket"))
        return specs

    monkeypatch.setattr(nodes.coref, "resolve", same_question)
    if scenario == "limit":
        monkeypatch.setattr(nodes.intent_mod.llm, "structured", lambda *a, **k: RunnableLambda(raise_rate_limit))
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
        if after_write:
            pause = next(event for event in events if event.get("event") == "interrupt")
            response = await client.post("/api/actions/resume", json={
                "conversation_id": pause["conversation_id"], "confirmed": True,
                "confirmation_id": pause["preview"]["confirmation_id"],
            })
            assert response.is_success
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
        assert any(event.get("replace") and answer in event["delta"] for event in events)
        if uses_faq:
            assert model_calls == 1
            assert searches == ([category, None] if faq_fallback else [None])
        actions = next(event["items"] for event in events if event.get("event") == "actions")
        expected_actions = ["transfer_human"] if after_write else ["retry_turn", "transfer_human"]
        assert [action["type"] for action in actions] == expected_actions
        if not after_write:
            assert actions[0]["draft"]["message"] == question
        done = next(event for event in events if event.get("event") == "done")
        cid = done["conversation_id"]

        feedback = await client.post("/api/feedback", json={
            "conversation_id": cid, "message_id": done["message_id"], "question": question, "rating": "down",
        })
        assert feedback.is_success and feedback.json()["pooled"] is False
        async with db_session_factory() as session:
            assert await session.scalar(select(func.count()).select_from(LowConfidenceQuestion)) == 0
            message = await session.get(Message, actions[-1]["message_id"])
            assert await session.scalar(select(func.count()).select_from(Ticket)) == int(after_write)
            assert message.handoff_context["service_failure"] == diagnosis
            reply = await session.get(Message, done["message_id"])
            assert reply.handoff_context["service_failure"] == diagnosis
            if uses_faq:
                audit = (await session.scalars(
                    select(ToolAuditLog).where(ToolAuditLog.tool_name == "query_faq")
                )).one()
                assert audit.tool_name == "query_faq"
                kind = diagnosis["kind"]
                assert audit.status == ("超时" if kind == "timeout" else "失败")
                assert audit.error_message == f"{diagnosis['stage']}:{kind}"
                assert audit.retry_count == 0
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
