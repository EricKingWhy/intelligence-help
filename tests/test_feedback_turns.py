import json

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from pymilvus.exceptions import MilvusException
from sqlalchemy import select

from app.db.models import LowConfidenceQuestion
from app.graph import nodes, runtime
from app.graph.build import build_graph
from app.graph.state import ConversationState
from app.main import app


async def test_feedback_keeps_each_repeated_question_bound_to_its_reply(db_session_factory, monkeypatch):
    evidence = [[{"id": 101, "question": "旧证据"}], [{"id": 202, "question": "新证据"}]]

    def answer(state):
        turn = sum(message.type == "human" for message in state["messages"])
        if turn <= 2:
            return {"answer": f"答复 {turn}", "retrieved_snapshot": evidence[turn - 1]}
        return {"answer": "上游超时", "service_failure": {"stage": "generation", "kind": "timeout"}}

    async def skip_summary(cid):
        pass

    builder = StateGraph(ConversationState)
    builder.add_node("script_reply", answer)
    builder.add_node("log", nodes.log_node)
    builder.add_edge(START, "script_reply")
    builder.add_edge("script_reply", "log")
    graph = builder.compile(checkpointer=InMemorySaver())
    monkeypatch.setattr(runtime, "get_graph", lambda: graph)
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", skip_summary)

    question = "退货运费谁承担"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        async def chat(cid=None):
            response = await client.post("/api/chat", json={
                "conversation_id": cid, "user_id": "u1", "message": question,
            })
            response.raise_for_status()
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
            return next(event for event in events if event.get("event") == "done")

        async def down(cid, message_id):
            return await client.post("/api/feedback", json={
                "conversation_id": cid, "message_id": message_id, "question": question, "rating": "down",
            })

        first = await chat()
        cid = first["conversation_id"]
        second = await chat(cid)
        assert (await down(cid, second["message_id"])).json()["pooled"] is True
        failed = await chat(cid)

        history = f"/api/conversations/{cid}/messages"
        page = (await client.get(history, params={"limit": 2})).json()
        assert page["items"][-1]["service_failed"] is True
        assert page["items"][-1]["content"] == failed["final_text"] == "上游超时"
        page = (await client.get(history, params={"limit": 2, "before_id": page["next_before_id"]})).json()
        assert page["items"][-1]["id"] == second["message_id"]
        assert page["items"][-1]["content"] == second["final_text"] == "答复 2"
        page = (await client.get(history, params={"limit": 2, "before_id": page["next_before_id"]})).json()
        user_message, old_reply = page["items"]
        assert old_reply["id"] == first["message_id"]
        assert old_reply["content"] == first["final_text"] == "答复 1"
        assert (await down(cid, old_reply["id"])).json()["pooled"] is True
        assert (await down(cid, failed["message_id"])).json()["pooled"] is False
        assert (await down(cid, user_message["id"])).status_code == 404
        assert (await down(cid + 1, old_reply["id"])).status_code == 404

    async with db_session_factory() as session:
        rows = (await session.scalars(
            select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == cid)
            .order_by(LowConfidenceQuestion.id)
        )).all()
        assert [(row.raw_question, row.source, row.retrieved_chunks) for row in rows] == [
            (question, "user_feedback", evidence[1]), (question, "user_feedback", evidence[0]),
        ]

async def test_refund_feedback_keeps_ranked_evidence_for_the_selected_reply(db_session_factory, monkeypatch):
    def hit(chunk_id, answer, score):
        return {"id": chunk_id, "question": "如何退货", "answer": answer,
                "rerank_score": score, "section_path": "退款政策"}

    def snapshot(chunk):
        return {key: chunk[key] for key in ("question", "answer", "rerank_score", "section_path")}

    initial = hit(101, "低分命中", 0.60)
    best = hit(101, "签收后七天", 0.95)
    second_best = hit(102, "需要保留包装", 0.85)
    third_best = hit(103, "退款原路退回", 0.75)
    later = hit(201, "特殊商品仅支持换货", 0.99)
    results = iter(([initial, second_best], [best, third_best], [later], []))

    async def resolve(query, history):
        return query

    async def classify(query, history):
        return {"intent": "退款退货", "confidence": 0.95}

    async def expand(query):
        return [query, f"{query} 政策"]

    async def search(*args, **kwargs):
        try:
            return next(results)
        except StopIteration:
            raise MilvusException(message="knowledge unavailable")

    async def tool_specs():
        return []

    async def reply(messages):
        return AIMessage(content="请查看这条退款政策。")

    class Model:
        def bind_tools(self, tools):
            return RunnableLambda(reply)

    async def skip_summary(cid):
        pass

    monkeypatch.setattr(nodes.coref, "resolve", resolve)
    monkeypatch.setattr(nodes.intent_mod, "classify", classify)
    monkeypatch.setattr(nodes.query_understanding, "expand_queries", expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", search)
    monkeypatch.setattr(nodes.registry, "get_all_specs", tool_specs)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kwargs: Model())
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", skip_summary)
    graph = build_graph(checkpointer=InMemorySaver())
    monkeypatch.setattr(runtime, "get_graph", lambda: graph)

    question = "订单1001可以退货吗"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        async def chat(conversation_id=None):
            response = await client.post("/api/chat", json={
                "conversation_id": conversation_id, "user_id": "u1", "message": question,
            })
            response.raise_for_status()
            frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
            assert frames[-1]["event"] == "done"
            citations = next((frame["items"] for frame in frames if frame.get("event") == "citations"), [])
            return frames[-1], [(item["n"], item["id"], item["answer"]) for item in citations]

        async def down(message_id):
            response = await client.post("/api/feedback", json={
                "conversation_id": cid, "message_id": message_id, "question": question, "rating": "down",
            })
            response.raise_for_status()
            return response.json()["pooled"]

        first, first_citations = await chat()
        cid = first["conversation_id"]
        second, second_citations = await chat(cid)
        assert first_citations == [(1, 101, best["answer"]), (2, 103, third_best["answer"]),
                                   (3, 102, second_best["answer"])]
        assert second_citations == [(1, 201, later["answer"])]
        assert first["message_id"] != second["message_id"]
        assert await down(first["message_id"]) is True

        failed, failed_citations = await chat(cid)
        assert not failed_citations
        assert await down(failed["message_id"]) is False

    async with db_session_factory() as session:
        rows = (await session.scalars(
            select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == cid)
        )).all()
        assert [(row.raw_question, row.source, row.retrieved_chunks) for row in rows] == [
            (question, "user_feedback", [snapshot(best), snapshot(second_best), snapshot(third_best)]),
        ]


@pytest.mark.parametrize("resume", [False, True])
async def test_stream_completion_matches_saved_agent_reply(resume, db_session_factory, monkeypatch):
    async def resolve(query, history):
        return query

    async def classify(query, history):
        return {"intent": "退款退货" if resume else "订单", "confidence": 0.95}

    async def expand(query):
        return [query]

    async def policies(*args, **kwargs):
        return []

    async def tool_specs():
        return [nodes.registry.get_builtin_spec("query_order")]

    async def model_reply(messages):
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content=[{"type": "text", "text": "订单"}, {"type": "text", "text": "已发货"}])
        return AIMessage(content="我先查一下", tool_calls=[
            {"name": "query_order", "id": "read-order", "args": {"order_id": "1001"}},
        ])

    class Model:
        def bind_tools(self, tools):
            return RunnableLambda(model_reply)

    async def skip_summary(cid):
        return None

    monkeypatch.setattr(nodes.coref, "resolve", resolve)
    monkeypatch.setattr(nodes.intent_mod, "classify", classify)
    monkeypatch.setattr(nodes.query_understanding, "expand_queries", expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", policies)
    monkeypatch.setattr(nodes.registry, "get_all_specs", tool_specs)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kwargs: Model())
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", skip_summary)
    graph = build_graph(checkpointer=InMemorySaver())
    monkeypatch.setattr(runtime, "get_graph", lambda: graph)

    def events(response):
        response.raise_for_status()
        assert response.text.endswith("data: [DONE]\n\n")
        return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/chat", json={"user_id": "u1", "message": "看一下订单"})
        frames = events(response)
        if resume:
            pause = frames[-1]
            assert pause["event"] == "interrupt"
            response = await client.post("/api/actions/resume", json={
                "conversation_id": pause["conversation_id"], "order_id": "1001",
                "interrupt_id": pause["interrupt_id"],
            })
            frames = events(response)

        assert any(frame.get("delta") == "我先查一下" for frame in frames)
        assert {"event": "tool", "name": "query_order"} in frames
        done = frames[-1]
        assert done["event"] == "done"
        history = await client.get(f"/api/conversations/{done['conversation_id']}/messages")
        saved = history.json()["items"][-1]
        assert done["message_id"] == saved["id"]
        assert done["final_text"] == saved["content"] == "订单已发货"
