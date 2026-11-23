import json

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from sqlalchemy import select

from app.db.models import LowConfidenceQuestion
from app.graph import nodes, runtime
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
            return next(event for event in events if event["event"] == "done")

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
        page = (await client.get(history, params={"limit": 2, "before_id": page["next_before_id"]})).json()
        assert page["items"][-1]["id"] == second["message_id"]
        page = (await client.get(history, params={"limit": 2, "before_id": page["next_before_id"]})).json()
        user_message, old_reply = page["items"]
        assert old_reply["id"] == first["message_id"]
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
