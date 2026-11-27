# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import asyncio
import json
from itertools import count
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import interrupt

from app.api.auth import identity
from app.graph import nodes, runtime
from app.graph.build import build_graph
from app.graph.state import ConversationState
from app.main import app


@pytest.mark.parametrize("authenticated", [True, False])
async def test_order_card_only_resumes_its_own_pause(monkeypatch, authenticated):
    selected_orders = []

    def choose_order(state):
        selected = interrupt({"type": "select_order", "orders": [{"order_id": "1001"}]})
        selected_orders.append(selected)
        return {"answer": "已选择订单 " + selected}

    builder = StateGraph(ConversationState)
    builder.add_node("script_reply", choose_order)
    builder.add_edge(START, "script_reply")
    graph = builder.compile(checkpointer=InMemorySaver())

    async def conversation(cid):
        return SimpleNamespace(user_id="u1", summary=None, summary_upto_msg_id=None)

    async def append_message(cid, role, *, content):
        return 1 if content == "第一轮查询" else 2

    async def no_summary(cid):
        pass

    monkeypatch.setattr(runtime.repository, "get_conversation", conversation)
    monkeypatch.setattr(runtime.repository, "append_message", append_message)
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", no_summary)
    monkeypatch.setattr(runtime, "get_graph", lambda: graph)
    monkeypatch.setitem(app.dependency_overrides, identity, lambda: ("u1", "customer") if authenticated else None)
    config = {"configurable": {"thread_id": "7"}}

    async with _client() as client:
        first_card = await client.post("/api/chat", json={
            "user_id": "u1", "conversation_id": 7, "message": "第一轮查询",
        })
        first_pause = (await graph.aget_state(config)).interrupts[0].id
        body = {"conversation_id": 7, "order_id": "1001", "interrupt_id": first_pause}
        first_result = await client.post("/api/actions/resume", json=body)
        assert "已选择订单 1001" in first_result.text

        second_card = await client.post("/api/chat", json={
            "user_id": "u1", "conversation_id": 7, "message": "第二轮查询",
        })
        before = await graph.aget_state(config)
        current_pause = before.interrupts[0].id
        assert current_pause != first_pause
        current = {**body, "interrupt_id": current_pause}

        for invalid, status in (
            (body, 409),
            ({**current, "order_id": "2002"}, 409),
            ({"conversation_id": 7, "order_id": "1001"}, 409),
            ({**current, "order_id": "999999"}, 404),
        ):
            rejected = await client.post("/api/actions/resume", json=invalid)
            assert rejected.status_code == status
            after = await graph.aget_state(config)
            assert after.config == before.config and after.interrupts == before.interrupts
            assert selected_orders == ["1001"]

        for response, pause in ((first_card, first_pause), (second_card, current_pause)):
            frame = next(p for p in _data_payloads(response.text.splitlines()) if p.get("kind") == "select_order")
            assert frame["interrupt_id"] == pause
            assert frame["orders"] == [{"order_id": "1001"}]
        continued = await client.post("/api/actions/resume", json=current)
        assert continued.status_code == 200 and "已选择订单 1001" in continued.text
        assert selected_orders == ["1001", "1001"]
        assert not (await graph.aget_state(config)).interrupts


@pytest.mark.parametrize("new_conversation", [False, True])
async def test_conversation_excludes_overlapping_http_graph_operations(monkeypatch, new_conversation):
    started = asyncio.Event()
    finish = asyncio.Event()
    received = []

    async def get_conversation(cid):
        return SimpleNamespace(summary=None, summary_upto_msg_id=None)

    async def create_conversation(user_id):
        return 7

    async def append_message(cid, role, *, content):
        received.append((cid, content))
        return len(received)

    async def no_summary(cid):
        pass

    class Graph:
        pause = SimpleNamespace(
            id="current-order-selection",
            value={"type": "select_order", "orders": [{"order_id": "1001"}]},
        )

        async def aget_state(self, config):
            return SimpleNamespace(values={"user_id": "u1"}, interrupts=(self.pause,))

        async def astream(self, source, config, **kwargs):
            cid = int(config["configurable"]["thread_id"])
            if cid != 7:
                return
            if isinstance(source, dict):
                started.set()
                await finish.wait()
                yield "updates", {"__interrupt__": [self.pause]}
            else:
                assert source.resume == {self.pause.id: "1001"}
                yield "updates", {"script_reply": {"answer": "在路上"}}

    monkeypatch.setattr(runtime.repository, "get_conversation", get_conversation)
    monkeypatch.setattr(runtime.repository, "create_conversation", create_conversation)
    monkeypatch.setattr(runtime.repository, "append_message", append_message)
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", no_summary)
    monkeypatch.setattr(runtime, "get_graph", Graph)

    async with _client() as client:
        message = {"user_id": "u1", "conversation_id": 7, "message": "查询订单"}
        resume = {"conversation_id": 7, "order_id": "1001", "interrupt_id": Graph.pause.id}
        first_body = {**message, "conversation_id": None} if new_conversation else message
        first = asyncio.create_task(client.post("/api/chat", json=first_body))
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            for path, body in (
                ("/api/chat", {**message, "message": "重复发送"}),
                ("/api/agent", message),
                ("/api/actions/resume", resume),
            ):
                assert (await client.post(path, json=body)).status_code == 409
            other = await client.post("/api/chat", json={**message, "conversation_id": 8})
            assert other.status_code == 200
        finally:
            finish.set()
            result = await first
        assert '"kind": "select_order"' in result.text
        continued = await client.post("/api/actions/resume", json=resume)
        assert continued.status_code == 200 and "在路上" in continued.text
    assert received == [(7, "查询订单"), (8, "查询订单")]


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def _collect_sse(resp) -> list[str]:
    return [ln async for ln in resp.aiter_lines()]


def _data_payloads(lines: list[str]) -> list[dict]:
    out = []
    for ln in lines:
        if ln.startswith("data: "):
            payload = ln[len("data: "):]
            if payload != "[DONE]":
                out.append(json.loads(payload))
    return out


def _fake_stream_turn(events: list[dict]):
    async def fake(user_id, message, conversation_id):
        for ev in events:
            yield ev
    return fake


async def test_chat_tool_path_streams_badge_and_answer(monkeypatch):
    events = [
        {"type": "tool", "name": "query_logistics"},
        {"type": "delta", "text": "您的订单 "},
        {"type": "delta", "text": "1001 "},
        {"type": "delta", "text": "正在运输中。"},
        {"type": "done", "conversation_id": 7},
    ]
    monkeypatch.setattr(runtime, "stream_turn", _fake_stream_turn(events))

    async with _client() as client:
        async with client.stream(
            "POST", "/api/chat",
            json={"user_id": "u1", "message": "订单 1001 到哪了"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            lines = await _collect_sse(resp)

    payloads = _data_payloads(lines)
    tools = [p for p in payloads if p.get("event") == "tool"]
    assert tools and tools[0]["name"] == "query_logistics"     # 工具轨迹徽章帧
    deltas = "".join(p["delta"] for p in payloads if "delta" in p)
    assert deltas == "您的订单 1001 正在运输中。"                 # 逐 token 流式最终答
    done = [p for p in payloads if p.get("event") == "done"]
    assert done and done[0]["conversation_id"] == 7
    assert "data: [DONE]" in lines


async def test_chat_no_tool_returns_answer(monkeypatch):
    events = [
        {"type": "delta", "text": "你好,喵~"},
        {"type": "done", "conversation_id": 3},
    ]
    monkeypatch.setattr(runtime, "stream_turn", _fake_stream_turn(events))

    async with _client() as client:
        async with client.stream(
            "POST", "/api/chat", json={"user_id": "u1", "message": "你好"},
        ) as resp:
            lines = await _collect_sse(resp)

    payloads = _data_payloads(lines)
    assert [p for p in payloads if p.get("event") == "tool"] == []
    assert "".join(p["delta"] for p in payloads if "delta" in p) == "你好,喵~"
    cid = next(p["conversation_id"] for p in payloads if p.get("event") == "done")
    assert cid == 3


async def test_chat_stream_includes_citations(monkeypatch):
    events = [
        {"type": "delta", "text": "根据知识库,退货需 7 天内。"},
        {"type": "citations", "items": [{"chunk_id": 1, "title": "退货政策"}]},
        {"type": "done", "conversation_id": 4},
    ]
    monkeypatch.setattr(runtime, "stream_turn", _fake_stream_turn(events))

    async with _client() as client:
        async with client.stream(
            "POST", "/api/chat", json={"user_id": "u1", "message": "怎么退货"},
        ) as resp:
            lines = await _collect_sse(resp)

    payloads = _data_payloads(lines)
    citations = [p for p in payloads if p.get("event") == "citations"]
    assert citations and citations[0]["items"] == [{"chunk_id": 1, "title": "退货政策"}]


async def test_chat_stream_includes_actions(monkeypatch):
    async def fake_stream(user_id, message, conversation_id):
        yield {"type": "delta", "text": "抱歉"}
        yield {"type": "actions", "items": [{"type": "transfer_human"},
                                            {"type": "create_ticket", "draft": {}}]}
        yield {"type": "done", "conversation_id": 5}

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)

    async with _client() as client:
        async with client.stream(
            "POST", "/api/chat", json={"user_id": "u1", "message": "我要投诉"},
        ) as resp:
            lines = await _collect_sse(resp)

    body = "\n".join(lines)
    assert '"event": "actions"' in body
    assert '"delta": "抱歉"' in body
    assert "[DONE]" in body

    payloads = _data_payloads(lines)
    actions = next(p for p in payloads if p.get("event") == "actions")
    assert actions["items"] == [{"type": "transfer_human"},
                                 {"type": "create_ticket", "draft": {}}]
    done = next(p for p in payloads if p.get("event") == "done")
    assert done["conversation_id"] == 5


async def test_chat_unknown_conversation_emits_error_frame(monkeypatch):
    async def fake_stream(user_id, message, conversation_id):
        raise runtime.ConversationNotFound(conversation_id)
        yield  # pragma: no cover - 让函数保持生成器形态

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)

    async with _client() as client:
        async with client.stream(
            "POST", "/api/chat",
            json={"user_id": "u1", "message": "在吗", "conversation_id": 999999},
        ) as resp:
            assert resp.status_code == 200
            lines = await _collect_sse(resp)

    assert "event: error" in lines
    idx = lines.index("event: error")
    payload = next(json.loads(l[len("data: "):]) for l in lines[idx + 1:] if l.startswith("data: "))
    assert payload["message"] == "会话不存在"
    assert "data: [DONE]" not in lines             # 错误流不发终止帧


async def test_chat_db_error_emits_error_frame(monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError

    async def fake_stream(user_id, message, conversation_id):
        raise SQLAlchemyError("boom")
        yield  # pragma: no cover

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)

    async with _client() as client:
        async with client.stream(
            "POST", "/api/chat", json={"user_id": "u1", "message": "在吗"},
        ) as resp:
            assert resp.status_code == 200
            lines = await _collect_sse(resp)

    assert "event: error" in lines
    idx = lines.index("event: error")
    payload = next(json.loads(l[len("data: "):]) for l in lines[idx + 1:] if l.startswith("data: "))
    assert payload["message"] == "数据库暂时不可用,请稍后重试"
    assert "data: [DONE]" not in lines


async def test_chat_generic_error_emits_error_frame(monkeypatch):
    async def fake_stream(user_id, message, conversation_id):
        raise RuntimeError("上游挂了")
        yield  # pragma: no cover

    monkeypatch.setattr(runtime, "stream_turn", fake_stream)

    async with _client() as client:
        async with client.stream(
            "POST", "/api/chat", json={"user_id": "u1", "message": "在吗"},
        ) as resp:
            assert resp.status_code == 200
            lines = await _collect_sse(resp)

    assert "event: error" in lines
    idx = lines.index("event: error")
    payload = next(json.loads(l[len("data: "):]) for l in lines[idx + 1:] if l.startswith("data: "))
    assert payload["message"] == "上游模型暂时不可用,请稍后重试"
    assert "data: [DONE]" not in lines


async def test_chat_validates_empty_message():
    async with _client() as client:
        resp = await client.post("/api/chat", json={"user_id": "u1", "message": ""})
    assert resp.status_code == 422


@pytest.mark.parametrize(("confirm", "legacy_card"), [(False, False), (True, False), (True, True)])
async def test_pending_ticket_blocks_new_turns_until_current_card_is_answered(monkeypatch, confirm, legacy_card):
    cfg = {"configurable": {"thread_id": "7"}}
    graph = build_graph(checkpointer=InMemorySaver())
    writes = {}
    model_calls = 0
    message_ids = count(1)

    async def append(cid, role, **kwargs):
        return next(message_ids)

    async def create_ticket(cid, desc, ticket_type, *, submission_key):
        writes[submission_key] = "T-SYNTHETIC-1"
        return writes[submission_key]

    async def get_result(cid, key):
        return writes.get(key)

    async def coref(query, history):
        return query

    async def classify(query, history):
        return {"intent": "订单", "confidence": 0.95}

    async def specs():
        return [nodes.registry.get_builtin_spec("create_ticket")]

    class Model:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, config):
            nonlocal model_calls
            model_calls += 1
            if model_calls == 1:
                return AIMessage(content="", tool_calls=[{"name": "create_ticket", "id": "ticket", "args": {
                    "ticket_type": "咨询", "description": "核对订单状态",
                }}])
            assert any(isinstance(message, ToolMessage) and message.tool_call_id == "ticket"
                       for message in messages)
            return AIMessage(content="会话已继续")

    monkeypatch.setattr(runtime, "get_graph", lambda: graph)
    monkeypatch.setattr(runtime.repository, "get_conversation", AsyncMock(
        return_value=SimpleNamespace(user_id="u1", summary=None, summary_upto_msg_id=None),
    ))
    monkeypatch.setattr(runtime.repository, "append_message", append)
    monkeypatch.setattr(runtime.repository, "create_ticket", create_ticket)
    monkeypatch.setattr(runtime.repository, "get_ticket_result", get_result)
    monkeypatch.setattr(runtime.repository, "insert_tool_audit", AsyncMock())
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", AsyncMock())
    monkeypatch.setattr(nodes.coref, "resolve", coref)
    monkeypatch.setattr(nodes.intent_mod, "classify", classify)
    monkeypatch.setattr(nodes.registry, "get_all_specs", specs)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kwargs: Model())
    monkeypatch.setitem(app.dependency_overrides, identity, lambda: ("u1", "customer"))

    def frames(response):
        return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]

    if legacy_card:
        def legacy_interrupt(payload):
            preview = {key: value for key, value in payload["preview"].items() if key != "confirmation_id"}
            return interrupt({**payload, "preview": preview})

        monkeypatch.setattr(nodes, "interrupt", legacy_interrupt)

    body = {"user_id": "u1", "conversation_id": 7, "message": "核对订单"}
    async with _client() as tab, _client() as other_tab:
        first = await tab.post("/api/chat", json=body)
        pause = frames(first)[-1]
        assert pause["event"] == "interrupt"
        checkpoint = await graph.aget_state(cfg)
        draft = {**body, "message": "另一条消息"}
        rejected = await other_tab.post("/api/chat", json=draft)
        error = frames(rejected)[0]
        assert error["code"] == "ticket_confirmation_pending"
        preview = error["preview"]
        assert preview["confirmation_id"]
        assert all(preview[key] == value for key, value in pause["preview"].items())
        assert "data: [DONE]" not in rejected.text

        rejected = await other_tab.post("/api/agent", json=draft)
        assert rejected.status_code == 409 and rejected.json()["detail"]["preview"] == preview
        assert (await graph.aget_state(cfg)).config == checkpoint.config
        assert next(message_ids) == 2
        assert model_calls == 1
        assert not writes

        decision = {"conversation_id": 7, "confirmed": confirm, "confirmation_id": preview["confirmation_id"]}
        resolved = await tab.post("/api/actions/resume", json=decision)
        assert resolved.is_success and frames(resolved)[-1]["event"] == "done"

        stale_cancel = await other_tab.post("/api/actions/resume", json={**decision, "confirmed": False})
        assert stale_cancel.status_code == 409
        if confirm:
            replay = await other_tab.post("/api/actions/resume", json=decision)
            assert replay.is_success and "T-SYNTHETIC-1" in replay.text
        assert len(writes) == int(confirm)

        next_turn = await other_tab.post("/api/chat", json=draft)
        assert next_turn.is_success and frames(next_turn)[-1]["final_text"] == "会话已继续"
