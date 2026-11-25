import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import langfuse
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import interrupt
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.config import settings
from app.core import observability
from app.graph import runtime
from app.graph.state import ConversationState
from app.main import app


QUESTION = "SYNTHETIC_CUSTOMER_QUESTION"
REPLY = "SYNTHETIC_CUSTOMER_REPLY"
TOOL_ARGUMENT = "SYNTHETIC_TOOL_ARGUMENT"
TOOL_RESULT = "SYNTHETIC_TOOL_RESULT"
CUSTOMER = "SYNTHETIC_CUSTOMER_ID"
CID = 921640728


@pytest.mark.parametrize("mask_fails", [False, True])
async def test_chat_agent_and_resume_export_safe_trace_data(monkeypatch, mask_fails):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    real_client = langfuse.Langfuse
    attempted_requests = []

    def reject_request(request):
        attempted_requests.append(request.url.path)
        return httpx.Response(400)

    api_client = httpx.Client(transport=httpx.MockTransport(reject_request))

    def in_memory_client(**kwargs):
        return real_client(**kwargs, tracer_provider=provider, span_exporter=exporter, httpx_client=api_client)

    def broken_mask(*, params):
        raise RuntimeError("synthetic mask failure")

    if mask_fails:
        monkeypatch.setattr(observability, "_mask_otel_spans", broken_mask)
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "true")

    monkeypatch.setattr(langfuse, "Langfuse", in_memory_client)
    monkeypatch.setattr(observability, "_client", None)
    monkeypatch.setattr(settings, "langfuse_public_key", f"pk-test-{uuid4()}")
    monkeypatch.setattr(settings, "langfuse_secret_key", "synthetic-deployment-secret")
    monkeypatch.setattr(settings, "langfuse_base_url", "http://127.0.0.1:1")
    monkeypatch.setattr(runtime.repository, "get_conversation", AsyncMock(return_value=SimpleNamespace(
        user_id=CUSTOMER, summary=None, summary_upto_msg_id=0,
    )))
    monkeypatch.setattr(runtime.repository, "append_message", AsyncMock(return_value=10))
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", AsyncMock())

    @tool
    def synthetic_tool(reference: str, attachment: str) -> str:
        """Return synthetic tool data."""
        assert reference == TOOL_ARGUMENT
        assert attachment.startswith("data:")
        return TOOL_RESULT

    async def scripted_node(state):
        observability.tag_intent("订单", 0.9)
        if state["messages"][-1].content == QUESTION:
            interrupt({"type": "confirm_ticket", "preview": {}})
        await synthetic_tool.ainvoke({
            "reference": TOOL_ARGUMENT, "attachment": "data:text/plain;base64,c3ludGhldGlj",
        })
        model = GenericFakeChatModel(disable_streaming=True, messages=iter([AIMessage(
            REPLY, response_metadata={"usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}},
        )]))
        reply = await model.ainvoke(state["messages"], config={
            "metadata": {"ls_model_name": "synthetic-chat-model"},
        })
        return {"answer": reply.content, "messages": [reply]}

    builder = StateGraph(ConversationState)
    builder.add_node("script_reply", scripted_node)
    builder.add_edge(START, "script_reply")
    graph = observability.attach_observability(builder.compile(checkpointer=InMemorySaver()))
    monkeypatch.setattr(runtime, "get_graph", lambda: graph)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            body = {"user_id": CUSTOMER, "conversation_id": CID, "message": QUESTION}
            chat = await client.post("/api/chat", json=body)
            assert chat.is_success and "confirm_ticket" in chat.text
            resume = await client.post("/api/actions/resume", json={"conversation_id": CID, "confirmed": False})
            assert resume.is_success and REPLY in resume.text
            agent = await client.post("/api/agent", json={**body, "message": QUESTION + "_JSON"})
            assert agent.is_success and agent.json()["answer"] == REPLY

        langfuse_client = observability.get_langfuse()
        with langfuse_client.start_as_current_observation(name="synthetic-failed-observation") as failed_span:
            failed_span.update(level="ERROR", status_message=TOOL_RESULT)
        langfuse_client.flush()
        spans = exporter.get_finished_spans()
        assert not attempted_requests, "trace media must not be uploaded"
        if mask_fails:
            assert not spans
            return
        serialized = "\n".join(span.to_json() for span in spans)
        for sensitive in (QUESTION, REPLY, TOOL_ARGUMENT, TOOL_RESULT, CUSTOMER, str(CID)):
            assert sensitive not in serialized

        roots = [span for span in spans if span.parent is None]
        assert len(roots) == 3
        assert len({span.attributes["session.id"] for span in roots}) == 1
        assert any(span.name == "synthetic_tool" for span in spans)
        tagged_traces = {
            span.context.trace_id for span in spans
            if "intent:订单" in span.attributes.get("langfuse.trace.tags", ())
        }
        generations = [span for span in spans if span.attributes.get("langfuse.observation.type") == "generation"]
        assert len(generations) == 2
        assert all(span.end_time >= span.start_time for span in generations)
        for span in generations:
            assert span.context.trace_id in tagged_traces
            assert span.attributes["langfuse.observation.model.name"] == "synthetic-chat-model"
            assert json.loads(span.attributes["langfuse.observation.usage_details"])["total"] == 16
    finally:
        provider.shutdown()
        api_client.close()
