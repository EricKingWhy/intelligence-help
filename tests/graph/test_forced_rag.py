# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import nodes
from app.graph.build import build_graph


@pytest.mark.asyncio
async def test_forced_rag_strong_builds_evidence(monkeypatch):
    hits = [{"id": 1, "question": "退货政策", "answer": "7天无理由", "rerank_score": 0.9,
             "section_path": "政策/退货", "content_type": "policy"}]
    monkeypatch.setattr(nodes.query_understanding, "understand",
                        _aval({"standard": "退货政策", "expanded": []}))
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", _aval(hits))
    monkeypatch.setattr(nodes.retrieval, "arrange_head_tail", lambda h: h)
    monkeypatch.setattr(nodes.selfcheck, "check_sufficient", _aval({"useful": True, "reason": ""}))

    out = await nodes.retrieve_knowledge({"messages": [HumanMessage("退货政策是什么")]})
    assert out["evidence_strong"] is True
    assert "[1]" in out["evidence"]
    assert out["citations"][0]["n"] == 1
    assert out["trace"]["forced_rag"] is True


@pytest.mark.asyncio
async def test_forced_rag_weak_when_below_threshold(monkeypatch):
    monkeypatch.setattr(nodes.query_understanding, "understand",
                        _aval({"standard": "注销账号", "expanded": []}))
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", _aval([]))
    out = await nodes.retrieve_knowledge({"messages": [HumanMessage("怎么注销账号")]})
    assert out["evidence_strong"] is False
    assert out["trace"]["forced_rag"] is True


async def test_forced_rag_uses_product_from_previous_turn(monkeypatch):
    async def understand(query):
        return {"standard": query, "expanded": []}

    async def resolve(query, history):
        if query == "那它有保修吗？" and "Pro 猫砂盆" in history:
            return "Pro 猫砂盆有保修吗？"
        return query

    async def search(query, **kwargs):
        if "Pro 猫砂盆" not in query:
            return []
        return [{"id": 17, "question": "Pro 猫砂盆保修", "answer": "整机保修一年",
                 "rerank_score": 0.95, "section_path": "Pro / 保修", "content_type": "policy"}]

    monkeypatch.setattr(nodes.coref, "resolve", resolve)
    monkeypatch.setattr(nodes.intent_mod, "classify",
                        _aval({"intent": "商品咨询", "confidence": 0.95}))
    monkeypatch.setattr(nodes.query_understanding, "understand", understand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", search)
    monkeypatch.setattr(nodes.selfcheck, "check_sufficient", _aval({"useful": True, "reason": ""}))
    monkeypatch.setattr(nodes, "main_agent",
                        _aval({"messages": [AIMessage("依据[1]，整机保修一年")]}))
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "customer"}}
    first = await graph.ainvoke({"messages": [HumanMessage("Pro 猫砂盆整机保修多久？")]}, config)
    assert first["citations"][0]["id"] == 17
    second = await graph.ainvoke({"messages": [HumanMessage("那它有保修吗？")], "citations": []}, config)
    assert second["evidence_strong"] is True
    assert second["citations"][0]["id"] == 17
    assert "[1]" in nodes.resolve_answer(second)


@pytest.mark.asyncio
async def test_classify_intent_sets_route_in_state(monkeypatch):
    # route 必须落进 State,否则 _agent_messages 判不出知识路、强制检索证据不会注入
    async def fake_classify(query, history=""):
        return {"intent": "退款退货", "confidence": 0.8}
    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({"messages": [HumanMessage("我要退货")]})
    assert out["intent"] == "退款退货"
    assert out["route"] == "refund_flow"      # ch06:退款退货 → refund_flow 出口
    assert out["trace"]["route"] == "refund_flow"


@pytest.mark.asyncio
async def test_confidence_check_traces_decision():
    assert (await nodes.confidence_check({"evidence_strong": True}))["trace"]["confidence"] == "strong"
    assert (await nodes.confidence_check({"evidence_strong": False}))["trace"]["confidence"] == "weak"


def _aval(value):
    async def _f(*a, **k):
        return value
    return _f
