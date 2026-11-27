import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from scripts import eval_rag_tradeoffs


async def test_report_distinguishes_structured_self_check_failure_from_refusal(monkeypatch, tmp_path):
    async def understand(query):
        return {"standard": query, "expanded": []}

    async def search(*args, **kwargs):
        return [{"question": "保修", "answer": "保修一年"}]

    async def check(query, texts):
        result = {"useful": False, "reason": "自评供应商的提示文案已更换"}
        if query == "服务故障":
            result["service_failure"] = {"stage": "evidence_check", "kind": "timeout"}
        return result

    samples = [
        {"id": "fault", "bucket": "no_evidence", "query": "服务故障", "should_refuse": True},
        {"id": "weak", "bucket": "no_evidence", "query": "证据不足", "should_refuse": True},
    ]
    monkeypatch.setattr(eval_rag_tradeoffs, "_load", lambda: samples)
    monkeypatch.setattr(eval_rag_tradeoffs, "FOLLOWUPS", [])
    monkeypatch.setattr(eval_rag_tradeoffs.query_understanding, "understand", understand)
    monkeypatch.setattr(eval_rag_tradeoffs.retrieval, "search_knowledge", search)
    monkeypatch.setattr(eval_rag_tradeoffs.selfcheck, "check_sufficient", check)
    monkeypatch.setattr(eval_rag_tradeoffs.llm, "get_chat_model", lambda **kwargs: None)
    monkeypatch.setattr(eval_rag_tradeoffs.llm, "structured", lambda schema: None)

    args = SimpleNamespace(output=tmp_path / "report.json", kb_version="fixture",
                           concurrency=1, timeout=10, limit_per_bucket=None)
    assert await eval_rag_tradeoffs.run(args) == 1
    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["complete"] is False and report["recommended_default"] is None
    for row in report["samples"]:
        if row["id"] == "fault":
            assert row["service_failure"] == {"stage": "evidence_check", "kind": "timeout"}
            assert row["error"] == "EvidenceCheckFailure"
            assert "refused" not in row
        else:
            assert row["refused"] is True and "error" not in row
    for totals in report["totals"].values():
        assert (totals["n"], totals["failed"], totals["timeouts"]) == (2, 1, 1)
        assert (totals["refusal_n"], totals["refusal_rate"]) == (1, 1)
    for buckets in report["buckets"].values():
        assert buckets["no_evidence"]["timeouts"] == 1


@pytest.mark.parametrize("stage", ["preprocessing", "strategy_evaluation"])
async def test_report_counts_client_timeouts_in_totals_and_buckets(stage, monkeypatch, tmp_path):
    scenarios = {
        "deadline": TimeoutError,
        "read": httpx.ReadTimeout,
        "connect": httpx.ConnectError,
    }
    samples = [
        {"id": case, "bucket": "injected", "query": case, "should_refuse": True, "expect_points": []}
        for case in scenarios
    ] + [{"id": "judge", "bucket": "scoring", "query": "judge",
          "should_refuse": True, "expect_points": []}]

    async def understand(query):
        if stage == "preprocessing" and query in scenarios:
            raise scenarios[query]("fixture error")
        return {"standard": query, "expanded": []}

    async def search(query, **kwargs):
        if stage == "strategy_evaluation" and query in scenarios:
            raise scenarios[query]("fixture error")
        return [{"question": "政策是什么", "answer": "支持退货"}]

    async def check(query, texts):
        return {"useful": True}

    async def answer(prompt):
        return AIMessage(content="支持退货")

    async def judge(prompt):
        raise httpx.ReadTimeout("fixture judge error")

    monkeypatch.setattr(eval_rag_tradeoffs, "_load", lambda: samples)
    monkeypatch.setattr(eval_rag_tradeoffs, "FOLLOWUPS", [])
    monkeypatch.setattr(eval_rag_tradeoffs.query_understanding, "understand", understand)
    monkeypatch.setattr(eval_rag_tradeoffs.retrieval, "search_knowledge", search)
    monkeypatch.setattr(eval_rag_tradeoffs.selfcheck, "check_sufficient", check)
    monkeypatch.setattr(eval_rag_tradeoffs.llm, "get_chat_model", lambda **kwargs: RunnableLambda(answer))
    monkeypatch.setattr(eval_rag_tradeoffs.llm, "structured", lambda schema: RunnableLambda(judge))

    args = SimpleNamespace(output=tmp_path / "report.json", kb_version="fixture",
                           concurrency=1, timeout=10, limit_per_bucket=None)
    assert await eval_rag_tradeoffs.run(args) == 1
    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["complete"] is False and report["recommended_default"] is None
    assert set(report["totals"]) == set(eval_rag_tradeoffs.STRATEGIES)
    for strategy, totals in report["totals"].items():
        assert (totals["n"], totals["failed"], totals["timeouts"], totals["scoring_failed"]) == (4, 3, 2, 1)
        injected = report["buckets"][strategy]["injected"]
        assert (injected["n"], injected["failed"], injected["timeouts"]) == (3, 3, 2)
        scoring = report["buckets"][strategy]["scoring"]
        assert (scoring["n"], scoring["failed"], scoring["timeouts"], scoring["scoring_failed"]) == (1, 0, 0, 1)
        assert scoring["answer_calls"] == 1

    for row in report["samples"]:
        case = row["id"]
        if case == "judge":
            assert row["scoring_error"] == "ReadTimeout"
            assert "error" not in row
        else:
            assert row["error"] == scenarios[case].__name__
            assert row["service_failure"] == {
                "stage": stage, "kind": "unavailable" if case == "connect" else "timeout",
            }
