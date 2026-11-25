import json
from types import SimpleNamespace

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
