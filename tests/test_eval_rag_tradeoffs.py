import pytest

from scripts import eval_rag_tradeoffs


async def test_eval_distinguishes_structured_self_check_failure_from_refusal(monkeypatch):
    async def search(*args, **kwargs):
        return [{"question": "保修", "answer": "保修一年"}]

    result = {"useful": False, "reason": "自评供应商的提示文案已更换"}

    async def check(*args, **kwargs):
        return result

    monkeypatch.setattr(eval_rag_tradeoffs.retrieval, "search_knowledge", search)
    monkeypatch.setattr(eval_rag_tradeoffs.selfcheck, "check_sufficient", check)

    args = ("hybrid_rerank", {"should_refuse": True}, ("退货政策", "退货政策", 0), None, None, None)
    assert (await eval_rag_tradeoffs.evaluate_one(*args))["refused"] is True

    result["service_failure"] = {"stage": "evidence_check", "kind": "timeout"}
    with pytest.raises(RuntimeError, match="evidence_check:timeout"):
        await eval_rag_tradeoffs.evaluate_one(*args)
