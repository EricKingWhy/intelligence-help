# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""意图分类的两处约束:提示词不许再提「正文 JSON」(机制只认 tool_calls),
且当评委用的链路必须显式 temperature=0。"""
from app.core import intent, llm
from app.core.intent import _Intent
from app.core.prompts import INTENT_CLASSIFY_SYSTEM


def test_提示词不再要求正文_JSON():
    # structured() 走 function_calling,schema 由 bind_tools 的 JSON Schema 承载;
    # 提示词再说「按 JSON 格式返回」会把模型引到正文那条路,tool_calls 为空 → 整轮失败
    assert "JSON 格式" not in INTENT_CLASSIFY_SYSTEM
    assert "不得包含任何其他文本" not in INTENT_CLASSIFY_SYSTEM


def test_schema_仍然是九类加_0_1_confidence():
    props = _Intent.model_json_schema()["properties"]
    assert len(props["intent"]["enum"]) == 9
    assert props["confidence"]["minimum"] == 0.0
    assert props["confidence"]["maximum"] == 1.0


async def test_classify_用_temperature_0(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    seen = {}

    def fake_structured(schema, **kw):
        seen.update(kw)

        async def _run(_):
            return _Intent(intent="物流", confidence=0.9)

        return RunnableLambda(_run)

    monkeypatch.setattr(llm, "structured", fake_structured)
    out = await intent.classify("我的快递到哪了")
    assert seen["temperature"] == 0
    assert out == {"intent": "物流", "confidence": 0.9}
