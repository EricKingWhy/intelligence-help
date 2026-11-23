# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import pytest

from app.core import selfcheck


@pytest.mark.asyncio
async def test_check_sufficient_parses_flat(monkeypatch):
    class Fake:
        useful = False
        reason = "证据只讲运费,未涉及型号"

    async def fake_invoke(_):
        return Fake()

    class Chain:
        def __or__(self, o):
            return self

        async def ainvoke(self, _):
            return Fake()

    monkeypatch.setattr("app.core.selfcheck._chain", lambda: Chain())
    out = await selfcheck.check_sufficient("Pro型号功能", ["满99包邮"])
    assert out == {"useful": False, "reason": "证据只讲运费,未涉及型号"}


def test_证据自评链也传_temperature_0(monkeypatch):
    # 自评是评委链路,和 intent 同级别:默认 0.3 会给同一份证据抖出不同结论
    from langchain_core.runnables import RunnableLambda

    from app.core import llm
    seen = {}

    def fake_structured(schema, **kw):
        seen.update(kw)
        return RunnableLambda(lambda _: None)

    monkeypatch.setattr(llm, "structured", fake_structured)
    selfcheck._chain()
    assert seen["temperature"] == 0
