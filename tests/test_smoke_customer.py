import json
import sys

import httpx
import pytest

from scripts import smoke_customer


LOGISTICS = [
    {"event": "tool", "name": "query_order"},
    {"event": "tool", "name": "query_logistics"},
    {"delta": "快递已在成都揽件。"},
]
POLICY = [
    {"event": "citations", "items": [
        {"n": 1, "answer": "自签收之日起 7 天内，商品完好且不影响二次销售的支持无理由退货。生鲜类和定制类商品除外。本条的 7 天自签收日起算，只管无理由退货;价保的 7 天从下单起算。"},
    ]},
    {"delta": "支持七天无理由退货，商品需完好。[1]"},
]
REFUSAL = [
    {"delta": "从现有知识中没有查到确切信息，建议联系人工客服确认。"},
    {"event": "actions", "items": [{"type": "transfer_human"}]},
]


def wire(events, *, final_text=None, marker=True):
    terminal = {"event": "done", "conversation_id": 7}
    if final_text is not None:
        terminal["final_text"] = final_text
    framed = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in [*events, terminal])
    return framed + ("data: [DONE]\n\n" if marker else "")


def run_smoke(monkeypatch, responses):
    bodies = iter(responses)

    def post(request):
        assert request.url.path == "/api/chat"
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=next(bodies))

    real_client = httpx.Client
    monkeypatch.setattr(smoke_customer.httpx, "Client",
                        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(post)))
    monkeypatch.setattr(sys, "argv", ["smoke_customer", "--base-url", "http://example.test", "--timeout", "2"])
    smoke_customer.main()


def test_valid_demo_replies_pass_with_final_text_and_legacy_replacement(monkeypatch, capsys):
    responses = [
        wire([*LOGISTICS[:2], {"delta": "处理中"}, {"delta": "快递已揽收。", "replace": True}]),
        wire([POLICY[0], {"delta": "旧草稿"}], final_text=POLICY[1]["delta"]),
        wire(REFUSAL),
    ]
    run_smoke(monkeypatch, responses)
    assert capsys.readouterr().out.count("通过") == 3


@pytest.mark.parametrize(("scenario", "slot", "body"), [
    ("物流", 0, wire(LOGISTICS, marker=False)),
    ("物流", 0, wire(LOGISTICS[:2])),
    ("物流", 0, wire(LOGISTICS, final_text="还没确认快递是否已揽件。")),
    ("物流", 0, wire(LOGISTICS, final_text="快递已签收。")),
    ("物流", 0, wire(LOGISTICS, final_text="快递已在成都揽件。当前状态已签收。")),
    ("物流", 0, "event: error\ndata: {\"message\": \"上游超时\"}\n\n" + wire(LOGISTICS)),
    ("政策", 1, wire(POLICY, final_text="只支持换货。")),
    ("政策", 1, wire(POLICY, final_text="生鲜商品也支持七天无理由退货，签收后七天内可退。[1]")),
    ("政策", 1, wire(POLICY, final_text="下单后七天内支持无理由退货，商品需完好。[1]")),
    ("政策", 1, wire(POLICY, final_text="自发货之日起七天内，商品需完好，可无理由退货。[1]")),
    ("政策", 1, wire(POLICY, final_text="签收满七天后才能申请无理由退货，商品需完好。[1]")),
    ("政策", 1, wire(POLICY, final_text="商品拆封、不完好也支持七天无理由退货。[1]")),
    ("政策", 1, wire(POLICY, final_text="签收后七天内支持无理由退货，商品需完好；生鲜和定制商品同样适用。[1]")),
    ("政策", 1, wire([{"event": "citations", "items": ["错误引用对象"]}, POLICY[1]])),
    ("政策", 1, wire(POLICY, final_text="商品需完好，但不支持七天无理由退货。[1]")),
    ("政策", 1, wire([POLICY[0], {"delta": "支持七天无理由退货，商品需完好。[2]"}])),
    ("政策", 1, wire([{"event": "citations", "items": [{"n": 1, "answer": "满九十九元包邮"}]}, POLICY[1]])),
    ("库外拒答", 2, wire([
        {"delta": "这次查询等待超时，您可以稍后重试或转人工。", "replace": True},
        {"event": "actions", "items": [{"type": "retry_turn"}, {"type": "transfer_human"}]},
    ])),
])
def test_unqualified_demo_exits_nonzero(monkeypatch, scenario, slot, body):
    responses = [wire(LOGISTICS), wire(POLICY), wire(REFUSAL)]
    responses[slot] = body
    with pytest.raises(SystemExit, match=scenario) as failed:
        run_smoke(monkeypatch, responses)
    assert failed.value.code != 0
