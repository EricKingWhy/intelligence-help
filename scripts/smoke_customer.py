"""Check the three customer scenarios against a running demo."""
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Allow the documented module command and direct execution from a checkout.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.business import order_snapshot
from mcp_servers.logistics_server import logistics_snapshot


_LOGISTICS_STATUS = {
    "PICKED_UP": r"已(?:(?:在|被).{0,8})?揽[件收]|已取件",
    "IN_TRANSIT": r"运输中|正在运输|在途",
    "DELIVERING": r"派送中|正在派送|正在配送",
    "DELIVERED": r"已签收|已送达",
}

_SEVEN_DAYS = r"(?:(?<!\d)7\s*[天日]|(?<![十百零])七[天日])"
_RETURN_DEADLINE = rf"(?:自?签收(?:之?日)?[起后]\s*)?{_SEVEN_DAYS}(?:之?内)?"
_NO_REASON_RETURN = r"无理由(?:申请)?退货"
_RETURN_OFFER = re.compile(
    rf"(?<![不未])(?:支持|提供|允许|适用|可(?:以)?|享受)\s*(?:在\s*)?"
    rf"(?:{_RETURN_DEADLINE}\s*)?(?:申请|办理|享受|发起)?\s*"
    rf"(?:{_RETURN_DEADLINE}\s*)?{_NO_REASON_RETURN}"
    rf"|{_NO_REASON_RETURN}(?:政策|规则|服务|条款)?\s*"
    r"(?:(?:仍然?|照常|继续)?(?:有效|生效)|(?:仍然?|照常|继续)(?:适用|提供|支持))"
)
_RETURN_DENIAL = (
    r"(?:自[^，,]{1,8}起)?(?:目前|现在|现已|已经|已|正式|全面|被|并){0,4}"
    r"(?:不再?(?:支持|提供|允许|适用|能|可|享受|执行|有效|生效|受理)|"
    r"(?:取消|撤销|废止|废除|作废|失效|停用|停止|暂停|终止)(?:提供|执行)?)"
)
_REVOKED_RETURN = re.compile(
    rf"(?<!未)(?<!没有){_RETURN_DENIAL}.{{0,16}}{_NO_REASON_RETURN}"
    rf"|{_NO_REASON_RETURN}(?:政策|规则|服务|条款)?\s*{_RETURN_DENIAL}"
)
_EXCLUDED_PRODUCT = r"(?:生鲜|定制)(?:类)?(?:商品)?"
_EXCLUDED_SUBJECT = re.compile(rf"{_EXCLUDED_PRODUCT}(?:[和与、及]{_EXCLUDED_PRODUCT})*")
_CITATION_GROUP = re.compile(r"(\[\d+\](?:[ \t,，、]*\[\d+\])*)")


@dataclass
class Reply:
    answer: str = ""
    tools: set[str] = field(default_factory=set)
    citations: list[dict] = field(default_factory=list)
    actions: set[str] = field(default_factory=set)
    replaced: bool = False


def _frame_items(event: dict) -> list[dict]:
    items = event.get("items")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise RuntimeError(f"SSE {event['event']} 项目格式无效")
    return items


def ask(client: httpx.Client, message: str) -> Reply:
    response = client.post("/api/chat", json={"user_id": "u1", "message": message})
    response.raise_for_status()
    if response.headers.get("content-type", "").split(";")[0] != "text/event-stream":
        raise RuntimeError("响应不是 SSE 流")
    frames = response.text.replace("\r\n", "\n").split("\n\n")
    if frames.pop().strip():
        raise RuntimeError("SSE 尾部帧未闭合")

    reply = Reply()
    done = terminated = False
    for frame in frames:
        lines = frame.splitlines()
        if any(line.startswith("event:") and line[6:].strip() == "error" for line in lines):
            raise RuntimeError("收到 SSE 错误事件")
        payload = "\n".join(line[5:].removeprefix(" ") for line in lines if line.startswith("data:"))
        if not payload:
            continue
        if payload == "[DONE]" and done and not terminated:
            terminated = True
            continue
        if done or payload == "[DONE]":
            raise RuntimeError("SSE 结束帧顺序错误")
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("SSE 数据不是有效 JSON") from exc
        if not isinstance(event, dict):
            raise RuntimeError("SSE 数据不是对象")
        kind = event.get("event")
        if kind == "interrupt":
            raise RuntimeError("演示请求在等待用户操作")
        if kind == "done":
            cid = event.get("conversation_id")
            if type(cid) is not int or cid <= 0:
                raise RuntimeError("完成帧缺少有效会话标识")
            done = True
            if "final_text" in event:
                if not isinstance(event["final_text"], str):
                    raise RuntimeError("完成帧的最终正文格式无效")
                reply.answer = event["final_text"]
        elif kind == "tool" and isinstance(event.get("name"), str):
            reply.tools.add(event["name"])
        elif kind == "citations":
            reply.citations = _frame_items(event)
        elif kind == "actions":
            reply.actions.update(item["type"] for item in _frame_items(event))
        elif isinstance(event.get("delta"), str):
            if event.get("replace"):
                reply.answer = ""
                reply.citations.clear()
                reply.replaced = True
            reply.answer += event["delta"]
    if not terminated:
        raise RuntimeError("回复缺少完整的 done 和 [DONE]")
    return reply


def _logistics_ok(reply: Reply) -> bool:
    expected = logistics_snapshot(order_snapshot("1001")["tracking_no"])["status_code"]
    if not {"query_order", "query_logistics"} <= reply.tools or "retry_turn" in reply.actions:
        return False
    reported = set()
    for clause in re.split(r"[，,。；;！？\n]", reply.answer):
        if re.search(r"无法|查不到|查无|超时|暂不可用|是否|不确定|还没|未|不是|不在|没有|预计|历史", clause):
            continue
        reported.update(code for code, pattern in _LOGISTICS_STATUS.items() if re.search(pattern, clause))
    return reported == {expected}


def _states_return_rule(text: str, *, require_condition: bool = True) -> bool:
    text = re.sub(r"[*`]", "", text)
    stated_rule = has_condition = False
    ineligible = r"(?:不完好|(?<![非不])(?:生鲜|定制|影响二次销售))"
    admits = r"(?<!不)(?:支持|适用|可以|可享)"
    for sentence in re.findall(r"[^。；;！？!?\n]+[？?]?", text):
        if sentence.endswith(("？", "?")):
            continue
        if "价保" in sentence and "无理由" not in sentence:
            continue
        # Detect contradictory eligibility even when it is stated in a separate sentence.
        clauses = re.split(r"[，,]", sentence)
        offers_return = seven_day_context = False
        for idx, clause in enumerate(clauses):
            ineligible_product = (
                re.search(ineligible, clause)
                or (idx > 0 and _EXCLUDED_SUBJECT.fullmatch(clauses[idx - 1].strip()))
            )
            denied = _REVOKED_RETURN.search(clause)
            if denied and not (ineligible_product or re.search(rf"(?:超出|超过)\s*{_SEVEN_DAYS}|逾期", clause)):
                return False
            if re.search(r"(?:无需|无须|不必|不用|不需要)(?:仍|再)?(?:保持)?(?:商品)?完好", clause):
                return False
            excluded = denied or re.search(r"除外|不适用|不支持|不享受|除.{0,20}以外", clause)
            if (excluded and ineligible_product) or re.search(r"(?<!不)完好|不影响二次销售", clause):
                has_condition = True
            if (re.search(rf"{ineligible}.{{0,12}}{admits}", clause)
                    and not excluded):
                return False
            if excluded or "价保" in clause:
                continue
            if re.search(_SEVEN_DAYS, clause) and not re.search(rf"(?:超出|超过)\s*{_SEVEN_DAYS}", clause):
                seven_day_context = True
            if _RETURN_OFFER.search(clause):
                offers_return = True

        stated_rule = stated_rule or (seven_day_context and offers_return)
        if not (re.search(_SEVEN_DAYS, sentence) and re.search(_NO_REASON_RETURN, sentence)):
            continue
        if re.search(rf"(?:满|超过){_SEVEN_DAYS}(?:后)?(?:才|仍|也)(?:能|可|支持)"
                     rf"|{_SEVEN_DAYS}后才(?:能|可|支持)", sentence):
            return False
        for clause in clauses:
            if ("价保" not in clause and not re.search(r"不是|不从|不按", clause)
                    and re.search(rf"(?:下单|付款|发货).{{0,10}}(?:起算|[起后].{{0,5}}{_SEVEN_DAYS})", clause)):
                return False
    return stated_rule and (has_condition or not require_condition)


def _policy_ok(reply: Reply) -> bool:
    if not _states_return_rule(reply.answer):
        return False
    supporting = {
        str(item.get("n")) for item in reply.citations if _states_return_rule(item.get("answer") or "")
    }
    # The numbered references support the preceding passage, including inline references.
    passages = _CITATION_GROUP.split(reply.answer)
    return any(
        supporting.intersection(re.findall(r"\[(\d+)\]", references))
        and _states_return_rule(passage, require_condition=False)
        for passage, references in zip(passages[::2], passages[1::2])
    )


def _refusal_ok(reply: Reply) -> bool:
    return (
        not reply.replaced
        and "transfer_human" in reply.actions and "retry_turn" not in reply.actions
        and re.search(r"没有查到|没查到|未查到|未找到|无法.{0,12}(?:确认|核实)|现有知识不足", reply.answer) is not None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    cases = (
        ("物流", "订单 1001 的物流到哪了？", _logistics_ok, "缺少订单、物流调用或最终物流状态"),
        ("政策", "退货政策是什么？", _policy_ok, "缺少七天无理由结论、适用条件或对应编号引用"),
        ("库外拒答", "火星车能参加以旧换新吗？", _refusal_ok, "缺少明确的知识不足说明或人工兜底"),
    )
    failures = []
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        for label, question, accepts, failure in cases:
            try:
                reply = ask(client, question)
                if accepts(reply):
                    print(f"{label}通过", flush=True)
                else:
                    failures.append(f"{label}：{failure}")
            except (httpx.HTTPError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                failures.append(f"{label}：{exc}")

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
