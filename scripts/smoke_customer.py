"""Check the three customer scenarios against a running demo."""
import argparse
import json
import re
from dataclasses import dataclass, field

import httpx


@dataclass
class Reply:
    answer: str = ""
    tools: set[str] = field(default_factory=set)
    citations: list[dict] = field(default_factory=list)
    actions: set[str] = field(default_factory=set)
    replaced: bool = False


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
            reply.citations = event["items"]
        elif kind == "actions":
            reply.actions.update(item["type"] for item in event["items"])
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
    statuses = r"已揽[件收]|已取件|运输中|正在运输|在途|派送中|正在派送|正在配送|已签收|已送达"
    return (
        {"query_order", "query_logistics"} <= reply.tools
        and "retry_turn" not in reply.actions
        and not re.search(r"无法|查不到|未查到|超时|暂不可用", reply.answer)
        and re.search(statuses, reply.answer) is not None
    )


def _states_return_rule(text: str) -> bool:
    for sentence in re.split(r"[。；！？\n]", text):
        if (re.search(r"(?:7\s*[天日]|七[天日])", sentence) and "无理由" in sentence and "退货" in sentence
                and not re.search(r"(?:不支持|不提供|不能|不可).{0,16}无理由", sentence)):
            return any(term in text for term in ("完好", "二次销售", "签收", "定制", "生鲜"))
    return False


def _policy_ok(reply: Reply) -> bool:
    referred = set(re.findall(r"\[(\d+)\]", reply.answer))
    return _states_return_rule(reply.answer) and any(
        str(item.get("n")) in referred and _states_return_rule(item.get("answer") or "")
        for item in reply.citations
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
