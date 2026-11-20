"""Check the three customer scenarios against a running demo."""
import argparse
import json

import httpx


def ask(client, message):
    response = client.post("/api/chat", json={"user_id": "u1", "message": message})
    response.raise_for_status()
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: {")]
    if not any(event.get("event") == "done" for event in events):
        raise RuntimeError(f"Chat did not finish for {message!r}")
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    failures = []
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        logistics = ask(client, "订单 1001 的物流到哪了？")
        invoked = {event["name"] for event in logistics if event.get("event") == "tool"}
        if {"query_order", "query_logistics"} <= invoked:
            print("物流通过：已调用订单和物流工具", flush=True)
        else:
            failures.append(f"物流：工具调用为 {sorted(invoked)}")

        policy = ask(client, "退货政策是什么？")
        answer = "".join(event.get("delta", "") for event in policy)
        if any(event.get("event") == "citations" and event["items"] for event in policy) and (
            ("7 天" in answer or "7天" in answer or "七天" in answer) and "无理由" in answer
        ):
            print("政策通过：回答七天无理由并附引用", flush=True)
        else:
            failures.append("政策：缺少七天无理由结论或引用")

        refusal = ask(client, "火星车能参加以旧换新吗？")
        if any(action["type"] == "transfer_human" for event in refusal
               if event.get("event") == "actions" for action in event["items"]):
            print("库外拒答通过：给出人工兜底", flush=True)
        else:
            failures.append("库外问题：未给出人工兜底")

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
