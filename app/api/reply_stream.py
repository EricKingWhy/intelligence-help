"""Encode conversation graph events for both streaming HTTP entry points."""

import json
from collections.abc import AsyncIterator


def _data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def error_frame(message: str, *, code: str | None = None, preview: dict | None = None) -> str:
    payload: dict = {"message": message}
    if code is not None:
        payload["code"] = code
    if preview is not None:
        payload["preview"] = preview
    return "event: error\n" + _data(payload)


async def stream_frames(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    finished = False
    async for event in events:
        if finished:
            raise ValueError("Graph emitted an event after its terminal outcome")
        kind = event["type"]
        if kind == "delta":
            payload = {"delta": event["text"]}
            if event.get("replace"):
                payload["replace"] = True
        else:
            payload = {"event": kind, **{key: value for key, value in event.items() if key != "type"}}
        finished = kind in {"done", "interrupt"}
        yield _data(payload)
    if finished:
        yield "data: [DONE]\n\n"
    else:
        yield error_frame("回复接收中断，请先回看会话确认处理结果", code="incomplete_reply")
