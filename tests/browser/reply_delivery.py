"""Run with Playwright installed: python tests/browser/reply_delivery.py."""

import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright, expect

URL = "http://localhost:19355"
PAGE = Path(__file__).resolve().parents[2] / "app/static/index.html"
DELTA = {"delta": "这段文字尚未确认"}
DONE = {"event": "done", "conversation_id": 7, "message_id": 9}
PAUSE = {"event": "interrupt", "kind": "confirm_ticket", "conversation_id": 7,
         "preview": {"description": "测试工单", "confirmation_id": "a" * 64}}


def wire(*frames):
    return "".join(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n" for frame in frames)


async def check(browser, resume, body, expected):
    page = await browser.new_page()
    requests = []

    async def respond(route):
        path = route.request.url.removeprefix(URL)
        if path == "/":
            await route.fulfill(body=PAGE.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")
        elif path in {"/api/chat", "/api/actions/resume"}:
            requests.append(path)
            payload = wire(PAUSE) + "data: [DONE]\n\n" if resume and path == "/api/chat" else body
            await route.fulfill(body=payload, content_type="text/event-stream; charset=utf-8")
        else:
            await route.fulfill(json={"items": []})

    try:
        await page.route(f"{URL}/**", respond)
        await page.goto(URL)
        await page.locator("#input").fill("测试")
        send = page.get_by_role("button", name="发送")
        await send.click()
        if resume:
            await page.get_by_role("button", name="确认提交").click()
        await expect(send).to_be_enabled()
        bubble = page.locator(".row.bot").last
        await expect(bubble).to_contain_text(expected)
        if "中断" in expected:
            await expect(bubble.get_by_role("button", name="这条回答没帮助")).to_have_count(0)
        assert requests == (["/api/chat", "/api/actions/resume"] if resume else ["/api/chat"])
    finally:
        await page.close()


async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel=os.getenv("PLAYWRIGHT_CHANNEL"))
        try:
            for resume in (False, True):
                for body, expected in [
                    (wire(DELTA), "回复接收中断"),
                    (wire(DELTA, DONE), "回复接收中断"),
                    (wire(DELTA) + "data: [DONE]\n\n", "回复接收中断"),
                    (wire(DELTA) + 'data: {"event":', "回复接收中断"),
                    (wire(PAUSE) + "data: [DONE]\n\n", "请确认工单信息"),
                    (wire(DELTA, DONE) + "data: [DONE]\n\n", DELTA["delta"]),
                    ('event: error\ndata: {"message":"暂时不可用"}\n\n', "失败"),
                    ('event: error\ndata: {"code":"incomplete_reply"}\n\n', "回复接收中断"),
                ]:
                    await check(browser, resume, body, expected)
                print(f"Passed 8 stream cases (resume={resume})")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
