"""Run with Playwright installed: python tests/browser/history_ticket.py."""

import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright, expect

URL = "http://localhost:19354"
PAGE = Path(__file__).resolve().parents[2] / "app/static/index.html"
LEGACY_TEXT = "非常抱歉给您带来了不好的体验,我理解您的心情。您可以选择转接人工客服,或让我为您登记一张工单跟进处理。"


async def check(browser, wording):
    page = await browser.new_page()
    await page.add_init_script("localStorage.setItem('mewhelp_conversation_id', '7')")
    reply = {"id": 9, "role": "assistant", "content": wording, "can_create_ticket": True}

    async def respond(route):
        path = route.request.url.removeprefix(URL)
        if path == "/":
            await route.fulfill(body=PAGE.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")
        elif path == "/api/chat":
            frames = [
                {"delta": wording}, {"event": "actions", "items": [{"type": "create_ticket"}]},
                {"event": "done", "conversation_id": 7, "message_id": 9},
            ]
            await route.fulfill(content_type="text/event-stream",
                                body="".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
                                + "data: [DONE]\n\n")
        elif path == "/api/conversations/7/messages":
            await route.fulfill(json={"items": [
                reply, {"id": 10, "role": "assistant", "content": LEGACY_TEXT, "can_create_ticket": False},
            ], "next_before_id": 9})
        elif path == "/api/conversations/7/messages?before_id=9":
            await route.fulfill(json={"items": [
                {"id": 5, "role": "assistant", "content": "较早的投诉答复", "can_create_ticket": True},
                {"id": 6, "role": "assistant", "content": LEGACY_TEXT},
            ]})
        else:
            await route.fulfill(json={"items": []})

    try:
        await page.route(f"{URL}/**", respond)
        await page.goto(URL)
        await expect(page.get_by_text(wording)).to_be_visible()
        # Start a live reply in the same conversation after the initial history load.
        await page.locator("#input").fill("我要投诉")
        await page.get_by_role("button", name="发送").click()
        await expect(page.locator(".row.bot").last.get_by_role("button", name="建工单")).to_be_visible()

        await page.reload()
        await expect(page.locator(".row.bot").first.get_by_role("button", name="建工单")).to_be_visible()
        await expect(page.locator(".row.bot").last.get_by_role("button", name="建工单")).to_have_count(0)
        await page.get_by_role("button", name="加载更早的消息").click()
        await expect(page.locator(".row.bot")).to_have_count(4)
        for index in (0, 1, 2):
            await expect(page.locator(".row.bot").nth(index).get_by_role("button", name="建工单")).to_be_visible()
        await page.locator(".row.bot").first.get_by_role("button", name="建工单").click()
        await expect(page.get_by_role("dialog", name="创建工单")).to_be_visible()
    finally:
        await page.close()


async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel=os.getenv("PLAYWRIGHT_CHANNEL"))
        try:
            for wording in ("我会记录这次投诉并跟进。", "请确认是否为这件事登记工单。"):
                await check(browser, wording)
                print(f"Passed live reply, reload, and older page: {wording}")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
