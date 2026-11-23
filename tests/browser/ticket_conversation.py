"""Run with Playwright installed: python tests/browser/ticket_conversation.py."""

import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright, expect

URL = "http://localhost:19353"
PAGE = Path(__file__).resolve().parents[2] / "app/static/index.html"


async def open_form(page, action):
    await page.get_by_role("button", name="建工单" if action == "ticket" else "提交退款工单",
                           exact=True).click()
    if action == "ticket":
        await page.locator("#ticketType").select_option("咨询")
        await page.locator("#ticketDesc").fill("测试描述")
    else:
        await page.locator("#refundReason").select_option(label="质量问题")


async def check(browser, action, succeeds, switch):
    page = await browser.new_page()
    submitted = asyncio.Event()
    release = asyncio.Event()
    requests = []
    button = page.locator(f"#{action}Submit")
    error = page.locator(f"#{action}Err")
    mask = page.locator(f"#{action}Mask")
    endpoint = f"/api/actions/create-{action}"
    submission = {"type": "create_ticket"} if action == "ticket" else {
        "type": "refund_form", "draft": {"order_id": "1001"},
    }

    async def respond(route):
        path = route.request.url.split("?", 1)[0].removeprefix(URL)
        if path == "/":
            await route.fulfill(body=PAGE.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")
        elif path == "/api/conversations":
            await route.fulfill(json={"items": [{"id": 1, "preview": "会话 A"}, {"id": 2, "preview": "会话 B"}]})
        elif path.endswith("/messages"):
            await route.fulfill(json={"items": [{"role": "user", "content": "当前会话历史"}]})
        elif path == "/api/chat":
            cid = route.request.post_data_json["conversation_id"]
            frames = [{"delta": "测试回复"}, {"event": "actions", "items": [submission]},
                      {"event": "done", "conversation_id": cid}]
            body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
            await route.fulfill(body=body, content_type="text/event-stream")
        elif path == endpoint:
            requests.append((route.request.post_data_json["conversation_id"],
                             route.request.headers["idempotency-key"]))
            submitted.set()
            await release.wait()
            success = succeeds or len(requests) > 1
            await route.fulfill(json={"ticket_no": "QA-17"} if success else {"detail": "test failure"},
                                status=200 if success else 502)
        else:
            await route.fulfill(json={"items": []})

    async def ask(label):
        await page.locator(".conv-item").filter(has_text=label).click()
        await expect(page.get_by_text("当前会话历史")).to_be_visible()
        await page.locator("#input").fill("测试请求")
        await page.get_by_role("button", name="发送").click()
        await open_form(page, action)

    try:
        await page.route(f"{URL}/**", respond)
        await page.goto(URL)
        await ask("会话 A")
        await button.click()
        await asyncio.wait_for(submitted.wait(), timeout=5)
        if switch:
            await page.keyboard.press("Escape")
            await ask("会话 B")
        async with page.expect_response(lambda response: response.url.endswith(endpoint)):
            release.set()
        await page.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        assert requests[0][0] == 1
        if switch:
            await expect(mask).to_be_visible()
            await expect(error).to_be_empty()
            await expect(page.get_by_text("QA-17", exact=False)).to_have_count(0)
        else:
            if not succeeds:
                await expect(error).to_contain_text("失败")
                await button.click()
            await expect(page.get_by_text("QA-17", exact=False)).to_be_visible()
            if not succeeds:
                assert requests == [requests[0], requests[0]]
    finally:
        release.set()
        await page.close()


async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel=os.getenv("PLAYWRIGHT_CHANNEL"))
        try:
            for action in ("ticket", "refund"):
                for succeeds, switch in ((True, True), (False, True), (False, False)):
                    await check(browser, action, succeeds, switch)
                    print(f"Passed {action}, first response succeeds={succeeds}, switch={switch}")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
