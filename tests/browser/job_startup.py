"""Run the shared job control in Edge with deterministic API responses."""

import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright, expect

URL = "http://localhost:19353"
SCRIPT = Path(__file__).resolve().parents[2] / "app/static/acceptance.js"
TITLE = "材料清单与切块预览"
STARTING = {"name": "kb-preview", "title": TITLE, "cmd": "make kb-preview",
            "status": "starting", "pid": None, "returncode": None}


async def check(browser, outcome):
    page = await browser.new_page()
    poll_count = 0
    allow_running = asyncio.Event()
    allow_finish = asyncio.Event()
    states = [{**STARTING, "log": "新作业已发起"}]
    if outcome == "ok":
        states.append({**STARTING, "status": "running", "pid": 222, "log": "新作业运行中"})
    states.append({
        **STARTING, "status": outcome, "pid": 222 if outcome == "ok" else None,
        "log": "新作业结束", "returncode": 0 if outcome == "ok" else None,
    })
    responses = iter(states)

    async def respond(route):
        path = route.request.url.removeprefix(URL)
        if path == "/":
            await route.fulfill(content_type="text/html; charset=utf-8", body=f"""
                <div id="controls"></div><pre id="log"></pre><output id="finished"></output>
                <script src="/static/acceptance.js"></script>
                <script>
                  JOB_LOGS["kb-preview"] = "上一轮日志";
                  document.getElementById("controls").appendChild(jobButton(
                    {json.dumps(STARTING, ensure_ascii=False)}, document.getElementById("log"),
                    () => document.getElementById("finished").textContent += "done"
                  ));
                </script>
            """)
        elif path == "/static/acceptance.js":
            await route.fulfill(content_type="text/javascript; charset=utf-8",
                                body=SCRIPT.read_text(encoding="utf-8"))
        elif path == "/api/jobs/kb-preview":
            nonlocal poll_count
            poll_count += 1
            state = next(responses)
            if state["status"] == "running":
                await allow_running.wait()
            elif state["status"] == outcome:
                await allow_finish.wait()
            await route.fulfill(json=state)
        else:
            await route.abort()

    try:
        await page.route(f"{URL}/**", respond)
        await page.goto(URL)
        button = page.get_by_role("button")
        await expect(button).to_contain_text("启动中")
        await expect(button).to_be_disabled()
        await expect(page.locator("#log")).not_to_contain_text("上一轮日志")
        await expect(page.locator("#log")).to_contain_text("新作业已发起")
        if outcome == "ok":
            allow_running.set()
            await expect(button).to_contain_text("运行中")
            await expect(button).to_be_disabled()
        await expect(page.locator("#finished")).to_be_empty()
        allow_finish.set()
        await expect(page.locator("#log")).to_contain_text("新作业结束")
        await expect(button).to_be_enabled()
        await expect(page.locator("#finished")).to_have_text("done")
        assert poll_count == len(states)
    finally:
        allow_running.set()
        allow_finish.set()
        await page.close()


async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel=os.getenv("PLAYWRIGHT_CHANNEL"))
        try:
            for outcome in ("ok", "failed"):
                await check(browser, outcome)
                print(f"Passed backend job polling (outcome={outcome})")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
