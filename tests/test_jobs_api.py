# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""作业运行器的安全边界:名字不在白名单里就 404,同名作业没跑完不许重入。

这两条是「页面上有重跑按钮但没有 shell」的实证。另加一条注册表自检:每个作业都得是
仓库 Makefile 里真有的目标,免得页面按钮指向一个不存在的配方。
"""
import asyncio
import pathlib
import re
import threading

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import jobs as jobs_api
from app.core import jobs


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(jobs_api.router)
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


async def test_unknown_job_is_rejected(client):
    """前端只传作业名。名字对不上就到此为止——命令片段根本没有地方可以传进来。"""
    for path, method in (("/api/jobs/rm -rf", "post"), ("/api/jobs/rm -rf", "get"),
                         ("/api/jobs/未注册/stop", "post")):
        resp = await getattr(client, method)(path)
        assert resp.status_code == 404, path
        assert "未注册的作业" in resp.json()["detail"]


async def test_running_job_refuses_reentry(client, monkeypatch):
    """同名作业在跑就拒绝再发起,否则两份进程抢同一批产物。"""
    monkeypatch.setitem(jobs._runs, "kb-build",
                        jobs.JobRun(status="running", pid=4242))
    resp = await client.post("/api/jobs/kb-build")
    assert resp.status_code == 409 and "正在运行中" in resp.json()["detail"]


@pytest.mark.parametrize("first_launch_fails", [False, True])
async def test_job_refuses_reentry_during_startup(client, monkeypatch, tmp_path, first_launch_fails):
    name = "kb-preview"
    monkeypatch.setitem(jobs._runs, name, jobs.JobRun(
        status="ok", pid=111, started_at="2026-01-01T10:00:00",
        finished_at="2026-01-01T10:01:00", returncode=0,
    ))
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    (tmp_path / f"{name}.log").write_text("上一轮日志", encoding="utf-8")
    entered = asyncio.Event()
    release = asyncio.Event()
    launches = []

    class Process:
        pid = 4242

        async def wait(self):
            return 0

    async def launch(*args, stdout, **kwargs):
        launches.append(stdout)
        if len(launches) == 1:
            entered.set()
            await release.wait()
            if first_launch_fails:
                raise FileNotFoundError("make")
        return Process()

    monkeypatch.setattr(jobs.asyncio, "create_subprocess_exec", launch)
    url = f"/api/jobs/{name}"
    first = asyncio.create_task(client.post(url))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        starting = (await asyncio.wait_for(client.get(url), timeout=2)).json()
        listed = (await client.get("/api/jobs")).json()["jobs"]
        second = await asyncio.wait_for(client.post(url), timeout=2)
    finally:
        release.set()
        response = await first
        if jobs._runs[name].watcher is not None:
            await jobs._runs[name].watcher

    assert starting["status"] == "starting"
    assert all(starting[key] is None for key in ("pid", "finished_at", "returncode"))
    assert starting["started_at"] != "2026-01-01T10:00:00"
    assert starting["log"].startswith("$ make kb-preview")
    assert next(item for item in listed if item["name"] == name)["status"] == "starting"
    assert second.status_code == 409
    assert len(launches) == 1 and launches[0].closed
    assert response.status_code == (500 if first_launch_fails else 200)
    if first_launch_fails:
        failed = (await client.get(url)).json()
        assert failed["status"] == "failed" and failed["pid"] is None and failed["returncode"] is None
        assert failed["finished_at"] and failed["log"] == starting["log"]
        assert (await client.post(url)).status_code == 200
        await jobs._runs[name].watcher
        assert len(launches) == 2

    terminal = (await client.get(url)).json()
    assert terminal["status"] == "ok" and terminal["pid"] == 4242 and terminal["returncode"] == 0
    assert terminal["log"].startswith("$ make kb-preview")


async def test_status_all_covers_registry(client):
    body = (await client.get("/api/jobs")).json()
    assert {j["name"] for j in body["jobs"]} == set(jobs.JOBS)
    for j in body["jobs"]:
        assert j["cmd"].startswith("make ")     # 一律走 make,页面按的和终端敲的是同一条命令


def test_every_job_target_exists_in_makefile():
    text = (pathlib.Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-zA-Z0-9_.-]+):", text, re.MULTILINE))
    missing = [spec.argv[1] for spec in jobs.JOBS.values() if spec.argv[1] not in targets]
    assert not missing, f"作业指向的 make 目标不存在:{missing}"


async def test_log_poll_preserves_small_log_and_appends(client, monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    url = "/api/jobs/kb-preview"
    path = tmp_path / "kb-preview.log"

    missing = (await client.get(url)).json()
    assert missing["log"] == "" and missing["log_mtime"] is None

    path.touch()
    assert (await client.get(url)).json()["log"] == ""
    path.write_bytes("准备完成\r\n正在".encode())
    assert (await client.get(url)).json()["log"] == "准备完成\n正在"

    with path.open("ab") as log:
        log.write("处理\n最后一行".encode())
    assert (await client.get(url)).json()["log"] == "准备完成\n正在处理\n最后一行"


async def test_log_poll_reads_at_most_256_kib(client, monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    path = tmp_path / "kb-preview.log"
    with path.open("wb") as log:
        log.write(b"old\n" * (1024 * 512))
        log.write("界".encode() * (1024 * 128))
        log.write("未完".encode())

    original_open = pathlib.Path.open
    sizes = []

    class CountingFile:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def read(self, size=-1):
            data = self.file.read(size)
            sizes.append(len(data.encode("utf-8")) if isinstance(data, str) else len(data))
            return data

    def open_log(file_path, *args, **kwargs):
        file = original_open(file_path, *args, **kwargs)
        return CountingFile(file) if file_path == path else file

    monkeypatch.setattr(pathlib.Path, "open", open_log)
    response = await client.get("/api/jobs/kb-preview")
    assert response.status_code == 200
    assert 0 < sum(sizes) <= 256 * 1024
    tail = response.json()["log"]
    assert tail.startswith("界") and tail.endswith("未完") and "�" not in tail


async def test_job_log_keeps_latest_400_lines(client, monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    rows = [f"进度 {i}" for i in range(450)]
    (tmp_path / "kb-preview.log").write_text("\n".join(rows) + "\n", encoding="utf-8")
    response = await client.get("/api/jobs/kb-preview")
    assert response.json()["log"] == "\n".join(rows[-400:])


async def test_job_polls_leave_event_loop_available(app, client, monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    path = tmp_path / "kb-preview.log"
    path.write_text("正在运行", encoding="utf-8")
    original_open = pathlib.Path.open
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    blocked = threading.Event()

    @app.get("/ready")
    async def ready():
        return {"ready": True}

    def open_log(file_path, *args, **kwargs):
        if file_path == path:
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(2):
                blocked.set()
                # Let the old synchronous path fail promptly rather than blocking all polls.
                release.set()
        return original_open(file_path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", open_log)
    polls = [asyncio.create_task(client.get("/api/jobs/kb-preview")) for _ in range(4)]
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        response = await asyncio.wait_for(client.get("/ready"), timeout=1)
        assert response.json() == {"ready": True}
        assert not blocked.is_set()
        assert any(not poll.done() for poll in polls)
    finally:
        release.set()
        responses = await asyncio.gather(*polls)
    assert all(response.json()["log"] == "正在运行" for response in responses)


async def test_poll_does_not_pair_new_log_with_previous_run(client, monkeypatch, tmp_path):
    name = "kb-preview"
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    monkeypatch.setitem(jobs._runs, name, jobs.JobRun(status="ok", pid=111))
    path = tmp_path / f"{name}.log"
    path.write_text("上一轮", encoding="utf-8")
    url = f"/api/jobs/{name}"
    original_open = pathlib.Path.open
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()

    def open_log(file_path, mode="r", *args, **kwargs):
        if file_path == path and mode == "rb" and not release.is_set():
            loop.call_soon_threadsafe(entered.set)
            release.wait(3)
        return original_open(file_path, mode, *args, **kwargs)

    class Process:
        pid = 222

        async def wait(self):
            return 0

    async def launch(*args, **kwargs):
        return Process()

    monkeypatch.setattr(pathlib.Path, "open", open_log)
    monkeypatch.setattr(jobs.asyncio, "create_subprocess_exec", launch)
    poll = asyncio.create_task(client.get(url))
    start = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        start = asyncio.create_task(client.post(url))
        # Wait for the new run to replace the file before letting either reader proceed.
        async with asyncio.timeout(2):
            while jobs._runs[name].pid != 222:
                await asyncio.sleep(0)
    finally:
        release.set()
        response = await poll
        if start is not None:
            await start
            await jobs._runs[name].watcher

    assert response.json()["pid"] == 222
    assert response.json()["log"] == ""
    current = (await client.get(url)).json()
    assert current["log"].startswith("$ make kb-preview")
