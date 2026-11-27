# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""后台管理页作业运行器:把 make 目标搬到浏览器上按,日志轮询回显,长活能停。

安全边界:只能跑 JOBS 注册表里的目标,argv 全部写死在本模块。前端只传一个 job 名,
传不进任何命令片段、参数或路径——页面有「重跑」按钮,但没有 shell。
命令一律走 make:配方是仓库里那一份,页面按的和终端敲的必须是同一条命令,不许分叉。
"""
import asyncio
import contextlib
import datetime as dt
import logging
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "log" / "acceptance"
TAIL_LINES = 400
TAIL_BYTES = 256 * 1024


@dataclass(frozen=True)
class JobSpec:
    name: str
    title: str
    argv: tuple[str, ...]
    needs: str                    # 前置条件,页面按钮上直接提示,免得点了才发现上游没配好
    heavy: bool = False           # 分钟级重活:页面二次确认才发起


JOBS: dict[str, JobSpec] = {
    j.name: j for j in (
        # ch03 知识库:离线建库那四步
        JobSpec("kb-preview", "材料清单与切块预览", ("make", "kb-preview"), "本地跑,不写库"),
        JobSpec("kb-build", "离线建库(文档切块 → pending)", ("make", "kb-build"), "需 mysql"),
        JobSpec("kb-mine", "对话挖知识(抽 QA → 去重 → pending)", ("make", "kb-mine"),
                "需 mysql + 聊天上游", heavy=True),
        JobSpec("kb-vectorize", "向量化(嵌入 → Milvus → 回标 done)", ("make", "kb-vectorize"),
                "需 mysql + 嵌入上游 + Milvus"),
        # 改了 data/kb/*.md 之后走这条:按(小节路径、第几块)对齐,只把正文变了的块原地改掉,
        # id 不变,再按 kb-vectorize 同 id upsert 覆盖向量。不能用 kb-reset 顶替——那会清掉飞轮写回的块
        JobSpec("kb-repatch", "补丁式重嵌(md 改动 → 原地改正文)", ("make", "kb-repatch"),
                "需 mysql;改完再按「向量化待补块」"),
        JobSpec("seed-conv", "灌历史会话种子", ("make", "seed-conv"), "需 docker mysql 容器"),
        JobSpec("kb-reset", "清库重建(清两表 + drop 集合)", ("make", "kb-reset"),
                "需 mysql + Milvus;会清空知识库", heavy=True),
        # ch04 检索质量:四策略对照评估(检索段确定性 + 生成段过 LLM 裁判)
        JobSpec("eval-rag", "RAG 评估(四策略对照)", ("make", "eval-rag"),
                "需 Milvus + 已建库 + 聊天上游,分钟级", heavy=True),
        # ch09 观测与成本:意图成本账、评估趋势、置信度阈值校准
        JobSpec("cost-report", "意图成本账", ("make", "cost-report"),
                "需 Langfuse 在跑且窗口内有 trace"),
        JobSpec("eval-flywheel", "评估流水线(落一轮趋势)", ("make", "eval-flywheel"),
                "需 mysql + Milvus + 聊天上游,分钟级", heavy=True),
        JobSpec("calibrate-confidence", "置信度阈值校准", ("make", "calibrate-confidence"),
                "需 Milvus + 已建库 + 重排上游,分钟级", heavy=True),
        # ch10 分类器
        JobSpec("ch10-golden", "黄金样例闸", ("make", "ch10-golden"), "需聊天上游"),
        JobSpec("ch10-corpus", "语料流水线", ("make", "ch10-corpus"),
                "需 mysql + 聊天上游", heavy=True),
        JobSpec("ch10-dataset", "数据集划分与增强", ("make", "ch10-dataset"),
                "需聊天上游", heavy=True),
        JobSpec("ch10-train", "全参微调训练", ("make", "ch10-train"),
                "需 ml 依赖,分钟级", heavy=True),
        JobSpec("ch10-eval", "测试集评测", ("make", "ch10-eval"), "需 ml 依赖 + 已训练权重"),
        JobSpec("ch10-export", "ONNX 导出与一致性校验", ("make", "ch10-export"),
                "需 ml 依赖 + 已训练权重", heavy=True),
        JobSpec("ch10-threshold-scan", "阈值扫描重演", ("make", "ch10-threshold-scan"),
                "需分类器服务 :8110"),
        JobSpec("classifier-up", "拉起分类器服务", ("make", "classifier-up"), "需已导出 ONNX"),
        JobSpec("classifier-down", "停止分类器服务", ("make", "classifier-down"), "—"),
        JobSpec("classify-pool", "旁路批量归类", ("make", "classify-pool"),
                "需 mysql + :8110"),
        JobSpec("classify-pool-force", "旁路批量归类(不足一批强跑)",
                ("make", "classify-pool", "FORCE=1"), "需 mysql + :8110"),
    )
}


@dataclass
class JobRun:
    """一次运行的状态。进程对象与 watcher 任务留在内存,重启服务后归零——
    但产物里的 ran_at 与日志文件都在盘上,页面照样能说出「上次什么时候跑的、跑成什么样」。"""
    status: str = "idle"          # idle | starting | running | ok | failed | stopped
    pid: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    log_revision: int = field(default=0, repr=False)
    proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    watcher: asyncio.Task | None = field(default=None, repr=False)


_runs: dict[str, JobRun] = {}


def _run(name: str) -> JobRun:
    return _runs.setdefault(name, JobRun())


def log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


async def start(name: str) -> JobRun:
    """发起一次运行。同名作业启动中或尚未退出时拒绝重入。"""
    spec = JOBS[name]
    run = _run(name)
    if run.status == "starting":
        raise RuntimeError(f"{spec.title}正在启动中，请稍后再试")
    if run.status == "running" or run.proc is not None:
        raise RuntimeError(f"{spec.title}正在运行中(pid {run.pid}),等它跑完再发起")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = log_path(name)
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    header = f"$ {' '.join(spec.argv)}\n# {started_at} 由后台页发起\n\n"
    fh = path.open("w", encoding="utf-8", buffering=1)  # 窗口只显示本次日志
    run.log_revision += 1
    run.status, run.pid = "starting", None
    run.started_at = started_at
    run.finished_at = run.returncode = None
    run.watcher = None
    # ponytail: 作业状态在单个服务进程内共享，多 worker 部署需要共享作业状态。
    try:
        fh.write(header)
        proc = await asyncio.create_subprocess_exec(
            *spec.argv, cwd=REPO_ROOT, stdout=fh, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            # 独立会话:kill 时能连 make → uv → python 整条链一起收,不留孤儿进程
            start_new_session=True,
        )
    except BaseException:
        run.status = "failed"
        run.finished_at = dt.datetime.now().isoformat(timespec="seconds")
        with contextlib.suppress(OSError):
            fh.close()
        raise

    run.status, run.pid = "running", proc.pid
    run.proc = proc
    run.watcher = asyncio.create_task(_watch(name, run, proc, fh))
    logger.info("验收页发起作业 %s pid=%s argv=%s", name, proc.pid, " ".join(spec.argv))
    return run


async def _watch(name: str, run: JobRun, proc: asyncio.subprocess.Process, fh) -> None:
    try:
        rc = await proc.wait()
    finally:
        with contextlib.suppress(Exception):
            fh.close()
    run.returncode = rc
    run.finished_at = dt.datetime.now().isoformat(timespec="seconds")
    # stopped 保留:人为 kill 的退出码也是非 0,别报成「作业失败」冤枉它
    if run.status != "stopped":
        run.status = "ok" if rc == 0 else "failed"
    run.proc = None
    logger.info("验收页作业 %s 结束 rc=%s status=%s", name, rc, run.status)


async def stop(name: str) -> None:
    run = _run(name)
    proc = run.proc
    if run.status != "running" or proc is None:
        raise RuntimeError("该作业当前没有在运行")
    run.status = "stopped"
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def tail(name: str, lines: int = TAIL_LINES) -> str:
    try:
        with log_path(name).open("rb", buffering=0) as log:
            end = log.seek(0, os.SEEK_END)
            start = max(0, end - TAIL_BYTES)
            log.seek(start)
            data = log.read(end - start)
    except FileNotFoundError:
        return ""

    if start:
        # The byte limit can start in the middle of a UTF-8 character.
        first = 0
        while first < min(3, len(data)) and data[first] & 0b11000000 == 0b10000000:
            first += 1
        data = data[first:]
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-lines:])


def _run_status(name: str) -> dict:
    spec, run = JOBS[name], _run(name)
    return {
        "name": name, "title": spec.title, "cmd": " ".join(spec.argv),
        "needs": spec.needs, "heavy": spec.heavy,
        "status": run.status, "pid": run.pid,
        "started_at": run.started_at, "finished_at": run.finished_at,
        "returncode": run.returncode,
    }


def _log_status(name: str, with_log: bool) -> dict:
    try:
        mtime = log_path(name).stat().st_mtime
    except FileNotFoundError:
        mtime = None
    # 作业状态重启后归零时，文件修改时间仍可用于显示上次运行时间。
    out = {"log_mtime": dt.datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
           if mtime is not None else None}
    if with_log:
        out["log"] = tail(name)
    return out


def status(name: str, with_log: bool = False) -> dict:
    return _run_status(name) | _log_status(name, with_log)


async def status_with_log(name: str) -> dict:
    run = _run(name)
    revision = run.log_revision
    log = await asyncio.to_thread(_log_status, name, True)
    if revision != run.log_revision:
        # A new run replaced the file during the read. The next poll will show its log.
        log = {"log_mtime": None, "log": ""}
    return _run_status(name) | log


def status_all() -> list[dict]:
    return [status(n) for n in JOBS]
