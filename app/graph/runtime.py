# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing, contextmanager, nullcontext

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.config import settings
from app.core import summarizer
from app.core.observability import attach_observability, trace_metadata
from app.db import repository
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

# 会流式吐答复 token 的节点(其余节点的模型调用 token 不进 delta)
ANSWER_NODES = {"main_agent"}
# 确定性节点把答复写在 state['answer'],整块作为 delta 吐出
DETERMINISTIC_ANSWER_NODES = {"script_reply", "complaint_reply", "fallback_reply", "handoff_reply"}
# 产出编号引用的检索节点(知识路 retrieve_knowledge + 退款政策路 retrieve_policy)
CITATION_NODES = {"retrieve_knowledge", "retrieve_policy"}

_graph = None
_cm = None  # AsyncSqliteSaver context manager,持有以免被 GC
# ponytail: the SQLite checkpointer deployment runs one server worker.
_active: set[int] = set()


class ConversationNotFound(Exception):
    pass


class ConversationBusy(Exception):
    pass


@contextmanager
def hold_conversation(conversation_id: int):
    if conversation_id in _active:
        raise ConversationBusy("此会话正在处理另一条消息，请稍后重试")
    _active.add(conversation_id)
    try:
        yield
    finally:
        _active.remove(conversation_id)


async def init_graph() -> None:
    """lifespan 起:开持久 checkpointer + setup + 编译图。"""
    global _graph, _cm
    _cm = AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path)
    checkpointer = await _cm.__aenter__()
    await checkpointer.setup()
    # ch09:编译时挂一次 Langfuse 回调,全图自动 trace(未配置时原样返回,零依赖)
    _graph = attach_observability(build_graph(checkpointer=checkpointer))
    logger.info("图已编译,checkpointer=%s", settings.checkpointer_db_path)


async def close_graph() -> None:
    global _graph, _cm
    if _cm is not None:
        await _cm.__aexit__(None, None, None)
        _cm = None
    _graph = None


def get_graph():
    if _graph is None:
        raise RuntimeError("图未初始化,应在 FastAPI lifespan 调 init_graph()")
    return _graph


def dedup_actions(actions: list) -> list:
    """跨多个 ReAct 轮次累积的 suggested_actions 可能重复(如 agent 多次提议
    create_ticket);按 type 去重,保留每种 type 首次出现的那条。"""
    seen = set()
    out = []
    for action in actions:
        t = action.get("type")
        if t in seen:
            continue
        seen.add(t)
        out.append(action)
    return out


async def _ensure_conversation(user_id: str, conversation_id: int | None) -> tuple[int, str, int]:
    """返回 (cid, summary, summary_upto_msg_id):入口一次库读把 ch07 摘要两字段一并带出。"""
    if conversation_id is None:
        return await repository.create_conversation(user_id), "", 0
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        raise ConversationNotFound(conversation_id)
    return conversation_id, conv.summary or "", conv.summary_upto_msg_id or 0


def _graph_input(user_id: str, message: str, cid: int, msg_id: int,
                 summary: str, summary_upto: int) -> dict:
    # 显式重置每轮的输出通道:checkpointer 按 thread_id 持久化 State,这些无 reducer 的标量
    # 会跨轮残留(如上一轮闲聊/投诉的 answer、建工单可选项串到本轮),须在入口清零。
    # messages 有 add_messages reducer(追加=跨轮历史),不清;steps/tokens_used 是本轮预算。
    # ch07:用户消息带 db-{msg_id} 锚点(滑窗与摘要边界对齐);summary 两字段每轮从库刷新。
    return {"messages": [HumanMessage(message, id=f"db-{msg_id}")], "user_id": user_id,
            "conversation_id": cid, "reply_message_id": None, "reply_text": None, "steps": 0, "tokens_used": 0,
            "answer": "", "suggested_actions": [], "evidence": "",
            "evidence_strong": False, "citations": [],
            "evidence_confidence": 0.0, "fallback_source": "", "retrieved_snapshot": [], "service_failure": None,
            "resolved_query": "", "intent_confidence": 0.0,
            "order_id": "", "order_data": {},
            "summary": summary, "summary_upto_msg_id": summary_upto,
            "trace": None}  # merge 通道:None=重置哨兵(见 merge_dict),清上一轮 trace 残留


def _interrupt_payload(state: dict):
    """从终态提取中断负载(整个 __interrupt__[0].value:含 type + orders/preview 等);
    无中断返回 None。ch08 起中断不止订单选择器,负载透传、消费方按 type 分发。"""
    intr = state.get("__interrupt__")
    if not intr:
        return None
    item = intr[0]
    payload = item.value
    if payload.get("type") == "select_order":
        return {**payload, "interrupt_id": item.id}
    return payload


def _graph_config(conversation_id: int) -> dict:
    return {
        "configurable": {"thread_id": str(conversation_id)},
        "metadata": trace_metadata(conversation_id),
    }


async def run_turn(user_id, message, conversation_id) -> dict:
    """非流式:落 user 消息 → ainvoke → 返回终态(供 /api/agent、eval)。遇 interrupt 带 orders。"""
    cid, summary, upto = await _ensure_conversation(user_id, conversation_id)
    # The HTTP request already owns an existing conversation. A new one has just received its ID.
    with hold_conversation(cid) if conversation_id is None else nullcontext():
        msg_id = await repository.append_message(cid, "user", content=message)
        config = _graph_config(cid)
        final = await get_graph().ainvoke(
            _graph_input(user_id, message, cid, msg_id, summary, upto), config)
        await summarizer.maybe_schedule_summary(cid)
    return {"conversation_id": cid, "state": final, "interrupt": _interrupt_payload(final)}


async def resume_turn(conversation_id: int, resume_value) -> dict:
    """非流式续跑(供 /api/agent、eval):Command(resume) 回填后跑到下一个中断或结束。"""
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    config = _graph_config(conversation_id)
    final = await get_graph().ainvoke(Command(resume=resume_value), config)
    await summarizer.maybe_schedule_summary(conversation_id)
    return {"conversation_id": conversation_id, "state": final,
            "interrupt": _interrupt_payload(final)}


async def _stream_events(cid: int, stream_source) -> AsyncIterator[dict]:
    """astream 多模式 → 事件 dict。stream_source 为 _graph_input(...)(新一轮)或 Command(resume=...)(续跑)。
    遇 interrupt(updates 块含 __interrupt__,Task 1 冒烟钉死)→ 吐 interrupt 事件并结束本次流
    (前端点选后走 /api/actions/resume 续流)。"""
    config = _graph_config(cid)
    actions: list = []
    reply_message_id = None
    reply_text = None
    events = get_graph().astream(stream_source, config, stream_mode=["messages", "updates"])
    interrupt = None
    async with aclosing(events):
        async for mode, chunk in events:
            if interrupt is not None:
                continue
            if mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") in ANSWER_NODES:
                    text = msg.content if isinstance(msg.content, str) else ""
                    if text:
                        yield {"type": "delta", "text": text}
            elif mode == "updates":
                if "__interrupt__" in chunk:
                    payload = _interrupt_payload(chunk)
                    # 带上 conversation_id:中断时不发 done 帧,前端要靠这个才知道续跑用哪个会话。
                    # ch08:负载按中断类型透传(select_order 带 orders,confirm_ticket 带 preview),
                    # 前端按 kind 分发。
                    ev = {"type": "interrupt", "kind": payload.get("type", ""), "conversation_id": cid}
                    for k in ("orders", "preview", "interrupt_id"):
                        if payload.get(k) is not None:
                            ev[k] = payload[k]
                    interrupt = ev
                    continue
                for node, upd in chunk.items():
                    if not isinstance(upd, dict):
                        continue
                    if node == "log":
                        reply_message_id = upd.get("reply_message_id")
                        reply_text = upd.get("reply_text")
                    if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                        event = {"type": "delta", "text": upd["answer"]}
                        if node == "fallback_reply" and upd.get("trace", {}).get("service_failure"):
                            event["replace"] = True
                        yield event
                    if node in CITATION_NODES and upd.get("citations"):
                        yield {"type": "citations", "items": upd["citations"]}
                    if node == "agent_tools":
                        for m in upd.get("messages", []):
                            name = getattr(m, "name", None)
                            # submit_refund 仍是「拦成前端表单」的合成消息,不发工具帧;
                            # create_ticket ch08 起确认后真执行,照常发帧
                            if name and name != "submit_refund":
                                yield {"type": "tool", "name": name}
                    if upd.get("suggested_actions"):
                        actions.extend(upd["suggested_actions"])
    if interrupt is not None:
        yield interrupt
        return
    if actions:
        yield {"type": "actions", "items": dedup_actions(actions)}
    done = {"type": "done", "conversation_id": cid}
    if reply_message_id is not None:
        done["message_id"] = reply_message_id
        if reply_text is not None:
            done["final_text"] = reply_text
    yield done


async def stream_turn(user_id, message, conversation_id) -> AsyncIterator[dict]:
    """流式:落 user 消息 → astream 多模式 → 事件 dict(供 /api/chat 转 SSE)。"""
    cid, summary, upto = await _ensure_conversation(user_id, conversation_id)
    with hold_conversation(cid) if conversation_id is None else nullcontext():
        msg_id = await repository.append_message(cid, "user", content=message)
        graph_input = _graph_input(user_id, message, cid, msg_id, summary, upto)
        async with aclosing(_stream_events(cid, graph_input)) as stream:
            async for ev in stream:
                yield ev
        await summarizer.maybe_schedule_summary(cid)


async def stream_resume(conversation_id: int, resume_value) -> AsyncIterator[dict]:
    """前端点选订单后续跑同一会话的暂停流(Command(resume) 续 astream,事件与 stream_turn 同构)。"""
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    async with aclosing(_stream_events(conversation_id, Command(resume=resume_value))) as stream:
        async for ev in stream:
            yield ev
    await summarizer.maybe_schedule_summary(conversation_id)
