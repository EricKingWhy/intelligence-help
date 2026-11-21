# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.auth import Principal, owned_conversation
from app.db import repository
from app.graph import runtime
from app.tools.business import owns_order
from app.schemas.actions import (
    CreateRefundRequest,
    CreateRefundResponse,
    CreateTicketRequest,
    CreateTicketResponse,
    ResumeRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()
SubmissionKey = Annotated[str | None, Header(alias="Idempotency-Key", min_length=1,
                                              max_length=128, pattern=r"^[a-zA-Z0-9_.:-]+$")]


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket_action(
    req: CreateTicketRequest, principal: Principal, key: SubmissionKey = None,
) -> CreateTicketResponse:
    """建工单按钮:用户点了才写 tickets 表(复用 ch02 工单能力)。"""
    await owned_conversation(principal, req.conversation_id)
    try:
        options = {"submission_key": f"ticket:{key}"} if key else {}
        ticket_no = await repository.create_ticket(
            req.conversation_id, req.description, req.ticket_type, **options)
    except repository.SubmissionConflict as exc:
        raise HTTPException(409, "提交标识已用于不同内容") from exc
    except SQLAlchemyError:
        logger.exception("建工单失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="工单系统暂时不可用,请稍后重试")
    return CreateTicketResponse(ticket_no=ticket_no)


@router.post("/api/actions/create-refund", response_model=CreateRefundResponse)
async def create_refund_action(
    req: CreateRefundRequest, principal: Principal, key: SubmissionKey = None,
) -> CreateRefundResponse:
    """退款表单提交:写 tickets(ticket_type='退款'),描述带订单号 + 固定类目原因。复用 ch02 工单能力。"""
    await owned_conversation(principal, req.conversation_id)
    if principal is not None and not owns_order(principal[0], req.order_id):
        raise HTTPException(404, "订单不存在")
    desc = f"退款申请 订单号={req.order_id} 原因={req.reason}"
    try:
        options = {"submission_key": f"refund:{key}"} if key else {}
        ticket_no = await repository.create_ticket(req.conversation_id, desc, "退款", **options)
    except repository.SubmissionConflict as exc:
        raise HTTPException(409, "提交标识已用于不同内容") from exc
    except SQLAlchemyError:
        logger.exception("退款单创建失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="退款系统暂时不可用,请稍后重试")
    return CreateRefundResponse(ticket_no=ticket_no)


@router.post("/api/actions/resume")
async def resume_action(req: ResumeRequest, principal: Principal):
    """中断续跑(SSE,事件与 /api/chat 同构):订单选择器点选回 order_id;
    工单预览卡回 confirmed(ch08,resume 值为 {"confirmed": bool},agent_tools 按此放行/拒绝)。"""
    await owned_conversation(principal, req.conversation_id)
    if req.order_id is not None and principal is not None and not owns_order(principal[0], req.order_id):
        raise HTTPException(404, "订单不存在")
    if req.order_id is None and req.confirmed is None:
        raise HTTPException(status_code=400, detail="order_id 与 confirmed 至少传一个")
    if req.order_id is not None and req.confirmed is not None:
        raise HTTPException(400, "请选择一种续跑操作")
    if principal is not None and req.confirmed is not None and req.confirmation_id is None:
        raise HTTPException(422, "工单确认需要确认标识")
    if req.confirmed and req.confirmation_id:
        ticket = await repository.get_ticket_result(req.conversation_id, f"agent:{req.confirmation_id}")
        if ticket:
            async def replay():
                yield _sse({"delta": f"工单已创建:{ticket}"})
                yield _sse({"event": "done", "conversation_id": req.conversation_id})
                yield "data: [DONE]\n\n"

            return StreamingResponse(replay(), media_type="text/event-stream")
    if principal is not None or req.confirmation_id:
        state = await runtime.get_graph().aget_state(
            {"configurable": {"thread_id": str(req.conversation_id)}},
        )
        if principal is not None and state.values.get("user_id") != principal[0]:
            raise HTTPException(409, "会话没有可续跑的步骤")
        if principal is not None and req.order_id is not None and not any(
            isinstance(item.value, dict) and item.value.get("type") == "select_order"
            for item in state.interrupts
        ):
            raise HTTPException(409, "会话没有待选择的订单")
        if req.confirmation_id and not any(
            isinstance(item.value, dict)
            and (item.value.get("preview") or {}).get("confirmation_id") == req.confirmation_id
            for item in state.interrupts
        ):
            raise HTTPException(409, "工单确认已失效")
    resume_value = req.order_id
    if resume_value is None:
        resume_value = {"confirmed": bool(req.confirmed)}
        if req.confirmation_id:
            resume_value["confirmation_id"] = req.confirmation_id

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev in runtime.stream_resume(req.conversation_id, resume_value):
                if ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "citations":
                    yield _sse({"event": "citations", "items": ev["items"]})
                elif ev["type"] == "actions":
                    yield _sse({"event": "actions", "items": ev["items"]})
                elif ev["type"] == "interrupt":
                    yield _sse({"event": "interrupt",
                                **{k: v for k, v in ev.items() if k != "type"}})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except runtime.ConversationNotFound:
            yield "event: error\n"
            yield _sse({"message": "会话不存在"})
            return
        except Exception:
            logger.exception("续跑失败 conv=%s", req.conversation_id)
            yield "event: error\n"
            yield _sse({"message": "上游暂时不可用,请稍后重试"})
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
