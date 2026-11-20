"""Customer handoff submission and the operator queue."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.auth import Principal
from app.db import repository

customer_router = APIRouter()
operator_router = APIRouter(prefix="/api/operator/handoffs")


class HandoffRequest(BaseModel):
    user_id: str | None = Field(default=None, max_length=64)
    message_id: int = Field(gt=0)


class ResolutionRequest(BaseModel):
    resolution: str = Field(min_length=1, max_length=2000)


async def verify_owner(conversation_id: int, principal: Principal, user_id: str | None) -> None:
    owner = principal[0] if principal else user_id
    conversation = await repository.get_conversation(conversation_id)
    if not owner or conversation is None or conversation.user_id != owner:
        raise HTTPException(404, "会话不存在")


def view(ticket, *, operator_view: bool = False) -> dict:
    status = ("已处理" if ticket.status == "已处理" else
              "处理中" if ticket.assigned_to else "待接单")
    result = {
        "ticket_no": ticket.ticket_no,
        "status": status,
        "description": ticket.description,
        "resolution": ticket.resolution,
        "created_at": ticket.created_at,
    }
    if operator_view:
        result.update(conversation_id=ticket.conversation_id, assigned_to=ticket.assigned_to,
                      context=ticket.handoff_context)
    return result


@customer_router.post("/api/conversations/{conversation_id}/handoffs")
async def submit(conversation_id: int, req: HandoffRequest, principal: Principal):
    await verify_owner(conversation_id, principal, req.user_id)
    message = await repository.get_handoff_message(conversation_id, req.message_id)
    if message is None:
        raise HTTPException(409, "这条消息没有可提交的转接请求")
    ticket_no = await repository.create_ticket(
        conversation_id, message.handoff_context["question"], "咨询",
        submission_key=f"handoff:{message.id}", handoff_context=message.handoff_context,
    )
    return view(await repository.get_handoff_ticket(ticket_no))


@customer_router.get("/api/conversations/{conversation_id}/handoffs")
async def customer_tickets(conversation_id: int, principal: Principal, user_id: str | None = None):
    await verify_owner(conversation_id, principal, user_id)
    return {"items": [view(row) for row in await repository.list_handoffs(
        conversation_id=conversation_id,
    )]}


@operator_router.get("")
async def queue(
    principal: Principal,
    status: Literal["open", "handled"] = "open",
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    active_operator = principal[0] if principal else "demo-operator"
    return {"operator": active_operator, "items": [view(row, operator_view=True)
            for row in await repository.list_handoffs(status=status, limit=limit, offset=offset)]}


async def update(ticket_no: str, principal: Principal, resolution: str | None = None):
    active_operator = principal[0] if principal else "demo-operator"
    try:
        ticket = await repository.update_handoff(ticket_no, active_operator, resolution)
    except LookupError as exc:
        raise HTTPException(404, "转接工单不存在") from exc
    except repository.SubmissionConflict as exc:
        raise HTTPException(409, "工单状态已变化，请刷新队列") from exc
    return view(ticket, operator_view=True)


@operator_router.post("/{ticket_no}/claim")
async def claim(ticket_no: str, principal: Principal):
    return await update(ticket_no, principal)


@operator_router.post("/{ticket_no}/resolve")
async def resolve(ticket_no: str, req: ResolutionRequest, principal: Principal):
    resolution = req.resolution.strip()
    if not resolution:
        raise HTTPException(422, "请填写给客户的处理结论")
    return await update(ticket_no, principal, resolution)
