# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch07 验收配套:会话列表 + 历史消息回载(只读,前端多会话切换用)。"""
from fastapi import APIRouter, HTTPException

from app.api.auth import Principal, owned_conversation
from app.db import repository

router = APIRouter()


@router.get("/api/conversations")
async def list_conversations(user_id: str, principal: Principal):
    if principal is not None:
        user_id = principal[0]
    items = await repository.list_conversations(user_id)
    return {"items": items}


@router.get("/api/conversations/{conversation_id}/messages")
async def list_messages(conversation_id: int, principal: Principal):
    await owned_conversation(principal, conversation_id)
    if await repository.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    msgs = await repository.list_dialog_messages(conversation_id)
    return {"items": [
        {"role": m.role, "content": m.content or "",
         "created_at": m.created_at.isoformat() if m.created_at else None,
         **({"handoff_message_id": m.id} if (m.handoff_context or {}).get("offered") else {}),
         **({"retryable": True} if m.role == "user" and (m.handoff_context or {}).get("retryable") else {}),
         **({"service_failed": True} if m.role == "assistant"
            and (m.handoff_context or {}).get("service_failure") else {})}
        for m in msgs
    ]}
