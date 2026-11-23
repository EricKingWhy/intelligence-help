"""Keep a conversation's HTTP graph operations exclusive through response completion."""

from contextlib import contextmanager
from typing import Annotated

from fastapi import Depends, HTTPException

from app.api.auth import Principal, owned_conversation
from app.schemas.actions import ResumeRequest
from app.schemas.agent import AgentRequest
from app.schemas.chat import ChatRequest

# ponytail: the SQLite checkpointer deployment runs one server worker.
_active: set[int] = set()


@contextmanager
def _exclusive(conversation_id: int | None):
    if conversation_id is None:
        # The runtime allocates a distinct conversation for each first turn.
        yield
        return
    if conversation_id in _active:
        raise HTTPException(409, "此会话正在处理另一条消息，请稍后重试")
    _active.add(conversation_id)
    try:
        yield
    finally:
        _active.remove(conversation_id)


async def _chat(req: ChatRequest, principal: Principal):
    await owned_conversation(principal, req.conversation_id)
    with _exclusive(req.conversation_id):
        yield req


async def _agent(req: AgentRequest, principal: Principal):
    await owned_conversation(principal, req.conversation_id)
    with _exclusive(req.conversation_id):
        yield req


async def _resume(req: ResumeRequest, principal: Principal):
    await owned_conversation(principal, req.conversation_id)
    with _exclusive(req.conversation_id):
        yield req


ChatExecution = Annotated[ChatRequest, Depends(_chat, scope="request")]
AgentExecution = Annotated[AgentRequest, Depends(_agent, scope="request")]
ResumeExecution = Annotated[ResumeRequest, Depends(_resume, scope="request")]
