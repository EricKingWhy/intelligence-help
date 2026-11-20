"""Deployment credentials and authorization shared by HTTP and SSE entry points."""
import hashlib
import json
import os
from functools import cache
from hmac import compare_digest
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.db import repository

_basic = HTTPBasic(auto_error=False)


@cache
def credentials() -> dict[str, dict[str, str]] | None:
    mode = os.environ.get("MEWHELP_AUTH_MODE", "demo")
    if mode == "demo":
        return None
    if mode != "required":
        raise ValueError("MEWHELP_AUTH_MODE must be demo or required")
    users = json.loads(Path(os.environ["MEWHELP_AUTH_FILE"]).read_text(encoding="utf-8"))
    if not isinstance(users, dict) or not users:
        raise ValueError("MEWHELP_AUTH_FILE must contain at least one account")
    for name, record in users.items():
        if not name or not name.isascii() or ":" in name or len(name) > 64:
            raise ValueError("Invalid account identifier")
        if (record["role"] not in {"customer", "operator"}
                or len(bytes.fromhex(record["sha256"])) != 32):
            raise ValueError("Invalid account role or credential digest")
    return users


def identity(request: Request, supplied: Annotated[HTTPBasicCredentials | None, Depends(_basic)]):
    users = credentials()
    if users is None:
        return None
    if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get(
        "sec-fetch-site"
    ) == "cross-site":
        raise HTTPException(403, "不接受跨站操作")
    if supplied is not None:
        record = users.get(supplied.username)
        digest = hashlib.sha256(supplied.password.encode()).digest()
        expected = bytes.fromhex(record["sha256"]) if record else bytes(32)
        if compare_digest(digest, expected) and record is not None:
            return supplied.username, record["role"]
    raise HTTPException(401, "请登录", headers={"WWW-Authenticate": 'Basic realm="MewHelp"'})


Principal = Annotated[tuple[str, str] | None, Depends(identity)]


def operator(principal: Principal) -> None:
    if principal is not None and principal[1] != "operator":
        raise HTTPException(403, "需要运营权限")


async def owned_conversation(principal: tuple[str, str] | None, conversation_id: int | None):
    if principal is None or conversation_id is None:
        return
    conversation = await repository.get_conversation(conversation_id)
    if conversation is None or conversation.user_id != principal[0]:
        raise HTTPException(404, "会话不存在")
