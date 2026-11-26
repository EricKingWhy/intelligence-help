# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.auth import Principal
from app.api.conversation_execution import ChatExecution
from app.api.reply_stream import error_frame, stream_frames
from app.graph import runtime

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/chat")
async def chat(req: ChatExecution, principal: Principal):
    if principal is not None:
        req = req.model_copy(update={"user_id": principal[0]})
    async def event_stream() -> AsyncIterator[str]:
        try:
            async with aclosing(runtime.stream_turn(req.user_id, req.message, req.conversation_id)) as stream:
                async for frame in stream_frames(stream):
                    yield frame
        except runtime.PendingTicketConfirmation as error:
            yield error_frame(str(error), code="ticket_confirmation_pending", preview=error.preview)
        except runtime.ConversationBusy as error:
            yield error_frame(str(error))
        except runtime.ConversationNotFound:
            yield error_frame("会话不存在")
        except SQLAlchemyError:
            logger.exception("数据库错误 user_id=%s", req.user_id)
            yield error_frame("数据库暂时不可用,请稍后重试")
        except Exception:
            logger.exception("图编排失败 user_id=%s", req.user_id)
            yield error_frame("上游模型暂时不可用,请稍后重试")

    return StreamingResponse(event_stream(), media_type="text/event-stream")
