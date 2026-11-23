# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""Record feedback against the selected persisted reply."""
import logging

from fastapi import APIRouter, HTTPException

from app.db import repository
from app.api.auth import Principal, owned_conversation
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest, principal: Principal) -> FeedbackResponse:
    await owned_conversation(principal, req.conversation_id)
    question = req.question
    snapshot = None
    if req.message_id is not None:
        reply = await repository.get_feedback_reply(req.conversation_id, req.message_id)
        if reply is None:
            raise HTTPException(404, "回复不存在")
        context = reply.handoff_context or {}
        if context.get("service_failure"):
            return FeedbackResponse(pooled=False)
        if feedback := context.get("feedback"):
            question = feedback["question"]
            snapshot = feedback["evidence"] or None

    if req.rating == "up":
        logger.info("feedback up conv=%s q=%r(只记日志,不落库)", req.conversation_id, question[:40])
        return FeedbackResponse(pooled=False)

    await repository.insert_low_confidence(
        req.conversation_id, question, "user_feedback", "用户反馈未解决",
        retrieved_chunks=snapshot,
    )
    logger.info("feedback down conv=%s 已落池 快照=%s", req.conversation_id,
                "有" if snapshot else "无")
    return FeedbackResponse(pooled=True)
