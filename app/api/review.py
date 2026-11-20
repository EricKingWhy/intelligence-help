# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch09 审核后台 API:待审队列列表/详情/通过(写回知识库)/驳回。
通过 = 核准答案以 QA chunk 走 ch03 落库流程(write_pending → vectorize_pending 同步),
向量化成功才置「通过」——保证审核页点了通过,下一问就能检索命中(验收 3)。"""
import logging

from fastapi import APIRouter, HTTPException

from app.db import repository
from app.kb import dualwrite
from app.db.models import KnowledgeChunk, ReviewQueue
from app.kb.documents import is_key
from app.schemas.review import ApproveRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _item_out(r) -> dict:
    publishing = r.review_status == "待审" and r.publication_chunk_id is not None
    return {"id": r.id, "normalized_question": r.normalized_question,
            "ai_suggested_answer": r.ai_suggested_answer,
            "occurrence_count": r.occurrence_count, "review_status": r.review_status,
            "publishing": publishing,
            "publication_answer": r.approved_answer if publishing else None,
            "created_at": r.created_at.isoformat() if r.created_at else None}


@router.get("/api/review/queue")
async def review_queue(status: str | None = None) -> dict:
    """列表按 occurrence_count 降序:一个缺口被越多用户反复问到,越该优先补(README)。"""
    rows = await repository.list_review_queue(status)
    return {"items": [_item_out(r) for r in rows]}


@router.get("/api/review/{review_id}")
async def review_detail(review_id: int) -> dict:
    """详情带归并原话 + 各自的召回快照:审核人对着快照判断「真缺知识,还是有但没检到」。"""
    detail = await repository.get_review_detail(review_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="缺口不存在")
    item, raws = detail
    out = _item_out(item)
    out["approved_answer"] = item.approved_answer
    out["raws"] = [{"raw_question": r.raw_question, "source": r.source, "reason": r.reason,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "retrieved_chunks": r.retrieved_chunks}
                   for r in raws]
    return out


@router.post("/api/review/{review_id}/approve")
async def approve(review_id: int, req: ApproveRequest) -> dict:
    def make_chunk(item):
        return KnowledgeChunk(
            category="飞轮沉淀", questions=item.normalized_question, answer=req.approved_answer,
            section_path=f"飞轮沉淀 / {item.normalized_question}"[:512], content_type="faq",
            is_key_clause=is_key(item.normalized_question, req.approved_answer),
        )

    try:
        chunk_id = await repository.prepare_publication(
            ReviewQueue, review_id, make_chunk, answer=req.approved_answer,
        )
        await dualwrite.vectorize_pending(chunk_ids=[chunk_id])
        await repository.finish_publication(ReviewQueue, review_id, chunk_id)
    except LookupError as exc:
        raise HTTPException(404, "缺口不存在") from exc
    except repository.SubmissionConflict as exc:
        raise HTTPException(409, "审核状态或已登记的答案发生冲突") from exc
    except Exception as exc:
        logger.exception("审核写回未完成 review=%s", review_id)
        raise HTTPException(502, "写回未完成，请使用相同答案重试") from exc
    return {"ok": True, "chunk_ids": [chunk_id]}


@router.post("/api/review/{review_id}/reject")
async def reject(review_id: int) -> dict:
    if not await repository.update_review_status(review_id, "驳回"):
        detail = await repository.get_review_detail(review_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="缺口不存在")
        raise HTTPException(status_code=409, detail="仅待审状态可驳回")
    return {"ok": True}
