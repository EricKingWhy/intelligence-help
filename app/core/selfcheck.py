# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import logging

from pydantic import BaseModel, Field

from app.core import llm
from app.core.upstream import failure
from app.core.prompts import SELF_CHECK_PROMPT

logger = logging.getLogger(__name__)


class _Check(BaseModel):
    useful: bool = Field(description="证据是否足以回答")
    reason: str = Field(default="", description="判断依据")


def _chain():
    return SELF_CHECK_PROMPT | llm.structured(_Check)


async def check_sufficient(query: str, evidence_texts: list[str]) -> dict:
    """生成前证据充分性自评。扁平字段 useful/reason(避开 glm-5.2 嵌套数组 502)。

    自评失败时附上服务故障，交给会话图决定如何恢复。成功判为证据不足时进入知识审核。"""
    evidence = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(evidence_texts)) or "(无证据)"
    try:
        r: _Check = await _chain().ainvoke({"query": query, "evidence": evidence})
    except Exception as exc:
        return {"useful": False, "reason": "证据自评失败",
                "service_failure": failure(exc, "evidence_check")}
    return {"useful": bool(r.useful), "reason": r.reason or ""}
