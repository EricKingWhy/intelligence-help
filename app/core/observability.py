# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""Export operational telemetry without conversation and tool content."""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult
    from opentelemetry.sdk.trace import ReadableSpan

logger = logging.getLogger(__name__)
_client = None


_SESSION_ID = re.compile(r"mewhelp-[0-9a-f]{32}")
# Permit only the SDK fields used to group traces and report cost, timing and status.
_EXPORT_ATTRIBUTES = frozenset({
    "langfuse.environment",
    "langfuse.internal.as_root",
    "langfuse.internal.is_app_root",
    "langfuse.observation.completion_start_time",
    "langfuse.observation.cost_details",
    "langfuse.observation.level",
    "langfuse.observation.model.name",
    "langfuse.observation.type",
    "langfuse.observation.usage_details",
    "langfuse.trace.metadata.intent",
    "langfuse.trace.metadata.intent_confidence",
    "langfuse.trace.tags",
})


def _export_chat_span(span: ReadableSpan) -> bool:
    # The SDK can patch attributes only. Exclude spans whose events, links or status
    # descriptions may contain raw context, as well as other OTel instrumentation.
    from langfuse.span_filter import is_langfuse_span

    return is_langfuse_span(span) and not (span.events or span.links or span.status.description)


def _mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for identifier, span in params.spans.items():
        allowed = _EXPORT_ATTRIBUTES
        session = span.attributes.get("session.id")
        if isinstance(session, str) and _SESSION_ID.fullmatch(session):
            allowed = allowed | {"session.id"}
        patches[identifier] = OtelSpanPatch(
            delete_attributes=tuple(key for key in span.attributes if key not in allowed),
        )
    return MaskOtelSpansResult(span_patches=patches)


def trace_metadata(conversation_id: int) -> dict[str, str]:
    if not langfuse_enabled():
        return {}
    digest = hmac.new(
        settings.langfuse_secret_key.encode("utf-8"),
        f"mewhelp:conversation:{conversation_id}".encode("ascii"),
        hashlib.sha256,
    ).hexdigest()[:32]
    return {"langfuse_session_id": f"mewhelp-{digest}"}


def langfuse_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key
                and settings.langfuse_base_url)


def _init_client():
    """初始化(或复用)Langfuse 单例。"""
    global _client
    if _client is None:
        from langfuse import Langfuse

        # The SDK handles media before applying the span export mask.
        os.environ["LANGFUSE_MEDIA_UPLOAD_ENABLED"] = "false"
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
            should_export_span=_export_chat_span,
            mask_otel_spans=_mask_otel_spans,
        )
    return _client


def _make_handler():
    from langfuse.langchain import CallbackHandler

    class _InlineCallbackHandler(CallbackHandler):
        """run_inline=True:让 LangChain 在调用方协程内联执行回调,而不是丢进独立 task。
        没有它,handler 的 OTel context.attach 挂在别的 task 上,节点代码里没有活动
        span 上下文,tag_intent 的 propagate_attributes 写不进当前 trace(实测踩坑)。"""
        run_inline = True

    return _InlineCallbackHandler(public_key=settings.langfuse_public_key)


def get_langfuse():
    """启用时返回已初始化 client(cost 脚本用 client.api 查询);未启用返回 None。"""
    if not langfuse_enabled():
        return None
    return _init_client()


def attach_observability(graph):
    """编译后挂一次 Langfuse 回调,全图自动 trace(README:「编译时挂一次,节点里一行不用动」)。
    未配置时原样返回——不挂、不 import langfuse。"""
    if not langfuse_enabled():
        return graph
    try:
        _init_client()
        return graph.with_config({"callbacks": [_make_handler()]})
    except Exception as exc:
        logger.warning("Langfuse 初始化失败，已跳过追踪 (%s)", type(exc).__name__)
        return graph


def tag_intent(intent: str, confidence: float) -> None:
    """把意图写进当前 trace 的 metadata + tag(Cost Control 按意图分堆的钩子)。
    任何异常静默降级——观测失败不许影响业务流。"""
    if not langfuse_enabled() or _client is None:
        return
    try:
        from langfuse import propagate_attributes
        # 不能只 with...pass 写"当前活动 span"——LangGraph 异步调度下它可能已 ended,
        # OTel 对 ended span 的写入静默丢弃(实测踩坑)。改为在传播上下文内新建一个
        # event 观测:新 span 必然存活,trace 级属性(metadata/tags)随它上提到整条 trace。
        with propagate_attributes(
            metadata={"intent": intent, "intent_confidence": f"{confidence:.2f}"},
            tags=[f"intent:{intent}"],
        ):
            _client.create_event(name="intent-tagged")
    except Exception as exc:
        logger.warning("Langfuse 意图标记失败，已跳过 (%s)", type(exc).__name__)
