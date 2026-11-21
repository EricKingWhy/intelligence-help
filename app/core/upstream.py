"""Normalize external failures into customer-safe recovery categories."""
import httpx
from langchain_core.exceptions import OutputParserException
from openai import APIError, APITimeoutError
from pymilvus.exceptions import MilvusException

UPSTREAM_ERRORS = (TimeoutError, httpx.HTTPError, APIError, MilvusException, OutputParserException)


def failure(exc: Exception, stage: str) -> dict[str, str]:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    if status == 429:
        kind = "rate_limited"
    elif isinstance(exc, (TimeoutError, httpx.TimeoutException, APITimeoutError)) or status in {408, 504}:
        kind = "timeout"
    elif isinstance(exc, MilvusException):
        kind = "knowledge_unavailable"
    else:
        kind = "unavailable"
    return {"stage": stage, "kind": kind}
