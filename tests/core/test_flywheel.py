# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch09 飞轮核心:标准化+查重一次输出;命中累加不新建、不复活;幻觉 id 跳过;串行防同批重复。"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core import flywheel
from app.core.flywheel import NormalizeResult
from app.db import repository


def _stub_llm(monkeypatch, results):
    """按序弹出预设结果;可混入 Exception 模拟坏 JSON/解析失败。"""
    queue = list(results)

    async def fake(raw_question, candidates):
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(flywheel, "normalize_and_match", fake)


async def test_new_question_creates_pending_item(db_session_factory, monkeypatch):
    await repository.insert_low_confidence(None, "猫窝能水洗吗", "retrieval_low_conf", None)
    _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="猫窝是否支持水洗", matched_question_id=None,
        ai_suggested_answer="以平台售后规则为准")])
    stats = await flywheel.process_pending()
    assert stats == {"processed": 1, "merged": 0, "created": 1, "skipped": 0}
    rows = await repository.list_review_queue("待审")
    assert rows[0].normalized_question == "猫窝是否支持水洗"
    assert not await repository.fetch_unmatched_low_conf(10)   # 游标已推进


async def test_match_merges_and_never_revives(db_session_factory, monkeypatch):
    rid = await repository.insert_review_item("猫窝是否支持水洗", None)
    await repository.update_review_status(rid, "驳回")        # 已驳回=终审
    await repository.insert_low_confidence(None, "猫窝可以洗吗", "user_feedback", None)
    _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="猫窝是否支持水洗", matched_question_id=rid, ai_suggested_answer="")])
    stats = await flywheel.process_pending()
    assert stats["merged"] == 1 and stats["created"] == 0
    item, raws = await repository.get_review_detail(rid)
    assert item.occurrence_count == 2
    assert item.review_status == "驳回"                        # 只累加,不复活
    assert [r.raw_question for r in raws] == ["猫窝可以洗吗"]   # 归并落点记回原话


async def test_hallucinated_id_skips_and_keeps_cursor(db_session_factory, monkeypatch):
    await repository.insert_low_confidence(None, "猫窝可以洗吗", "self_check", None)
    _stub_llm(monkeypatch, [NormalizeResult(
        normalized_question="猫窝是否支持水洗", matched_question_id=99999, ai_suggested_answer="")])
    stats = await flywheel.process_pending()
    assert stats["skipped"] == 1 and stats["processed"] == 0
    assert len(await repository.fetch_unmatched_low_conf(10)) == 1   # 游标留在原地,下轮重试
    assert await repository.list_review_queue(None) == []


async def test_llm_failure_skips_row(db_session_factory, monkeypatch):
    await repository.insert_low_confidence(None, "问题一", "self_check", None)
    await repository.insert_low_confidence(None, "问题二", "self_check", None)
    _stub_llm(monkeypatch, [RuntimeError("坏 JSON"),
                            NormalizeResult(normalized_question="问题二标准版",
                                            matched_question_id=None, ai_suggested_answer="")])
    stats = await flywheel.process_pending()
    assert stats == {"processed": 1, "merged": 0, "created": 1, "skipped": 1}


async def test_same_batch_duplicates_merge_serially(db_session_factory, monkeypatch):
    """同批两条同义:串行处理,第二条的候选里已有第一条刚建的行 → 归并而非重复建行。"""
    await repository.insert_low_confidence(None, "猫窝能水洗吗", "retrieval_low_conf", None)
    await repository.insert_low_confidence(None, "猫窝可以洗吗", "user_feedback", None)
    seen_candidates = []

    async def fake(raw_question, candidates):
        seen_candidates.append(list(candidates))
        if candidates:
            return NormalizeResult(normalized_question="猫窝是否支持水洗",
                                   matched_question_id=candidates[0]["id"], ai_suggested_answer="")
        return NormalizeResult(normalized_question="猫窝是否支持水洗",
                               matched_question_id=None, ai_suggested_answer="")
    monkeypatch.setattr(flywheel, "normalize_and_match", fake)
    stats = await flywheel.process_pending()
    assert stats == {"processed": 2, "merged": 1, "created": 1, "skipped": 0}
    assert seen_candidates[0] == [] and len(seen_candidates[1]) == 1
    rows = await repository.list_review_queue(None)
    assert len(rows) == 1 and rows[0].occurrence_count == 2


@pytest.mark.parametrize("existing_status", [None, "通过", "驳回"])
async def test_candidate_and_source_roll_back_together(db_session_factory, monkeypatch, existing_status):
    review_id = None
    if existing_status:
        review_id = await repository.insert_review_item("猫窝能否清洗", None)
        await repository.update_review_status(review_id, existing_status)
    await repository.insert_low_confidence(None, "猫窝怎么洗", "self_check", None)
    result = NormalizeResult(normalized_question="猫窝能否清洗", matched_question_id=review_id)
    _stub_llm(monkeypatch, [result, result])

    with monkeypatch.context() as failed_write:
        failed_write.setattr(repository, "set_matched_review", AsyncMock(side_effect=RuntimeError("cursor failed")))
        with pytest.raises(RuntimeError, match="cursor failed"):
            await flywheel.process_pending()

    old = await repository.list_review_queue(None)
    assert [(row.id, row.review_status, row.occurrence_count) for row in old] == (
        [(review_id, existing_status, 1)] if existing_status else []
    )
    assert len(await repository.fetch_unmatched_low_conf(10)) == 1

    assert (await flywheel.process_pending())["processed"] == 1
    [candidate] = await repository.list_review_queue(None)
    assert candidate.review_status == (existing_status or "待审")
    assert candidate.occurrence_count == (2 if existing_status else 1)
    assert await repository.fetch_unmatched_low_conf(10) == []


@pytest.mark.parametrize("has_candidate", [False, True])
async def test_two_batches_apply_one_update_for_the_same_source(db_session_factory, monkeypatch, has_candidate):
    review_id = await repository.insert_review_item("猫窝能否清洗", None) if has_candidate else None
    await repository.insert_low_confidence(None, "猫窝怎么洗", "self_check", None)
    model_calls = 0
    both_picked_source = asyncio.Event()

    async def both_finish_matching(question, candidates):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 2:
            both_picked_source.set()
        await asyncio.wait_for(both_picked_source.wait(), 5)
        return NormalizeResult(normalized_question="猫窝能否清洗", matched_question_id=review_id)

    monkeypatch.setattr(flywheel, "normalize_and_match", both_finish_matching)
    batches = await asyncio.gather(flywheel.process_pending(), flywheel.process_pending())
    assert sum(batch["processed"] for batch in batches) == 1
    [candidate] = await repository.list_review_queue(None)
    assert candidate.occurrence_count == (2 if has_candidate else 1)
    assert await repository.fetch_unmatched_low_conf(10) == []
