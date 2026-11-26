# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import asyncio
import uuid

import pytest

from app.db import repository
from app.kb import dualwrite, milvus_client
from app.kb.documents import Chunk

URI = "http://localhost:19530"


def _chunk(q, a):
    return Chunk(category="c", questions=q, answer=a, section_path="c / " + q, content_type="policy")


@pytest.fixture()
def milvus():
    client = milvus_client.get_client(uri=URI)
    name = f"test_kb_{uuid.uuid4().hex[:8]}"
    milvus_client.ensure_collection(client, collection=name)
    yield client, name
    milvus_client.drop(client, name)


async def test_write_pending_links_neighbors(db_session_factory):
    ids = await dualwrite.write_pending([_chunk("q1", "a1"), _chunk("q2", "a2"), _chunk("q3", "a3")])
    assert len(ids) == 3
    by_id = {c.id: c for c in await repository.list_pending_chunks()}
    assert by_id[ids[1]].prev_chunk_id == ids[0]
    assert by_id[ids[1]].next_chunk_id == ids[2]
    assert by_id[ids[0]].prev_chunk_id is None


@pytest.mark.parametrize("stage", ["upsert_vectors", "flush", "mark_chunk_vectorized"])
async def test_vector_write_failure_can_be_retried(db_session_factory, milvus, monkeypatch, stage):
    client, name = milvus
    ids = await dualwrite.write_pending([_chunk("保修政策", "整机保修一年"), _chunk("其他", "待处理")])

    async def embed(texts):
        return [[0.05] * milvus_client.DIM for _ in texts]

    target = repository if stage == "mark_chunk_vectorized" else milvus_client
    original = getattr(target, stage)

    def fail(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", embed)
    monkeypatch.setattr(target, stage, fail)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        await dualwrite.vectorize_pending(client, collection=name, chunk_ids=ids[:1])
    assert await repository.count_chunks_by_status("pending") == 2

    monkeypatch.setattr(target, stage, original)
    assert await dualwrite.vectorize_pending(client, collection=name, chunk_ids=ids[:1]) == 1
    assert await repository.count_chunks_by_status("pending") == 1
    client.load_collection(name)
    for _ in range(10):
        hits = await milvus_client.acall(
            lambda: milvus_client.bm25_search(client, "保修政策", top_k=2, collection=name),
        )
        if ids[0] in [hit["id"] for hit in hits]:
            break
        await asyncio.sleep(0.3)
    else:
        pytest.fail("Recovered knowledge chunk did not become searchable")


async def test_vectorize_resumes_after_crash(db_session_factory, milvus, monkeypatch):
    client, name = milvus
    await dualwrite.write_pending([_chunk(f"q{i}", f"a{i}") for i in range(4)])

    # 第一趟:embed 在第 2 批抛错(batch_size=2 → 前 2 条 done,后 2 条仍 pending)
    calls = {"n": 0}

    async def flaky_embed(texts):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("嵌入服务中断")
        return [[float(i), 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3) for i, _ in enumerate(texts)]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", flaky_embed)
    with pytest.raises(RuntimeError):
        await dualwrite.vectorize_pending(client, batch_size=2, collection=name)
    assert await repository.count_chunks_by_status("done") == 2
    assert await repository.count_chunks_by_status("pending") == 2

    # 第二趟:恢复正常 → 只捡剩下 2 条 pending 补齐;Milvus 按 id upsert 无重
    async def ok_embed(texts):
        return [[9.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3) for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", ok_embed)
    done_now = await dualwrite.vectorize_pending(client, batch_size=2, collection=name)
    assert done_now == 2
    assert await repository.count_chunks_by_status("pending") == 0
    assert await repository.count_chunks_by_status("done") == 4

    client.load_collection(name)
    import time; time.sleep(2)  # 等新段对 query 可见
    assert milvus_client.count(client, collection=name) == 4  # 无重无漏


@pytest.mark.parametrize("new_finishes_first", [False, True])
async def test_edited_policy_is_searchable_even_when_old_embedding_returns_last(
    db_session_factory, milvus, monkeypatch, new_finishes_first,
):
    client, name = milvus
    [chunk_id] = await dualwrite.write_pending([_chunk("保修期限", "旧政策保修一年")])
    old_embedding = asyncio.Event()
    release_old = asyncio.Event()

    async def embed(texts):
        if texts[0].endswith("旧政策保修一年"):
            old_embedding.set()
            await release_old.wait()
        return [[0.05] * milvus_client.DIM for _ in texts]

    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", embed)
    old_task = asyncio.create_task(dualwrite.vectorize_pending(client, collection=name, chunk_ids=[chunk_id]))
    try:
        await asyncio.wait_for(old_embedding.wait(), 5)
        await asyncio.wait_for(repository.repend_chunk_text(chunk_id, "保修期限", "新政策保修两年"), 5)
        if new_finishes_first:
            assert await dualwrite.vectorize_pending(client, collection=name, chunk_ids=[chunk_id]) == 1
    finally:
        release_old.set()
        old_count = await old_task

    assert old_count == 0
    if not new_finishes_first:
        [pending] = await repository.list_pending_chunks([chunk_id])
        assert pending.answer == "新政策保修两年"
        assert await dualwrite.vectorize_pending(client, collection=name, chunk_ids=[chunk_id]) == 1

    assert await repository.list_pending_chunks([chunk_id]) == []
    await milvus_client.acall(client.load_collection, name)
    for _ in range(10):
        hits = await milvus_client.acall(milvus_client.bm25_search, client, "保修期限", 2, collection=name)
        if hits:
            assert [(hit["id"], hit["answer"]) for hit in hits] == [(chunk_id, "新政策保修两年")]
            break
        await asyncio.sleep(0.3)
    else:
        pytest.fail("Updated policy did not become searchable")
