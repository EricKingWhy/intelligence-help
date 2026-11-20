# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
from app.core import embeddings
from app.db import repository
from app.kb import milvus_client
from app.kb.documents import Chunk


async def write_pending(chunks: list[Chunk]) -> list[int]:
    """按顺序把一份文档的 chunks 写入 MySQL(status=pending),并在同文档内连 prev/next 指针。"""
    ids: list[int] = []
    for c in chunks:
        cid = await repository.insert_knowledge_chunk(
            c.category, c.questions, c.answer,
            section_path=c.section_path, content_type=c.content_type,
            is_key_clause=c.is_key_clause,
        )
        ids.append(cid)
    for i, cid in enumerate(ids):
        prev_id = ids[i - 1] if i > 0 else None
        next_id = ids[i + 1] if i < len(ids) - 1 else None
        await repository.set_chunk_neighbors(cid, prev_id, next_id)
    return ids


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


async def vectorize_pending(
    client=None, batch_size: int = 64, collection: str = milvus_client.COLLECTION,
    *, chunk_ids: list[int] | None = None,
) -> int:
    """幂等可重跑:取 pending → 拼 category+questions+answer → 嵌入 + BM25 text → Milvus upsert(PK=id)
    → 回填 vector_id、status=done。
    ch09:client 缺省时在 Milvus 专用线程内惰性创建,upsert/flush 也走 acall——
    server 进程(审核通过写回)与脚本两个场景共用一条安全路径(gRPC 客户端跨线程
    共享在 async server 下会间歇空返回,见 retrieval.py 注)。"""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pending = (await repository.list_pending_chunks() if chunk_ids is None
               else await repository.list_pending_chunks(chunk_ids))
    done = 0
    for batch in _batches(pending, batch_size):
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await embeddings.embed_texts(texts)
        if len(vectors) != len(batch):
            raise ValueError(f"Expected {len(batch)} embeddings, received {len(vectors)}")
        rows = [
            {"id": r.id, "dense": v, "text": t,
             "question": r.questions, "answer": r.answer,
             "section_path": r.section_path or "", "content_type": r.content_type or "",
             "category": r.category or ""}
            for r, v, t in zip(batch, vectors, texts)
        ]

        def write_batch():
            c = client or milvus_client.get_client()
            milvus_client.ensure_collection(c, collection=collection)
            milvus_client.upsert_vectors(c, rows, collection=collection)
            milvus_client.flush(c, collection=collection)
        await milvus_client.acall(write_batch)
        for r in batch:
            await repository.mark_chunk_vectorized(r.id, str(r.id))
        done += len(batch)
    return done
