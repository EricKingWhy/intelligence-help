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


def _vector_fields(chunk) -> dict:
    return {
        "id": chunk.id,
        "text": f"{chunk.category}\n{chunk.questions}\n{chunk.answer}",
        "question": chunk.questions,
        "answer": chunk.answer,
        "section_path": chunk.section_path or "",
        "content_type": chunk.content_type or "",
        "category": chunk.category or "",
    }


def _write_vectors(client, rows: list[dict], collection: str) -> None:
    client = client or milvus_client.get_client()
    milvus_client.ensure_collection(client, collection=collection)
    milvus_client.upsert_vectors(client, rows, collection=collection)
    milvus_client.flush(client, collection=collection)


async def vectorize_pending(
    client=None, batch_size: int = 64, collection: str = milvus_client.COLLECTION,
    *, chunk_ids: list[int] | None = None,
) -> int:
    """Embed pending snapshots, then publish only rows that are still current."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pending = (await repository.list_pending_chunks() if chunk_ids is None
               else await repository.list_pending_chunks(chunk_ids))
    done = 0
    for batch in _batches(pending, batch_size):
        snapshots = [_vector_fields(chunk) for chunk in batch]
        vectors = await embeddings.embed_texts([snapshot["text"] for snapshot in snapshots])
        if len(vectors) != len(batch):
            raise ValueError(f"Expected {len(batch)} embeddings, received {len(vectors)}")

        async with repository.lock_pending_chunks([chunk.id for chunk in batch]) as (session, current):
            rows = []
            for snapshot, vector in zip(snapshots, vectors):
                row = current.get(snapshot["id"])
                if row is not None and snapshot == _vector_fields(row):
                    rows.append({**snapshot, "dense": vector})
            if not rows:
                continue

            await milvus_client.acall(_write_vectors, client, rows, collection, wait_on_cancel=True)
            for row in rows:
                await repository.mark_chunk_vectorized(row["id"], str(row["id"]), session=session)
        done += len(rows)
    return done
