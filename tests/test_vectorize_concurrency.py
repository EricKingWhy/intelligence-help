import asyncio
from threading import Event

import pytest

from app.db import repository
from app.kb import dualwrite, milvus_client
from app.kb.documents import Chunk


class RecordingIndex:
    def __init__(self):
        self.rows = {}
        self.writes = []

    def upsert(self, collection, rows):
        for row in rows:
            self.rows[row["id"]] = row
            self.writes.append(row["answer"])

    def flush(self, collection):
        pass


async def add_policy():
    return (await dualwrite.write_pending([
        Chunk(category="保修", questions="保修期", answer="旧政策", section_path="政策 / 保修", content_type="policy"),
    ]))[0]


@pytest.mark.parametrize("cancel_publish", [False, True])
async def test_edit_waits_for_in_flight_flush_and_preserves_pending(
    db_session_factory, monkeypatch, cancel_publish,
):
    chunk_id = await add_policy()
    in_flush = asyncio.Event()
    release_flush = Event()
    loop = asyncio.get_running_loop()

    class PausedIndex(RecordingIndex):
        def flush(self, collection):
            loop.call_soon_threadsafe(in_flush.set)
            if not release_flush.wait(5):
                raise RuntimeError("flush test barrier timed out")

    async def embed(texts):
        return [[0.05]] * len(texts)

    index = PausedIndex()
    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", embed)
    monkeypatch.setattr(milvus_client, "ensure_collection", lambda *args, **kwargs: None)

    publish = asyncio.create_task(dualwrite.vectorize_pending(index, chunk_ids=[chunk_id]))
    edit = None
    try:
        await asyncio.wait_for(in_flush.wait(), 5)
        if cancel_publish:
            publish.cancel()
        edit = asyncio.create_task(repository.repend_chunk_text(chunk_id, "保修期", "新政策"))
        # The database write must wait even if the request awaiting the threaded flush is cancelled.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(edit), 0.2)
    finally:
        release_flush.set()
        if cancel_publish:
            with pytest.raises(asyncio.CancelledError):
                await publish
        else:
            assert await publish == 1
        if edit is not None:
            await asyncio.wait_for(edit, 5)

    [pending] = await repository.list_pending_chunks([chunk_id])
    assert pending.answer == "新政策"
    assert await dualwrite.vectorize_pending(index, chunk_ids=[chunk_id]) == 1
    assert index.writes == ["旧政策", "新政策"]
