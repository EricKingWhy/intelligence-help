import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.api.auth import credentials
from app.db import repository
from app.graph import runtime
from app.main import app


def test_deployed_customer_and_operator_permissions(monkeypatch, tmp_path):
    accounts = {"customer-a": "customer", "customer-b": "customer", "agent": "operator"}
    password = "test-credential-only"
    path = tmp_path / "accounts.json"
    path.write_text(json.dumps({name: {
        "role": role, "sha256": hashlib.sha256(password.encode()).hexdigest(),
    } for name, role in accounts.items()}), encoding="utf-8")
    monkeypatch.setenv("MEWHELP_AUTH_MODE", "required")
    monkeypatch.setenv("MEWHELP_AUTH_FILE", str(path))
    credentials.cache_clear()
    try:
        monkeypatch.setattr(repository, "get_conversation",
                            AsyncMock(return_value=SimpleNamespace(user_id="customer-a")))
        monkeypatch.setattr(repository, "list_dialog_messages", AsyncMock(return_value=[]))
        list_conversations = AsyncMock(return_value=[])
        monkeypatch.setattr(repository, "list_conversations", list_conversations)

        async def stream(user_id, message, conversation_id):
            assert user_id == "customer-a"
            yield {"type": "done", "conversation_id": conversation_id}

        monkeypatch.setattr(runtime, "stream_turn", stream)
        client = TestClient(app)
        a, b, operator = ((name, password) for name in accounts)
        assert client.get("/").status_code == 401
        assert client.get("/api/conversations", params={"user_id": "customer-a"}, auth=a).is_success
        list_conversations.assert_awaited_once_with("customer-a")
        assert client.get("/api/conversations/1/messages", auth=a).is_success
        assert client.get("/api/conversations/1/messages", auth=b).status_code == 404
        chat = client.post("/api/chat", json={"user_id": "customer-a", "message": "你好",
                                             "conversation_id": 1}, auth=a)
        assert chat.is_success and '"conversation_id": 1' in chat.text
        assert client.get("/admin", auth=a).status_code == 403
        assert client.get("/admin", auth=operator).is_success
        assert client.get("/api/jobs", auth=operator).is_success
    finally:
        credentials.cache_clear()


async def test_authenticated_sidebar_matches_exact_account_ids(db_session_factory, monkeypatch, tmp_path):
    accounts = {
        "test-customer": ["较早的会话", "最近的会话"],
        "Test-Customer": ["独立会话"],
    }
    password = "fixture-credential"
    digest = hashlib.sha256(password.encode()).hexdigest()
    path = tmp_path / "accounts.json"
    path.write_text(json.dumps({
        user: {"role": "customer", "sha256": digest}
        for user in accounts
    }), encoding="utf-8")
    monkeypatch.setenv("MEWHELP_AUTH_MODE", "required")
    monkeypatch.setenv("MEWHELP_AUTH_FILE", str(path))
    expected = {}
    for user, previews in accounts.items():
        expected[user] = []
        for preview in previews:
            cid = await repository.create_conversation(user)
            await repository.append_message(cid, "user", content=preview)
            expected[user].insert(0, (cid, preview))

    credentials.cache_clear()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for user, conversations in expected.items():
                auth = (user, password)
                response = await client.get("/api/conversations", params={"user_id": user}, auth=auth)
                response.raise_for_status()
                assert [(item["id"], item["preview"]) for item in response.json()["items"]] == conversations
                for cid, preview in conversations:
                    detail = await client.get(f"/api/conversations/{cid}/messages", auth=auth)
                    detail.raise_for_status()
                    assert detail.json()["items"][0]["content"] == preview
    finally:
        credentials.cache_clear()
