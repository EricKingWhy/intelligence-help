import asyncio
import hashlib
import json

from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage

from app.api.auth import credentials
from app.db import repository
from app.graph import nodes
from app.main import app


async def test_confirmed_handoff_can_be_claimed_and_resolved(db_session_factory, monkeypatch, tmp_path):
    accounts = {"customer-a": "customer", "customer-b": "customer",
                "operator-a": "operator", "operator-b": "operator"}
    password = "synthetic-test-password"
    accounts_path = tmp_path / "users.json"
    accounts_path.write_text(json.dumps({name: {
        "role": role, "sha256": hashlib.sha256(password.encode()).hexdigest(),
    } for name, role in accounts.items()}), encoding="utf-8")
    monkeypatch.setenv("MEWHELP_AUTH_MODE", "required")
    monkeypatch.setenv("MEWHELP_AUTH_FILE", str(accounts_path))
    credentials.cache_clear()

    cid = await repository.create_conversation("customer-a")
    message_id = await repository.append_message(cid, "user", "订单1001一直没解决")
    state = {"conversation_id": cid, "user_id": "customer-a", "intent": "投诉",
             "messages": [HumanMessage("订单1001一直没解决", id=f"db-{message_id}")],
             "retrieved_snapshot": [{"section_path": "售后规则", "answer": "人工复核"}]}
    await nodes.log_node({**state, **await nodes.complaint_reply(state)})
    next_id = await repository.append_message(cid, "user", "后来又问了别的")
    await repository.create_ticket(cid, "普通工单", "咨询")

    a, b, op1, op2 = ((name, password) for name in accounts)
    customer_url = f"/api/conversations/{cid}/handoffs"
    queue_url = "/api/operator/handoffs"
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as request:
            assert (await request.get(queue_url, auth=a)).status_code == 403
            assert (await request.get(customer_url, auth=a)).json()["items"] == []
            assert (await request.get(queue_url, auth=op1)).json()["items"] == []
            assert (await request.post(customer_url, auth=a, json={"message_id": next_id})).status_code == 409
            proposed = {"message_id": message_id}
            assert (await request.post(customer_url, auth=b, json=proposed)).status_code == 404
            first, duplicate = await asyncio.gather(
                request.post(customer_url, auth=a, json=proposed),
                request.post(customer_url, auth=a, json=proposed),
            )
            assert first.status_code == duplicate.status_code == 200
            ticket = first.json()
            assert duplicate.json()["ticket_no"] == ticket["ticket_no"]
            assert ticket["description"] == "订单1001一直没解决" and ticket["status"] == "待接单"
            ref = ticket["ticket_no"]
            assert "context" not in ticket

            item = (await request.get(queue_url, auth=op1)).json()["items"][0]
            assert item["context"]["order"]["order_id"] == "1001"
            assert item["context"]["evidence"][0]["section_path"] == "售后规则"
            claim_url = f"{queue_url}/{ref}/claim"
            claims = await asyncio.gather(request.post(claim_url, auth=op1),
                                          request.post(claim_url, auth=op2))
            assert sorted(r.status_code for r in claims) == [200, 409]
            winner, loser = (op1, op2) if claims[0].is_success else (op2, op1)
            customer = (await request.get(customer_url, auth=a)).json()["items"][0]
            assert customer["status"] == "处理中"
            resolve = f"{queue_url}/{ref}/resolve"
            body = {"resolution": "已核实并跟进。"}
            assert (await request.post(resolve, auth=loser, json=body)).status_code == 409
            assert (await request.post(resolve, auth=winner, json=body)).is_success
            assert (await request.post(resolve, auth=winner, json=body)).is_success
            closed = (await request.get(customer_url, auth=a)).json()["items"][0]
            assert (closed["status"], closed["resolution"]) == ("已处理", body["resolution"])
    finally:
        credentials.cache_clear()
