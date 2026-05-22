from fastapi.testclient import TestClient

from traffic_law_v2.api import create_app


def test_auth_register_login_me() -> None:
    client = TestClient(create_app())
    email = "duc@example.com"
    registered = client.post("/api/v1/auth/register", json={"email": email, "password": "secret123"})
    assert registered.status_code == 200
    token = registered.json()["access_token"]
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["user"]["email"] == email
    login = client.post("/api/v1/auth/login", json={"email": email, "password": "secret123"})
    assert login.status_code == 200


def test_thread_memory_flow() -> None:
    client = TestClient(create_app())
    created = client.post("/api/v1/threads", params={"user_id": "u1", "title": "Smoke"}).json()
    thread_id = created["id"]
    first = client.post(
        f"/api/v1/threads/{thread_id}/chat",
        json={"user_id": "u1", "query": "tôi tên là Đức", "index_dir": "data/index", "top_k": 3},
    )
    assert first.status_code == 200
    second = client.post(
        f"/api/v1/threads/{thread_id}/chat",
        json={"user_id": "u1", "query": "tôi tên là gì?", "index_dir": "data/index", "top_k": 3},
    )
    assert second.status_code == 200
    body = second.json()
    assert "answer" in body
    assert isinstance(body["answer"], str)
