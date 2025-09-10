from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

def register_and_login(email: str, password: str) -> str:
    # Attempt to register (may fail if already registered)
    reg_resp = client.post("/api/register", json={"email": email, "password": password})
    assert reg_resp.status_code in (201, 400)
    # Login
    login_resp = client.post("/api/login", json={"email": email, "password": password})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    assert token
    return token

def test_full_flow() -> None:
    email = "test@example.com"
    password = "secret123"
    token = register_and_login(email, password)
    headers = {"Authorization": f"Bearer {token}"}

    # Create two predictions: v1 and v2
    p1 = client.post(
        "/api/predictions",
        headers=headers,
        json={"text": "Hello world!", "model_version": "v1"},
    )
    assert p1.status_code == 200, p1.text
    j1 = p1.json()
    assert j1["model_version"] == "v1"
    assert j1["label"] in ("POSITIVE", "NEGATIVE")
    assert isinstance(j1["score"], float)

    p2 = client.post(
        "/api/predictions",
        headers=headers,
        json={"text": "Another input", "model_version": "v2"},
    )
    assert p2.status_code == 200, p2.text
    j2 = p2.json()
    assert j2["model_version"] == "v2"

    # List predictions
    listing = client.get("/api/predictions", headers=headers)
    assert listing.status_code == 200
    arr = listing.json()
    assert len(arr) >= 2
    # Ensure each record has required keys
    for rec in arr:
        assert "id" in rec
        assert rec["model_version"] in ("v1", "v2")
        assert isinstance(rec["input_data"], dict)
        assert isinstance(rec["output_data"], dict)

    # Get prediction by ID
    pred_id = arr[0]["id"]
    get_one = client.get(f"/api/predictions/{pred_id}", headers=headers)
    assert get_one.status_code == 200
    rec = get_one.json()
    assert rec["id"] == pred_id

    # Stats
    stats_resp = client.get("/api/stats", headers=headers)
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total"] >= 2
    assert "by_class" in stats and "by_model_version" in stats

if __name__ == "__main__":
    test_full_flow()
    print("Success")