def test_health_returns_ok_status(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"


def test_health_reports_vector_store_not_ready_initially(client):
    response = client.get("/health")
    assert response.json()["vector_store_ready"] is False
