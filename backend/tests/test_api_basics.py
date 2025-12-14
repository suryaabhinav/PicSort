def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_status_404_not_found(client):
    # UUID pattern but not existing
    run_id = "non-existent-run-id"
    response = client.get(f"/api/runs/{run_id}/status")
    assert response.status_code == 404
    assert response.json() == {"detail": "Run not found"}
