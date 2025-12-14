def test_start_pipeline(client):
    payload = {"root": "/tmp/test_images", "batch_size": 16, "dry_run": True, "run_faces": False}
    response = client.post("/api/pipeline/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert len(data["run_id"]) > 0


def test_start_pipeline_invalid_payload(client):
    # Missing root
    payload = {"batch_size": 16}
    response = client.post("/api/pipeline/start", json=payload)
    assert response.status_code == 422


def test_pipeline_execution_mocked(client, mock_pipeline_background):
    # Start a run
    payload = {"root": "/tmp/images"}
    response = client.post("/api/pipeline/start", json=payload)
    run_id = response.json()["run_id"]

    # Check status - initially it might be created or running
    status_response = client.get(f"/api/runs/{run_id}/status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["run_id"] == run_id

    # Since background tasks run in threadpool/processpool in FastAPI,
    # and we mocked the function, we can check if our mock was called.
    # Note: FastAPI BackgroundTasks might not complete instantly in test client without block.
    # However, since we patched it, we just verify the call if possible or side effects.
    # For now, let's verify status structure is correct.
    assert "stage" in status_data
    assert "progress" in status_data


def test_stop_pipeline(client):
    # Create a run first
    payload = {"root": "/tmp/images"}
    run_resp = client.post("/api/pipeline/start", json=payload)
    run_id = run_resp.json()["run_id"]

    # Stop it
    stop_resp = client.post(f"/api/pipeline/stop/{run_id}")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "cancelling"


def test_stop_pipeline_404(client):
    response = client.post("/api/pipeline/stop/non-existent-id")
    assert response.status_code == 404


def test_stream_events_404(client):
    response = client.get("/api/runs/bad-id/events")
    assert response.status_code == 404
