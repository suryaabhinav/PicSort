def test_move_endpoint(client, mock_move_artifact):
    # We need a run in registry that has completed successfully for move to work
    # So we need to manually inject a completed run into the registry or mock registry

    # Option 1: Mock registry.get to return a mock run object
    # This requires knowing internal structure of registry

    # Option 2: Start a run and simulate completion.
    # Since we are mocking the background task, the run object stays in memory.
    # We can fetch it and manually set its state.

    # Let's go with Option 2
    from api.progress import registry

    # 1. Start run
    payload = {"root": "/tmp/test_move"}
    resp = client.post("/api/pipeline/start", json=payload)
    run_id = resp.json()["run_id"]

    # 2. Get the run object and fake its state to "completed"
    run = registry.get(run_id)
    assert run is not None
    # The endpoint checks if 'final' is in run.paths
    run.paths["final"] = "/tmp/test_move/runs/final.parquet"

    # 3. Call move
    # dry_run=True by default for safety in check
    move_resp = client.post(f"/api/move/{run_id}?dry_run=true")
    assert move_resp.status_code == 200
    assert move_resp.json()["result"]["moved"] == 5

    # Verify mock was called
    mock_move_artifact.assert_called()


def test_move_missing_run(client):
    response = client.post("/api/move/bad-id")
    assert response.status_code == 404


def test_move_incomplete_run(client):
    # Start run but don't set paths
    payload = {"root": "/tmp/test_move_fail"}
    resp = client.post("/api/pipeline/start", json=payload)
    run_id = resp.json()["run_id"]

    response = client.post(f"/api/move/{run_id}")
    # Expect 400 because 'final' path is missing
    assert response.status_code == 400
