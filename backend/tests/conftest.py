import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path to allow imports from backend
root_path = Path(__file__).resolve().parent.parent
if str(root_path.parent) not in sys.path:
    sys.path.append(str(root_path.parent))

# Mock the entire app.py module's dependencies BEFORE importing app
# This is crucial because app.py imports these at the top level
from api import app as app_module


# Create a TestClient fixture
@pytest.fixture(scope="module")
def client():
    from api.app import app

    with TestClient(app) as c:
        yield c


# Mock background tasks to avoid running actual ML pipeline
@pytest.fixture(autouse=True)
def mock_pipeline_background():
    with patch("api.app.run_pipeline_background") as mock:
        # returns simple analytics dict
        mock.return_value = {"total": {"images": 10}, "status": "mocked_success"}
        yield mock


# Mock move operation
@pytest.fixture(autouse=True)
def mock_move_artifact():
    with patch("api.app.move_with_run_artifact") as mock:
        mock.return_value = {"moved": 5}
        yield mock


# Mock Path.mkdir to avoid creating directories during tests
@pytest.fixture(autouse=True)
def mock_fs_mkdir():
    with patch("pathlib.Path.mkdir") as mock:
        yield mock


# Mock logging to clean up output
@pytest.fixture(autouse=True)
def mock_logging():
    with patch("api.app.RotatingFileHandler"), patch("api.app.get_logger"):
        yield
