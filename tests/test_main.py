"""Tests for the main FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from cogniforge.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Welcome to CogniForge!"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "cogniforge"


def test_info_endpoint(client):
    """Test the info endpoint that shows library versions."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    
    # Check that libraries section exists
    assert "libraries" in data
    libraries = data["libraries"]
    
    # Check that all expected libraries are present
    expected_libraries = [
        "numpy", "torch", "pybullet", 
        "gymnasium", "pillow", "stable_baselines3"
    ]
    for lib in expected_libraries:
        assert lib in libraries
    
    # Check CUDA information is present
    assert "cuda_available" in data
    assert "torch_device" in data
    assert isinstance(data["cuda_available"], bool)