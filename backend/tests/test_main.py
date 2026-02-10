"""
Tests for main application
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data


def test_api_docs_accessible(client):
    """Test that API documentation is accessible"""
    response = client.get("/api/v1/docs")
    assert response.status_code == 200
