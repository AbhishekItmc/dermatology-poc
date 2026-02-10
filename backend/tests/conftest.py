"""
Pytest configuration and fixtures
"""
import pytest
from fastapi.testclient import TestClient

# Temporarily commented out to allow testing without full dependencies
# from app.main import app


# @pytest.fixture
# def client():
#     """Test client fixture"""
#     return TestClient(app)


@pytest.fixture
def sample_image_set():
    """Sample image set for testing"""
    # TODO: Implement sample image generation
    return []
