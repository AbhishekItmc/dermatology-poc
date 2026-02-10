"""
Tests for API endpoints
"""
import pytest
import numpy as np
import cv2
from io import BytesIO
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from app.main import app
from app.core.storage import ImageStorage


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_storage():
    """Create mock storage"""
    storage = Mock(spec=ImageStorage)
    storage.save_image_set = Mock(return_value=True)
    storage.load_image_set = Mock(return_value=[])
    return storage


@pytest.fixture
def sample_image_bytes():
    """Create sample image as bytes"""
    # Create a simple test image
    image = np.ones((512, 512, 3), dtype=np.uint8) * 180
    cv2.circle(image, (256, 256), 100, (200, 170, 150), -1)
    
    # Encode to JPEG
    success, encoded = cv2.imencode('.jpg', image)
    return encoded.tobytes()


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_api_docs_accessible(client):
    """Test that API documentation is accessible"""
    response = client.get("/api/v1/docs")
    assert response.status_code == 200


def test_upload_images_success(client, sample_image_bytes, mock_storage):
    """Test successful image upload"""
    # Login to get token
    login_response = client.post(
        "/api/v1/auth/login",
        json={"username": "clinician", "password": "clinician123"}
    )
    token = login_response.json()["access_token"]
    
    with patch('app.api.v1.endpoints.patients.get_image_storage', return_value=mock_storage):
        # Create file upload data
        files = [
            ("images", ("image1.jpg", BytesIO(sample_image_bytes), "image/jpeg")),
            ("images", ("image2.jpg", BytesIO(sample_image_bytes), "image/jpeg")),
            ("images", ("image3.jpg", BytesIO(sample_image_bytes), "image/jpeg")),
        ]
        
        response = client.post(
            "/api/v1/patients/test_patient_001/images",
            files=files,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "test_patient_001"
        assert data["image_count"] == 3
        assert data["status"] == "uploaded"
        assert "image_set_id" in data


def test_upload_images_insufficient_count(client, sample_image_bytes, mock_storage):
    """Test image upload with insufficient images"""
    # Login to get token
    login_response = client.post(
        "/api/v1/auth/login",
        json={"username": "clinician", "password": "clinician123"}
    )
    token = login_response.json()["access_token"]
    
    with patch('app.api.v1.endpoints.patients.get_image_storage', return_value=mock_storage):
        # Only 2 images (minimum is 3)
        files = [
            ("images", ("image1.jpg", BytesIO(sample_image_bytes), "image/jpeg")),
            ("images", ("image2.jpg", BytesIO(sample_image_bytes), "image/jpeg")),
        ]
        
        response = client.post(
            "/api/v1/patients/test_patient_001/images",
            files=files,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 400
        assert "At least 3 images required" in response.json()["detail"]


def test_upload_images_too_many(client, sample_image_bytes, mock_storage):
    """Test image upload with too many images"""
    # Login to get token
    login_response = client.post(
        "/api/v1/auth/login",
        json={"username": "clinician", "password": "clinician123"}
    )
    token = login_response.json()["access_token"]
    
    with patch('app.api.v1.endpoints.patients.get_image_storage', return_value=mock_storage):
        # 11 images (maximum is 10)
        files = [
            ("images", (f"image{i}.jpg", BytesIO(sample_image_bytes), "image/jpeg"))
            for i in range(11)
        ]
        
        response = client.post(
            "/api/v1/patients/test_patient_001/images",
            files=files,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 400
        assert "Maximum 10 images allowed" in response.json()["detail"]


def test_create_analysis(client, mock_storage):
    """Test creating an analysis"""
    # Login to get token
    login_response = client.post(
        "/api/v1/auth/login",
        json={"username": "clinician", "password": "clinician123"}
    )
    token = login_response.json()["access_token"]
    
    # Mock the analysis service and background task processing
    with patch('app.api.v1.endpoints.analyses.get_analysis_service') as mock_service_getter:
        with patch('app.api.v1.endpoints.analyses.get_image_storage', return_value=mock_storage):
            with patch('app.api.v1.endpoints.analyses.process_analysis') as mock_process:
                # Create mock analysis service
                mock_service = MagicMock()
                mock_service_getter.return_value = mock_service
                
                # Create analysis request
                response = client.post(
                    "/api/v1/analyses/",
                    json={
                        "patient_id": "test_patient_001",
                        "image_set_id": "test_image_set_001"
                    },
                    headers={"Authorization": f"Bearer {token}"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "analysis_id" in data
                assert data["status"] == "queued"
                assert data["progress"] == 0.0


def test_get_analysis_status(client):
    """Test getting analysis status"""
    # Login to get token
    login_response = client.post(
        "/api/v1/auth/login",
        json={"username": "clinician", "password": "clinician123"}
    )
    token = login_response.json()["access_token"]
    
    # First create an analysis
    with patch('app.api.v1.endpoints.analyses.get_analysis_service') as mock_service_getter:
        with patch('app.api.v1.endpoints.analyses.get_image_storage') as mock_storage_getter:
            with patch('app.api.v1.endpoints.analyses.process_analysis') as mock_process:
                mock_service = MagicMock()
                mock_storage = Mock(spec=ImageStorage)
                mock_service_getter.return_value = mock_service
                mock_storage_getter.return_value = mock_storage
                
                # Create analysis
                response = client.post(
                    "/api/v1/analyses/",
                    json={
                        "patient_id": "test_patient_001",
                        "image_set_id": "test_image_set_001"
                    },
                    headers={"Authorization": f"Bearer {token}"}
                )
                
                analysis_id = response.json()["analysis_id"]
                
                # Get status
                response = client.get(f"/api/v1/analyses/{analysis_id}/status")
                
                assert response.status_code == 200
                data = response.json()
                assert data["analysis_id"] == analysis_id
                assert "status" in data
                assert "progress" in data


def test_get_analysis_not_found(client):
    """Test getting non-existent analysis"""
    # Login to get token
    login_response = client.post(
        "/api/v1/auth/login",
        json={"username": "clinician", "password": "clinician123"}
    )
    token = login_response.json()["access_token"]
    
    response = client.get(
        "/api/v1/analyses/nonexistent_id",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404


def test_get_analysis_status_not_found(client):
    """Test getting status of non-existent analysis"""
    response = client.get("/api/v1/analyses/nonexistent_id/status")
    assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
