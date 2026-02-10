"""
Integration tests for the complete analysis pipeline.

Tests the integration of all detection modules:
- Image preprocessing
- Landmark detection
- Pigmentation detection
- Wrinkle detection

Note: These tests use mocked landmark detection to avoid model download delays.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

from app.services.analysis_service import AnalysisService, AnalysisResult


@pytest.fixture
def analysis_service():
    """Create an analysis service instance with mocked landmark detector."""
    # Mock LandmarkDetector class to avoid model download during initialization
    with patch('app.services.analysis_service.LandmarkDetector') as MockLandmarkDetector:
        # Create a mock instance
        mock_detector = MagicMock()
        mock_detector.detect_landmarks = Mock(return_value=None)
        MockLandmarkDetector.return_value = mock_detector
        
        # Create service with mocked landmark detector
        service = AnalysisService()
        
        # Also ensure reconstructor is available
        assert service.reconstructor_3d is not None
        
        yield service


@pytest.fixture
def sample_face_image():
    """Create a simple sample facial image."""
    # Create a simple 512x512 image (smaller for speed)
    image = np.ones((512, 512, 3), dtype=np.uint8) * 180
    
    # Simple face ellipse
    cv2.ellipse(image, (256, 256), (150, 200), 0, 0, 360, (200, 170, 150), -1)
    
    # Simple eyes
    cv2.circle(image, (206, 206), 20, (50, 50, 50), -1)
    cv2.circle(image, (306, 206), 20, (50, 50, 50), -1)
    
    # Simple mouth
    cv2.ellipse(image, (256, 300), (40, 15), 0, 0, 180, (120, 80, 80), -1)
    
    # Add a pigmentation spot
    cv2.circle(image, (200, 150), 10, (140, 110, 90), -1)
    
    # Add a wrinkle line
    cv2.line(image, (150, 125), (350, 125), (160, 140, 120), 2)
    
    return image


@pytest.fixture
def sample_image_set(sample_face_image):
    """Create a set of 5 images with different angles."""
    images = []
    
    # Create 5 variations (simulating different angles)
    for i in range(5):
        # Slight variations in brightness to simulate different angles
        variation = sample_face_image.copy()
        brightness_factor = 0.9 + (i * 0.05)
        variation = (variation * brightness_factor).astype(np.uint8)
        images.append(variation)
    
    return images


def test_analysis_service_initialization(analysis_service):
    """Test that analysis service initializes all modules."""
    assert analysis_service.preprocessor is not None
    assert analysis_service.landmark_detector is not None
    assert analysis_service.pigmentation_detector is not None
    assert analysis_service.wrinkle_detector is not None
    assert analysis_service.reconstructor_3d is not None


def test_single_image_analysis(analysis_service, sample_face_image):
    """Test analysis of a single image."""
    result = analysis_service.analyze_single_image(
        image=sample_face_image,
        patient_id="test_patient_001",
        analysis_id="test_analysis_001"
    )
    
    # Check result structure
    assert isinstance(result, AnalysisResult)
    assert result.patient_id == "test_patient_001"
    assert result.analysis_id == "test_analysis_001"
    assert result.status in ["success", "partial", "failed"]
    
    # Check validation
    assert result.validation is not None
    assert "valid" in result.validation
    assert "image_count" in result.validation
    
    # Check pigmentation data
    assert result.pigmentation is not None
    assert "total_areas" in result.pigmentation
    assert "coverage_percentage" in result.pigmentation
    
    # Check wrinkles data
    assert result.wrinkles is not None
    assert "total_wrinkle_count" in result.wrinkles
    assert "texture_grade" in result.wrinkles


def test_multi_image_analysis(analysis_service, sample_image_set):
    """Test analysis of multiple images (180-degree coverage)."""
    result = analysis_service.analyze_patient(
        images=sample_image_set,
        patient_id="test_patient_002",
        analysis_id="test_analysis_002"
    )
    
    # Check result structure
    assert isinstance(result, AnalysisResult)
    assert result.patient_id == "test_patient_002"
    assert result.analysis_id == "test_analysis_002"
    
    # Check validation
    assert result.validation is not None
    assert result.validation["image_count"] == 5
    
    # If validation passed, check detection results
    if result.status == "success":
        # Check landmarks
        if result.landmarks:
            assert result.landmarks["detected"] is True
            assert "pixel_to_mm_scale" in result.landmarks
        
        # Check pigmentation
        if result.pigmentation:
            assert "total_areas" in result.pigmentation
            assert "severity_distribution" in result.pigmentation
            assert "areas" in result.pigmentation
        
        # Check wrinkles
        if result.wrinkles:
            assert "total_wrinkle_count" in result.wrinkles
            assert "regional_density" in result.wrinkles
            assert "wrinkles" in result.wrinkles


def test_result_serialization(analysis_service, sample_face_image):
    """Test that analysis result can be serialized to dict."""
    result = analysis_service.analyze_single_image(
        image=sample_face_image,
        patient_id="test_patient_003",
        analysis_id="test_analysis_003"
    )
    
    # Convert to dict
    result_dict = result.to_dict()
    
    # Check dict structure
    assert isinstance(result_dict, dict)
    assert "analysis_id" in result_dict
    assert "patient_id" in result_dict
    assert "validation" in result_dict
    assert "pigmentation" in result_dict
    assert "wrinkles" in result_dict
    assert "status" in result_dict


def test_error_handling_empty_images(analysis_service):
    """Test error handling with empty image list."""
    result = analysis_service.analyze_patient(
        images=[],
        patient_id="test_patient_004",
        analysis_id="test_analysis_004"
    )
    
    assert result.status == "failed"
    assert len(result.errors) > 0
    assert not result.validation["valid"]


def test_error_handling_low_quality_image(analysis_service):
    """Test error handling with low quality image."""
    # Create a very small, low quality image
    low_quality_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = analysis_service.analyze_single_image(
        image=low_quality_image,
        patient_id="test_patient_005",
        analysis_id="test_analysis_005"
    )
    
    # Should still complete but may have warnings
    assert isinstance(result, AnalysisResult)
    assert result.status in ["success", "partial", "failed"]


def test_pipeline_consistency(analysis_service, sample_face_image):
    """Test that running analysis twice gives consistent results."""
    result1 = analysis_service.analyze_single_image(
        image=sample_face_image,
        patient_id="test_patient_006",
        analysis_id="test_analysis_006a"
    )
    
    result2 = analysis_service.analyze_single_image(
        image=sample_face_image,
        patient_id="test_patient_006",
        analysis_id="test_analysis_006b"
    )
    
    # Results should be consistent
    assert result1.status == result2.status
    
    if result1.pigmentation and result2.pigmentation:
        # Pigmentation counts should be the same
        assert result1.pigmentation["total_areas"] == result2.pigmentation["total_areas"]
    
    if result1.wrinkles and result2.wrinkles:
        # Wrinkle counts should be the same
        assert result1.wrinkles["total_wrinkle_count"] == result2.wrinkles["total_wrinkle_count"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
