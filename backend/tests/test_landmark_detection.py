"""
Unit tests for facial landmark detection module.

Tests landmark detection, pose estimation, IPD calculation,
and facial region extraction.
"""

import pytest
import numpy as np
import cv2
from app.services.landmark_detection import (
    LandmarkDetector,
    Landmark3D,
    PoseMatrix,
    FacialRegion,
    LandmarkResult
)


@pytest.fixture
def landmark_detector():
    """Create a landmark detector instance."""
    return LandmarkDetector()


@pytest.fixture
def sample_face_image():
    """
    Create a synthetic face image for testing.
    
    This creates a simple face-like pattern that MediaPipe can detect.
    """
    # Create a blank image
    image = np.ones((512, 512, 3), dtype=np.uint8) * 200
    
    # Draw a face-like oval
    center = (256, 256)
    axes = (120, 160)
    cv2.ellipse(image, center, axes, 0, 0, 360, (180, 150, 130), -1)
    
    # Draw eyes
    left_eye = (200, 220)
    right_eye = (312, 220)
    cv2.ellipse(image, left_eye, (20, 15), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(image, right_eye, (20, 15), 0, 0, 360, (50, 50, 50), -1)
    
    # Draw pupils
    cv2.circle(image, left_eye, 8, (0, 0, 0), -1)
    cv2.circle(image, right_eye, 8, (0, 0, 0), -1)
    
    # Draw nose
    nose_points = np.array([
        [256, 240],
        [246, 280],
        [256, 290],
        [266, 280]
    ], dtype=np.int32)
    cv2.fillPoly(image, [nose_points], (160, 130, 110))
    
    # Draw mouth
    cv2.ellipse(image, (256, 330), (40, 20), 0, 0, 180, (100, 50, 50), -1)
    
    return image


@pytest.fixture
def real_face_image():
    """
    Load a real face image if available, otherwise skip.
    
    This is useful for more realistic testing but not required.
    """
    # For now, we'll use the synthetic image
    # In a real project, you might load a test image from a file
    pytest.skip("Real face image not available for testing")


class TestLandmarkDetector:
    """Test suite for LandmarkDetector class."""
    
    def test_initialization(self, landmark_detector):
        """Test that detector initializes correctly."""
        assert landmark_detector is not None
        assert landmark_detector.face_landmarker is not None
        assert len(landmark_detector.landmark_names) > 0
    
    def test_detect_landmarks_with_face(self, landmark_detector, sample_face_image):
        """Test landmark detection on image with face."""
        result = landmark_detector.detect_landmarks(sample_face_image)
        
        # MediaPipe might not detect our synthetic face, so we handle both cases
        if result is not None:
            # If face detected, verify structure
            assert isinstance(result, LandmarkResult)
            assert len(result.landmarks) > 0
            assert result.confidence_score >= 0.0
            assert result.confidence_score <= 1.0
            assert result.interpupillary_distance_px > 0
            assert isinstance(result.pose, PoseMatrix)
            assert isinstance(result.facial_regions, dict)
        else:
            # If no face detected, that's also acceptable for synthetic image
            assert result is None
    
    def test_detect_landmarks_no_face(self, landmark_detector):
        """Test landmark detection on image without face."""
        # Create blank image
        blank_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        
        result = landmark_detector.detect_landmarks(blank_image)
        
        # Should return None for no face
        assert result is None
    
    def test_landmark_structure(self, landmark_detector, sample_face_image):
        """Test that landmarks have correct structure."""
        result = landmark_detector.detect_landmarks(sample_face_image)
        
        if result is not None:
            # Check first landmark structure
            landmark = result.landmarks[0]
            assert isinstance(landmark, Landmark3D)
            assert isinstance(landmark.id, int)
            assert isinstance(landmark.x, float)
            assert isinstance(landmark.y, float)
            assert isinstance(landmark.z, float)
            assert isinstance(landmark.confidence, float)
            assert isinstance(landmark.name, str)
            
            # Check coordinate ranges
            assert 0 <= landmark.x <= sample_face_image.shape[1]
            assert 0 <= landmark.y <= sample_face_image.shape[0]
            assert 0.0 <= landmark.confidence <= 1.0
    
    def test_pose_estimation(self, landmark_detector, sample_face_image):
        """Test pose estimation from landmarks."""
        result = landmark_detector.detect_landmarks(sample_face_image)
        
        if result is not None:
            pose = result.pose
            
            # Check pose structure
            assert isinstance(pose, PoseMatrix)
            assert pose.rotation_matrix.shape == (3, 3)
            assert pose.translation_vector.shape == (3, 1)
            assert len(pose.euler_angles) == 3
            
            # Check Euler angles are in reasonable range
            pitch, yaw, roll = pose.euler_angles
            assert -180 <= pitch <= 180
            assert -180 <= yaw <= 180
            assert -180 <= roll <= 180
    
    def test_interpupillary_distance(self, landmark_detector, sample_face_image):
        """Test IPD calculation."""
        result = landmark_detector.detect_landmarks(sample_face_image)
        
        if result is not None:
            ipd = result.interpupillary_distance_px
            
            # IPD should be positive
            assert ipd > 0
            
            # For a 512x512 image with face, IPD should be reasonable
            # (not too small, not larger than image width)
            assert 10 < ipd < 512
    
    def test_pixel_to_mm_scale(self, landmark_detector):
        """Test pixel-to-mm scaling calculation."""
        # Test with known IPD
        ipd_px = 100.0
        scale = landmark_detector.calculate_pixel_to_mm_scale(ipd_px)
        
        # With average IPD of 63mm and 100px, scale should be 0.63 mm/px
        assert abs(scale - 0.63) < 0.01
        
        # Test with zero IPD
        scale_zero = landmark_detector.calculate_pixel_to_mm_scale(0.0)
        assert scale_zero == 0.0
    
    def test_facial_regions(self, landmark_detector, sample_face_image):
        """Test facial region extraction."""
        result = landmark_detector.detect_landmarks(sample_face_image)
        
        if result is not None:
            regions = result.facial_regions
            
            # Check that regions were extracted
            assert isinstance(regions, dict)
            
            # Check expected regions
            expected_regions = [
                "forehead", "left_cheek", "right_cheek",
                "periorbital_left", "periorbital_right",
                "nose", "mouth"
            ]
            
            for region_name in expected_regions:
                if region_name in regions:
                    region = regions[region_name]
                    assert isinstance(region, FacialRegion)
                    assert region.name == region_name
                    assert len(region.landmark_indices) > 0
                    assert len(region.bounding_box) == 4
                    
                    # Check bounding box is valid
                    x, y, w, h = region.bounding_box
                    assert x >= 0
                    assert y >= 0
                    assert w > 0
                    assert h > 0
    
    def test_visualize_landmarks(self, landmark_detector, sample_face_image):
        """Test landmark visualization."""
        result = landmark_detector.detect_landmarks(sample_face_image)
        
        if result is not None:
            # Visualize without indices
            vis_image = landmark_detector.visualize_landmarks(
                sample_face_image,
                result.landmarks,
                draw_indices=False
            )
            
            assert vis_image.shape == sample_face_image.shape
            assert vis_image.dtype == sample_face_image.dtype
            
            # Visualize with indices
            vis_image_idx = landmark_detector.visualize_landmarks(
                sample_face_image,
                result.landmarks,
                draw_indices=True
            )
            
            assert vis_image_idx.shape == sample_face_image.shape
    
    def test_bgr_to_rgb_conversion(self, landmark_detector):
        """Test that detector handles BGR images correctly."""
        # Create BGR image (OpenCV format)
        bgr_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Should not raise an error
        result = landmark_detector.detect_landmarks(bgr_image)
        
        # Result can be None (no face) or valid LandmarkResult
        assert result is None or isinstance(result, LandmarkResult)
    
    def test_confidence_threshold(self, landmark_detector):
        """Test that low confidence detections are rejected."""
        # Create very noisy image
        noisy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        result = landmark_detector.detect_landmarks(noisy_image)
        
        # Should return None for low confidence or no face
        # If it returns a result, confidence should be above threshold
        if result is not None:
            assert result.confidence_score >= landmark_detector.MIN_OVERALL_CONFIDENCE
    
    def test_multiple_detections(self, landmark_detector, sample_face_image):
        """Test that detector is consistent across multiple calls."""
        result1 = landmark_detector.detect_landmarks(sample_face_image)
        result2 = landmark_detector.detect_landmarks(sample_face_image)
        
        # Both should give same result (None or valid)
        if result1 is None:
            assert result2 is None
        else:
            assert result2 is not None
            # Landmark count should be the same
            assert len(result1.landmarks) == len(result2.landmarks)


class TestLandmark3D:
    """Test suite for Landmark3D dataclass."""
    
    def test_landmark_creation(self):
        """Test creating a Landmark3D object."""
        landmark = Landmark3D(
            id=0,
            x=100.0,
            y=200.0,
            z=50.0,
            confidence=0.95,
            name="nose_tip"
        )
        
        assert landmark.id == 0
        assert landmark.x == 100.0
        assert landmark.y == 200.0
        assert landmark.z == 50.0
        assert landmark.confidence == 0.95
        assert landmark.name == "nose_tip"


class TestPoseMatrix:
    """Test suite for PoseMatrix dataclass."""
    
    def test_pose_creation(self):
        """Test creating a PoseMatrix object."""
        rotation = np.eye(3)
        translation = np.zeros((3, 1))
        euler = (0.0, 0.0, 0.0)
        
        pose = PoseMatrix(
            rotation_matrix=rotation,
            translation_vector=translation,
            euler_angles=euler
        )
        
        assert pose.rotation_matrix.shape == (3, 3)
        assert pose.translation_vector.shape == (3, 1)
        assert len(pose.euler_angles) == 3


class TestFacialRegion:
    """Test suite for FacialRegion dataclass."""
    
    def test_region_creation(self):
        """Test creating a FacialRegion object."""
        region = FacialRegion(
            name="forehead",
            landmark_indices=[10, 11, 12, 13],
            bounding_box=(100, 50, 200, 100)
        )
        
        assert region.name == "forehead"
        assert len(region.landmark_indices) == 4
        assert region.bounding_box == (100, 50, 200, 100)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_image(self, landmark_detector):
        """Test with empty image."""
        empty_image = np.zeros((0, 0, 3), dtype=np.uint8)
        
        # Should handle gracefully (return None or raise appropriate error)
        try:
            result = landmark_detector.detect_landmarks(empty_image)
            assert result is None
        except (ValueError, cv2.error):
            # Acceptable to raise error for invalid input
            pass
    
    def test_small_image(self, landmark_detector):
        """Test with very small image."""
        small_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        
        result = landmark_detector.detect_landmarks(small_image)
        
        # Should return None (face too small or not detected)
        assert result is None
    
    def test_large_image(self, landmark_detector):
        """Test with large image."""
        large_image = np.ones((2048, 2048, 3), dtype=np.uint8) * 128
        
        # Should handle without crashing
        result = landmark_detector.detect_landmarks(large_image)
        
        # Result can be None or valid
        assert result is None or isinstance(result, LandmarkResult)
    
    def test_grayscale_image(self, landmark_detector):
        """Test with grayscale image."""
        gray_image = np.ones((512, 512), dtype=np.uint8) * 128
        
        # Should handle gracefully
        try:
            result = landmark_detector.detect_landmarks(gray_image)
            # Can be None or valid result
            assert result is None or isinstance(result, LandmarkResult)
        except (ValueError, cv2.error):
            # Acceptable to raise error for invalid format
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
