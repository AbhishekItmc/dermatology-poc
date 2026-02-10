"""
Unit tests for image preprocessing and validation.

Tests cover:
- Image set validation (coverage, count, format)
- Resolution and quality checks
- Face detection and cropping

Note: These tests use mocked face detection to focus on validation logic.
Real face detection should be tested with actual facial images.
"""

import pytest
import numpy as np
import cv2
from typing import List, Tuple
from unittest.mock import Mock, patch

from app.services.image_preprocessing import (
    ImagePreprocessor,
    ValidationResult,
    ValidationIssue,
    QualityMetrics,
    ImageInfo
)


@pytest.fixture
def preprocessor():
    """Create an ImagePreprocessor instance."""
    return ImagePreprocessor()


@pytest.fixture
def create_test_image():
    """Factory fixture to create test images with specific properties."""
    def _create_image(
        width: int = 1024,
        height: int = 1024,
        add_face: bool = True,
        face_size: float = 0.4,
        face_position: Tuple[float, float] = (0.5, 0.5),
        blur: bool = False,
        uneven_lighting: bool = False
    ) -> np.ndarray:
        """
        Create a synthetic test image.
        
        Args:
            width: Image width
            height: Image height
            add_face: Whether to add a face-like region
            face_size: Size of face relative to image (0-1)
            face_position: Face center position (x, y) normalized (0-1)
            blur: Whether to blur the image
            uneven_lighting: Whether to add uneven lighting
        """
        # Create base image with gradient
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if uneven_lighting:
            # Create strong gradient for uneven lighting
            for i in range(height):
                intensity = int(255 * (i / height))
                image[i, :] = [intensity, intensity, intensity]
        else:
            # Uniform gray background
            image[:] = [128, 128, 128]
        
        if add_face:
            # Add a more realistic face-like structure
            center_x = int(width * face_position[0])
            center_y = int(height * face_position[1])
            axes_x = int(width * face_size / 2)
            axes_y = int(height * face_size / 2)
            
            # Draw face (ellipse with skin tone)
            cv2.ellipse(
                image,
                (center_x, center_y),
                (axes_x, axes_y),
                0, 0, 360,
                (180, 150, 120),  # Skin tone
                -1
            )
            
            # Add more facial features for better detection
            # Eyes
            eye_y = center_y - axes_y // 3
            eye_offset = axes_x // 3
            eye_size = axes_x // 8
            cv2.circle(image, (center_x - eye_offset, eye_y), eye_size, (255, 255, 255), -1)  # White
            cv2.circle(image, (center_x + eye_offset, eye_y), eye_size, (255, 255, 255), -1)
            cv2.circle(image, (center_x - eye_offset, eye_y), eye_size // 2, (50, 50, 50), -1)  # Pupil
            cv2.circle(image, (center_x + eye_offset, eye_y), eye_size // 2, (50, 50, 50), -1)
            
            # Nose
            nose_y = center_y
            nose_points = np.array([
                [center_x, nose_y - axes_y // 6],
                [center_x - axes_x // 12, nose_y + axes_y // 12],
                [center_x + axes_x // 12, nose_y + axes_y // 12]
            ], np.int32)
            cv2.fillPoly(image, [nose_points], (160, 130, 100))
            
            # Mouth
            mouth_y = center_y + axes_y // 3
            cv2.ellipse(
                image,
                (center_x, mouth_y),
                (axes_x // 3, axes_y // 6),
                0, 0, 180,
                (100, 50, 50),
                -1
            )
            
            # Add some texture/edges for better focus detection
            if not blur:
                # Add eyebrows
                cv2.line(image, 
                        (center_x - eye_offset - eye_size, eye_y - eye_size),
                        (center_x - eye_offset + eye_size, eye_y - eye_size),
                        (80, 60, 40), 3)
                cv2.line(image,
                        (center_x + eye_offset - eye_size, eye_y - eye_size),
                        (center_x + eye_offset + eye_size, eye_y - eye_size),
                        (80, 60, 40), 3)
        
        if blur:
            # Apply strong blur
            image = cv2.GaussianBlur(image, (21, 21), 10)
        
        return image
    
    return _create_image


class TestImageSetValidation:
    """Tests for image set validation."""
    
    def test_valid_image_set(self, preprocessor, create_test_image):
        """Test validation of a valid image set."""
        # Create 5 images with faces at different positions (reduced size for speed)
        images = [
            create_test_image(width=512, height=512, face_position=(0.15, 0.5)),
            create_test_image(width=512, height=512, face_position=(0.325, 0.5)),
            create_test_image(width=512, height=512, face_position=(0.5, 0.5)),
            create_test_image(width=512, height=512, face_position=(0.675, 0.5)),
            create_test_image(width=512, height=512, face_position=(0.85, 0.5)),
        ]
        
        # Mock face detection to return consistent results with good spread
        with patch.object(preprocessor, '_detect_face') as mock_detect:
            # Return face detected with good coverage for all images
            mock_detect.side_effect = [
                (True, (int(1024 * pos), 300, 400, 500), 0.4)
                for pos in [0.15, 0.325, 0.5, 0.675, 0.85]
            ]
            
            result = preprocessor.validate_image_set(images)
        
        assert result.valid == True  # Use == instead of is for numpy bool
        assert result.image_count == 5
        assert result.angular_coverage >= preprocessor.MIN_ANGULAR_COVERAGE
        assert len(result.images) == 5
        assert all(img.face_detected for img in result.images)
    
    def test_insufficient_images(self, preprocessor, create_test_image):
        """Test validation fails with too few images."""
        images = [
            create_test_image(width=512, height=512),
            create_test_image(width=512, height=512),
        ]
        
        with patch.object(preprocessor, '_detect_face') as mock_detect:
            mock_detect.return_value = (True, (300, 300, 400, 500), 0.4)
            result = preprocessor.validate_image_set(images)
        
        assert result.valid is False
        assert result.image_count == 2
        assert any(
            issue[0] == ValidationIssue.INSUFFICIENT_IMAGES
            for issue in result.issues
        )
    
    def test_insufficient_coverage(self, preprocessor, create_test_image):
        """Test validation fails with insufficient angular coverage."""
        # Create 5 images with faces at same position (no coverage)
        images = [
            create_test_image(width=512, height=512, face_position=(0.5, 0.5))
            for _ in range(5)
        ]
        
        with patch.object(preprocessor, '_detect_face') as mock_detect:
            # All faces at same position
            mock_detect.return_value = (True, (512, 300, 400, 500), 0.4)
            result = preprocessor.validate_image_set(images)
        
        assert result.valid is False
        assert any(
            issue[0] == ValidationIssue.INSUFFICIENT_COVERAGE
            for issue in result.issues
        )
    
    def test_low_resolution_images(self, preprocessor, create_test_image):
        """Test validation detects low resolution images."""
        # Use even smaller images for faster test
        images = [
            create_test_image(width=256, height=256)
            for _ in range(5)
        ]
        
        with patch.object(preprocessor, '_detect_face') as mock_detect:
            mock_detect.return_value = (True, (200, 200, 200, 250), 0.4)
            result = preprocessor.validate_image_set(images)
        
        assert result.valid is False
        assert any(
            issue[0] == ValidationIssue.LOW_RESOLUTION
            for issue in result.issues
        )
    
    def test_no_face_detected(self, preprocessor, create_test_image):
        """Test validation detects missing faces."""
        images = [
            create_test_image(width=512, height=512, add_face=False)
            for _ in range(5)
        ]
        
        with patch.object(preprocessor, '_detect_face') as mock_detect:
            # Mock adds issues to the list
            def mock_detect_no_face(image, issues, image_id):
                issues.append(f"No face detected in image {image_id}. Please ensure face is clearly visible and properly framed.")
                return (False, None, 0.0)
            
            mock_detect.side_effect = mock_detect_no_face
            result = preprocessor.validate_image_set(images)
        
        assert result.valid is False
        # Check that issues were recorded (either as NO_FACE_DETECTED or in the issues list)
        assert len(result.issues) > 0
        assert all(not img.face_detected for img in result.images)
    
    def test_poor_lighting(self, preprocessor, create_test_image):
        """Test validation detects poor lighting."""
        images = [
            create_test_image(width=512, height=512, uneven_lighting=True)
            for _ in range(5)
        ]
        
        with patch.object(preprocessor, '_detect_face') as mock_detect:
            mock_detect.return_value = (True, (300, 300, 400, 500), 0.4)
            result = preprocessor.validate_image_set(images)
        
        # Should detect lighting issues
        assert any(
            issue[0] == ValidationIssue.POOR_LIGHTING
            for issue in result.issues
        )
    
    def test_out_of_focus(self, preprocessor, create_test_image):
        """Test validation detects out of focus images."""
        images = [
            create_test_image(width=512, height=512, blur=True)
            for _ in range(5)
        ]
        
        with patch.object(preprocessor, '_detect_face') as mock_detect:
            mock_detect.return_value = (True, (300, 300, 400, 500), 0.4)
            result = preprocessor.validate_image_set(images)
        
        # Should detect focus issues
        assert any(
            issue[0] == ValidationIssue.OUT_OF_FOCUS
            for issue in result.issues
        )


class TestSingleImageValidation:
    """Tests for individual image validation."""
    
    def test_high_quality_image(self, preprocessor, create_test_image):
        """Test validation of a high quality image."""
        image = create_test_image(width=1024, height=1024)
        
        with patch.object(preprocessor, '_detect_face') as mock_detect:
            mock_detect.return_value = (True, (300, 300, 400, 500), 0.4)
            info = preprocessor._validate_single_image(image, "test_img", 0)
        
        assert info.face_detected is True
        assert info.face_bbox is not None
        assert info.quality_metrics.resolution_score >= 0.5
        assert info.quality_metrics.overall_score >= preprocessor.MIN_OVERALL_QUALITY
        assert len(info.quality_metrics.issues) == 0
    
    def test_low_resolution_detection(self, preprocessor, create_test_image):
        """Test detection of low resolution."""
        image = create_test_image(width=256, height=256)
        
        info = preprocessor._validate_single_image(image, "test_img", 0)
        
        assert info.resolution == (256, 256)
        assert info.quality_metrics.resolution_score < 1.0
        assert any("resolution" in issue.lower() for issue in info.quality_metrics.issues)
    
    def test_face_detection_success(self, preprocessor, create_test_image):
        """Test successful face detection."""
        image = create_test_image(width=512, height=512, face_size=0.5)
        
        with patch.object(preprocessor, '_detect_face') as mock_detect:
            mock_detect.return_value = (True, (150, 100, 200, 250), 0.4)
            info = preprocessor._validate_single_image(image, "test_img", 0)
        
        assert info.face_detected is True
        assert info.face_bbox is not None
        x, y, w, h = info.face_bbox
        assert w > 0 and h > 0
        assert info.quality_metrics.face_coverage > 0
    
    def test_face_detection_failure(self, preprocessor, create_test_image):
        """Test face detection failure."""
        image = create_test_image(width=512, height=512, add_face=False)
        
        info = preprocessor._validate_single_image(image, "test_img", 0)
        
        assert info.face_detected is False
        assert info.face_bbox is None
        assert info.quality_metrics.face_coverage == 0.0
        assert any("face" in issue.lower() for issue in info.quality_metrics.issues)


class TestQualityAssessment:
    """Tests for quality assessment functions."""
    
    def test_resolution_assessment(self, preprocessor):
        """Test resolution assessment for different resolutions."""
        # High resolution
        issues = []
        score = preprocessor._assess_resolution(2048, 2048, issues, "test")
        assert score >= 0.5 and len(issues) == 0
        
        # Low resolution
        issues = []
        score = preprocessor._assess_resolution(512, 512, issues, "test")
        assert score < 1.0 and len(issues) == 1
    
    def test_lighting_assessment(self, preprocessor, create_test_image):
        """Test lighting assessment for uniform and uneven lighting."""
        # Uniform lighting
        image = create_test_image(width=512, height=512, uneven_lighting=False)
        issues = []
        score = preprocessor._assess_lighting(image, issues, "test")
        assert score >= preprocessor.MIN_LIGHTING_UNIFORMITY
        
        # Uneven lighting
        image = create_test_image(width=512, height=512, uneven_lighting=True)
        issues = []
        score = preprocessor._assess_lighting(image, issues, "test")
        assert score < 0.8
    
    def test_focus_assessment(self, preprocessor, create_test_image):
        """Test focus assessment for sharp and blurry images."""
        # Sharp image
        image = create_test_image(width=512, height=512, blur=False)
        issues = []
        score = preprocessor._assess_focus(image, issues, "test")
        assert score >= preprocessor.MIN_FOCUS_SCORE
        
        # Blurry image
        image = create_test_image(width=512, height=512, blur=True)
        issues = []
        score = preprocessor._assess_focus(image, issues, "test")
        assert score < 0.8


class TestAngularCoverage:
    """Tests for angular coverage estimation."""
    
    def test_coverage_estimation(self, preprocessor):
        """Test coverage estimation with different face spreads."""
        # Good spread
        image_infos = [
            ImageInfo(
                id=f"img_{i}",
                resolution=(512, 512),
                quality_metrics=QualityMetrics(1.0, 1.0, 1.0, 0.5, 0.9, []),
                face_detected=True,
                face_bbox=(int(512 * pos), 150, 150, 200)
            )
            for i, pos in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
        ]
        coverage = preprocessor._estimate_angular_coverage(image_infos)
        assert coverage >= 120.0
        
        # No spread
        image_infos = [
            ImageInfo(
                id=f"img_{i}",
                resolution=(512, 512),
                quality_metrics=QualityMetrics(1.0, 1.0, 1.0, 0.5, 0.9, []),
                face_detected=True,
                face_bbox=(256, 150, 150, 200)
            )
            for i in range(5)
        ]
        coverage = preprocessor._estimate_angular_coverage(image_infos)
        assert coverage < 100.0


class TestFaceCropping:
    """Tests for face cropping functionality."""
    
    def test_crop_face(self, preprocessor, create_test_image):
        """Test face cropping with and without padding."""
        image = create_test_image(width=512, height=512)
        face_bbox = (150, 100, 200, 250)
        
        # Basic crop
        cropped = preprocessor.crop_face(image, face_bbox, padding=0.0)
        assert cropped.shape[0] == 250 and cropped.shape[1] == 200
        
        # With padding
        cropped = preprocessor.crop_face(image, face_bbox, padding=0.2)
        assert cropped.shape[0] > 250 and cropped.shape[1] > 200


class TestImageNormalization:
    """Tests for image normalization."""
    
    def test_normalize(self, preprocessor, create_test_image):
        """Test image normalization."""
        image = create_test_image(width=512, height=512)
        
        normalized = preprocessor.normalize_image(image, target_size=(256, 256))
        
        assert normalized.shape == (256, 256, 3)
        assert normalized.dtype == np.float32
        assert 0.0 <= normalized.min() and normalized.max() <= 1.0
        assert 0.1 < normalized.mean() < 0.9
    
    def test_normalize_with_srgb(self, preprocessor, create_test_image):
        """Test image normalization with sRGB conversion."""
        image = create_test_image(width=512, height=512)
        
        # Normalize with sRGB conversion
        normalized_srgb = preprocessor.normalize_image(
            image, target_size=(256, 256), to_srgb=True
        )
        
        # Normalize without sRGB conversion
        normalized_linear = preprocessor.normalize_image(
            image, target_size=(256, 256), to_srgb=False
        )
        
        assert normalized_srgb.shape == (256, 256, 3)
        assert normalized_linear.shape == (256, 256, 3)
        
        # sRGB and linear should be different (gamma correction applied)
        assert not np.allclose(normalized_srgb, normalized_linear)
        
        # Both should be in valid range
        assert 0.0 <= normalized_srgb.min() and normalized_srgb.max() <= 1.0
        assert 0.0 <= normalized_linear.min() and normalized_linear.max() <= 1.0
    
    def test_normalize_batch(self, preprocessor, create_test_image):
        """Test batch normalization of multiple images."""
        # Create a batch of images with different properties
        images = [
            create_test_image(width=1024, height=1024, face_position=(0.3, 0.5)),
            create_test_image(width=800, height=800, face_position=(0.5, 0.5)),
            create_test_image(width=1200, height=1200, face_position=(0.7, 0.5)),
        ]
        
        # Normalize batch
        batch = preprocessor.normalize_image_batch(
            images, target_size=(512, 512), to_srgb=True
        )
        
        # Check batch shape
        assert batch.shape == (3, 512, 512, 3)
        assert batch.dtype == np.float32
        
        # Check all images are normalized
        assert 0.0 <= batch.min() and batch.max() <= 1.0
        
        # Each image in batch should be different
        assert not np.allclose(batch[0], batch[1])
        assert not np.allclose(batch[1], batch[2])
    
    def test_normalize_batch_empty(self, preprocessor):
        """Test batch normalization with empty list."""
        batch = preprocessor.normalize_image_batch([])
        assert batch.shape == (0,)
    
    def test_normalize_batch_single_image(self, preprocessor, create_test_image):
        """Test batch normalization with single image."""
        image = create_test_image(width=1024, height=1024)
        
        batch = preprocessor.normalize_image_batch([image], target_size=(512, 512))
        
        assert batch.shape == (1, 512, 512, 3)
        assert batch.dtype == np.float32
    
    def test_normalize_batch_consistency(self, preprocessor, create_test_image):
        """Test that batch normalization produces same results as individual normalization."""
        images = [
            create_test_image(width=1024, height=1024, face_position=(0.3, 0.5)),
            create_test_image(width=800, height=800, face_position=(0.5, 0.5)),
        ]
        
        # Normalize individually
        individual_results = [
            preprocessor.normalize_image(img, target_size=(512, 512), to_srgb=True)
            for img in images
        ]
        
        # Normalize as batch
        batch_result = preprocessor.normalize_image_batch(
            images, target_size=(512, 512), to_srgb=True
        )
        
        # Results should be identical
        for i, individual in enumerate(individual_results):
            assert np.allclose(individual, batch_result[i], rtol=1e-5)
    
    def test_srgb_conversion(self, preprocessor):
        """Test sRGB gamma correction."""
        # Create test linear RGB values
        linear_rgb = np.array([
            [[0.0, 0.0031308, 0.1]],  # Low values
            [[0.5, 0.8, 1.0]]  # High values
        ], dtype=np.float32)
        
        srgb = preprocessor._convert_to_srgb(linear_rgb)
        
        # Check shape preserved
        assert srgb.shape == linear_rgb.shape
        
        # Check values in valid range
        assert 0.0 <= srgb.min() and srgb.max() <= 1.0
        
        # Check specific gamma correction behavior
        # For low values (< 0.0031308), sRGB = 12.92 * linear
        assert np.isclose(srgb[0, 0, 0], 0.0)
        assert np.isclose(srgb[0, 0, 1], 12.92 * 0.0031308, rtol=0.01)
        
        # For high values, sRGB should be greater than linear (gamma correction brightens)
        # This is because gamma = 1/2.4 ≈ 0.417, and the formula is 1.055 * x^(1/2.4) - 0.055
        assert srgb[1, 0, 0] > linear_rgb[1, 0, 0]
        assert srgb[1, 0, 1] > linear_rgb[1, 0, 1]
        
        # Maximum value should remain 1.0
        assert np.isclose(srgb[1, 0, 2], 1.0)
    
    def test_normalize_different_target_sizes(self, preprocessor, create_test_image):
        """Test normalization with different target sizes."""
        image = create_test_image(width=1024, height=1024)
        
        # Test various target sizes
        for size in [(256, 256), (512, 512), (1024, 1024)]:
            normalized = preprocessor.normalize_image(image, target_size=size)
            assert normalized.shape == (size[1], size[0], 3)
            assert 0.0 <= normalized.min() and normalized.max() <= 1.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_edge_cases(self, preprocessor, create_test_image):
        """Test various edge cases."""
        # Empty list
        result = preprocessor.validate_image_set([])
        assert result.valid is False and result.image_count == 0
        
        # Single image
        result = preprocessor.validate_image_set([create_test_image(width=512, height=512)])
        assert result.valid is False and result.image_count == 1
        
        # Custom IDs
        images = [create_test_image(width=512, height=512) for _ in range(5)]
        image_ids = [f"custom_{i}" for i in range(5)]
        result = preprocessor.validate_image_set(images, image_ids)
        assert all(img.id.startswith("custom_") for img in result.images)



# ============================================================================
# Property-Based Tests
# ============================================================================

from hypothesis import given, settings, strategies as st, HealthCheck
from hypothesis import assume


# Custom strategies for generating test data
def valid_image_strategy():
    """Strategy for generating valid test images."""
    return st.builds(
        lambda w, h, face_pos: np.zeros((h, w, 3), dtype=np.uint8),
        w=st.integers(min_value=1024, max_value=2048),
        h=st.integers(min_value=1024, max_value=2048),
        face_pos=st.floats(min_value=0.2, max_value=0.8)
    )


def angular_coverage_strategy():
    """Strategy for generating angular coverage values."""
    return st.floats(min_value=0.0, max_value=180.0)


class TestPropertyBasedValidation:
    """Property-based tests for image validation."""
    
    # Feature: dermatological-analysis-poc, Property 23: Image Coverage Validation
    @given(
        num_images=st.integers(min_value=5, max_value=10),
        coverage=st.floats(min_value=0.0, max_value=180.0)
    )
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture
        ]
    )
    def test_property_23_image_coverage_validation(
        self,
        preprocessor,
        create_test_image,
        num_images,
        coverage
    ):
        """
        **Validates: Requirements 10.2**
        
        Property 23: Image Coverage Validation
        For any image set with inadequate angular coverage (less than 150 degrees),
        the system should reject the input and provide a descriptive error message.
        
        This property verifies that:
        1. Image sets with coverage < 150 degrees are rejected (valid=False)
        2. A descriptive error message is provided
        3. The error message mentions "coverage" and provides actionable guidance
        """
        # Generate images with controlled face positions to achieve target coverage
        # Coverage is roughly proportional to the spread of face positions
        # spread of 0.5 (normalized) ≈ 90 degrees, spread of 1.0 ≈ 180 degrees
        target_spread = coverage / 180.0
        
        # Generate face positions with the target spread
        if num_images == 1:
            face_positions = [0.5]
        else:
            # Distribute faces across the target spread
            start_pos = 0.5 - target_spread / 2
            end_pos = 0.5 + target_spread / 2
            # Clamp to valid range [0.1, 0.9]
            start_pos = max(0.1, min(0.9, start_pos))
            end_pos = max(0.1, min(0.9, end_pos))
            
            face_positions = [
                start_pos + (end_pos - start_pos) * i / (num_images - 1)
                for i in range(num_images)
            ]
        
        # Create images with faces at calculated positions
        images = [
            create_test_image(face_position=(pos, 0.5))
            for pos in face_positions
        ]
        
        # Mock face detection to return consistent results
        with patch.object(preprocessor, '_detect_face') as mock_detect:
            # Return face detected with positions matching our target coverage
            mock_detect.side_effect = [
                (True, (int(1024 * pos), 300, 400, 500), 0.4)
                for pos in face_positions
            ]
            
            result = preprocessor.validate_image_set(images)
        
        # Property verification based on ACTUAL calculated coverage
        # (not the target coverage we tried to achieve)
        actual_coverage = result.angular_coverage
        
        if actual_coverage < preprocessor.MIN_ANGULAR_COVERAGE:
            # Should reject inadequate coverage
            assert result.valid is False, (
                f"Expected validation to fail for actual coverage {actual_coverage:.1f}° "
                f"(< {preprocessor.MIN_ANGULAR_COVERAGE}°), but it passed"
            )
            
            # Should provide descriptive error message
            coverage_issues = [
                issue for issue in result.issues
                if issue[0] == ValidationIssue.INSUFFICIENT_COVERAGE
            ]
            
            assert len(coverage_issues) > 0, (
                f"Expected INSUFFICIENT_COVERAGE issue for coverage {actual_coverage:.1f}°, "
                f"but got issues: {result.issues}"
            )
            
            # Check error message quality
            error_message = coverage_issues[0][1]
            
            # Should mention "coverage" or "degrees"
            assert any(
                keyword in error_message.lower()
                for keyword in ["coverage", "degrees", "angle"]
            ), (
                f"Error message should mention coverage/degrees/angle, "
                f"but got: {error_message}"
            )
            
            # Should provide actionable guidance
            assert any(
                keyword in error_message.lower()
                for keyword in ["capture", "additional", "side", "angle"]
            ), (
                f"Error message should provide actionable guidance, "
                f"but got: {error_message}"
            )
            
            # Should mention the actual coverage value
            assert any(
                str(int(actual_coverage)) in error_message or
                f"{actual_coverage:.1f}" in error_message
                for _ in [None]  # Just to make this a generator expression
            ), (
                f"Error message should mention actual coverage value "
                f"({actual_coverage:.1f}°), but got: {error_message}"
            )
            
            # Should mention the minimum required coverage
            assert str(int(preprocessor.MIN_ANGULAR_COVERAGE)) in error_message, (
                f"Error message should mention minimum required coverage "
                f"({preprocessor.MIN_ANGULAR_COVERAGE}°), but got: {error_message}"
            )
        
        else:
            # Adequate coverage - validation might still fail for other reasons,
            # but should NOT have insufficient coverage issue
            coverage_issues = [
                issue for issue in result.issues
                if issue[0] == ValidationIssue.INSUFFICIENT_COVERAGE
            ]
            
            assert len(coverage_issues) == 0, (
                f"Expected no INSUFFICIENT_COVERAGE issue for coverage {actual_coverage:.1f}° "
                f"(>= {preprocessor.MIN_ANGULAR_COVERAGE}°), "
                f"but got: {coverage_issues}"
            )
    
    # Feature: dermatological-analysis-poc, Property 26: Image Quality Validation
    @given(
        width=st.integers(min_value=256, max_value=2048),
        height=st.integers(min_value=256, max_value=2048),
        blur=st.booleans(),
        uneven_lighting=st.booleans()
    )
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture
        ]
    )
    def test_property_26_image_quality_validation(
        self,
        preprocessor,
        create_test_image,
        width,
        height,
        blur,
        uneven_lighting
    ):
        """
        **Validates: Requirements 10.6**
        
        Property 26: Image Quality Validation
        For any image with resolution below 1024x1024, poor lighting uniformity,
        or low focus score, the system should flag the quality issue and provide
        specific guidance.
        
        This property verifies that:
        1. Images with resolution < 1024x1024 are flagged with LOW_RESOLUTION issue
        2. Images with poor lighting uniformity are flagged with POOR_LIGHTING issue
        3. Images with low focus score are flagged with OUT_OF_FOCUS issue
        4. Each quality issue includes specific guidance for recapture
        5. The error messages are actionable and mention the specific problem
        """
        # Create test image with specified properties
        image = create_test_image(
            width=width,
            height=height,
            blur=blur,
            uneven_lighting=uneven_lighting
        )
        
        # Mock face detection to focus on quality validation
        with patch.object(preprocessor, '_detect_face') as mock_detect:
            mock_detect.return_value = (True, (width // 4, height // 4, width // 2, height // 2), 0.4)
            
            # Validate single image
            image_info = preprocessor._validate_single_image(image, "test_image", 0)
        
        # Determine expected quality issues based on input parameters
        min_dim = min(width, height)
        has_low_resolution = min_dim < preprocessor.MIN_RESOLUTION
        
        # Assess lighting (we need to calculate this to know if it should fail)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        cv_intensity = std_intensity / mean_intensity if mean_intensity > 0 else 1.0
        lighting_uniformity = max(0.0, 1.0 - cv_intensity)
        has_poor_lighting = lighting_uniformity < preprocessor.MIN_LIGHTING_UNIFORMITY
        
        # Assess focus
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        focus_score = min(1.0, variance / 200.0)
        has_low_focus = focus_score < preprocessor.MIN_FOCUS_SCORE
        
        # Verify resolution issue detection
        if has_low_resolution:
            # Should flag low resolution
            resolution_issues = [
                issue for issue in image_info.quality_metrics.issues
                if "resolution" in issue.lower()
            ]
            
            assert len(resolution_issues) > 0, (
                f"Expected LOW_RESOLUTION issue for {width}x{height} "
                f"(< {preprocessor.MIN_RESOLUTION}x{preprocessor.MIN_RESOLUTION}), "
                f"but got issues: {image_info.quality_metrics.issues}"
            )
            
            # Check error message quality
            error_message = resolution_issues[0]
            
            # Should mention "resolution" and the actual dimensions
            assert "resolution" in error_message.lower(), (
                f"Error message should mention 'resolution', but got: {error_message}"
            )
            
            # Should mention the minimum required resolution
            assert str(preprocessor.MIN_RESOLUTION) in error_message, (
                f"Error message should mention minimum resolution "
                f"({preprocessor.MIN_RESOLUTION}), but got: {error_message}"
            )
            
            # Should provide actionable guidance
            assert any(
                keyword in error_message.lower()
                for keyword in ["capture", "higher", "please"]
            ), (
                f"Error message should provide actionable guidance, "
                f"but got: {error_message}"
            )
        else:
            # Should NOT flag low resolution
            resolution_issues = [
                issue for issue in image_info.quality_metrics.issues
                if "resolution" in issue.lower()
            ]
            
            assert len(resolution_issues) == 0, (
                f"Expected no LOW_RESOLUTION issue for {width}x{height} "
                f"(>= {preprocessor.MIN_RESOLUTION}), "
                f"but got: {resolution_issues}"
            )
        
        # Verify lighting issue detection
        if has_poor_lighting:
            # Should flag poor lighting
            lighting_issues = [
                issue for issue in image_info.quality_metrics.issues
                if "lighting" in issue.lower()
            ]
            
            assert len(lighting_issues) > 0, (
                f"Expected POOR_LIGHTING issue for uniformity {lighting_uniformity:.2f} "
                f"(< {preprocessor.MIN_LIGHTING_UNIFORMITY}), "
                f"but got issues: {image_info.quality_metrics.issues}"
            )
            
            # Check error message quality
            error_message = lighting_issues[0]
            
            # Should mention "lighting" or "uniformity"
            assert any(
                keyword in error_message.lower()
                for keyword in ["lighting", "uniformity"]
            ), (
                f"Error message should mention lighting/uniformity, "
                f"but got: {error_message}"
            )
            
            # Should provide actionable guidance
            assert any(
                keyword in error_message.lower()
                for keyword in ["recapture", "consistent", "diffuse", "please"]
            ), (
                f"Error message should provide actionable guidance, "
                f"but got: {error_message}"
            )
        else:
            # Should NOT flag poor lighting
            lighting_issues = [
                issue for issue in image_info.quality_metrics.issues
                if "lighting" in issue.lower()
            ]
            
            assert len(lighting_issues) == 0, (
                f"Expected no POOR_LIGHTING issue for uniformity {lighting_uniformity:.2f} "
                f"(>= {preprocessor.MIN_LIGHTING_UNIFORMITY}), "
                f"but got: {lighting_issues}"
            )
        
        # Verify focus issue detection
        if has_low_focus:
            # Should flag low focus
            focus_issues = [
                issue for issue in image_info.quality_metrics.issues
                if "focus" in issue.lower()
            ]
            
            assert len(focus_issues) > 0, (
                f"Expected OUT_OF_FOCUS issue for focus score {focus_score:.2f} "
                f"(< {preprocessor.MIN_FOCUS_SCORE}), "
                f"but got issues: {image_info.quality_metrics.issues}"
            )
            
            # Check error message quality
            error_message = focus_issues[0]
            
            # Should mention "focus"
            assert "focus" in error_message.lower(), (
                f"Error message should mention 'focus', but got: {error_message}"
            )
            
            # Should provide actionable guidance
            assert any(
                keyword in error_message.lower()
                for keyword in ["recapture", "proper", "facial", "please"]
            ), (
                f"Error message should provide actionable guidance, "
                f"but got: {error_message}"
            )
        else:
            # Should NOT flag low focus
            focus_issues = [
                issue for issue in image_info.quality_metrics.issues
                if "focus" in issue.lower()
            ]
            
            assert len(focus_issues) == 0, (
                f"Expected no OUT_OF_FOCUS issue for focus score {focus_score:.2f} "
                f"(>= {preprocessor.MIN_FOCUS_SCORE}), "
                f"but got: {focus_issues}"
            )
        
        # Verify that quality metrics are calculated correctly
        assert 0.0 <= image_info.quality_metrics.resolution_score <= 1.0, (
            f"Resolution score should be in [0, 1], "
            f"but got: {image_info.quality_metrics.resolution_score}"
        )
        
        assert 0.0 <= image_info.quality_metrics.lighting_uniformity <= 1.0, (
            f"Lighting uniformity should be in [0, 1], "
            f"but got: {image_info.quality_metrics.lighting_uniformity}"
        )
        
        assert 0.0 <= image_info.quality_metrics.focus_score <= 1.0, (
            f"Focus score should be in [0, 1], "
            f"but got: {image_info.quality_metrics.focus_score}"
        )
        
        assert 0.0 <= image_info.quality_metrics.overall_score <= 1.0, (
            f"Overall score should be in [0, 1], "
            f"but got: {image_info.quality_metrics.overall_score}"
        )
