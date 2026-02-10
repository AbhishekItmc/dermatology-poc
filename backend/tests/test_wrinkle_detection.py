"""
Tests for wrinkle detection module.

This module tests the wrinkle detection system including:
- EdgeAwareCNN architecture
- Wrinkle detection and segmentation
- Attribute measurement (length, depth, width)
- Severity classification
- Regional density analysis
- Texture grading
- Property-based tests for detection completeness
"""

import pytest
import numpy as np
import cv2
from hypothesis import given, settings, strategies as st
from typing import List, Tuple

from app.services.wrinkle_detection import (
    WrinkleDetector,
    EdgeAwareCNN,
    TrainingPipeline,
    TrainingConfig,
    SeverityLevel,
    FacialRegion,
    TextureGrade,
    WrinkleAttributes,
    WrinkleAnalysis,
    RegionalDensity
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def detector():
    """Create a wrinkle detector instance."""
    return WrinkleDetector()


@pytest.fixture
def sample_image():
    """Create a sample normalized facial image."""
    # Create a 512x512 RGB image with some wrinkle-like features
    image = np.ones((512, 512, 3), dtype=np.float32) * 0.7
    
    # Add some line features (simulating wrinkles)
    # Horizontal lines (forehead wrinkles)
    cv2.line(image, (100, 100), (400, 105), (0.5, 0.5, 0.5), 2)
    cv2.line(image, (120, 130), (380, 135), (0.5, 0.5, 0.5), 2)
    
    # Diagonal lines (nasolabial folds)
    cv2.line(image, (200, 250), (180, 350), (0.4, 0.4, 0.4), 3)
    cv2.line(image, (312, 250), (332, 350), (0.4, 0.4, 0.4), 3)
    
    # Crow's feet
    cv2.line(image, (450, 200), (480, 190), (0.5, 0.5, 0.5), 1)
    cv2.line(image, (450, 210), (480, 220), (0.5, 0.5, 0.5), 1)
    
    return image


@pytest.fixture
def sample_image_with_wrinkles():
    """Create a sample image with known wrinkle patterns."""
    # Create base skin tone
    image = np.ones((512, 512, 3), dtype=np.float32) * 0.65
    
    # Add deep wrinkles (high severity)
    cv2.line(image, (100, 150), (400, 150), (0.3, 0.3, 0.3), 4)
    
    # Add medium wrinkles
    cv2.line(image, (100, 200), (400, 200), (0.45, 0.45, 0.45), 3)
    
    # Add fine lines (low severity)
    cv2.line(image, (100, 250), (400, 250), (0.55, 0.55, 0.55), 1)
    
    # Add micro-wrinkles (very fine)
    for y in range(300, 320, 5):
        cv2.line(image, (100, y), (400, y), (0.60, 0.60, 0.60), 1)
    
    return image


# ============================================================================
# Hypothesis Strategies
# ============================================================================

def normalized_image_strategy():
    """Strategy for generating normalized images - simplified for speed."""
    @st.composite
    def _strategy(draw):
        # Generate simple uniform image (much faster)
        base_color = draw(st.floats(min_value=0.4, max_value=0.7))
        image = np.ones((256, 256, 3), dtype=np.float32) * base_color  # Smaller size
        
        # Add just 1-3 lines (much faster)
        num_lines = draw(st.integers(min_value=0, max_value=3))
        for _ in range(num_lines):
            x1 = draw(st.integers(min_value=50, max_value=200))
            y1 = draw(st.integers(min_value=50, max_value=200))
            x2 = x1 + draw(st.integers(min_value=20, max_value=100))
            y2 = y1 + draw(st.integers(min_value=-20, max_value=20))
            cv2.line(image, (x1, y1), (x2, y2), (0.3, 0.3, 0.3), 2)
        
        return image
    
    return _strategy()


# ============================================================================
# Unit Tests - EdgeAwareCNN Architecture
# ============================================================================

def test_edge_aware_cnn_initialization():
    """Test EdgeAwareCNN model initialization."""
    model = EdgeAwareCNN(pretrained=True)
    
    assert model.pretrained is True
    assert model.input_size == (512, 512)
    assert len(model.feature_channels) == 5
    assert model.edge_kernel_size == 3
    assert model.depth_output_channels == 1
    assert model.is_trained is False
    assert model.training_epochs == 0


def test_edge_aware_cnn_forward_pass(sample_image):
    """Test EdgeAwareCNN forward pass."""
    model = EdgeAwareCNN()
    
    # Run forward pass
    wrinkle_mask, depth_map, edge_map = model.forward(sample_image)
    
    # Check output shapes
    assert wrinkle_mask.shape == (512, 512)
    assert depth_map.shape == (512, 512)
    assert edge_map.shape == (512, 512)
    
    # Check output types
    assert wrinkle_mask.dtype == np.uint8
    assert depth_map.dtype == np.uint8
    assert edge_map.dtype == np.uint8


# ============================================================================
# Unit Tests - Wrinkle Detection
# ============================================================================

def test_detector_initialization():
    """Test wrinkle detector initialization."""
    detector = WrinkleDetector()
    
    assert detector.model is not None
    assert detector.min_wrinkle_length_px == 10
    assert detector.micro_wrinkle_depth_threshold == 0.5


def test_detect_wrinkles_basic(detector, sample_image):
    """Test basic wrinkle detection."""
    analysis = detector.detect_wrinkles(sample_image, pixel_to_mm_scale=0.1)
    
    # Check analysis structure
    assert isinstance(analysis, WrinkleAnalysis)
    assert isinstance(analysis.wrinkles, list)
    assert isinstance(analysis.regional_density, dict)
    assert isinstance(analysis.texture_grade, TextureGrade)
    assert analysis.total_wrinkle_count >= 0
    assert analysis.micro_wrinkle_count >= 0
    assert analysis.average_depth_mm >= 0.0
    assert analysis.average_length_mm >= 0.0
    assert analysis.depth_map.shape == (512, 512)


def test_detect_wrinkles_with_known_patterns(detector, sample_image_with_wrinkles):
    """Test detection with known wrinkle patterns."""
    analysis = detector.detect_wrinkles(
        sample_image_with_wrinkles,
        pixel_to_mm_scale=0.1
    )
    
    # Should detect multiple wrinkles
    assert analysis.total_wrinkle_count > 0
    
    # Check wrinkle attributes
    for wrinkle in analysis.wrinkles:
        assert isinstance(wrinkle, WrinkleAttributes)
        assert wrinkle.length_mm > 0.0
        assert wrinkle.depth_mm >= 0.0
        assert wrinkle.width_mm > 0.0
        assert isinstance(wrinkle.severity, SeverityLevel)
        assert isinstance(wrinkle.region, FacialRegion)
        assert 0.0 <= wrinkle.confidence <= 1.0
        assert len(wrinkle.bounding_box) == 4


def test_severity_classification(detector, sample_image_with_wrinkles):
    """Test severity classification."""
    analysis = detector.detect_wrinkles(
        sample_image_with_wrinkles,
        pixel_to_mm_scale=0.1
    )
    
    # Should have wrinkles of different severities
    severities = {w.severity for w in analysis.wrinkles}
    assert len(severities) > 0
    
    # All severities should be valid
    valid_severities = {SeverityLevel.MICRO, SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH}
    assert severities.issubset(valid_severities)


def test_micro_wrinkle_classification(detector):
    """Test micro-wrinkle classification based on depth threshold (Requirement 2.7)."""
    # Test micro-wrinkle: depth < 0.5mm
    severity_micro = detector._classify_severity(
        length_mm=10.0,
        depth_mm=0.3,  # Below 0.5mm threshold
        width_mm=0.5
    )
    assert severity_micro == SeverityLevel.MICRO, \
        "Wrinkles with depth < 0.5mm should be classified as MICRO"
    
    # Test at exact threshold
    severity_at_threshold = detector._classify_severity(
        length_mm=10.0,
        depth_mm=0.49,  # Just below threshold
        width_mm=0.5
    )
    assert severity_at_threshold == SeverityLevel.MICRO, \
        "Wrinkles with depth < 0.5mm should be classified as MICRO"
    
    # Test just above threshold
    severity_above = detector._classify_severity(
        length_mm=10.0,
        depth_mm=0.51,  # Just above threshold
        width_mm=0.5
    )
    assert severity_above != SeverityLevel.MICRO, \
        "Wrinkles with depth >= 0.5mm should not be classified as MICRO"


def test_severity_based_on_attributes(detector):
    """Test severity classification based on length, depth, and width (Requirement 2.5)."""
    # Test LOW severity: small depth, length, width
    # Score = 0.6 * 2.0 + 5.0 * 0.1 + 0.3 * 0.5 = 1.2 + 0.5 + 0.15 = 1.85 < 2.0
    severity_low = detector._classify_severity(
        length_mm=5.0,
        depth_mm=0.6,  # Above micro threshold but small
        width_mm=0.3
    )
    assert severity_low == SeverityLevel.LOW, \
        "Small wrinkles should be classified as LOW severity"
    
    # Test MEDIUM severity: moderate attributes
    # Score = 1.0 * 2.0 + 15.0 * 0.1 + 0.8 * 0.5 = 2.0 + 1.5 + 0.4 = 3.9 < 4.0
    severity_medium = detector._classify_severity(
        length_mm=15.0,
        depth_mm=1.0,
        width_mm=0.8
    )
    assert severity_medium == SeverityLevel.MEDIUM, \
        "Moderate wrinkles should be classified as MEDIUM severity"
    
    # Test HIGH severity: large depth
    # Score = 2.5 * 2.0 + 10.0 * 0.1 + 1.0 * 0.5 = 5.0 + 1.0 + 0.5 = 6.5 >= 4.0
    severity_high_depth = detector._classify_severity(
        length_mm=10.0,
        depth_mm=2.5,  # Deep wrinkle
        width_mm=1.0
    )
    assert severity_high_depth == SeverityLevel.HIGH, \
        "Deep wrinkles should be classified as HIGH severity"
    
    # Test HIGH severity: long wrinkle
    # Score = 1.5 * 2.0 + 50.0 * 0.1 + 1.0 * 0.5 = 3.0 + 5.0 + 0.5 = 8.5 >= 4.0
    severity_high_length = detector._classify_severity(
        length_mm=50.0,  # Very long
        depth_mm=1.5,
        width_mm=1.0
    )
    assert severity_high_length == SeverityLevel.HIGH, \
        "Long wrinkles should be classified as HIGH severity"


def test_severity_depth_priority(detector):
    """Test that depth is the primary factor in severity classification."""
    # Two wrinkles with same length and width but different depths
    severity_shallow = detector._classify_severity(
        length_mm=20.0,
        depth_mm=0.7,
        width_mm=0.8
    )
    
    severity_deep = detector._classify_severity(
        length_mm=20.0,
        depth_mm=2.0,
        width_mm=0.8
    )
    
    # Deeper wrinkle should have higher severity
    severity_order = [SeverityLevel.MICRO, SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH]
    assert severity_order.index(severity_deep) > severity_order.index(severity_shallow), \
        "Depth should be the primary factor in severity classification"


def test_severity_consistency(detector):
    """Test that severity classification is consistent and deterministic."""
    # Same inputs should always produce same output
    # Score = 0.9 * 2.0 + 15.0 * 0.1 + 0.9 * 0.5 = 1.8 + 1.5 + 0.45 = 3.75 < 4.0 (MEDIUM)
    for _ in range(5):
        severity = detector._classify_severity(
            length_mm=15.0,
            depth_mm=0.9,
            width_mm=0.9
        )
        assert severity == SeverityLevel.MEDIUM, \
            "Severity classification should be deterministic"


def test_regional_density(detector, sample_image_with_wrinkles):
    """Test regional density calculation."""
    analysis = detector.detect_wrinkles(
        sample_image_with_wrinkles,
        pixel_to_mm_scale=0.1
    )
    
    # Check regional density structure
    assert len(analysis.regional_density) == len(FacialRegion)
    
    for region, density in analysis.regional_density.items():
        assert isinstance(region, FacialRegion)
        assert isinstance(density, RegionalDensity)
        assert density.wrinkle_count >= 0
        assert density.total_length_mm >= 0.0
        assert density.density_score >= 0.0
        assert density.average_depth_mm >= 0.0
        assert density.average_width_mm >= 0.0


def test_texture_grading(detector, sample_image_with_wrinkles):
    """Test texture grading."""
    analysis = detector.detect_wrinkles(
        sample_image_with_wrinkles,
        pixel_to_mm_scale=0.1
    )
    
    # Check overall texture grade
    assert isinstance(analysis.texture_grade, TextureGrade)
    assert analysis.texture_grade in {TextureGrade.SMOOTH, TextureGrade.MODERATE, TextureGrade.COARSE}
    
    # Check regional texture grades (Requirement 2.8)
    assert isinstance(analysis.regional_texture_grades, dict)
    assert len(analysis.regional_texture_grades) == len(FacialRegion), \
        "Should have texture grade for each facial region"
    
    # Verify all regions have valid texture grades
    for region, grade in analysis.regional_texture_grades.items():
        assert isinstance(region, FacialRegion), \
            f"Region key must be FacialRegion, got {type(region)}"
        assert isinstance(grade, TextureGrade), \
            f"Grade must be TextureGrade for region {region}, got {type(grade)}"
        assert grade in {TextureGrade.SMOOTH, TextureGrade.MODERATE, TextureGrade.COARSE}, \
            f"Invalid texture grade for region {region}: {grade}"


def test_regional_texture_grading_smooth_region(detector):
    """Test that regions with few wrinkles are graded as smooth."""
    # Create image with wrinkles only in one area (top)
    image = np.ones((512, 512, 3), dtype=np.float32) * 0.65
    
    # Add wrinkles only in forehead area (top 30% of image)
    for y in range(50, 150, 10):
        cv2.line(image, (100, y), (400, y), (0.4, 0.4, 0.4), 2)
    
    analysis = detector.detect_wrinkles(image, pixel_to_mm_scale=0.1)
    
    # Regions without wrinkles should be smooth
    regions_with_no_wrinkles = [
        region for region, density in analysis.regional_density.items()
        if density.wrinkle_count == 0
    ]
    
    for region in regions_with_no_wrinkles:
        assert analysis.regional_texture_grades[region] == TextureGrade.SMOOTH, \
            f"Region {region} with no wrinkles should be graded as SMOOTH"


def test_regional_texture_grading_coarse_region(detector):
    """Test that regions with many deep wrinkles are graded as coarse."""
    # Create image with many deep wrinkles in one area
    image = np.ones((512, 512, 3), dtype=np.float32) * 0.65
    
    # Add many deep wrinkles in forehead area
    for y in range(50, 150, 5):
        cv2.line(image, (100, y), (400, y), (0.2, 0.2, 0.2), 4)  # Deep, dark lines
    
    analysis = detector.detect_wrinkles(image, pixel_to_mm_scale=0.1)
    
    # At least one region should have wrinkles
    regions_with_wrinkles = [
        region for region, density in analysis.regional_density.items()
        if density.wrinkle_count > 5  # Significant number of wrinkles
    ]
    
    # Regions with many wrinkles should not all be smooth
    if regions_with_wrinkles:
        grades = [analysis.regional_texture_grades[r] for r in regions_with_wrinkles]
        assert any(g != TextureGrade.SMOOTH for g in grades), \
            "Regions with many wrinkles should not all be graded as SMOOTH"


def test_regional_texture_consistency(detector, sample_image_with_wrinkles):
    """Test that regional texture grading is consistent with regional density."""
    analysis = detector.detect_wrinkles(
        sample_image_with_wrinkles,
        pixel_to_mm_scale=0.1
    )
    
    # Regions with higher density should generally have coarser texture
    for region, grade in analysis.regional_texture_grades.items():
        density = analysis.regional_density[region]
        
        # If region has no wrinkles, it should be smooth
        if density.wrinkle_count == 0:
            assert grade == TextureGrade.SMOOTH, \
                f"Region {region} with no wrinkles should be SMOOTH, got {grade}"
        
        # If region has very high density, it should not be smooth
        if density.density_score > 20:  # High density threshold
            assert grade != TextureGrade.SMOOTH, \
                f"Region {region} with high density ({density.density_score}) " \
                f"should not be SMOOTH"


def test_task_9_5_skin_texture_grading_complete(detector):
    """
    Comprehensive test for Task 9.5: Implement skin texture grading
    
    Validates Requirement 2.8: "WHEN wrinkle analysis is complete, 
    THE Detection_Engine SHALL generate a skin texture grading for each facial region"
    
    This test verifies:
    1. Wrinkle distribution analysis is performed
    2. Micro-wrinkle count is considered
    3. Texture is graded as smooth/moderate/coarse
    4. Grading is provided for each facial region
    """
    # Create test image with varied texture across regions
    image = np.ones((512, 512, 3), dtype=np.float32) * 0.65
    
    # Forehead: Many fine lines (moderate texture)
    for y in range(50, 120, 8):
        cv2.line(image, (100, y), (400, y), (0.55, 0.55, 0.55), 1)
    
    # Cheeks: Few wrinkles (smooth texture)
    cv2.line(image, (100, 300), (150, 320), (0.60, 0.60, 0.60), 1)
    
    # Nasolabial: Deep wrinkles (coarse texture)
    cv2.line(image, (200, 250), (180, 350), (0.3, 0.3, 0.3), 4)
    cv2.line(image, (312, 250), (332, 350), (0.3, 0.3, 0.3), 4)
    
    # Run analysis
    analysis = detector.detect_wrinkles(image, pixel_to_mm_scale=0.1)
    
    # Verify 1: Wrinkle distribution is analyzed
    assert analysis.total_wrinkle_count > 0, \
        "Should detect wrinkles for texture analysis"
    
    # Verify 2: Micro-wrinkle count is tracked
    assert analysis.micro_wrinkle_count >= 0, \
        "Should track micro-wrinkle count"
    
    # Verify 3: Overall texture grade is provided
    assert isinstance(analysis.texture_grade, TextureGrade), \
        "Should provide overall texture grade"
    assert analysis.texture_grade in {TextureGrade.SMOOTH, TextureGrade.MODERATE, TextureGrade.COARSE}, \
        f"Texture grade must be smooth/moderate/coarse, got {analysis.texture_grade}"
    
    # Verify 4: Regional texture grades are provided (Requirement 2.8)
    assert hasattr(analysis, 'regional_texture_grades'), \
        "Analysis must include regional_texture_grades attribute"
    
    assert isinstance(analysis.regional_texture_grades, dict), \
        "Regional texture grades must be a dictionary"
    
    assert len(analysis.regional_texture_grades) == len(FacialRegion), \
        f"Must provide texture grade for all {len(FacialRegion)} facial regions, " \
        f"got {len(analysis.regional_texture_grades)}"
    
    # Verify all regions have valid grades
    for region in FacialRegion:
        assert region in analysis.regional_texture_grades, \
            f"Missing texture grade for region {region.value}"
        
        grade = analysis.regional_texture_grades[region]
        assert isinstance(grade, TextureGrade), \
            f"Grade for region {region.value} must be TextureGrade, got {type(grade)}"
        
        assert grade in {TextureGrade.SMOOTH, TextureGrade.MODERATE, TextureGrade.COARSE}, \
            f"Grade for region {region.value} must be smooth/moderate/coarse, got {grade}"
    
    # Verify 5: Grading is based on wrinkle distribution and micro-wrinkle count
    # Regions with no wrinkles should be smooth
    for region, grade in analysis.regional_texture_grades.items():
        density = analysis.regional_density[region]
        region_wrinkles = [w for w in analysis.wrinkles if w.region == region]
        region_micro_count = sum(1 for w in region_wrinkles if w.severity == SeverityLevel.MICRO)
        
        if density.wrinkle_count == 0:
            assert grade == TextureGrade.SMOOTH, \
                f"Region {region.value} with no wrinkles should be SMOOTH"
        
        # Regions with many micro-wrinkles should not be smooth
        if region_micro_count > 5:
            assert grade != TextureGrade.SMOOTH, \
                f"Region {region.value} with {region_micro_count} micro-wrinkles " \
                f"should not be SMOOTH"
    
    # Verify 6: Grading considers depth (coarse texture has deeper wrinkles)
    for region, grade in analysis.regional_texture_grades.items():
        density = analysis.regional_density[region]
        
        if grade == TextureGrade.COARSE and density.wrinkle_count > 0:
            # Coarse regions should have significant depth or density
            assert density.average_depth_mm > 0.5 or density.density_score > 5, \
                f"Region {region.value} graded as COARSE should have " \
                f"significant depth or density"
    
    print(f"\n✓ Task 9.5 Complete: Skin texture grading implemented")
    print(f"  - Overall texture: {analysis.texture_grade.value}")
    print(f"  - Total wrinkles: {analysis.total_wrinkle_count}")
    print(f"  - Micro-wrinkles: {analysis.micro_wrinkle_count}")
    print(f"  - Regional grades:")
    for region, grade in analysis.regional_texture_grades.items():
        density = analysis.regional_density[region]
        print(f"    • {region.value}: {grade.value} " 
              f"({density.wrinkle_count} wrinkles, "
              f"density={density.density_score:.1f}/cm²)")



# ============================================================================
# Unit Tests - Attribute Measurement
# ============================================================================

def test_length_measurement(detector):
    """Test wrinkle length measurement."""
    # Create a simple straight line centerline
    centerline = np.array([[0, 0], [100, 0]], dtype=np.float32)
    
    length_mm = detector._measure_length(centerline, pixel_to_mm_scale=0.1)
    
    # Should be approximately 10mm (100 pixels * 0.1 mm/pixel)
    assert abs(length_mm - 10.0) < 0.1


def test_depth_measurement(detector):
    """Test wrinkle depth measurement."""
    # Create a depth map with known values
    depth_map = np.zeros((100, 100), dtype=np.uint8)
    depth_map[40:60, 40:60] = 128  # Medium depth region
    
    centerline = np.array([[50, 50], [50, 51], [50, 52]], dtype=np.float32)
    
    depth_mm = detector._measure_depth(centerline, depth_map, pixel_to_mm_scale=0.1)
    
    # Should be approximately 2.5mm (128/255 * 5mm)
    assert 2.0 < depth_mm < 3.0


def test_width_measurement(detector):
    """Test wrinkle width measurement."""
    # Create a wrinkle mask with known width
    wrinkle_mask = np.zeros((100, 100), dtype=np.uint8)
    wrinkle_mask[45:55, 20:80] = 255  # 10 pixels wide
    
    centerline = np.array([[50, 50], [51, 50], [52, 50]], dtype=np.float32)
    
    width_mm = detector._measure_width(centerline, wrinkle_mask, pixel_to_mm_scale=0.1)
    
    # Should be approximately 1mm (10 pixels * 0.1 mm/pixel)
    assert 0.5 < width_mm < 1.5


# ============================================================================
# Unit Tests - Training Pipeline
# ============================================================================

def test_training_config():
    """Test training configuration."""
    config = TrainingConfig(
        batch_size=16,
        learning_rate=2e-4,
        num_epochs=50
    )
    
    assert config.batch_size == 16
    assert config.learning_rate == 2e-4
    assert config.num_epochs == 50


def test_training_pipeline_initialization():
    """Test training pipeline initialization."""
    model = EdgeAwareCNN()
    config = TrainingConfig()
    pipeline = TrainingPipeline(model, config)
    
    assert pipeline.model is model
    assert pipeline.config is config
    assert isinstance(pipeline.training_history, list)


# ============================================================================
# Property-Based Tests
# ============================================================================

@given(image=normalized_image_strategy())
@settings(max_examples=5, deadline=None)
def test_property_complete_wrinkle_detection(image):
    """
    Property 6: Complete Wrinkle Detection
    
    **Validates: Requirements 2.1**
    
    This property verifies that wrinkle detection is complete and consistent:
    1. All detected wrinkles must have valid attributes
    2. Detection must be deterministic (same input → same output)
    3. All wrinkles must have valid spatial properties
    4. Regional coverage must be complete
    """
    detector = WrinkleDetector()
    
    # Run detection
    analysis = detector.detect_wrinkles(image, pixel_to_mm_scale=0.1)
    
    # Property 1: All wrinkles must have valid attributes
    for wrinkle in analysis.wrinkles:
        # Valid ID
        assert wrinkle.wrinkle_id > 0, "Wrinkle ID must be positive"
        
        # Valid centerline
        assert len(wrinkle.centerline) >= 2, "Centerline must have at least 2 points"
        assert wrinkle.centerline.shape[1] == 2, "Centerline must be (N, 2)"
        
        # Valid measurements
        assert wrinkle.length_mm > 0.0, f"Length must be positive: {wrinkle.length_mm}"
        assert wrinkle.depth_mm >= 0.0, f"Depth must be non-negative: {wrinkle.depth_mm}"
        assert wrinkle.width_mm > 0.0, f"Width must be positive: {wrinkle.width_mm}"
        
        # Valid severity
        assert isinstance(wrinkle.severity, SeverityLevel), "Invalid severity type"
        
        # Valid region
        assert isinstance(wrinkle.region, FacialRegion), "Invalid region type"
        
        # Valid confidence
        assert 0.0 <= wrinkle.confidence <= 1.0, \
            f"Confidence must be in [0, 1]: {wrinkle.confidence}"
        
        # Valid bounding box
        x, y, w, h = wrinkle.bounding_box
        assert w > 0 and h > 0, "Bounding box dimensions must be positive"
        assert 0 <= x < image.shape[1], "Bounding box x out of bounds"
        assert 0 <= y < image.shape[0], "Bounding box y out of bounds"
    
    # Property 2: Deterministic detection
    analysis2 = detector.detect_wrinkles(image, pixel_to_mm_scale=0.1)
    
    assert analysis.total_wrinkle_count == analysis2.total_wrinkle_count, \
        "Detection must be deterministic (same number of wrinkles)"
    
    # Property 3: Aggregate statistics must be consistent
    if len(analysis.wrinkles) > 0:
        assert analysis.total_wrinkle_count == len(analysis.wrinkles), \
            "Total count must match wrinkle list length"
        
        micro_count = sum(1 for w in analysis.wrinkles if w.severity == SeverityLevel.MICRO)
        assert analysis.micro_wrinkle_count == micro_count, \
            "Micro wrinkle count must be accurate"
        
        avg_depth = np.mean([w.depth_mm for w in analysis.wrinkles])
        assert abs(analysis.average_depth_mm - avg_depth) < 0.01, \
            "Average depth must be accurate"
        
        avg_length = np.mean([w.length_mm for w in analysis.wrinkles])
        assert abs(analysis.average_length_mm - avg_length) < 0.01, \
            "Average length must be accurate"
    
    # Property 4: Regional coverage must be complete
    assert len(analysis.regional_density) == len(FacialRegion), \
        "All facial regions must be analyzed"
    
    for region in FacialRegion:
        assert region in analysis.regional_density, \
            f"Region {region} missing from analysis"


@given(image=normalized_image_strategy())
@settings(max_examples=5, deadline=None)
def test_property_accurate_attribute_measurement(image):
    """
    Property 7: Accurate Wrinkle Attribute Measurement
    
    **Validates: Requirements 2.2, 2.3, 2.4**
    
    This property verifies that attribute measurements are accurate and consistent:
    1. Measurements must be physically plausible
    2. Measurements must scale correctly with pixel-to-mm conversion
    3. Measurements must be consistent across multiple runs
    4. Severity classification must correlate with measurements
    """
    detector = WrinkleDetector()
    
    # Run detection with known scale
    scale1 = 0.1  # 0.1 mm per pixel
    analysis1 = detector.detect_wrinkles(image, pixel_to_mm_scale=scale1)
    
    # Property 1: Measurements must be physically plausible
    for wrinkle in analysis1.wrinkles:
        # Length: typical facial wrinkles are 1-100mm
        assert 0.1 <= wrinkle.length_mm <= 200.0, \
            f"Length out of plausible range: {wrinkle.length_mm}mm"
        
        # Depth: typical wrinkles are 0-5mm deep
        assert 0.0 <= wrinkle.depth_mm <= 10.0, \
            f"Depth out of plausible range: {wrinkle.depth_mm}mm"
        
        # Width: typical wrinkles are 0.1-5mm wide
        assert 0.05 <= wrinkle.width_mm <= 10.0, \
            f"Width out of plausible range: {wrinkle.width_mm}mm"
    
    # Property 2: Measurements must scale correctly
    scale2 = 0.2  # Double the scale
    analysis2 = detector.detect_wrinkles(image, pixel_to_mm_scale=scale2)
    
    # Number of detected wrinkles should be the same (scale-invariant detection)
    assert analysis1.total_wrinkle_count == analysis2.total_wrinkle_count, \
        "Number of wrinkles should be scale-invariant"
    
    # Measurements should scale linearly
    if len(analysis1.wrinkles) > 0 and len(analysis2.wrinkles) > 0:
        scale_ratio = scale2 / scale1
        
        # Average length should scale linearly
        ratio = analysis2.average_length_mm / (analysis1.average_length_mm + 1e-8)
        assert abs(ratio - scale_ratio) / scale_ratio < 0.2, \
            f"Length should scale linearly: expected {scale_ratio}, got {ratio}"
        
        # Average width should scale linearly
        avg_width1 = np.mean([w.width_mm for w in analysis1.wrinkles])
        avg_width2 = np.mean([w.width_mm for w in analysis2.wrinkles])
        if avg_width1 > 0:
            ratio = avg_width2 / avg_width1
            assert abs(ratio - scale_ratio) / scale_ratio < 0.2, \
                f"Width should scale linearly: expected {scale_ratio}, got {ratio}"
    
    # Property 3: Severity must correlate with measurements
    for wrinkle in analysis1.wrinkles:
        if wrinkle.severity == SeverityLevel.MICRO:
            assert wrinkle.depth_mm < 0.5, \
                "Micro wrinkles must have depth < 0.5mm"
        elif wrinkle.severity == SeverityLevel.HIGH:
            # High severity should have significant depth or length
            assert wrinkle.depth_mm > 1.0 or wrinkle.length_mm > 20.0, \
                "High severity wrinkles must have significant depth or length"


def wrinkle_attributes_strategy():
    """Strategy for generating random wrinkle attributes."""
    return st.builds(
        lambda l, d, w: (l, d, w),
        l=st.floats(min_value=0.5, max_value=100.0),  # length_mm
        d=st.floats(min_value=0.0, max_value=5.0),    # depth_mm
        w=st.floats(min_value=0.1, max_value=3.0)     # width_mm
    )


@given(attributes=wrinkle_attributes_strategy())
@settings(max_examples=100, deadline=None)
def test_property_consistent_wrinkle_classification(attributes):
    """
    Property 8: Consistent Wrinkle Classification
    
    **Validates: Requirements 2.5**
    
    This property verifies that wrinkle severity classification is consistent
    with the measured attributes (length, depth, width) according to defined
    thresholds. The classification must be:
    1. Deterministic: same attributes always produce same classification
    2. Consistent with thresholds: classifications follow the defined rules
    3. Monotonic: increasing severity attributes should not decrease severity
    4. Complete: all valid attribute combinations produce a valid severity
    """
    detector = WrinkleDetector()
    length_mm, depth_mm, width_mm = attributes
    
    # Property 1: Deterministic classification
    # Same inputs should always produce the same output
    severity1 = detector._classify_severity(length_mm, depth_mm, width_mm)
    severity2 = detector._classify_severity(length_mm, depth_mm, width_mm)
    
    assert severity1 == severity2, \
        f"Classification must be deterministic: {severity1} != {severity2}"
    
    # Property 2: Consistent with threshold rules
    # Micro-wrinkle threshold: depth < 0.5mm
    if depth_mm < 0.5:
        assert severity1 == SeverityLevel.MICRO, \
            f"Wrinkles with depth {depth_mm}mm < 0.5mm must be MICRO, got {severity1}"
    else:
        assert severity1 != SeverityLevel.MICRO, \
            f"Wrinkles with depth {depth_mm}mm >= 0.5mm must not be MICRO, got {severity1}"
    
    # Property 3: Severity score calculation consistency
    # Calculate expected severity score
    severity_score = depth_mm * 2.0 + length_mm * 0.1 + width_mm * 0.5
    
    if depth_mm >= 0.5:  # Not a micro-wrinkle
        if severity_score < 2.0:
            expected = SeverityLevel.LOW
        elif severity_score < 4.0:
            expected = SeverityLevel.MEDIUM
        else:
            expected = SeverityLevel.HIGH
        
        assert severity1 == expected, \
            f"Classification inconsistent with score {severity_score:.2f}: " \
            f"expected {expected}, got {severity1} " \
            f"(length={length_mm:.2f}, depth={depth_mm:.2f}, width={width_mm:.2f})"
    
    # Property 4: Valid severity level
    valid_severities = {SeverityLevel.MICRO, SeverityLevel.LOW, 
                       SeverityLevel.MEDIUM, SeverityLevel.HIGH}
    assert severity1 in valid_severities, \
        f"Classification must produce valid severity level, got {severity1}"
    
    # Property 5: Monotonicity - increasing depth should not decrease severity
    # (when depth is above micro threshold)
    if depth_mm >= 0.5 and depth_mm < 4.5:  # Leave room to increase
        increased_depth = depth_mm + 0.5
        severity_increased = detector._classify_severity(length_mm, increased_depth, width_mm)
        
        severity_order = [SeverityLevel.MICRO, SeverityLevel.LOW, 
                         SeverityLevel.MEDIUM, SeverityLevel.HIGH]
        
        # Severity should not decrease when depth increases
        if severity1 in severity_order and severity_increased in severity_order:
            assert severity_order.index(severity_increased) >= severity_order.index(severity1), \
                f"Increasing depth from {depth_mm:.2f} to {increased_depth:.2f} " \
                f"decreased severity from {severity1} to {severity_increased}"
    
    # Property 6: Boundary consistency
    # Test that values near thresholds are handled correctly
    if abs(depth_mm - 0.5) < 0.01:  # Near micro threshold
        # Just below threshold should be MICRO
        severity_below = detector._classify_severity(length_mm, 0.49, width_mm)
        assert severity_below == SeverityLevel.MICRO, \
            "Depth just below 0.5mm threshold should be MICRO"
        
        # Just above threshold should not be MICRO
        severity_above = detector._classify_severity(length_mm, 0.51, width_mm)
        assert severity_above != SeverityLevel.MICRO, \
            "Depth just above 0.5mm threshold should not be MICRO"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_empty_image():
    """Test detection on uniform image (no wrinkles)."""
    detector = WrinkleDetector()
    
    # Uniform image - smaller for speed
    image = np.ones((256, 256, 3), dtype=np.float32) * 0.7
    
    analysis = detector.detect_wrinkles(image, pixel_to_mm_scale=0.1)
    
    # Should detect few or no wrinkles
    assert analysis.total_wrinkle_count >= 0
    assert analysis.texture_grade == TextureGrade.SMOOTH


def test_very_textured_image():
    """Test detection on highly textured image."""
    detector = WrinkleDetector()
    
    # Add lots of noise (texture) - use smaller image for speed
    image = np.random.rand(256, 256, 3).astype(np.float32) * 0.3 + 0.4
    
    analysis = detector.detect_wrinkles(image, pixel_to_mm_scale=0.1)
    
    # Should detect wrinkles (texture grade may vary based on detection)
    assert isinstance(analysis.texture_grade, TextureGrade)


def test_small_image():
    """Test detection on small image."""
    detector = WrinkleDetector()
    
    # Small image
    image = np.ones((128, 128, 3), dtype=np.float32) * 0.7
    cv2.line(image, (20, 60), (100, 60), (0.4, 0.4, 0.4), 2)
    
    analysis = detector.detect_wrinkles(image, pixel_to_mm_scale=0.1)
    
    # Should still work
    assert analysis.depth_map.shape == (128, 128)


def test_with_landmarks():
    """Test detection with facial landmarks."""
    detector = WrinkleDetector()
    
    # Create sample image - smaller for speed
    image = np.ones((256, 256, 3), dtype=np.float32) * 0.7
    cv2.line(image, (50, 50), (200, 50), (0.4, 0.4, 0.4), 3)
    
    # Create mock landmarks (468 points) - scaled to smaller image
    landmarks = np.random.rand(468, 3) * 256
    
    analysis = detector.detect_wrinkles(image, landmarks=landmarks, pixel_to_mm_scale=0.1)
    
    # Should work with landmarks
    assert analysis.total_wrinkle_count >= 0


# ============================================================================
# Property-Based Test for Regional Density
# ============================================================================

@given(image=normalized_image_strategy())
@settings(max_examples=10, deadline=None)
def test_property_accurate_regional_density_calculation(image):
    """
    Property 9: Accurate Regional Density Calculation
    
    **Validates: Requirements 2.6**
    
    This property verifies that regional wrinkle density calculation is accurate
    and consistent:
    1. All facial regions must be analyzed
    2. Density scores must accurately reflect wrinkle count and region area
    3. Regional statistics must be consistent with detected wrinkles
    4. Density calculations must be non-negative and physically plausible
    5. Sum of regional wrinkle counts must equal total wrinkle count
    """
    detector = WrinkleDetector()
    
    # Run detection
    analysis = detector.detect_wrinkles(image, pixel_to_mm_scale=0.1)
    
    # Property 1: All facial regions must be analyzed
    assert len(analysis.regional_density) == len(FacialRegion), \
        f"All {len(FacialRegion)} facial regions must be analyzed, " \
        f"got {len(analysis.regional_density)}"
    
    for region in FacialRegion:
        assert region in analysis.regional_density, \
            f"Region {region.value} must be present in regional density analysis"
    
    # Property 2: Regional statistics must be valid
    for region, density in analysis.regional_density.items():
        # Valid region type
        assert isinstance(region, FacialRegion), \
            f"Region key must be FacialRegion enum, got {type(region)}"
        
        # Valid density object
        assert isinstance(density, RegionalDensity), \
            f"Density value must be RegionalDensity, got {type(density)}"
        
        # Non-negative counts
        assert density.wrinkle_count >= 0, \
            f"Wrinkle count must be non-negative for {region.value}, " \
            f"got {density.wrinkle_count}"
        
        # Non-negative measurements
        assert density.total_length_mm >= 0.0, \
            f"Total length must be non-negative for {region.value}, " \
            f"got {density.total_length_mm}"
        
        assert density.density_score >= 0.0, \
            f"Density score must be non-negative for {region.value}, " \
            f"got {density.density_score}"
        
        assert density.average_depth_mm >= 0.0, \
            f"Average depth must be non-negative for {region.value}, " \
            f"got {density.average_depth_mm}"
        
        assert density.average_width_mm >= 0.0, \
            f"Average width must be non-negative for {region.value}, " \
            f"got {density.average_width_mm}"
        
        # Plausible density scores (wrinkles per cm²)
        # Typical facial skin: 0-50 wrinkles per cm² is reasonable
        assert density.density_score <= 100.0, \
            f"Density score seems implausibly high for {region.value}: " \
            f"{density.density_score} wrinkles/cm²"
    
    # Property 3: Sum of regional counts must equal total count
    total_regional_count = sum(
        density.wrinkle_count 
        for density in analysis.regional_density.values()
    )
    
    assert total_regional_count == analysis.total_wrinkle_count, \
        f"Sum of regional wrinkle counts ({total_regional_count}) must equal " \
        f"total wrinkle count ({analysis.total_wrinkle_count})"
    
    # Property 4: Regional statistics must be consistent with wrinkles in that region
    for region, density in analysis.regional_density.items():
        # Get wrinkles in this region
        wrinkles_in_region = [w for w in analysis.wrinkles if w.region == region]
        
        # Count must match
        assert density.wrinkle_count == len(wrinkles_in_region), \
            f"Region {region.value} density count ({density.wrinkle_count}) " \
            f"must match actual wrinkles in region ({len(wrinkles_in_region)})"
        
        if len(wrinkles_in_region) > 0:
            # Total length must match sum of wrinkle lengths
            expected_total_length = sum(w.length_mm for w in wrinkles_in_region)
            assert abs(density.total_length_mm - expected_total_length) < 0.01, \
                f"Region {region.value} total length ({density.total_length_mm}mm) " \
                f"must match sum of wrinkle lengths ({expected_total_length}mm)"
            
            # Average depth must match mean of wrinkle depths
            expected_avg_depth = np.mean([w.depth_mm for w in wrinkles_in_region])
            assert abs(density.average_depth_mm - expected_avg_depth) < 0.01, \
                f"Region {region.value} average depth ({density.average_depth_mm}mm) " \
                f"must match mean of wrinkle depths ({expected_avg_depth}mm)"
            
            # Average width must match mean of wrinkle widths
            expected_avg_width = np.mean([w.width_mm for w in wrinkles_in_region])
            assert abs(density.average_width_mm - expected_avg_width) < 0.01, \
                f"Region {region.value} average width ({density.average_width_mm}mm) " \
                f"must match mean of wrinkle widths ({expected_avg_width}mm)"
        else:
            # If no wrinkles, all statistics should be zero
            assert density.total_length_mm == 0.0, \
                f"Region {region.value} with no wrinkles should have zero total length"
            assert density.average_depth_mm == 0.0, \
                f"Region {region.value} with no wrinkles should have zero average depth"
            assert density.average_width_mm == 0.0, \
                f"Region {region.value} with no wrinkles should have zero average width"
            assert density.density_score == 0.0, \
                f"Region {region.value} with no wrinkles should have zero density score"
    
    # Property 5: Density score calculation must be consistent
    # Density = count / area, so if we know count and density, we can verify area is reasonable
    for region, density in analysis.regional_density.items():
        if density.density_score > 0:
            # Calculate implied area from count and density
            implied_area_cm2 = density.wrinkle_count / density.density_score
            
            # Facial regions should be between 1 cm² and 200 cm²
            assert 0.1 <= implied_area_cm2 <= 500.0, \
                f"Region {region.value} implied area ({implied_area_cm2:.2f} cm²) " \
                f"seems implausible (count={density.wrinkle_count}, " \
                f"density={density.density_score:.2f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
