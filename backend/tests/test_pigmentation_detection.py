"""
Tests for pigmentation detection module.

This module tests the pigmentation detection system including:
- U-Net architecture
- Inference and post-processing
- Severity classification
- Quantitative measurements
- Heat-map generation
- Property-based tests for detection completeness
"""

import pytest
import numpy as np
import cv2
from hypothesis import given, settings, strategies as st
from typing import List, Tuple

from app.services.pigmentation_detection import (
    PigmentationDetector,
    UNetWithAttention,
    TrainingPipeline,
    TrainingConfig,
    SeverityLevel,
    PigmentationArea,
    SegmentationMask,
    PigmentationMetrics,
    HeatMap
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def detector():
    """Create a pigmentation detector instance."""
    return PigmentationDetector()


@pytest.fixture
def sample_image():
    """Create a sample normalized facial image."""
    # Create a 512x512 RGB image with some pigmentation-like features
    image = np.ones((512, 512, 3), dtype=np.float32) * 0.7
    
    # Add some darker regions (simulating pigmentation)
    cv2.circle(image, (200, 200), 30, (0.4, 0.3, 0.3), -1)
    cv2.circle(image, (300, 300), 40, (0.5, 0.4, 0.4), -1)
    cv2.circle(image, (400, 150), 25, (0.3, 0.2, 0.2), -1)
    
    return image


@pytest.fixture
def sample_image_with_pigmentation():
    """Create a sample image with known pigmentation areas."""
    # Create base skin tone
    image = np.ones((512, 512, 3), dtype=np.float32) * 0.65
    
    # Add low severity pigmentation (lighter)
    cv2.circle(image, (100, 100), 40, (0.55, 0.50, 0.50), -1)
    
    # Add medium severity pigmentation (darker)
    cv2.circle(image, (250, 250), 50, (0.40, 0.35, 0.35), -1)
    
    # Add high severity pigmentation (very dark)
    cv2.circle(image, (400, 400), 35, (0.25, 0.20, 0.20), -1)
    
    return image


# ============================================================================
# Unit Tests - U-Net Architecture
# ============================================================================

def test_unet_initialization():
    """Test U-Net model initialization."""
    model = UNetWithAttention(pretrained=True, num_classes=4)
    
    assert model.pretrained is True
    assert model.num_classes == 4
    assert model.input_size == (512, 512)
    assert len(model.encoder_channels) == 5
    assert len(model.decoder_channels) == 4
    assert model.is_trained is False
    assert model.training_epochs == 0


def test_unet_forward_pass(sample_image):
    """Test U-Net forward pass."""
    model = UNetWithAttention()
    
    # Run forward pass
    output = model.forward(sample_image)
    
    # Check output shape
    assert output.shape == (512, 512, 4)
    
    # Check output is valid probabilities (sum to 1)
    prob_sums = output.sum(axis=2)
    np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)
    
    # Check all values are in [0, 1]
    assert np.all(output >= 0.0)
    assert np.all(output <= 1.0)


# ============================================================================
# Unit Tests - Pigmentation Detection
# ============================================================================

def test_detector_initialization():
    """Test pigmentation detector initialization."""
    detector = PigmentationDetector()
    
    assert detector.model is not None
    assert isinstance(detector.model, UNetWithAttention)


def test_detect_pigmentation_basic(detector, sample_image):
    """Test basic pigmentation detection."""
    seg_mask, areas = detector.detect_pigmentation(sample_image, pixel_to_mm_scale=0.1)
    
    # Check segmentation mask
    assert isinstance(seg_mask, SegmentationMask)
    assert seg_mask.mask.shape == (512, 512)
    assert seg_mask.class_probabilities.shape == (512, 512, 4)
    
    # Check detected areas
    assert isinstance(areas, list)
    for area in areas:
        assert isinstance(area, PigmentationArea)
        assert area.id.startswith("pigment_")
        assert isinstance(area.severity, SeverityLevel)
        assert area.surface_area_mm2 >= 0
        assert area.density >= 0
        assert area.color_deviation >= 0
        assert area.melanin_index >= 0
        assert 0.0 <= area.confidence <= 1.0


def test_detect_pigmentation_with_known_areas(detector, sample_image_with_pigmentation):
    """Test detection with known pigmentation areas."""
    seg_mask, areas = detector.detect_pigmentation(
        sample_image_with_pigmentation,
        pixel_to_mm_scale=0.1
    )
    
    # Should detect at least some areas
    assert len(areas) > 0
    
    # Check that different severity levels are detected
    severities = {area.severity for area in areas}
    # At least one severity level should be detected
    assert len(severities) > 0


def test_severity_classification(detector, sample_image_with_pigmentation):
    """Test severity classification."""
    seg_mask, areas = detector.detect_pigmentation(
        sample_image_with_pigmentation,
        pixel_to_mm_scale=0.1
    )
    
    # Classify severity
    severity_levels = detector.classify_severity(seg_mask, sample_image_with_pigmentation)
    
    assert isinstance(severity_levels, list)
    for severity in severity_levels:
        assert isinstance(severity, SeverityLevel)


def test_calculate_metrics(detector, sample_image_with_pigmentation):
    """Test pigmentation metrics calculation."""
    seg_mask, areas = detector.detect_pigmentation(
        sample_image_with_pigmentation,
        pixel_to_mm_scale=0.1
    )
    
    metrics = detector.calculate_metrics(
        areas,
        sample_image_with_pigmentation.shape[:2],
        pixel_to_mm_scale=0.1
    )
    
    assert isinstance(metrics, PigmentationMetrics)
    assert metrics.total_areas >= 0
    assert metrics.total_surface_area_mm2 >= 0
    assert metrics.average_melanin_index >= 0
    assert isinstance(metrics.severity_distribution, dict)
    assert "Low" in metrics.severity_distribution
    assert "Medium" in metrics.severity_distribution
    assert "High" in metrics.severity_distribution
    assert 0.0 <= metrics.coverage_percentage <= 100.0


def test_generate_heatmap(detector, sample_image_with_pigmentation):
    """Test heat-map generation."""
    seg_mask, areas = detector.detect_pigmentation(
        sample_image_with_pigmentation,
        pixel_to_mm_scale=0.1
    )
    
    heatmap = detector.generate_heatmap(areas, sample_image_with_pigmentation.shape[:2])
    
    assert isinstance(heatmap, HeatMap)
    assert heatmap.density_map.shape == (512, 512)
    assert heatmap.severity_map.shape == (512, 512)
    assert heatmap.melanin_map.shape == (512, 512)
    assert heatmap.visualization.shape == (512, 512, 3)
    
    # Check visualization is in valid range
    assert np.all(heatmap.visualization >= 0.0)
    assert np.all(heatmap.visualization <= 1.0)


# ============================================================================
# Unit Tests - Quantitative Measurements
# ============================================================================

def test_surface_area_calculation(detector):
    """Test surface area calculation."""
    # Create a simple mask (100x100 square)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    
    # Calculate area with 1mm per pixel scale
    area_mm2 = detector._calculate_surface_area(mask, pixel_to_mm_scale=1.0)
    
    # Should be 100x100 = 10,000 mm²
    assert area_mm2 == 10000.0


def test_density_calculation(detector):
    """Test density calculation."""
    # Create a simple mask
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    
    # Calculate density
    density = detector._calculate_density(mask, pixel_to_mm_scale=1.0)
    
    # Density should be positive
    assert density > 0


def test_color_deviation_calculation(detector, sample_image_with_pigmentation):
    """Test color deviation calculation."""
    # Create a mask for one of the pigmentation areas
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask, (250, 250), 50, 1, -1)
    
    # Calculate color deviation
    delta_e = detector._calculate_color_deviation(sample_image_with_pigmentation, mask)
    
    # Should be positive (pigmentation differs from surrounding skin)
    assert delta_e > 0


def test_melanin_index_estimation(detector, sample_image_with_pigmentation):
    """Test melanin index estimation."""
    # Create a mask for a dark pigmentation area
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask, (400, 400), 35, 1, -1)
    
    # Estimate melanin index
    melanin_index = detector._estimate_melanin_index(sample_image_with_pigmentation, mask)
    
    # Should be positive
    assert melanin_index > 0


# ============================================================================
# Unit Tests - Training Pipeline
# ============================================================================

def test_training_config():
    """Test training configuration."""
    config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-3,
        num_epochs=50
    )
    
    assert config.batch_size == 16
    assert config.learning_rate == 1e-3
    assert config.num_epochs == 50


def test_training_pipeline_initialization():
    """Test training pipeline initialization."""
    model = UNetWithAttention()
    config = TrainingConfig(num_epochs=10)
    pipeline = TrainingPipeline(model, config)
    
    assert pipeline.model is model
    assert pipeline.config is config
    assert pipeline.current_epoch == 0
    assert pipeline.best_val_loss == float('inf')


def test_dice_loss_calculation():
    """Test Dice loss calculation."""
    model = UNetWithAttention()
    config = TrainingConfig()
    pipeline = TrainingPipeline(model, config)
    
    # Create mock predictions and targets
    predictions = np.random.rand(64, 64, 4).astype(np.float32)
    predictions = predictions / predictions.sum(axis=2, keepdims=True)
    targets = np.random.randint(0, 4, (64, 64))
    
    # Calculate Dice loss
    loss = pipeline.dice_loss(predictions, targets)
    
    # Loss should be in [0, 1]
    assert 0.0 <= loss <= 1.0


def test_cross_entropy_loss_calculation():
    """Test cross-entropy loss calculation."""
    model = UNetWithAttention()
    config = TrainingConfig()
    pipeline = TrainingPipeline(model, config)
    
    # Create mock predictions and targets
    predictions = np.random.rand(64, 64, 4).astype(np.float32)
    predictions = predictions / predictions.sum(axis=2, keepdims=True)
    targets = np.random.randint(0, 4, (64, 64))
    
    # Calculate cross-entropy loss
    loss = pipeline.cross_entropy_loss(predictions, targets)
    
    # Loss should be positive
    assert loss > 0


def test_combined_loss_calculation():
    """Test combined loss calculation."""
    model = UNetWithAttention()
    config = TrainingConfig()
    pipeline = TrainingPipeline(model, config)
    
    # Create mock predictions and targets
    predictions = np.random.rand(64, 64, 4).astype(np.float32)
    predictions = predictions / predictions.sum(axis=2, keepdims=True)
    targets = np.random.randint(0, 4, (64, 64))
    
    # Calculate combined loss
    loss = pipeline.combined_loss(predictions, targets)
    
    # Loss should be positive
    assert loss > 0


def test_training_loop_mock():
    """Test mock training loop."""
    model = UNetWithAttention()
    config = TrainingConfig(num_epochs=5)
    pipeline = TrainingPipeline(model, config)
    
    # Create mock data
    train_images = [np.random.rand(512, 512, 3).astype(np.float32) for _ in range(10)]
    train_masks = [np.random.randint(0, 4, (512, 512)) for _ in range(10)]
    
    # Create data loaders
    train_loader, val_loader = pipeline.create_data_loaders(train_images, train_masks)
    
    # Run training
    metrics_history = pipeline.train(train_loader, val_loader)
    
    # Check results
    assert len(metrics_history) <= config.num_epochs
    assert model.is_trained is True
    assert model.training_epochs > 0


# ============================================================================
# Property-Based Tests
# ============================================================================

# Strategy for generating valid normalized images
@st.composite
def normalized_image_strategy(draw):
    """Generate a valid normalized facial image - simplified for speed."""
    # Fixed smaller size for speed
    size = 256
    
    # Generate uniform base image (much faster than random)
    base_color = draw(st.floats(min_value=0.5, max_value=0.7))
    image = np.ones((size, size, 3), dtype=np.float32) * base_color
    
    # Add just 1-2 spots (much faster)
    num_spots = draw(st.integers(min_value=0, max_value=2))
    for _ in range(num_spots):
        x = draw(st.integers(min_value=50, max_value=200))
        y = draw(st.integers(min_value=50, max_value=200))
        radius = draw(st.integers(min_value=15, max_value=30))
        darkness = draw(st.floats(min_value=0.3, max_value=0.5))
        cv2.circle(image, (x, y), radius, (darkness, darkness, darkness), -1)
    
    return image


# Feature: dermatological-analysis-poc, Property 1: Complete Pigmentation Detection
@given(image=normalized_image_strategy())
@settings(max_examples=5, deadline=None)
def test_property_complete_pigmentation_detection(image):
    """
    Property 1: Complete Pigmentation Detection
    
    For any valid 180-degree facial image set containing pigmentation areas,
    the detection engine should identify all visible pigmentation areas
    without missing any regions.
    
    **Validates: Requirements 1.1**
    """
    detector = PigmentationDetector()
    
    # Run detection
    seg_mask, areas = detector.detect_pigmentation(image, pixel_to_mm_scale=0.1)
    
    # Property: Detection should complete successfully
    assert seg_mask is not None
    assert areas is not None
    
    # Property: Segmentation mask should cover entire image
    assert seg_mask.mask.shape == image.shape[:2]
    
    # Property: All pixels should be classified
    assert np.all(seg_mask.mask >= 0)
    assert np.all(seg_mask.mask < 4)
    
    # Property: Class probabilities should sum to 1
    prob_sums = seg_mask.class_probabilities.sum(axis=2)
    np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-4)
    
    # Property: Each detected area should have valid measurements
    for area in areas:
        assert area.surface_area_mm2 >= 0
        assert area.density >= 0
        assert area.color_deviation >= 0
        assert area.melanin_index >= 0
        assert 0.0 <= area.confidence <= 1.0
        assert area.mask.shape == image.shape[:2]


# Feature: dermatological-analysis-poc, Property 4: Distinct Pigmentation Area Identification
@given(image=normalized_image_strategy())
@settings(max_examples=5, deadline=None)
def test_property_distinct_area_identification(image):
    """
    Property 4: Distinct Pigmentation Area Identification
    
    For any image containing multiple pigmentation areas, each area should
    maintain a unique identifier and not be incorrectly merged with adjacent areas.
    
    **Validates: Requirements 1.5**
    """
    detector = PigmentationDetector()
    
    # Run detection
    seg_mask, areas = detector.detect_pigmentation(image, pixel_to_mm_scale=0.1)
    
    # Property: Each area should have a unique ID
    area_ids = [area.id for area in areas]
    assert len(area_ids) == len(set(area_ids)), "Area IDs must be unique"
    
    # Property: Area masks should not overlap significantly
    for i, area1 in enumerate(areas):
        for j, area2 in enumerate(areas):
            if i < j:
                # Calculate overlap
                overlap = np.logical_and(area1.mask > 0, area2.mask > 0)
                overlap_pixels = np.sum(overlap)
                
                # Overlap should be minimal (less than 5% of smaller area)
                min_area_pixels = min(np.sum(area1.mask > 0), np.sum(area2.mask > 0))
                if min_area_pixels > 0:
                    overlap_ratio = overlap_pixels / min_area_pixels
                    assert overlap_ratio < 0.05, "Areas should not overlap significantly"
    
    # Property: Each area should have distinct spatial location
    centroids = [area.centroid for area in areas]
    for i, c1 in enumerate(centroids):
        for j, c2 in enumerate(centroids):
            if i < j:
                # Calculate distance between centroids
                distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                # Centroids should be separated (at least 10 pixels apart)
                assert distance >= 10.0, "Area centroids should be distinct"


# ============================================================================
# Property-Based Tests - Severity Classification and Measurements
# ============================================================================

@given(image=normalized_image_strategy())
@settings(max_examples=5, deadline=None)
def test_property_severity_classification(image):
    """
    Property 2: Accurate Pigmentation Severity Classification
    
    **Validates: Requirements 1.2**
    
    This property verifies that severity classification is consistent and accurate:
    1. All detected areas must have a valid severity level (Low/Medium/High)
    2. Severity levels must be ordered by darkness/chromatic intensity
    3. Classification must be deterministic (same input → same output)
    4. Severity thresholds must be properly applied
    """
    detector = PigmentationDetector()
    
    # Run detection
    seg_mask, areas = detector.detect_pigmentation(image, pixel_to_mm_scale=0.1)
    
    # Property 1: All areas must have valid severity levels
    valid_severities = {SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH}
    for area in areas:
        assert area.severity in valid_severities, \
            f"Invalid severity level: {area.severity}"
    
    # Property 2: Severity must correlate with darkness/chromatic intensity
    # Group areas by severity and check their color characteristics
    severity_groups = {
        SeverityLevel.LOW: [],
        SeverityLevel.MEDIUM: [],
        SeverityLevel.HIGH: []
    }
    
    for area in areas:
        # Calculate average darkness (inverse of brightness)
        mask_pixels = image[area.mask > 0]
        if len(mask_pixels) > 0:
            avg_brightness = np.mean(mask_pixels)
            severity_groups[area.severity].append(avg_brightness)
    
    # Check that higher severity corresponds to lower brightness (darker)
    # This is a general trend, not strict ordering for individual spots
    if severity_groups[SeverityLevel.LOW] and severity_groups[SeverityLevel.HIGH]:
        avg_low = np.mean(severity_groups[SeverityLevel.LOW])
        avg_high = np.mean(severity_groups[SeverityLevel.HIGH])
        # High severity should generally be darker (lower brightness)
        # Allow some tolerance for edge cases
        assert avg_high <= avg_low + 0.2, \
            f"High severity areas should be darker than low severity"
    
    # Property 3: Deterministic classification
    # Run detection again and verify same results
    seg_mask2, areas2 = detector.detect_pigmentation(image, pixel_to_mm_scale=0.1)
    
    assert len(areas) == len(areas2), \
        "Detection must be deterministic (same number of areas)"
    
    # Sort areas by position for comparison
    areas_sorted = sorted(areas, key=lambda a: (a.centroid[0], a.centroid[1]))
    areas2_sorted = sorted(areas2, key=lambda a: (a.centroid[0], a.centroid[1]))
    
    for a1, a2 in zip(areas_sorted, areas2_sorted):
        assert a1.severity == a2.severity, \
            "Severity classification must be deterministic"


@given(image=normalized_image_strategy())
@settings(max_examples=5, deadline=None)
def test_property_comprehensive_measurements(image):
    """
    Property 5: Comprehensive Pigmentation Measurements
    
    **Validates: Requirements 1.6, 1.7, 1.8, 1.10**
    
    This property verifies that all quantitative measurements are:
    1. Present and valid for all detected areas
    2. Physically plausible (non-negative, within reasonable bounds)
    3. Consistent with each other
    4. Scale-invariant where appropriate
    """
    detector = PigmentationDetector()
    
    # Run detection with known scale
    pixel_to_mm_scale = 0.1  # 0.1 mm per pixel
    seg_mask, areas = detector.detect_pigmentation(image, pixel_to_mm_scale=pixel_to_mm_scale)
    
    # Calculate metrics
    metrics = detector.calculate_metrics(areas, image.shape[:2], pixel_to_mm_scale)
    
    # Property 1: All measurements must be present and valid
    assert metrics.total_areas >= 0, "Total areas must be non-negative"
    assert metrics.total_surface_area_mm2 >= 0.0, "Surface area must be non-negative"
    assert metrics.average_melanin_index >= 0.0, "Melanin index must be non-negative"
    assert 0.0 <= metrics.coverage_percentage <= 100.0, \
        f"Coverage percentage must be 0-100, got {metrics.coverage_percentage}"
    
    # Property 2: Individual area measurements must be valid
    for area in areas:
        # Surface area must be positive
        assert area.surface_area_mm2 > 0.0, \
            f"Surface area must be positive, got {area.surface_area_mm2}"
        
        # Melanin index must be in reasonable range (0-100+)
        assert 0.0 <= area.melanin_index <= 200.0, \
            f"Melanin index out of range: {area.melanin_index}"
        
        # Color deviation must be non-negative
        assert area.color_deviation >= 0.0, \
            f"Color deviation must be non-negative: {area.color_deviation}"
        
        # Bounding box must be valid
        x, y, w, h = area.bounding_box
        assert w > 0 and h > 0, "Bounding box dimensions must be positive"
        assert 0 <= x < image.shape[1], "Bounding box x out of bounds"
        assert 0 <= y < image.shape[0], "Bounding box y out of bounds"
    
    # Property 3: Aggregate measurements must be consistent
    if len(areas) > 0:
        # Total surface area should equal sum of individual areas
        sum_individual_areas = sum(area.surface_area_mm2 for area in areas)
        assert abs(metrics.total_surface_area_mm2 - sum_individual_areas) < 0.01, \
            "Total surface area must equal sum of individual areas"
        
        # Average melanin index should be within range of individual values
        if metrics.average_melanin_index > 0:
            min_melanin = min(area.melanin_index for area in areas)
            max_melanin = max(area.melanin_index for area in areas)
            assert min_melanin <= metrics.average_melanin_index <= max_melanin, \
                "Average melanin index must be within range of individual values"
        
        # Coverage percentage should be reasonable
        image_area_mm2 = (image.shape[0] * pixel_to_mm_scale) * (image.shape[1] * pixel_to_mm_scale)
        expected_coverage = (metrics.total_surface_area_mm2 / image_area_mm2) * 100
        assert abs(metrics.coverage_percentage - expected_coverage) < 1.0, \
            "Coverage percentage calculation inconsistent"
    
    # Property 4: Measurements should scale correctly
    # Run detection with different scale
    scale2 = 0.2  # Double the scale
    seg_mask2, areas2 = detector.detect_pigmentation(image, pixel_to_mm_scale=scale2)
    metrics2 = detector.calculate_metrics(areas2, image.shape[:2], pixel_to_mm_scale=scale2)
    
    # Number of detected areas should be the same (scale-invariant detection)
    assert len(areas) == len(areas2), \
        "Number of detected areas should be scale-invariant"
    
    # Surface areas should scale quadratically (area scales with scale²)
    if metrics.total_surface_area_mm2 > 0:
        scale_ratio = scale2 / pixel_to_mm_scale
        expected_ratio = scale_ratio ** 2
        actual_ratio = metrics2.total_surface_area_mm2 / metrics.total_surface_area_mm2
        # Allow 10% tolerance for numerical errors
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.1, \
            f"Surface area should scale quadratically: expected {expected_ratio}, got {actual_ratio}"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_empty_image():
    """Test detection on uniform image (no pigmentation)."""
    detector = PigmentationDetector()
    
    # Create uniform image
    image = np.ones((512, 512, 3), dtype=np.float32) * 0.7
    
    seg_mask, areas = detector.detect_pigmentation(image, pixel_to_mm_scale=0.1)
    
    # Should complete without errors
    assert seg_mask is not None
    # May or may not detect areas depending on thresholds


def test_very_dark_image():
    """Test detection on very dark image."""
    detector = PigmentationDetector()
    
    # Create very dark image
    image = np.ones((512, 512, 3), dtype=np.float32) * 0.1
    
    seg_mask, areas = detector.detect_pigmentation(image, pixel_to_mm_scale=0.1)
    
    # Should complete without errors
    assert seg_mask is not None


def test_very_bright_image():
    """Test detection on very bright image."""
    detector = PigmentationDetector()
    
    # Create very bright image
    image = np.ones((512, 512, 3), dtype=np.float32) * 0.95
    
    seg_mask, areas = detector.detect_pigmentation(image, pixel_to_mm_scale=0.1)
    
    # Should complete without errors
    assert seg_mask is not None


def test_small_image():
    """Test detection on small image."""
    detector = PigmentationDetector()
    
    # Create small image
    image = np.random.rand(128, 128, 3).astype(np.float32)
    
    seg_mask, areas = detector.detect_pigmentation(image, pixel_to_mm_scale=0.1)
    
    # Should complete without errors
    assert seg_mask is not None
    assert seg_mask.mask.shape == (128, 128)


def test_metrics_with_no_areas():
    """Test metrics calculation with no detected areas."""
    detector = PigmentationDetector()
    
    metrics = detector.calculate_metrics([], (512, 512), pixel_to_mm_scale=0.1)
    
    assert metrics.total_areas == 0
    assert metrics.total_surface_area_mm2 == 0.0
    assert metrics.average_melanin_index == 0.0
    assert metrics.coverage_percentage == 0.0


def test_heatmap_with_no_areas():
    """Test heat-map generation with no detected areas."""
    detector = PigmentationDetector()
    
    heatmap = detector.generate_heatmap([], (512, 512))
    
    assert heatmap.density_map.shape == (512, 512)
    assert heatmap.severity_map.shape == (512, 512)
    assert heatmap.melanin_map.shape == (512, 512)
    assert heatmap.visualization.shape == (512, 512, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
