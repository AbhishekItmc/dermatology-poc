"""
Tests for anomaly overlay engine
"""
import pytest
import numpy as np

from app.services.anomaly_overlay import (
    AnomalyOverlayEngine,
    AnomalyType,
    VertexLabel,
    ColorMap,
    LayeredTexture
)
from app.services.reconstruction_3d import Mesh3D, CameraParameters


@pytest.fixture
def overlay_engine():
    """Create anomaly overlay engine instance"""
    return AnomalyOverlayEngine()


@pytest.fixture
def sample_mesh():
    """Create sample mesh"""
    # Simple triangle mesh
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=np.int32)
    
    normals = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ], dtype=np.float32)
    
    uv_coords = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=np.float32)
    
    return Mesh3D(
        vertices=vertices,
        faces=faces,
        normals=normals,
        uv_coords=uv_coords
    )


@pytest.fixture
def sample_camera():
    """Create sample camera parameters"""
    rotation = np.eye(3, dtype=np.float32)
    translation = np.zeros((3, 1), dtype=np.float32)
    intrinsic = np.array([
        [500, 0, 256],
        [0, 500, 256],
        [0, 0, 1]
    ], dtype=np.float32)
    distortion = np.zeros((5, 1), dtype=np.float32)
    
    return CameraParameters(
        rotation_matrix=rotation,
        translation_vector=translation,
        intrinsic_matrix=intrinsic,
        distortion_coeffs=distortion
    )


@pytest.fixture
def sample_segmentation_mask():
    """Create sample segmentation mask"""
    mask = np.zeros((512, 512), dtype=np.uint8)
    # Add some anomaly regions
    mask[100:200, 100:200] = 255
    mask[300:400, 300:400] = 128
    return mask


def test_overlay_engine_initialization(overlay_engine):
    """Test that overlay engine initializes correctly"""
    assert overlay_engine is not None
    assert overlay_engine.color_map is not None
    assert isinstance(overlay_engine.color_map, ColorMap)


def test_color_map_default():
    """Test default color map creation"""
    color_map = ColorMap.default()
    
    assert AnomalyType.NONE in color_map.colors
    assert AnomalyType.PIGMENTATION_LOW in color_map.colors
    assert AnomalyType.PIGMENTATION_MEDIUM in color_map.colors
    assert AnomalyType.PIGMENTATION_HIGH in color_map.colors
    assert AnomalyType.WRINKLE_MICRO in color_map.colors
    assert AnomalyType.WRINKLE_REGULAR in color_map.colors
    
    # Check that colors are RGB tuples
    for color in color_map.colors.values():
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)


def test_vertex_label_creation():
    """Test VertexLabel dataclass creation"""
    label = VertexLabel(
        vertex_index=0,
        anomaly_type=AnomalyType.PIGMENTATION_LOW,
        confidence=0.8,
        view_count=2
    )
    
    assert label.vertex_index == 0
    assert label.anomaly_type == AnomalyType.PIGMENTATION_LOW
    assert label.confidence == 0.8
    assert label.view_count == 2


def test_project_vertices_to_image(overlay_engine, sample_mesh, sample_camera):
    """Test vertex projection to image space"""
    projected = overlay_engine._project_vertices_to_image(
        sample_mesh.vertices,
        sample_camera
    )
    
    assert projected.shape == (len(sample_mesh.vertices), 2)
    assert projected.dtype == np.int32


def test_project_2d_to_3d(overlay_engine, sample_mesh, sample_segmentation_mask, sample_camera):
    """Test 2D to 3D projection"""
    labels = overlay_engine.project_2d_to_3d(
        mesh=sample_mesh,
        segmentation_mask=sample_segmentation_mask,
        anomaly_type=AnomalyType.PIGMENTATION_LOW,
        camera_params=sample_camera
    )
    
    assert isinstance(labels, list)
    assert all(isinstance(label, VertexLabel) for label in labels)
    assert all(label.anomaly_type == AnomalyType.PIGMENTATION_LOW for label in labels)


def test_merge_multi_view_projections(overlay_engine):
    """Test merging projections from multiple views"""
    # Create sample projections from 3 views
    projection1 = [
        VertexLabel(0, AnomalyType.PIGMENTATION_LOW, 0.8, 1),
        VertexLabel(1, AnomalyType.PIGMENTATION_LOW, 0.7, 1),
    ]
    
    projection2 = [
        VertexLabel(0, AnomalyType.PIGMENTATION_LOW, 0.9, 1),
        VertexLabel(2, AnomalyType.WRINKLE_MICRO, 0.6, 1),
    ]
    
    projection3 = [
        VertexLabel(0, AnomalyType.PIGMENTATION_MEDIUM, 0.5, 1),
        VertexLabel(1, AnomalyType.PIGMENTATION_LOW, 0.8, 1),
    ]
    
    merged = overlay_engine.merge_multi_view_projections([
        projection1, projection2, projection3
    ])
    
    assert isinstance(merged, dict)
    assert 0 in merged
    assert 1 in merged
    assert 2 in merged
    
    # Vertex 0 should have PIGMENTATION_LOW (2 votes vs 1)
    assert merged[0].anomaly_type == AnomalyType.PIGMENTATION_LOW
    assert merged[0].view_count == 3


def test_build_adjacency_list(overlay_engine, sample_mesh):
    """Test adjacency list building"""
    adjacency = overlay_engine._build_adjacency_list(
        sample_mesh.faces,
        len(sample_mesh.vertices)
    )
    
    assert isinstance(adjacency, dict)
    assert len(adjacency) == len(sample_mesh.vertices)
    
    # Check that each vertex has neighbors
    for vertex_idx, neighbors in adjacency.items():
        assert isinstance(neighbors, list)


def test_smooth_boundaries(overlay_engine, sample_mesh):
    """Test boundary smoothing"""
    # Create initial labels
    vertex_labels = {
        0: VertexLabel(0, AnomalyType.PIGMENTATION_LOW, 0.8, 1),
        1: VertexLabel(1, AnomalyType.PIGMENTATION_LOW, 0.7, 1),
        2: VertexLabel(2, AnomalyType.WRINKLE_MICRO, 0.6, 1),
        3: VertexLabel(3, AnomalyType.PIGMENTATION_LOW, 0.9, 1),
    }
    
    smoothed = overlay_engine.smooth_boundaries(
        mesh=sample_mesh,
        vertex_labels=vertex_labels,
        iterations=2
    )
    
    assert isinstance(smoothed, dict)
    assert len(smoothed) == len(vertex_labels)
    
    # Check that labels are still valid
    for label in smoothed.values():
        assert isinstance(label, VertexLabel)
        assert isinstance(label.anomaly_type, AnomalyType)


def test_generate_color_coded_overlay(overlay_engine, sample_mesh):
    """Test color-coded overlay generation"""
    vertex_labels = {
        0: VertexLabel(0, AnomalyType.PIGMENTATION_LOW, 0.8, 1),
        1: VertexLabel(1, AnomalyType.PIGMENTATION_MEDIUM, 0.9, 1),
        2: VertexLabel(2, AnomalyType.WRINKLE_MICRO, 0.7, 1),
    }
    
    vertex_colors = overlay_engine.generate_color_coded_overlay(
        mesh=sample_mesh,
        vertex_labels=vertex_labels
    )
    
    assert vertex_colors.shape == (len(sample_mesh.vertices), 3)
    assert vertex_colors.dtype == np.uint8
    
    # Check that labeled vertices have different colors than base
    base_color = overlay_engine.color_map.colors[AnomalyType.NONE]
    assert not np.array_equal(vertex_colors[0], base_color)
    assert not np.array_equal(vertex_colors[1], base_color)
    assert not np.array_equal(vertex_colors[2], base_color)


def test_create_layered_textures(overlay_engine, sample_mesh):
    """Test layered texture creation"""
    pigmentation_labels = {
        0: VertexLabel(0, AnomalyType.PIGMENTATION_LOW, 0.8, 1),
        1: VertexLabel(1, AnomalyType.PIGMENTATION_MEDIUM, 0.9, 1),
    }
    
    wrinkle_labels = {
        2: VertexLabel(2, AnomalyType.WRINKLE_MICRO, 0.7, 1),
        3: VertexLabel(3, AnomalyType.WRINKLE_REGULAR, 0.6, 1),
    }
    
    layered = overlay_engine.create_layered_textures(
        mesh=sample_mesh,
        pigmentation_labels=pigmentation_labels,
        wrinkle_labels=wrinkle_labels,
        texture_size=(512, 512)
    )
    
    assert isinstance(layered, LayeredTexture)
    assert layered.base_texture.shape == (512, 512, 3)
    assert layered.pigmentation_overlay.shape == (512, 512, 3)
    assert layered.wrinkle_overlay.shape == (512, 512, 3)
    assert layered.combined.shape == (512, 512, 3)


def test_render_labels_to_texture(overlay_engine, sample_mesh):
    """Test rendering labels to texture"""
    texture = np.zeros((512, 512, 4), dtype=np.uint8)
    
    vertex_labels = {
        0: VertexLabel(0, AnomalyType.PIGMENTATION_LOW, 0.8, 1),
        1: VertexLabel(1, AnomalyType.PIGMENTATION_MEDIUM, 0.9, 1),
    }
    
    overlay_engine._render_labels_to_texture(
        texture=texture,
        mesh=sample_mesh,
        vertex_labels=vertex_labels,
        texture_size=(512, 512)
    )
    
    # Check that some pixels were rendered
    assert np.any(texture[:, :, 3] > 0)  # Some alpha values set


def test_color_blending(overlay_engine, sample_mesh):
    """Test that colors are blended based on confidence"""
    # Low confidence label
    low_conf_labels = {
        0: VertexLabel(0, AnomalyType.PIGMENTATION_HIGH, 0.3, 1),
    }
    
    # High confidence label
    high_conf_labels = {
        0: VertexLabel(0, AnomalyType.PIGMENTATION_HIGH, 0.9, 1),
    }
    
    low_conf_colors = overlay_engine.generate_color_coded_overlay(
        sample_mesh, low_conf_labels
    )
    
    high_conf_colors = overlay_engine.generate_color_coded_overlay(
        sample_mesh, high_conf_labels
    )
    
    # High confidence should be closer to pure anomaly color
    base_color = np.array(overlay_engine.color_map.colors[AnomalyType.NONE])
    anomaly_color = np.array(overlay_engine.color_map.colors[AnomalyType.PIGMENTATION_HIGH])
    
    # High confidence should be further from base
    dist_low = np.linalg.norm(low_conf_colors[0] - base_color)
    dist_high = np.linalg.norm(high_conf_colors[0] - base_color)
    
    assert dist_high > dist_low


def test_edge_case_empty_labels(overlay_engine, sample_mesh):
    """Test with no labels"""
    vertex_colors = overlay_engine.generate_color_coded_overlay(
        mesh=sample_mesh,
        vertex_labels={}
    )
    
    # Should return base colors for all vertices
    base_color = overlay_engine.color_map.colors[AnomalyType.NONE]
    assert np.all(vertex_colors == base_color)


def test_edge_case_all_same_type(overlay_engine, sample_mesh):
    """Test with all vertices having same anomaly type"""
    vertex_labels = {
        i: VertexLabel(i, AnomalyType.PIGMENTATION_LOW, 0.8, 1)
        for i in range(len(sample_mesh.vertices))
    }
    
    vertex_colors = overlay_engine.generate_color_coded_overlay(
        sample_mesh, vertex_labels
    )
    
    # All vertices should have similar colors
    assert np.std(vertex_colors, axis=0).max() < 50  # Low variance


def test_edge_case_mesh_without_uv(overlay_engine):
    """Test with mesh that has no UV coordinates"""
    mesh_no_uv = Mesh3D(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        normals=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32),
        uv_coords=None
    )
    
    # Should handle gracefully
    layered = overlay_engine.create_layered_textures(
        mesh=mesh_no_uv,
        pigmentation_labels={},
        wrinkle_labels={},
        texture_size=(512, 512)
    )
    
    assert isinstance(layered, LayeredTexture)


def test_multiple_views_voting(overlay_engine):
    """Test that voting works correctly with conflicting views"""
    # 3 views vote for PIGMENTATION_LOW, 1 votes for PIGMENTATION_HIGH
    projections = [
        [VertexLabel(0, AnomalyType.PIGMENTATION_LOW, 0.8, 1)],
        [VertexLabel(0, AnomalyType.PIGMENTATION_LOW, 0.7, 1)],
        [VertexLabel(0, AnomalyType.PIGMENTATION_LOW, 0.9, 1)],
        [VertexLabel(0, AnomalyType.PIGMENTATION_HIGH, 0.6, 1)],
    ]
    
    merged = overlay_engine.merge_multi_view_projections(projections)
    
    # Should select PIGMENTATION_LOW (3 votes vs 1)
    assert merged[0].anomaly_type == AnomalyType.PIGMENTATION_LOW
    assert merged[0].view_count == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
