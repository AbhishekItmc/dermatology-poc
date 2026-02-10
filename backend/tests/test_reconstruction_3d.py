"""
Tests for 3D facial reconstruction module
"""
import pytest
import numpy as np
import cv2
import tempfile
import os

from app.services.reconstruction_3d import (
    FacialReconstructor,
    Mesh3D,
    CameraParameters,
    TextureMap,
    ReconstructionResult
)


@pytest.fixture
def reconstructor():
    """Create facial reconstructor instance"""
    return FacialReconstructor()


@pytest.fixture
def sample_landmarks():
    """Create sample 3D landmarks (simplified face)"""
    # Create a simple face-like point cloud
    landmarks = []
    
    # Create a grid of points on a face-like surface
    for i in range(20):
        for j in range(20):
            x = i * 10 + 100
            y = j * 10 + 100
            z = -50 + np.sin(i/5) * 20 + np.cos(j/5) * 20  # Curved surface
            landmarks.append([x, y, z])
    
    return np.array(landmarks, dtype=np.float32)


@pytest.fixture
def sample_image():
    """Create sample facial image"""
    image = np.ones((512, 512, 3), dtype=np.uint8) * 180
    
    # Draw simple face features
    cv2.circle(image, (256, 256), 100, (200, 170, 150), -1)
    cv2.circle(image, (220, 220), 15, (50, 50, 50), -1)  # Left eye
    cv2.circle(image, (292, 220), 15, (50, 50, 50), -1)  # Right eye
    cv2.ellipse(image, (256, 280), (30, 15), 0, 0, 180, (120, 80, 80), -1)  # Mouth
    
    return image


def test_reconstructor_initialization(reconstructor):
    """Test that reconstructor initializes correctly"""
    assert reconstructor is not None
    assert hasattr(reconstructor, 'reconstruct_from_landmarks')


def test_mesh_creation():
    """Test Mesh3D dataclass creation"""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    
    mesh = Mesh3D(vertices=vertices, faces=faces, normals=normals)
    
    assert len(mesh.vertices) == 3
    assert len(mesh.faces) == 1
    assert len(mesh.normals) == 3


def test_mesh_to_dict():
    """Test mesh serialization to dictionary"""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    
    mesh = Mesh3D(vertices=vertices, faces=faces, normals=normals)
    mesh_dict = mesh.to_dict()
    
    assert "vertices" in mesh_dict
    assert "faces" in mesh_dict
    assert "normals" in mesh_dict
    assert mesh_dict["vertex_count"] == 3
    assert mesh_dict["face_count"] == 1


def test_camera_parameters_creation():
    """Test CameraParameters dataclass creation"""
    rotation = np.eye(3, dtype=np.float32)
    translation = np.zeros((3, 1), dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    distortion = np.zeros((5, 1), dtype=np.float32)
    
    camera = CameraParameters(
        rotation_matrix=rotation,
        translation_vector=translation,
        intrinsic_matrix=intrinsic,
        distortion_coeffs=distortion
    )
    
    assert camera.rotation_matrix.shape == (3, 3)
    assert camera.translation_vector.shape == (3, 1)
    assert camera.intrinsic_matrix.shape == (3, 3)


def test_estimate_camera_parameters(reconstructor, sample_landmarks):
    """Test camera parameter estimation"""
    landmarks_list = [sample_landmarks, sample_landmarks, sample_landmarks]
    image_shapes = [(512, 512, 3), (512, 512, 3), (512, 512, 3)]
    
    camera_params = reconstructor._estimate_camera_parameters(
        landmarks_list,
        image_shapes
    )
    
    assert len(camera_params) == 3
    assert all(isinstance(cam, CameraParameters) for cam in camera_params)
    assert all(cam.intrinsic_matrix.shape == (3, 3) for cam in camera_params)


def test_generate_mesh_from_landmarks(reconstructor, sample_landmarks):
    """Test mesh generation from landmarks"""
    mesh = reconstructor._generate_mesh_from_landmarks(sample_landmarks)
    
    assert isinstance(mesh, Mesh3D)
    assert len(mesh.vertices) == len(sample_landmarks)
    assert len(mesh.faces) > 0
    assert len(mesh.normals) == len(mesh.vertices)
    assert mesh.uv_coords is not None
    assert len(mesh.uv_coords) == len(mesh.vertices)


def test_calculate_vertex_normals(reconstructor):
    """Test vertex normal calculation"""
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ], dtype=np.int32)
    
    normals = reconstructor._calculate_vertex_normals(vertices, faces)
    
    assert normals.shape == vertices.shape
    # Check that normals are normalized
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=0.1)


def test_generate_uv_coordinates(reconstructor, sample_landmarks):
    """Test UV coordinate generation"""
    uv_coords = reconstructor._generate_uv_coordinates(sample_landmarks)
    
    assert uv_coords.shape == (len(sample_landmarks), 2)
    # Check that UV coordinates are in [0, 1] range
    assert np.all(uv_coords >= 0.0)
    assert np.all(uv_coords <= 1.0)


def test_generate_texture_map(reconstructor, sample_landmarks, sample_image):
    """Test texture map generation"""
    mesh = reconstructor._generate_mesh_from_landmarks(sample_landmarks)
    texture = reconstructor._generate_texture_map(mesh, sample_image, sample_landmarks)
    
    assert isinstance(texture, TextureMap)
    assert texture.image.shape[0] == texture.resolution[1]
    assert texture.image.shape[1] == texture.resolution[0]
    assert texture.resolution == (1024, 1024)


def test_sample_texture_colors(reconstructor, sample_landmarks, sample_image):
    """Test texture color sampling"""
    mesh = reconstructor._generate_mesh_from_landmarks(sample_landmarks)
    texture = reconstructor._generate_texture_map(mesh, sample_image, sample_landmarks)
    
    vertex_colors = reconstructor._sample_texture_colors(mesh, texture, sample_image)
    
    assert vertex_colors.shape == (len(mesh.vertices), 3)
    assert vertex_colors.dtype == np.uint8


def test_calculate_reconstruction_confidence(reconstructor, sample_landmarks):
    """Test reconstruction confidence calculation"""
    landmarks_list = [sample_landmarks, sample_landmarks, sample_landmarks]
    mesh = reconstructor._generate_mesh_from_landmarks(sample_landmarks)
    
    confidence = reconstructor._calculate_reconstruction_confidence(landmarks_list, mesh)
    
    assert 0.0 <= confidence <= 1.0
    assert isinstance(confidence, float)


def test_reconstruct_from_landmarks_single_view(reconstructor, sample_landmarks, sample_image):
    """Test reconstruction from single view"""
    result = reconstructor.reconstruct_from_landmarks(
        landmarks_list=[sample_landmarks],
        images=[sample_image]
    )
    
    assert isinstance(result, ReconstructionResult)
    assert isinstance(result.mesh, Mesh3D)
    assert result.texture is not None
    assert len(result.camera_params) == 1
    assert 0.0 <= result.confidence_score <= 1.0


def test_reconstruct_from_landmarks_multi_view(reconstructor, sample_landmarks, sample_image):
    """Test reconstruction from multiple views"""
    # Create slight variations for different views
    landmarks_list = [
        sample_landmarks,
        sample_landmarks + np.random.randn(*sample_landmarks.shape) * 2,
        sample_landmarks + np.random.randn(*sample_landmarks.shape) * 2
    ]
    
    images = [sample_image, sample_image, sample_image]
    
    result = reconstructor.reconstruct_from_landmarks(
        landmarks_list=landmarks_list,
        images=images
    )
    
    assert isinstance(result, ReconstructionResult)
    assert isinstance(result.mesh, Mesh3D)
    assert len(result.camera_params) == 3
    assert result.confidence_score > 0.0


def test_export_mesh_obj(reconstructor, sample_landmarks):
    """Test mesh export to OBJ format"""
    mesh = reconstructor._generate_mesh_from_landmarks(sample_landmarks)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
        filepath = f.name
    
    try:
        # Export mesh
        success = reconstructor.export_mesh_obj(mesh, filepath)
        
        assert success
        assert os.path.exists(filepath)
        
        # Check file content
        with open(filepath, 'r') as f:
            content = f.read()
            assert 'v ' in content  # Vertices
            assert 'vn ' in content  # Normals
            assert 'f ' in content  # Faces
    
    finally:
        # Cleanup
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_mesh_quality_metrics(reconstructor, sample_landmarks):
    """Test that generated mesh has good quality metrics"""
    mesh = reconstructor._generate_mesh_from_landmarks(sample_landmarks)
    
    # Check vertex count
    assert len(mesh.vertices) > 0
    
    # Check face count
    assert len(mesh.faces) > 0
    
    # Check that all face indices are valid
    assert np.all(mesh.faces >= 0)
    assert np.all(mesh.faces < len(mesh.vertices))
    
    # Check that normals are unit vectors
    norms = np.linalg.norm(mesh.normals, axis=1)
    assert np.allclose(norms, 1.0, atol=0.1)


def test_reconstruction_with_custom_camera_params(reconstructor, sample_landmarks, sample_image):
    """Test reconstruction with provided camera parameters"""
    # Create custom camera parameters
    rotation = np.eye(3, dtype=np.float32)
    translation = np.zeros((3, 1), dtype=np.float32)
    intrinsic = np.array([
        [500, 0, 256],
        [0, 500, 256],
        [0, 0, 1]
    ], dtype=np.float32)
    distortion = np.zeros((5, 1), dtype=np.float32)
    
    camera_params = [CameraParameters(
        rotation_matrix=rotation,
        translation_vector=translation,
        intrinsic_matrix=intrinsic,
        distortion_coeffs=distortion
    )]
    
    result = reconstructor.reconstruct_from_landmarks(
        landmarks_list=[sample_landmarks],
        images=[sample_image],
        camera_params=camera_params
    )
    
    assert isinstance(result, ReconstructionResult)
    assert len(result.camera_params) == 1
    assert np.array_equal(result.camera_params[0].intrinsic_matrix, intrinsic)


def test_edge_case_minimal_landmarks(reconstructor, sample_image):
    """Test reconstruction with minimal number of landmarks"""
    # Create minimal landmark set (triangle)
    minimal_landmarks = np.array([
        [100, 100, 0],
        [200, 100, 0],
        [150, 200, 0]
    ], dtype=np.float32)
    
    result = reconstructor.reconstruct_from_landmarks(
        landmarks_list=[minimal_landmarks],
        images=[sample_image]
    )
    
    assert isinstance(result, ReconstructionResult)
    assert len(result.mesh.vertices) == 3


def test_edge_case_large_landmark_set(reconstructor, sample_image):
    """Test reconstruction with large landmark set"""
    # Create large landmark set
    large_landmarks = np.random.rand(1000, 3).astype(np.float32) * 500
    
    result = reconstructor.reconstruct_from_landmarks(
        landmarks_list=[large_landmarks],
        images=[sample_image]
    )
    
    assert isinstance(result, ReconstructionResult)
    assert len(result.mesh.vertices) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
