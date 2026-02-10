"""
3D Facial Reconstruction Module

This module implements 3D facial reconstruction from multi-view images.
For the PoC, we use MediaPipe landmarks as the foundation and create a
simplified mesh. Full Structure-from-Motion (SfM) can be added later.

Approach:
1. Use MediaPipe 468-point landmarks as 3D anchor points
2. Create mesh from landmark point cloud using Delaunay triangulation
3. Apply texture mapping from frontal view
4. Prepare structure for full SfM integration

Requirements: 10.4
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from scipy.spatial import Delaunay
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraParameters:
    """Camera parameters for a view"""
    rotation_matrix: np.ndarray  # 3x3
    translation_vector: np.ndarray  # 3x1
    intrinsic_matrix: np.ndarray  # 3x3
    distortion_coeffs: np.ndarray  # 5x1


@dataclass
class Mesh3D:
    """3D mesh representation"""
    vertices: np.ndarray  # Nx3 array of vertex positions
    faces: np.ndarray  # Mx3 array of triangle indices
    normals: np.ndarray  # Nx3 array of vertex normals
    uv_coords: Optional[np.ndarray] = None  # Nx2 array of UV coordinates
    vertex_colors: Optional[np.ndarray] = None  # Nx3 array of RGB colors
    
    def to_dict(self) -> Dict:
        """Convert mesh to dictionary for JSON serialization"""
        return {
            "vertices": self.vertices.tolist(),
            "faces": self.faces.tolist(),
            "normals": self.normals.tolist(),
            "uv_coords": self.uv_coords.tolist() if self.uv_coords is not None else None,
            "vertex_colors": self.vertex_colors.tolist() if self.vertex_colors is not None else None,
            "vertex_count": len(self.vertices),
            "face_count": len(self.faces)
        }


@dataclass
class TextureMap:
    """Texture map for mesh"""
    image: np.ndarray  # HxWx3 texture image
    resolution: Tuple[int, int]  # (width, height)


@dataclass
class ReconstructionResult:
    """Complete 3D reconstruction result"""
    mesh: Mesh3D
    texture: Optional[TextureMap]
    camera_params: List[CameraParameters]
    confidence_score: float  # Overall reconstruction quality (0-1)


class FacialReconstructor:
    """
    3D facial reconstruction from multi-view images.
    
    For PoC: Uses MediaPipe landmarks as foundation.
    Future: Full Structure-from-Motion (SfM) pipeline.
    """
    
    # MediaPipe face mesh topology (simplified for PoC)
    # These are approximate triangle connections for key facial features
    FACE_MESH_CONNECTIONS = [
        # Forehead region
        (10, 338, 297), (10, 297, 332), (10, 332, 284),
        # Left eye region
        (33, 133, 160), (33, 160, 159), (33, 159, 158),
        # Right eye region  
        (263, 362, 387), (263, 387, 386), (263, 386, 385),
        # Nose region
        (1, 2, 98), (1, 98, 327), (1, 327, 2),
        # Mouth region
        (61, 146, 91), (61, 91, 181), (61, 181, 84),
        (291, 375, 321), (291, 321, 405), (291, 405, 314),
        # Cheek regions
        (116, 123, 147), (116, 147, 187), (116, 187, 207),
        (345, 352, 376), (345, 376, 411), (345, 411, 427),
    ]
    
    def __init__(self):
        """Initialize the facial reconstructor"""
        logger.info("FacialReconstructor initialized")
    
    def reconstruct_from_landmarks(
        self,
        landmarks_list: List[np.ndarray],
        images: List[np.ndarray],
        camera_params: Optional[List[CameraParameters]] = None
    ) -> ReconstructionResult:
        """
        Reconstruct 3D mesh from MediaPipe landmarks.
        
        Args:
            landmarks_list: List of Nx3 landmark arrays (one per view)
            images: List of images corresponding to landmarks
            camera_params: Optional camera parameters (estimated if not provided)
            
        Returns:
            ReconstructionResult with mesh and texture
            
        Requirements: 10.4
        """
        logger.info(f"Starting 3D reconstruction from {len(landmarks_list)} views")
        
        # Use primary view (frontal) for mesh generation
        primary_landmarks = landmarks_list[len(landmarks_list) // 2]
        primary_image = images[len(images) // 2]
        
        # Estimate camera parameters if not provided
        if camera_params is None:
            camera_params = self._estimate_camera_parameters(
                landmarks_list,
                [img.shape for img in images]
            )
        
        # Generate mesh from landmarks
        mesh = self._generate_mesh_from_landmarks(primary_landmarks)
        
        # Generate texture from primary view
        texture = self._generate_texture_map(mesh, primary_image, primary_landmarks)
        
        # Apply texture colors to vertices
        mesh.vertex_colors = self._sample_texture_colors(mesh, texture, primary_image)
        
        # Calculate confidence score
        confidence = self._calculate_reconstruction_confidence(
            landmarks_list,
            mesh
        )
        
        logger.info(f"3D reconstruction completed with confidence: {confidence:.2f}")
        
        return ReconstructionResult(
            mesh=mesh,
            texture=texture,
            camera_params=camera_params,
            confidence_score=confidence
        )
    
    def _estimate_camera_parameters(
        self,
        landmarks_list: List[np.ndarray],
        image_shapes: List[Tuple[int, ...]]
    ) -> List[CameraParameters]:
        """
        Estimate camera parameters for each view.
        
        For PoC: Simple estimation based on image dimensions.
        Future: Full bundle adjustment.
        """
        camera_params = []
        
        for idx, (landmarks, shape) in enumerate(zip(landmarks_list, image_shapes)):
            height, width = shape[:2]
            
            # Simple camera intrinsics (pinhole model)
            focal_length = width  # Approximate
            cx, cy = width / 2, height / 2
            
            intrinsic_matrix = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Identity rotation for frontal view, slight rotation for others
            angle = (idx - len(landmarks_list) // 2) * 15  # degrees
            angle_rad = np.radians(angle)
            
            rotation_matrix = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ], dtype=np.float32)
            
            translation_vector = np.array([[0], [0], [500]], dtype=np.float32)
            
            distortion_coeffs = np.zeros((5, 1), dtype=np.float32)
            
            camera_params.append(CameraParameters(
                rotation_matrix=rotation_matrix,
                translation_vector=translation_vector,
                intrinsic_matrix=intrinsic_matrix,
                distortion_coeffs=distortion_coeffs
            ))
        
        return camera_params
    
    def _generate_mesh_from_landmarks(
        self,
        landmarks: np.ndarray
    ) -> Mesh3D:
        """
        Generate 3D mesh from landmark point cloud.
        
        Uses Delaunay triangulation on 2D projection, then lifts to 3D.
        """
        logger.info(f"Generating mesh from {len(landmarks)} landmarks")
        
        # Project landmarks to 2D (x, y) for triangulation
        points_2d = landmarks[:, :2]
        
        # Perform Delaunay triangulation
        try:
            tri = Delaunay(points_2d)
            faces = tri.simplices
        except Exception as e:
            logger.warning(f"Delaunay triangulation failed: {e}, using predefined topology")
            # Fallback to predefined connections
            faces = np.array(self.FACE_MESH_CONNECTIONS, dtype=np.int32)
        
        # Use 3D landmarks as vertices
        vertices = landmarks.copy()
        
        # Calculate vertex normals
        normals = self._calculate_vertex_normals(vertices, faces)
        
        # Generate UV coordinates (simple planar projection)
        uv_coords = self._generate_uv_coordinates(vertices)
        
        mesh = Mesh3D(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_coords=uv_coords
        )
        
        logger.info(f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces")
        
        return mesh
    
    def _calculate_vertex_normals(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """
        Calculate vertex normals from face normals.
        """
        # Initialize normals
        normals = np.zeros_like(vertices)
        
        # Calculate face normals and accumulate to vertices
        for face in faces:
            if len(face) != 3:
                continue
            
            # Get triangle vertices
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # Normalize
            norm = np.linalg.norm(face_normal)
            if norm > 1e-6:
                face_normal /= norm
            
            # Accumulate to vertex normals
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal
        
        # Normalize vertex normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0  # Avoid division by zero
        normals /= norms
        
        return normals
    
    def _generate_uv_coordinates(
        self,
        vertices: np.ndarray
    ) -> np.ndarray:
        """
        Generate UV coordinates using simple planar projection.
        
        Projects vertices onto XY plane and normalizes to [0, 1].
        """
        # Use X and Y coordinates
        uv = vertices[:, :2].copy()
        
        # Normalize to [0, 1]
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        uv_range = uv_max - uv_min
        uv_range[uv_range < 1e-6] = 1.0  # Avoid division by zero
        
        uv = (uv - uv_min) / uv_range
        
        return uv
    
    def _generate_texture_map(
        self,
        mesh: Mesh3D,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> TextureMap:
        """
        Generate texture map from image.
        
        For PoC: Use the frontal image directly.
        Future: Blend multiple views for better coverage.
        """
        # Use image as texture (resize if needed)
        texture_resolution = (1024, 1024)
        texture_image = cv2.resize(image, texture_resolution)
        
        return TextureMap(
            image=texture_image,
            resolution=texture_resolution
        )
    
    def _sample_texture_colors(
        self,
        mesh: Mesh3D,
        texture: TextureMap,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Sample texture colors for each vertex.
        
        Projects vertices back to image space and samples colors.
        """
        vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
        
        height, width = image.shape[:2]
        
        for i, vertex in enumerate(mesh.vertices):
            # Project vertex to image coordinates
            x, y = int(vertex[0]), int(vertex[1])
            
            # Clamp to image bounds
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            
            # Sample color
            vertex_colors[i] = image[y, x]
        
        return vertex_colors
    
    def _calculate_reconstruction_confidence(
        self,
        landmarks_list: List[np.ndarray],
        mesh: Mesh3D
    ) -> float:
        """
        Calculate confidence score for reconstruction quality.
        
        Based on:
        - Number of views
        - Landmark consistency across views
        - Mesh quality (vertex count, face count)
        """
        # View count score (more views = higher confidence)
        view_score = min(1.0, len(landmarks_list) / 5.0)
        
        # Mesh quality score
        vertex_count = len(mesh.vertices)
        face_count = len(mesh.faces)
        
        # Expect at least 400 vertices and 700 faces for good quality
        mesh_score = min(1.0, (vertex_count / 400.0 + face_count / 700.0) / 2.0)
        
        # Landmark consistency score (if multiple views)
        if len(landmarks_list) > 1:
            # Calculate variance in landmark positions across views
            landmarks_array = np.array(landmarks_list)
            variance = np.var(landmarks_array, axis=0).mean()
            consistency_score = 1.0 / (1.0 + variance / 100.0)
        else:
            consistency_score = 0.7  # Lower confidence for single view
        
        # Weighted average
        confidence = (
            0.3 * view_score +
            0.4 * mesh_score +
            0.3 * consistency_score
        )
        
        return float(confidence)
    
    def export_mesh_obj(
        self,
        mesh: Mesh3D,
        filepath: str
    ) -> bool:
        """
        Export mesh to OBJ file format.
        
        Args:
            mesh: Mesh to export
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                # Write vertices
                for vertex in mesh.vertices:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                
                # Write UV coordinates if available
                if mesh.uv_coords is not None:
                    for uv in mesh.uv_coords:
                        f.write(f"vt {uv[0]} {uv[1]}\n")
                
                # Write normals
                for normal in mesh.normals:
                    f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
                
                # Write faces (OBJ uses 1-based indexing)
                for face in mesh.faces:
                    if mesh.uv_coords is not None:
                        f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                               f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                               f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
                    else:
                        f.write(f"f {face[0]+1}//{face[0]+1} "
                               f"{face[1]+1}//{face[1]+1} "
                               f"{face[2]+1}//{face[2]+1}\n")
            
            logger.info(f"Exported mesh to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export mesh: {e}")
            return False
