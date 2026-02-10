"""
Anomaly Overlay Engine

This module projects 2D anomaly detections (pigmentation and wrinkles) onto
3D mesh surfaces, creating color-coded overlays for visualization.

Approach:
1. Project 2D segmentation masks onto 3D mesh using camera parameters
2. Aggregate labels from multiple views using voting
3. Generate color-coded vertex colors based on anomaly type and severity
4. Create layered texture maps for flexible visualization

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np
import cv2
import logging

from .reconstruction_3d import Mesh3D, CameraParameters

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies"""
    NONE = "none"
    PIGMENTATION_LOW = "pigmentation_low"
    PIGMENTATION_MEDIUM = "pigmentation_medium"
    PIGMENTATION_HIGH = "pigmentation_high"
    WRINKLE_MICRO = "wrinkle_micro"
    WRINKLE_REGULAR = "wrinkle_regular"


@dataclass
class VertexLabel:
    """Label for a single vertex"""
    vertex_index: int
    anomaly_type: AnomalyType
    confidence: float  # 0-1
    view_count: int  # Number of views that detected this


@dataclass
class ColorMap:
    """Color mapping for anomaly types"""
    colors: Dict[AnomalyType, Tuple[int, int, int]]  # RGB colors
    
    @staticmethod
    def default() -> 'ColorMap':
        """Create default color map"""
        return ColorMap(colors={
            AnomalyType.NONE: (200, 170, 150),  # Normal skin tone
            AnomalyType.PIGMENTATION_LOW: (255, 229, 180),  # Light yellow
            AnomalyType.PIGMENTATION_MEDIUM: (255, 165, 0),  # Orange
            AnomalyType.PIGMENTATION_HIGH: (139, 0, 0),  # Dark red
            AnomalyType.WRINKLE_MICRO: (173, 216, 230),  # Light blue
            AnomalyType.WRINKLE_REGULAR: (0, 0, 139),  # Dark blue
        })


@dataclass
class LayeredTexture:
    """Layered texture maps for different anomaly types"""
    base_texture: np.ndarray  # Base skin texture
    pigmentation_overlay: np.ndarray  # Pigmentation overlay
    wrinkle_overlay: np.ndarray  # Wrinkle overlay
    combined: np.ndarray  # Combined overlay


class AnomalyOverlayEngine:
    """
    Projects 2D anomaly detections onto 3D mesh surfaces.
    
    Creates color-coded overlays for visualization of pigmentation
    and wrinkles on the 3D facial mesh.
    """
    
    def __init__(self, color_map: Optional[ColorMap] = None):
        """
        Initialize the anomaly overlay engine.
        
        Args:
            color_map: Color mapping for anomaly types (uses default if None)
        """
        self.color_map = color_map or ColorMap.default()
        logger.info("AnomalyOverlayEngine initialized")
    
    def project_2d_to_3d(
        self,
        mesh: Mesh3D,
        segmentation_mask: np.ndarray,
        anomaly_type: AnomalyType,
        camera_params: CameraParameters
    ) -> List[VertexLabel]:
        """
        Project 2D segmentation mask onto 3D mesh.
        
        Args:
            mesh: 3D mesh to project onto
            segmentation_mask: Binary mask (H x W) where 1 = anomaly present
            anomaly_type: Type of anomaly in the mask
            camera_params: Camera parameters for projection
            
        Returns:
            List of vertex labels for detected anomalies
            
        Requirements: 4.1, 4.2
        """
        logger.info(f"Projecting {anomaly_type.value} mask onto mesh")
        
        vertex_labels = []
        
        # Project mesh vertices to image space
        projected_vertices = self._project_vertices_to_image(
            mesh.vertices,
            camera_params
        )
        
        height, width = segmentation_mask.shape
        
        # For each vertex, check if it projects onto an anomaly pixel
        for vertex_idx, (u, v) in enumerate(projected_vertices):
            # Check if projection is within image bounds
            if 0 <= u < width and 0 <= v < height:
                # Check if this pixel has an anomaly
                if segmentation_mask[v, u] > 0:
                    # Calculate confidence based on mask intensity
                    confidence = float(segmentation_mask[v, u]) / 255.0
                    
                    vertex_labels.append(VertexLabel(
                        vertex_index=vertex_idx,
                        anomaly_type=anomaly_type,
                        confidence=confidence,
                        view_count=1
                    ))
        
        logger.info(f"Projected {len(vertex_labels)} vertices with {anomaly_type.value}")
        
        return vertex_labels
    
    def _project_vertices_to_image(
        self,
        vertices: np.ndarray,
        camera_params: CameraParameters
    ) -> np.ndarray:
        """
        Project 3D vertices to 2D image coordinates.
        
        Args:
            vertices: Nx3 array of 3D vertex positions
            camera_params: Camera parameters
            
        Returns:
            Nx2 array of 2D image coordinates (u, v)
        """
        # Convert vertices to homogeneous coordinates
        vertices_homo = np.hstack([vertices, np.ones((len(vertices), 1))])
        
        # Apply extrinsic transformation (world to camera)
        extrinsic = np.hstack([
            camera_params.rotation_matrix,
            camera_params.translation_vector
        ])
        vertices_camera = vertices_homo @ extrinsic.T
        
        # Apply intrinsic transformation (camera to image)
        vertices_image = vertices_camera @ camera_params.intrinsic_matrix.T
        
        # Normalize by depth (perspective division)
        # Add small epsilon to avoid division by zero
        depth = vertices_image[:, 2:3]
        depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
        vertices_2d = vertices_image[:, :2] / depth
        
        return vertices_2d.astype(np.int32)
    
    def merge_multi_view_projections(
        self,
        projections: List[List[VertexLabel]]
    ) -> Dict[int, VertexLabel]:
        """
        Merge vertex labels from multiple views using voting.
        
        Args:
            projections: List of vertex label lists (one per view)
            
        Returns:
            Dictionary mapping vertex index to merged label
            
        Requirements: 4.1, 4.2
        """
        logger.info(f"Merging projections from {len(projections)} views")
        
        # Collect all labels by vertex index
        vertex_labels_by_index: Dict[int, List[VertexLabel]] = {}
        
        for projection in projections:
            for label in projection:
                if label.vertex_index not in vertex_labels_by_index:
                    vertex_labels_by_index[label.vertex_index] = []
                vertex_labels_by_index[label.vertex_index].append(label)
        
        # Merge labels using voting
        merged_labels = {}
        
        for vertex_idx, labels in vertex_labels_by_index.items():
            # Count votes for each anomaly type
            type_votes: Dict[AnomalyType, int] = {}
            type_confidences: Dict[AnomalyType, List[float]] = {}
            
            for label in labels:
                if label.anomaly_type not in type_votes:
                    type_votes[label.anomaly_type] = 0
                    type_confidences[label.anomaly_type] = []
                
                type_votes[label.anomaly_type] += 1
                type_confidences[label.anomaly_type].append(label.confidence)
            
            # Select anomaly type with most votes
            winning_type = max(type_votes.items(), key=lambda x: x[1])[0]
            
            # Average confidence for winning type
            avg_confidence = np.mean(type_confidences[winning_type])
            
            # Create merged label
            merged_labels[vertex_idx] = VertexLabel(
                vertex_index=vertex_idx,
                anomaly_type=winning_type,
                confidence=float(avg_confidence),
                view_count=len(labels)
            )
        
        logger.info(f"Merged labels for {len(merged_labels)} vertices")
        
        return merged_labels
    
    def smooth_boundaries(
        self,
        mesh: Mesh3D,
        vertex_labels: Dict[int, VertexLabel],
        iterations: int = 2
    ) -> Dict[int, VertexLabel]:
        """
        Smooth anomaly boundaries using bilateral filtering on mesh.
        
        Args:
            mesh: 3D mesh
            vertex_labels: Current vertex labels
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed vertex labels
        """
        logger.info(f"Smoothing boundaries with {iterations} iterations")
        
        # Build adjacency list from faces
        adjacency = self._build_adjacency_list(mesh.faces, len(mesh.vertices))
        
        smoothed_labels = vertex_labels.copy()
        
        for _ in range(iterations):
            new_labels = {}
            
            for vertex_idx in range(len(mesh.vertices)):
                if vertex_idx not in smoothed_labels:
                    continue
                
                # Get neighboring vertices
                neighbors = adjacency.get(vertex_idx, [])
                
                if not neighbors:
                    new_labels[vertex_idx] = smoothed_labels[vertex_idx]
                    continue
                
                # Collect neighbor labels
                neighbor_labels = [
                    smoothed_labels[n] for n in neighbors
                    if n in smoothed_labels
                ]
                
                if not neighbor_labels:
                    new_labels[vertex_idx] = smoothed_labels[vertex_idx]
                    continue
                
                # Vote among neighbors (including self)
                all_labels = neighbor_labels + [smoothed_labels[vertex_idx]]
                
                type_votes: Dict[AnomalyType, int] = {}
                for label in all_labels:
                    type_votes[label.anomaly_type] = type_votes.get(label.anomaly_type, 0) + 1
                
                # Keep current type if it has majority, otherwise switch
                winning_type = max(type_votes.items(), key=lambda x: x[1])[0]
                
                new_labels[vertex_idx] = VertexLabel(
                    vertex_index=vertex_idx,
                    anomaly_type=winning_type,
                    confidence=smoothed_labels[vertex_idx].confidence,
                    view_count=smoothed_labels[vertex_idx].view_count
                )
            
            smoothed_labels = new_labels
        
        return smoothed_labels
    
    def _build_adjacency_list(
        self,
        faces: np.ndarray,
        vertex_count: int
    ) -> Dict[int, List[int]]:
        """
        Build vertex adjacency list from faces.
        
        Args:
            faces: Mx3 array of triangle indices
            vertex_count: Total number of vertices
            
        Returns:
            Dictionary mapping vertex index to list of adjacent vertices
        """
        adjacency: Dict[int, set] = {i: set() for i in range(vertex_count)}
        
        for face in faces:
            # Add edges for this triangle
            adjacency[face[0]].add(face[1])
            adjacency[face[0]].add(face[2])
            adjacency[face[1]].add(face[0])
            adjacency[face[1]].add(face[2])
            adjacency[face[2]].add(face[0])
            adjacency[face[2]].add(face[1])
        
        # Convert sets to lists
        return {k: list(v) for k, v in adjacency.items()}
    
    def generate_color_coded_overlay(
        self,
        mesh: Mesh3D,
        vertex_labels: Dict[int, VertexLabel]
    ) -> np.ndarray:
        """
        Generate color-coded vertex colors based on anomaly labels.
        
        Args:
            mesh: 3D mesh
            vertex_labels: Vertex labels with anomaly types
            
        Returns:
            Nx3 array of RGB vertex colors
            
        Requirements: 4.3, 4.4, 4.5
        """
        logger.info("Generating color-coded overlay")
        
        # Initialize with base skin color
        vertex_colors = np.tile(
            self.color_map.colors[AnomalyType.NONE],
            (len(mesh.vertices), 1)
        ).astype(np.uint8)
        
        # Apply anomaly colors
        for vertex_idx, label in vertex_labels.items():
            if vertex_idx < len(vertex_colors):
                color = self.color_map.colors[label.anomaly_type]
                
                # Blend with base color based on confidence
                base_color = self.color_map.colors[AnomalyType.NONE]
                blended_color = (
                    np.array(color) * label.confidence +
                    np.array(base_color) * (1 - label.confidence)
                )
                
                vertex_colors[vertex_idx] = blended_color.astype(np.uint8)
        
        logger.info(f"Generated colors for {len(vertex_labels)} vertices")
        
        return vertex_colors
    
    def create_layered_textures(
        self,
        mesh: Mesh3D,
        pigmentation_labels: Dict[int, VertexLabel],
        wrinkle_labels: Dict[int, VertexLabel],
        texture_size: Tuple[int, int] = (1024, 1024)
    ) -> LayeredTexture:
        """
        Create layered texture maps for different anomaly types.
        
        Args:
            mesh: 3D mesh
            pigmentation_labels: Pigmentation vertex labels
            wrinkle_labels: Wrinkle vertex labels
            texture_size: Size of texture maps (width, height)
            
        Returns:
            LayeredTexture with separate maps for each anomaly type
            
        Requirements: 4.3, 4.4, 4.5
        """
        logger.info(f"Creating layered textures ({texture_size[0]}x{texture_size[1]})")
        
        width, height = texture_size
        
        # Create base texture (normal skin)
        base_color = self.color_map.colors[AnomalyType.NONE]
        base_texture = np.full((height, width, 3), base_color, dtype=np.uint8)
        
        # Create pigmentation overlay
        pigmentation_overlay = np.zeros((height, width, 4), dtype=np.uint8)
        self._render_labels_to_texture(
            pigmentation_overlay,
            mesh,
            pigmentation_labels,
            texture_size
        )
        
        # Create wrinkle overlay
        wrinkle_overlay = np.zeros((height, width, 4), dtype=np.uint8)
        self._render_labels_to_texture(
            wrinkle_overlay,
            mesh,
            wrinkle_labels,
            texture_size
        )
        
        # Create combined overlay
        combined = base_texture.copy()
        
        # Blend pigmentation
        for y in range(height):
            for x in range(width):
                if pigmentation_overlay[y, x, 3] > 0:  # Has alpha
                    alpha = pigmentation_overlay[y, x, 3] / 255.0
                    combined[y, x] = (
                        pigmentation_overlay[y, x, :3] * alpha +
                        combined[y, x] * (1 - alpha)
                    ).astype(np.uint8)
        
        # Blend wrinkles
        for y in range(height):
            for x in range(width):
                if wrinkle_overlay[y, x, 3] > 0:  # Has alpha
                    alpha = wrinkle_overlay[y, x, 3] / 255.0
                    combined[y, x] = (
                        wrinkle_overlay[y, x, :3] * alpha +
                        combined[y, x] * (1 - alpha)
                    ).astype(np.uint8)
        
        return LayeredTexture(
            base_texture=base_texture,
            pigmentation_overlay=pigmentation_overlay[:, :, :3],
            wrinkle_overlay=wrinkle_overlay[:, :, :3],
            combined=combined
        )
    
    def _render_labels_to_texture(
        self,
        texture: np.ndarray,
        mesh: Mesh3D,
        vertex_labels: Dict[int, VertexLabel],
        texture_size: Tuple[int, int]
    ):
        """
        Render vertex labels to texture using UV coordinates.
        
        Args:
            texture: RGBA texture to render into
            mesh: 3D mesh with UV coordinates
            vertex_labels: Vertex labels to render
            texture_size: Size of texture
        """
        if mesh.uv_coords is None:
            logger.warning("Mesh has no UV coordinates, skipping texture rendering")
            return
        
        width, height = texture_size
        
        # Render each labeled vertex
        for vertex_idx, label in vertex_labels.items():
            if vertex_idx >= len(mesh.uv_coords):
                continue
            
            # Get UV coordinates
            u, v = mesh.uv_coords[vertex_idx]
            
            # Convert to pixel coordinates
            x = int(u * width)
            y = int(v * height)
            
            # Clamp to texture bounds
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            
            # Get color for this anomaly type
            color = self.color_map.colors[label.anomaly_type]
            
            # Set pixel with alpha based on confidence
            alpha = int(label.confidence * 255 * 0.7)  # 0.7 transparency
            texture[y, x] = (*color, alpha)
