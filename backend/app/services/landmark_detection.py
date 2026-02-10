"""
Facial landmark detection module for dermatological analysis.

This module provides functions for detecting 468-point facial landmarks using
MediaPipe Face Landmarker, extracting confidence scores, and performing pose estimation
for 3D reconstruction and pixel-to-mm scaling.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2
import os
import urllib.request


@dataclass
class Landmark3D:
    """3D facial landmark with confidence score."""
    id: int
    x: float
    y: float
    z: float
    confidence: float
    name: str  # e.g., "left_eye_center", "nose_tip"


@dataclass
class PoseMatrix:
    """Head pose estimation matrix and angles."""
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    translation_vector: np.ndarray  # 3x1 translation vector
    euler_angles: Tuple[float, float, float]  # (pitch, yaw, roll) in degrees


@dataclass
class FacialRegion:
    """Facial region defined by landmark indices."""
    name: str
    landmark_indices: List[int]
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class LandmarkResult:
    """Result of landmark detection."""
    landmarks: List[Landmark3D]
    pose: PoseMatrix
    interpupillary_distance_px: float
    facial_regions: Dict[str, FacialRegion]
    confidence_score: float  # Overall confidence (0-1)


class LandmarkDetector:
    """
    Facial landmark detection using MediaPipe Face Landmarker.
    
    Detects 468 3D facial landmarks, estimates head pose, calculates
    interpupillary distance for pixel-to-mm conversion, and extracts
    facial regions for analysis.
    """
    
    # Minimum confidence threshold for landmark detection
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    MIN_OVERALL_CONFIDENCE = 0.7
    
    # MediaPipe model URL
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    
    # Key landmark indices in MediaPipe Face Mesh (468-point model)
    # These are approximate indices for key facial features
    LEFT_EYE_CENTER = 468  # Left iris center
    RIGHT_EYE_CENTER = 473  # Right iris center
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    CHIN = 152
    
    # Facial region definitions (landmark indices)
    FOREHEAD_INDICES = list(range(10, 67)) + list(range(297, 334))
    LEFT_CHEEK_INDICES = list(range(116, 123)) + list(range(147, 187))
    RIGHT_CHEEK_INDICES = list(range(345, 352)) + list(range(367, 411))
    PERIORBITAL_LEFT_INDICES = list(range(33, 133))
    PERIORBITAL_RIGHT_INDICES = list(range(263, 362))
    NOSE_INDICES = list(range(1, 9)) + list(range(168, 197))
    MOUTH_INDICES = list(range(61, 91)) + list(range(291, 321))
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the landmark detector with MediaPipe Face Landmarker.
        
        Args:
            model_path: Path to face landmarker model file (optional, will download if needed)
            num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        # Import MediaPipe here to avoid import errors if not needed
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from mediapipe import Image as MPImage, ImageFormat
            
            self.mp = mp
            self.python = python
            self.vision = vision
            self.MPImage = MPImage
            self.ImageFormat = ImageFormat
        except ImportError as e:
            raise ImportError(
                f"MediaPipe is not properly installed: {e}. "
                "Please install with: pip install mediapipe"
            )
        
        # Download model if not provided
        if model_path is None:
            model_path = self._get_model_path()
        
        # Create options for FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True
        )
        
        # Create FaceLandmarker
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Landmark name mapping for key points
        self.landmark_names = self._create_landmark_name_mapping()
    
    def _get_model_path(self) -> str:
        """
        Get path to MediaPipe face landmarker model, downloading if necessary.
        
        Returns:
            Path to model file
        """
        # Store model in app directory
        model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "face_landmarker.task")
        
        # Download if not exists
        if not os.path.exists(model_path):
            print(f"Downloading MediaPipe face landmarker model to {model_path}...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, model_path)
                print("Model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download MediaPipe model from {self.MODEL_URL}: {e}"
                )
        
        return model_path
    
    def _create_landmark_name_mapping(self) -> Dict[int, str]:
        """
        Create mapping from landmark indices to descriptive names.
        
        Returns:
            Dictionary mapping landmark index to name
        """
        names = {}
        
        # Key facial features
        names[self.NOSE_TIP] = "nose_tip"
        names[self.LEFT_EYE_OUTER] = "left_eye_outer"
        names[self.RIGHT_EYE_OUTER] = "right_eye_outer"
        names[self.LEFT_EYE_INNER] = "left_eye_inner"
        names[self.RIGHT_EYE_INNER] = "right_eye_inner"
        names[self.MOUTH_LEFT] = "mouth_left"
        names[self.MOUTH_RIGHT] = "mouth_right"
        names[self.CHIN] = "chin"
        
        # Default name for other landmarks
        for i in range(468):
            if i not in names:
                names[i] = f"landmark_{i}"
        
        return names
    
    def detect_landmarks(
        self,
        image: np.ndarray
    ) -> Optional[LandmarkResult]:
        """
        Detect 468-point facial landmarks in an image.
        
        Args:
            image: Image as numpy array (BGR format from cv2 or RGB)
            
        Returns:
            LandmarkResult with detected landmarks and derived information,
            or None if no face detected or confidence too low
            
        Validates: Requirements 10.3
        """
        # Convert BGR to RGB if needed (MediaPipe expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Create MediaPipe Image object
        mp_image = self.MPImage(image_format=self.ImageFormat.SRGB, data=image_rgb)
        
        # Detect face landmarks
        detection_result = self.face_landmarker.detect(mp_image)
        
        # Check if face was detected
        if not detection_result.face_landmarks:
            return None
        
        # Get the first face (we set num_faces=1)
        face_landmarks = detection_result.face_landmarks[0]
        
        # Extract landmarks with confidence scores
        landmarks = self._extract_landmarks(
            face_landmarks,
            image.shape[1],  # width
            image.shape[0]   # height
        )
        
        # Calculate overall confidence score
        confidence_score = self._calculate_overall_confidence(landmarks)
        
        # Check if confidence meets threshold
        if confidence_score < self.MIN_OVERALL_CONFIDENCE:
            return None
        
        # Estimate head pose
        pose = self._estimate_pose(landmarks, image.shape)
        
        # Calculate interpupillary distance
        ipd = self._calculate_interpupillary_distance(landmarks)
        
        # Extract facial regions
        facial_regions = self._extract_facial_regions(landmarks, image.shape)
        
        return LandmarkResult(
            landmarks=landmarks,
            pose=pose,
            interpupillary_distance_px=ipd,
            facial_regions=facial_regions,
            confidence_score=confidence_score
        )
    
    def _extract_landmarks(
        self,
        face_landmarks,
        image_width: int,
        image_height: int
    ) -> List[Landmark3D]:
        """
        Extract landmarks from MediaPipe results.
        
        Args:
            face_landmarks: MediaPipe face landmarks list
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            List of Landmark3D objects
        """
        landmarks = []
        
        for idx, landmark in enumerate(face_landmarks):
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * image_width
            y = landmark.y * image_height
            z = landmark.z * image_width  # z is relative to x scale
            
            # MediaPipe FaceLandmarker provides visibility/presence scores
            # Try to get presence, then visibility, then default to high confidence
            confidence = getattr(landmark, 'presence', None)
            if confidence is None:
                confidence = getattr(landmark, 'visibility', None)
            if confidence is None:
                # Default to high confidence if no score available
                confidence = 0.95
            
            # Get landmark name
            name = self.landmark_names.get(idx, f"landmark_{idx}")
            
            landmarks.append(Landmark3D(
                id=idx,
                x=x,
                y=y,
                z=z,
                confidence=float(confidence),  # Ensure it's a float
                name=name
            ))
        
        return landmarks
    
    def _calculate_overall_confidence(
        self,
        landmarks: List[Landmark3D]
    ) -> float:
        """
        Calculate overall confidence score from individual landmarks.
        
        Args:
            landmarks: List of detected landmarks
            
        Returns:
            Overall confidence score (0-1)
        """
        if not landmarks:
            return 0.0
        
        # Average confidence across all landmarks, filtering out None values
        confidences = [lm.confidence for lm in landmarks if lm.confidence is not None]
        
        if not confidences:
            # If no confidence scores available, return default high confidence
            return 0.95
        
        avg_confidence = sum(confidences) / len(confidences)
        
        return avg_confidence
    
    def _estimate_pose(
        self,
        landmarks: List[Landmark3D],
        image_shape: Tuple[int, ...]
    ) -> PoseMatrix:
        """
        Estimate head pose from facial landmarks.
        
        Uses solvePnP to estimate 3D pose from 2D-3D point correspondences.
        
        Args:
            landmarks: List of detected landmarks
            image_shape: Image shape (height, width, channels)
            
        Returns:
            PoseMatrix with rotation and translation
        """
        height, width = image_shape[:2]
        
        # Define 3D model points (generic face model in mm)
        # These are approximate 3D coordinates for key facial features
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float64)
        
        # Corresponding 2D image points from landmarks
        image_points = np.array([
            (landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y),
            (landmarks[self.CHIN].x, landmarks[self.CHIN].y),
            (landmarks[self.LEFT_EYE_OUTER].x, landmarks[self.LEFT_EYE_OUTER].y),
            (landmarks[self.RIGHT_EYE_OUTER].x, landmarks[self.RIGHT_EYE_OUTER].y),
            (landmarks[self.MOUTH_LEFT].x, landmarks[self.MOUTH_LEFT].y),
            (landmarks[self.MOUTH_RIGHT].x, landmarks[self.MOUTH_RIGHT].y)
        ], dtype=np.float64)
        
        # Camera matrix (assuming no lens distortion)
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP to get rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            # Return identity pose if PnP fails
            rotation_matrix = np.eye(3)
            translation_vector = np.zeros((3, 1))
            euler_angles = (0.0, 0.0, 0.0)
        else:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Calculate Euler angles from rotation matrix
            euler_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
        
        return PoseMatrix(
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
            euler_angles=euler_angles
        )
    
    def _rotation_matrix_to_euler_angles(
        self,
        R: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        # Calculate Euler angles from rotation matrix
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0
        
        # Convert to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)
        
        return (pitch, yaw, roll)
    
    def _calculate_interpupillary_distance(
        self,
        landmarks: List[Landmark3D]
    ) -> float:
        """
        Calculate interpupillary distance (IPD) in pixels.
        
        The IPD is used as a reference for pixel-to-mm conversion.
        Average human IPD is approximately 63mm.
        
        Args:
            landmarks: List of detected landmarks
            
        Returns:
            Interpupillary distance in pixels
        """
        # Get left and right eye centers
        # For MediaPipe Face Mesh, we use the inner eye corners as approximation
        left_eye = landmarks[self.LEFT_EYE_INNER]
        right_eye = landmarks[self.RIGHT_EYE_INNER]
        
        # Calculate Euclidean distance
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        dz = right_eye.z - left_eye.z
        
        ipd = np.sqrt(dx**2 + dy**2 + dz**2)
        
        return ipd
    
    def calculate_pixel_to_mm_scale(
        self,
        interpupillary_distance_px: float,
        average_ipd_mm: float = 63.0
    ) -> float:
        """
        Calculate pixel-to-millimeter scaling factor.
        
        Args:
            interpupillary_distance_px: IPD in pixels
            average_ipd_mm: Average human IPD in millimeters (default: 63mm)
            
        Returns:
            Scaling factor (mm per pixel)
        """
        if interpupillary_distance_px <= 0:
            return 0.0
        
        return average_ipd_mm / interpupillary_distance_px
    
    def _extract_facial_regions(
        self,
        landmarks: List[Landmark3D],
        image_shape: Tuple[int, ...]
    ) -> Dict[str, FacialRegion]:
        """
        Extract facial regions from landmarks.
        
        Args:
            landmarks: List of detected landmarks
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Dictionary mapping region name to FacialRegion
        """
        regions = {}
        
        # Define regions with their landmark indices
        region_definitions = {
            "forehead": self.FOREHEAD_INDICES,
            "left_cheek": self.LEFT_CHEEK_INDICES,
            "right_cheek": self.RIGHT_CHEEK_INDICES,
            "periorbital_left": self.PERIORBITAL_LEFT_INDICES,
            "periorbital_right": self.PERIORBITAL_RIGHT_INDICES,
            "nose": self.NOSE_INDICES,
            "mouth": self.MOUTH_INDICES
        }
        
        height, width = image_shape[:2]
        
        for region_name, indices in region_definitions.items():
            # Filter valid indices
            valid_indices = [i for i in indices if i < len(landmarks)]
            
            if not valid_indices:
                continue
            
            # Get landmark coordinates for this region
            region_landmarks = [landmarks[i] for i in valid_indices]
            
            # Calculate bounding box
            xs = [lm.x for lm in region_landmarks]
            ys = [lm.y for lm in region_landmarks]
            
            x_min = max(0, int(min(xs)))
            y_min = max(0, int(min(ys)))
            x_max = min(width, int(max(xs)))
            y_max = min(height, int(max(ys)))
            
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            regions[region_name] = FacialRegion(
                name=region_name,
                landmark_indices=valid_indices,
                bounding_box=bbox
            )
        
        return regions
    
    def visualize_landmarks(
        self,
        image: np.ndarray,
        landmarks: List[Landmark3D],
        draw_indices: bool = False
    ) -> np.ndarray:
        """
        Visualize landmarks on an image for debugging.
        
        Args:
            image: Image as numpy array (BGR format)
            landmarks: List of detected landmarks
            draw_indices: Whether to draw landmark indices
            
        Returns:
            Image with landmarks drawn
        """
        vis_image = image.copy()
        
        # Draw landmarks
        for landmark in landmarks:
            x, y = int(landmark.x), int(landmark.y)
            
            # Draw point
            cv2.circle(vis_image, (x, y), 1, (0, 255, 0), -1)
            
            # Draw index if requested
            if draw_indices:
                cv2.putText(
                    vis_image,
                    str(landmark.id),
                    (x + 2, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.2,
                    (255, 255, 255),
                    1
                )
        
        return vis_image
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker.close()
