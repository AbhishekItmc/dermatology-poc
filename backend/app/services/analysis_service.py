"""
Integrated Analysis Service

This service orchestrates the complete dermatological analysis pipeline,
combining image preprocessing, landmark detection, pigmentation detection,
wrinkle detection, and 3D reconstruction into a unified workflow.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

from .image_preprocessing import ImagePreprocessor, ValidationResult
from .landmark_detection import LandmarkDetector, LandmarkResult
from .pigmentation_detection import PigmentationDetector, PigmentationMetrics
from .wrinkle_detection import WrinkleDetector, WrinkleAnalysis
from .reconstruction_3d import FacialReconstructor, ReconstructionResult

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result for a patient."""
    analysis_id: str
    patient_id: str
    
    # Validation
    validation: Dict[str, Any]
    
    # Landmarks
    landmarks: Optional[Dict[str, Any]]
    
    # Pigmentation
    pigmentation: Optional[Dict[str, Any]]
    
    # Wrinkles
    wrinkles: Optional[Dict[str, Any]]
    
    # 3D Reconstruction
    reconstruction_3d: Optional[Dict[str, Any]]
    
    # Overall status
    status: str  # "success", "partial", "failed"
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class AnalysisService:
    """
    Integrated analysis service that orchestrates the complete pipeline.
    
    Pipeline:
    1. Image validation and preprocessing
    2. Facial landmark detection
    3. Pigmentation detection
    4. Wrinkle detection
    5. Result aggregation
    """
    
    def __init__(self):
        """Initialize all detection modules."""
        self.preprocessor = ImagePreprocessor()
        self.landmark_detector = LandmarkDetector()
        self.pigmentation_detector = PigmentationDetector()
        self.wrinkle_detector = WrinkleDetector()
        self.reconstructor_3d = FacialReconstructor()
        
        logger.info("AnalysisService initialized with all detection modules")
    
    def analyze_patient(
        self,
        images: List[np.ndarray],
        patient_id: str,
        analysis_id: str,
        image_ids: Optional[List[str]] = None
    ) -> AnalysisResult:
        """
        Perform complete dermatological analysis on patient images.
        
        Args:
            images: List of facial images (180-degree coverage)
            patient_id: Patient identifier
            analysis_id: Analysis session identifier
            image_ids: Optional list of image identifiers
        
        Returns:
            AnalysisResult with complete analysis data
        """
        errors = []
        warnings = []
        
        logger.info(f"Starting analysis {analysis_id} for patient {patient_id}")
        logger.info(f"Processing {len(images)} images")
        
        # Step 1: Validate image set
        logger.info("Step 1: Validating image set...")
        validation_result = self.preprocessor.validate_image_set(images, image_ids)
        
        validation_data = {
            "valid": validation_result.valid,
            "image_count": validation_result.image_count,
            "angular_coverage": validation_result.angular_coverage,
            "issues": [
                {"type": issue[0].value if hasattr(issue[0], 'value') else str(issue[0]), 
                 "message": issue[1]}
                for issue in validation_result.issues
            ]
        }
        
        if not validation_result.valid:
            logger.warning(f"Image validation failed: {len(validation_result.issues)} issues")
            errors.append("Image validation failed")
            return AnalysisResult(
                analysis_id=analysis_id,
                patient_id=patient_id,
                validation=validation_data,
                landmarks=None,
                pigmentation=None,
                wrinkles=None,
                reconstruction_3d=None,
                status="failed",
                errors=errors,
                warnings=warnings
            )
        
        # Step 2: Detect landmarks on primary image (frontal view)
        logger.info("Step 2: Detecting facial landmarks...")
        primary_image = images[len(images) // 2]  # Use middle image (frontal view)
        
        try:
            landmark_result = self.landmark_detector.detect_landmarks(primary_image)
            
            if landmark_result is None:
                warnings.append("Landmark detection failed - using fallback")
                landmarks_data = None
                landmarks_array = None
                pixel_to_mm_scale = 0.1  # Default fallback
            else:
                # Convert landmarks to numpy array
                landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmark_result.landmarks])
                pixel_to_mm_scale = landmark_result.pixel_to_mm_scale
                
                landmarks_data = {
                    "detected": True,
                    "landmark_count": len(landmark_result.landmarks),
                    "confidence": float(landmark_result.confidence_score),
                    "interpupillary_distance_px": float(landmark_result.interpupillary_distance_px),
                    "pixel_to_mm_scale": float(pixel_to_mm_scale),
                    "head_pose": {
                        "pitch": float(landmark_result.pose.pitch),
                        "yaw": float(landmark_result.pose.yaw),
                        "roll": float(landmark_result.pose.roll)
                    },
                    "regions": {
                        region_name: {
                            "landmark_count": len(region.landmark_indices),
                            "center": [float(x) for x in region.center]
                        }
                        for region_name, region in landmark_result.facial_regions.items()
                    }
                }
                
            logger.info(f"Landmarks detected: {landmarks_data is not None}")
            
        except Exception as e:
            logger.error(f"Landmark detection error: {e}")
            warnings.append(f"Landmark detection error: {str(e)}")
            landmarks_data = None
            landmarks_array = None
            pixel_to_mm_scale = 0.1
        
        # Step 3: Normalize images for detection
        logger.info("Step 3: Normalizing images...")
        normalized_images = self.preprocessor.normalize_image_batch(
            images,
            target_size=(512, 512),
            to_srgb=True
        )
        
        # Use primary normalized image for detection
        primary_normalized = normalized_images[len(normalized_images) // 2]
        
        # Step 4: Detect pigmentation
        logger.info("Step 4: Detecting pigmentation...")
        pigmentation_data = None
        
        try:
            seg_mask, pigmentation_areas = self.pigmentation_detector.detect_pigmentation(
                primary_normalized,
                pixel_to_mm_scale=pixel_to_mm_scale
            )
            
            pigmentation_metrics = self.pigmentation_detector.calculate_metrics(
                pigmentation_areas,
                primary_normalized.shape[:2],
                pixel_to_mm_scale
            )
            
            pigmentation_data = {
                "total_areas": pigmentation_metrics.total_areas,
                "total_surface_area_mm2": float(pigmentation_metrics.total_surface_area_mm2),
                "average_melanin_index": float(pigmentation_metrics.average_melanin_index),
                "coverage_percentage": float(pigmentation_metrics.coverage_percentage),
                "severity_distribution": {
                    severity: count
                    for severity, count in pigmentation_metrics.severity_distribution.items()
                },
                "areas": [
                    {
                        "id": area.id,
                        "severity": area.severity.value,
                        "surface_area_mm2": float(area.surface_area_mm2),
                        "density": float(area.density),
                        "color_deviation": float(area.color_deviation),
                        "melanin_index": float(area.melanin_index),
                        "confidence": float(area.confidence),
                        "centroid": [float(x) for x in area.centroid],
                        "bounding_box": [int(x) for x in area.bounding_box]
                    }
                    for area in pigmentation_areas[:10]  # Limit to first 10 for response size
                ]
            }
            
            logger.info(f"Pigmentation detected: {pigmentation_metrics.total_areas} areas")
            
        except Exception as e:
            logger.error(f"Pigmentation detection error: {e}")
            errors.append(f"Pigmentation detection failed: {str(e)}")
        
        # Step 5: Detect wrinkles
        logger.info("Step 5: Detecting wrinkles...")
        wrinkles_data = None
        
        try:
            wrinkle_analysis = self.wrinkle_detector.detect_wrinkles(
                primary_normalized,
                landmarks=landmarks_array,
                pixel_to_mm_scale=pixel_to_mm_scale
            )
            
            wrinkles_data = {
                "total_wrinkle_count": wrinkle_analysis.total_wrinkle_count,
                "micro_wrinkle_count": wrinkle_analysis.micro_wrinkle_count,
                "average_depth_mm": float(wrinkle_analysis.average_depth_mm),
                "average_length_mm": float(wrinkle_analysis.average_length_mm),
                "texture_grade": wrinkle_analysis.texture_grade.value,
                "regional_texture_grades": {
                    region.value: grade.value
                    for region, grade in wrinkle_analysis.regional_texture_grades.items()
                },
                "regional_density": {
                    region.value: {
                        "wrinkle_count": density.wrinkle_count,
                        "total_length_mm": float(density.total_length_mm),
                        "density_score": float(density.density_score),
                        "average_depth_mm": float(density.average_depth_mm),
                        "average_width_mm": float(density.average_width_mm)
                    }
                    for region, density in wrinkle_analysis.regional_density.items()
                },
                "wrinkles": [
                    {
                        "id": wrinkle.wrinkle_id,
                        "length_mm": float(wrinkle.length_mm),
                        "depth_mm": float(wrinkle.depth_mm),
                        "width_mm": float(wrinkle.width_mm),
                        "severity": wrinkle.severity.value,
                        "region": wrinkle.region.value,
                        "confidence": float(wrinkle.confidence),
                        "bounding_box": [int(x) for x in wrinkle.bounding_box]
                    }
                    for wrinkle in wrinkle_analysis.wrinkles[:10]  # Limit to first 10
                ]
            }
            
            logger.info(f"Wrinkles detected: {wrinkle_analysis.total_wrinkle_count} total")
            
        except Exception as e:
            logger.error(f"Wrinkle detection error: {e}")
            errors.append(f"Wrinkle detection failed: {str(e)}")
        
        # Step 6: 3D Reconstruction (if multiple views available)
        logger.info("Step 6: Performing 3D reconstruction...")
        reconstruction_data = None
        
        if len(images) >= 3:  # Need at least 3 views for reconstruction
            try:
                # Collect landmarks from all views
                all_landmarks = []
                for img in images:
                    lm_result = self.landmark_detector.detect_landmarks(img)
                    if lm_result is not None:
                        lm_array = np.array([[lm.x, lm.y, lm.z] for lm in lm_result.landmarks])
                        all_landmarks.append(lm_array)
                
                if len(all_landmarks) >= 3:
                    # Perform 3D reconstruction
                    reconstruction_result = self.reconstructor_3d.reconstruct_from_landmarks(
                        landmarks_list=all_landmarks,
                        images=normalized_images
                    )
                    
                    reconstruction_data = {
                        "mesh": reconstruction_result.mesh.to_dict(),
                        "confidence": float(reconstruction_result.confidence_score),
                        "camera_count": len(reconstruction_result.camera_params)
                    }
                    
                    logger.info(f"3D reconstruction completed with confidence: {reconstruction_result.confidence_score:.2f}")
                else:
                    warnings.append("Insufficient landmark detections for 3D reconstruction")
                    
            except Exception as e:
                logger.error(f"3D reconstruction error: {e}")
                warnings.append(f"3D reconstruction failed: {str(e)}")
        else:
            logger.info("Skipping 3D reconstruction (requires at least 3 views)")
            warnings.append("3D reconstruction skipped - requires at least 3 views")
        
        # Determine overall status
        if errors:
            status = "partial" if (pigmentation_data or wrinkles_data) else "failed"
        else:
            status = "success"
        
        logger.info(f"Analysis {analysis_id} completed with status: {status}")
        
        return AnalysisResult(
            analysis_id=analysis_id,
            patient_id=patient_id,
            validation=validation_data,
            landmarks=landmarks_data,
            pigmentation=pigmentation_data,
            wrinkles=wrinkles_data,
            reconstruction_3d=reconstruction_data,
            status=status,
            errors=errors,
            warnings=warnings
        )
    
    def analyze_single_image(
        self,
        image: np.ndarray,
        patient_id: str,
        analysis_id: str
    ) -> AnalysisResult:
        """
        Perform analysis on a single image (simplified workflow).
        
        Args:
            image: Single facial image
            patient_id: Patient identifier
            analysis_id: Analysis session identifier
        
        Returns:
            AnalysisResult with analysis data
        """
        logger.info(f"Starting single-image analysis {analysis_id} for patient {patient_id}")
        
        # Validate single image
        info = self.preprocessor._validate_single_image(image, "single_image", 0)
        
        validation_data = {
            "valid": info.quality_metrics.overall_score >= self.preprocessor.MIN_OVERALL_QUALITY,
            "image_count": 1,
            "angular_coverage": 0.0,
            "issues": [
                {"type": "quality", "message": issue}
                for issue in info.quality_metrics.issues
            ]
        }
        
        # Normalize image
        normalized = self.preprocessor.normalize_image(image, target_size=(512, 512), to_srgb=True)
        
        # Detect landmarks
        landmark_result = self.landmark_detector.detect_landmarks(image)
        
        if landmark_result is None:
            landmarks_array = None
            pixel_to_mm_scale = 0.1
            landmarks_data = None
        else:
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmark_result.landmarks])
            pixel_to_mm_scale = landmark_result.pixel_to_mm_scale
            landmarks_data = {
                "detected": True,
                "pixel_to_mm_scale": float(pixel_to_mm_scale)
            }
        
        # Detect pigmentation
        seg_mask, pigmentation_areas = self.pigmentation_detector.detect_pigmentation(
            normalized,
            pixel_to_mm_scale=pixel_to_mm_scale
        )
        
        pigmentation_metrics = self.pigmentation_detector.calculate_metrics(
            pigmentation_areas,
            normalized.shape[:2],
            pixel_to_mm_scale
        )
        
        pigmentation_data = {
            "total_areas": pigmentation_metrics.total_areas,
            "coverage_percentage": float(pigmentation_metrics.coverage_percentage)
        }
        
        # Detect wrinkles
        wrinkle_analysis = self.wrinkle_detector.detect_wrinkles(
            normalized,
            landmarks=landmarks_array,
            pixel_to_mm_scale=pixel_to_mm_scale
        )
        
        wrinkles_data = {
            "total_wrinkle_count": wrinkle_analysis.total_wrinkle_count,
            "texture_grade": wrinkle_analysis.texture_grade.value,
            "regional_texture_grades": {
                region.value: grade.value
                for region, grade in wrinkle_analysis.regional_texture_grades.items()
            }
        }
        
        return AnalysisResult(
            analysis_id=analysis_id,
            patient_id=patient_id,
            validation=validation_data,
            landmarks=landmarks_data,
            pigmentation=pigmentation_data,
            wrinkles=wrinkles_data,
            reconstruction_3d=None,  # Single image doesn't support 3D reconstruction
            status="success",
            errors=[],
            warnings=[]
        )
