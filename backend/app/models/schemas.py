"""
Pydantic schemas for API request/response models
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ImageUploadResponse(BaseModel):
    """Response for image upload"""
    patient_id: str
    image_set_id: str
    image_count: int
    status: str


class AnalysisCreateRequest(BaseModel):
    """Request to create a new analysis"""
    patient_id: str = Field(..., description="Patient identifier")
    image_set_id: str = Field(..., description="Image set identifier")


class AnalysisStatusResponse(BaseModel):
    """Analysis processing status"""
    analysis_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: Optional[str] = None


class ValidationInfo(BaseModel):
    """Image validation information"""
    valid: bool
    image_count: int
    angular_coverage: float
    issues: List[Dict[str, str]]


class LandmarkInfo(BaseModel):
    """Landmark detection information"""
    detected: bool
    landmark_count: Optional[int] = None
    confidence: Optional[float] = None
    interpupillary_distance_px: Optional[float] = None
    pixel_to_mm_scale: Optional[float] = None
    head_pose: Optional[Dict[str, float]] = None
    regions: Optional[Dict[str, Any]] = None


class PigmentationArea(BaseModel):
    """Individual pigmentation area"""
    id: str
    severity: str
    surface_area_mm2: float
    density: float
    color_deviation: float
    melanin_index: float
    confidence: float
    centroid: List[float]
    bounding_box: List[int]


class PigmentationInfo(BaseModel):
    """Pigmentation detection information"""
    total_areas: int
    total_surface_area_mm2: float
    average_melanin_index: float
    coverage_percentage: float
    severity_distribution: Dict[str, int]
    areas: List[PigmentationArea]


class WrinkleInfo(BaseModel):
    """Wrinkle detection information"""
    id: str
    length_mm: float
    depth_mm: float
    width_mm: float
    severity: str
    region: str
    confidence: float
    bounding_box: List[int]


class WrinkleAnalysisInfo(BaseModel):
    """Wrinkle analysis information"""
    total_wrinkle_count: int
    micro_wrinkle_count: int
    average_depth_mm: float
    average_length_mm: float
    texture_grade: str
    regional_texture_grades: Dict[str, str]  # Region name -> texture grade
    regional_density: Dict[str, Any]
    wrinkles: List[WrinkleInfo]


class AnalysisResultResponse(BaseModel):
    """Complete analysis result"""
    analysis_id: str
    patient_id: str
    created_at: datetime
    validation: ValidationInfo
    landmarks: Optional[LandmarkInfo] = None
    pigmentation: Optional[PigmentationInfo] = None
    wrinkles: Optional[WrinkleAnalysisInfo] = None
    status: str
    errors: List[str]
    warnings: List[str]


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    analysis_id: Optional[str] = None
