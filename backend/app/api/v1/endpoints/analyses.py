"""
Analysis endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from typing import Dict, Any
import uuid
from datetime import datetime
import logging

from app.models.schemas import (
    AnalysisCreateRequest,
    AnalysisStatusResponse,
    AnalysisResultResponse,
    ErrorResponse
)
from app.services import AnalysisService, ML_AVAILABLE
from app.core.storage import get_image_storage, ImageStorage
from app.core.auth import get_current_user, require_roles, TokenData, UserRole
from app.core.audit import log_audit_event, AuditAction

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for analysis results (replace with database in production)
analysis_results: Dict[str, Any] = {}
analysis_status: Dict[str, Dict[str, Any]] = {}


def get_analysis_service() -> AnalysisService:
    """Dependency to get analysis service instance"""
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Analysis service unavailable - ML dependencies not installed"
        )
    return AnalysisService()


@router.post("/", response_model=AnalysisStatusResponse)
async def create_analysis(
    request: AnalysisCreateRequest,
    req: Request,
    background_tasks: BackgroundTasks,
    storage: ImageStorage = Depends(get_image_storage),
    analysis_service: AnalysisService = Depends(get_analysis_service),
    current_user: TokenData = Depends(require_roles([UserRole.CLINICIAN, UserRole.ADMIN]))
):
    """
    Start a new analysis for a patient's image set
    
    Requirements: 11.1, 11.2, 12.1, 13.4
    """
    analysis_id = str(uuid.uuid4())
    
    # Log audit event
    log_audit_event(
        action=AuditAction.CREATE_ANALYSIS,
        user_id=current_user.user_id,
        resource_type="analysis",
        resource_id=analysis_id,
        details={
            "patient_id": request.patient_id,
            "image_set_id": request.image_set_id
        },
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("user-agent")
    )
    
    # Initialize status
    analysis_status[analysis_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "Analysis queued for processing"
    }
    
    # Queue analysis task in background
    background_tasks.add_task(
        process_analysis,
        analysis_id,
        request.patient_id,
        request.image_set_id,
        storage,
        analysis_service
    )
    
    logger.info(f"Created analysis {analysis_id} for patient {request.patient_id}")
    
    return AnalysisStatusResponse(
        analysis_id=analysis_id,
        status="queued",
        progress=0.0,
        message="Analysis queued for processing"
    )


async def process_analysis(
    analysis_id: str,
    patient_id: str,
    image_set_id: str,
    storage: ImageStorage,
    analysis_service: AnalysisService
):
    """Background task to process analysis"""
    try:
        # Update status to processing
        analysis_status[analysis_id] = {
            "status": "processing",
            "progress": 0.1,
            "message": "Loading images..."
        }
        
        # Load images from storage
        images = await storage.load_image_set(patient_id, image_set_id)
        
        if not images:
            raise ValueError("No images found for image set")
        
        # Update progress
        analysis_status[analysis_id]["progress"] = 0.3
        analysis_status[analysis_id]["message"] = "Analyzing images..."
        
        # Run analysis
        result = analysis_service.analyze_patient(
            images=images,
            patient_id=patient_id,
            analysis_id=analysis_id
        )
        
        # Store result
        analysis_results[analysis_id] = {
            "analysis_id": analysis_id,
            "patient_id": patient_id,
            "created_at": datetime.utcnow(),
            **result.to_dict()
        }
        
        # Update status to completed
        analysis_status[analysis_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "Analysis completed successfully"
        }
        
        logger.info(f"Analysis {analysis_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        analysis_status[analysis_id] = {
            "status": "failed",
            "progress": 0.0,
            "message": f"Analysis failed: {str(e)}"
        }


@router.get("/{analysis_id}", response_model=AnalysisResultResponse)
async def get_analysis(
    analysis_id: str,
    req: Request,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Get analysis results
    
    Requirements: 11.2, 13.4
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id]
    
    # Check if analysis is completed
    status = analysis_status.get(analysis_id, {}).get("status", "unknown")
    if status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis is not completed yet. Current status: {status}"
        )
    
    # Log audit event
    log_audit_event(
        action=AuditAction.VIEW_ANALYSIS,
        user_id=current_user.user_id,
        resource_type="analysis",
        resource_id=analysis_id,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("user-agent")
    )
    
    return AnalysisResultResponse(**result)


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(analysis_id: str):
    """
    Check analysis processing status
    
    Requirements: 11.1
    """
    if analysis_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    status_info = analysis_status[analysis_id]
    
    return AnalysisStatusResponse(
        analysis_id=analysis_id,
        **status_info
    )


@router.get("/{analysis_id}/mesh")
async def get_analysis_mesh(analysis_id: str):
    """
    Get 3D mesh data for an analysis
    
    Requirements: 3.1, 4.1, 4.2
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id]
    
    # Check if analysis is completed
    status = analysis_status.get(analysis_id, {}).get("status", "unknown")
    if status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis is not completed yet. Current status: {status}"
        )
    
    # Check if 3D reconstruction is available
    if "reconstruction_3d" not in result or result["reconstruction_3d"] is None:
        raise HTTPException(
            status_code=404,
            detail="3D reconstruction not available for this analysis"
        )
    
    return {
        "analysis_id": analysis_id,
        "mesh": result["reconstruction_3d"]["mesh"],
        "confidence": result["reconstruction_3d"]["confidence"]
    }


@router.get("/{analysis_id}/texture")
async def get_analysis_texture(analysis_id: str):
    """
    Get texture maps for an analysis
    
    Requirements: 3.1
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id]
    
    # Check if 3D reconstruction is available
    if "reconstruction_3d" not in result or result["reconstruction_3d"] is None:
        raise HTTPException(
            status_code=404,
            detail="3D reconstruction not available for this analysis"
        )
    
    return {
        "analysis_id": analysis_id,
        "texture_available": True,
        "message": "Texture data embedded in mesh vertex colors"
    }


@router.get("/{analysis_id}/anomalies")
async def get_analysis_anomalies(analysis_id: str):
    """
    Get anomaly labels for an analysis
    
    Requirements: 4.1, 4.2
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id]
    
    # Compile anomalies from pigmentation and wrinkles
    anomalies = {
        "pigmentation": result.get("pigmentation", {}),
        "wrinkles": result.get("wrinkles", {})
    }
    
    return {
        "analysis_id": analysis_id,
        "anomalies": anomalies
    }


@router.get("/{analysis_id}/recommendations")
async def get_treatment_recommendations(analysis_id: str):
    """
    Get treatment recommendations based on analysis
    
    Requirements: 9.7
    """
    # TODO: Implement recommendation generation
    return {
        "analysis_id": analysis_id,
        "recommendations": []
    }
