"""
Patient management endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from typing import List
import uuid
import logging

from app.models.schemas import ImageUploadResponse
from app.core.storage import get_image_storage, ImageStorage
from app.core.auth import get_current_user, require_roles, TokenData, UserRole
from app.core.audit import log_audit_event, AuditAction

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/{patient_id}/images", response_model=ImageUploadResponse)
async def upload_patient_images(
    patient_id: str,
    req: Request,
    images: List[UploadFile] = File(...),
    storage: ImageStorage = Depends(get_image_storage),
    current_user: TokenData = Depends(require_roles([UserRole.CLINICIAN, UserRole.ADMIN]))
):
    """
    Upload a 180-degree image set for a patient
    
    Requirements: 10.1, 13.1, 13.4
    """
    try:
        # Validate image count
        if len(images) < 3:
            raise HTTPException(
                status_code=400,
                detail="At least 3 images required for 180-degree coverage"
            )
        
        if len(images) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 images allowed per image set"
            )
        
        # Generate image set ID
        image_set_id = str(uuid.uuid4())
        
        # Read image bytes
        image_bytes_list = []
        for image in images:
            content = await image.read()
            image_bytes_list.append(content)
        
        # Save images to storage
        success = await storage.save_image_set(
            patient_id=patient_id,
            image_set_id=image_set_id,
            images=image_bytes_list
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save images"
            )
        
        # Log audit event
        log_audit_event(
            action=AuditAction.UPLOAD_IMAGE,
            user_id=current_user.user_id,
            resource_type="image_set",
            resource_id=image_set_id,
            details={
                "patient_id": patient_id,
                "image_count": len(images)
            },
            ip_address=req.client.host if req.client else None,
            user_agent=req.headers.get("user-agent")
        )
        
        logger.info(f"Uploaded {len(images)} images for patient {patient_id}")
        
        return ImageUploadResponse(
            patient_id=patient_id,
            image_set_id=image_set_id,
            image_count=len(images),
            status="uploaded"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload images: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload images: {str(e)}"
        )


@router.get("/{patient_id}/images")
async def list_patient_images(patient_id: str):
    """
    List all image sets for a patient
    
    Requirements: 11.1
    """
    # TODO: Implement image set listing from database
    return {
        "patient_id": patient_id,
        "image_sets": []
    }
