"""
Authentication endpoints

Requirements: 13.4
"""
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from pydantic import BaseModel
from datetime import timedelta

from app.core.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    TokenData
)
from app.core.config import settings
from app.core.audit import log_audit_event, AuditAction

router = APIRouter()
security = HTTPBasic()


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    token_type: str
    user_id: str
    username: str
    roles: list[str]


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT token
    
    Requirements: 13.4
    """
    # Authenticate user
    user = authenticate_user(request.username, request.password)
    
    if not user:
        # Log failed login attempt
        log_audit_event(
            action=AuditAction.LOGIN_FAILED,
            user_id=request.username,
            resource_type="auth",
            details={"reason": "invalid_credentials"}
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        # Log failed login attempt
        log_audit_event(
            action=AuditAction.LOGIN_FAILED,
            user_id=user.user_id,
            resource_type="auth",
            details={"reason": "inactive_user"}
        )
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.user_id,
            "username": user.username,
            "roles": user.roles
        },
        expires_delta=access_token_expires
    )
    
    # Log successful login
    log_audit_event(
        action=AuditAction.LOGIN,
        user_id=user.user_id,
        resource_type="auth",
        details={"username": user.username}
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user.user_id,
        username=user.username,
        roles=user.roles
    )


@router.post("/logout")
async def logout(current_user: TokenData = Depends(get_current_user)):
    """
    Logout user (client should discard token)
    
    Requirements: 13.4
    """
    # Log logout
    log_audit_event(
        action=AuditAction.LOGOUT,
        user_id=current_user.user_id,
        resource_type="auth",
        details={"username": current_user.username}
    )
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=TokenData)
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """
    Get current user information from token
    
    Requirements: 13.4
    """
    return current_user
