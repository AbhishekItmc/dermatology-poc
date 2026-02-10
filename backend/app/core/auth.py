"""
Authentication and authorization utilities

Requirements: 13.4
"""
from datetime import datetime, timedelta
from typing import Optional, List
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.core.config import settings

# Password hashing - using argon2 for better compatibility
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# JWT token security
security = HTTPBearer()

# User roles
class UserRole:
    ADMIN = "admin"
    CLINICIAN = "clinician"
    VIEWER = "viewer"


class TokenData(BaseModel):
    """JWT token payload data"""
    user_id: str
    username: str
    roles: List[str]
    exp: Optional[datetime] = None


class User(BaseModel):
    """User model"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    is_active: bool = True


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Requirements: 13.4
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt


def decode_access_token(token: str) -> TokenData:
    """
    Decode and validate a JWT access token
    
    Requirements: 13.4
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        roles: List[str] = payload.get("roles", [])
        
        if user_id is None or username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return TokenData(
            user_id=user_id,
            username=username,
            roles=roles
        )
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """
    Dependency to get current authenticated user from JWT token
    
    Requirements: 13.4
    """
    token = credentials.credentials
    return decode_access_token(token)


def require_roles(required_roles: List[str]):
    """
    Dependency factory to require specific roles
    
    Requirements: 13.4
    
    Usage:
        @router.get("/admin-only")
        async def admin_endpoint(user: TokenData = Depends(require_roles([UserRole.ADMIN]))):
            ...
    """
    async def role_checker(user: TokenData = Depends(get_current_user)) -> TokenData:
        if not any(role in user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {required_roles}"
            )
        return user
    
    return role_checker


# In-memory user database (replace with real database in production)
# Using lazy initialization to avoid bcrypt issues at module load
_fake_users_db = None


def _get_fake_users_db():
    """Get or initialize fake users database"""
    global _fake_users_db
    if _fake_users_db is None:
        _fake_users_db = {
            "admin": {
                "user_id": "user_001",
                "username": "admin",
                "email": "admin@example.com",
                "hashed_password": get_password_hash("admin123"),
                "roles": [UserRole.ADMIN, UserRole.CLINICIAN],
                "is_active": True
            },
            "clinician": {
                "user_id": "user_002",
                "username": "clinician",
                "email": "clinician@example.com",
                "hashed_password": get_password_hash("clinician123"),
                "roles": [UserRole.CLINICIAN],
                "is_active": True
            },
            "viewer": {
                "user_id": "user_003",
                "username": "viewer",
                "email": "viewer@example.com",
                "hashed_password": get_password_hash("viewer123"),
                "roles": [UserRole.VIEWER],
                "is_active": True
            }
        }
    return _fake_users_db


def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user by username and password
    
    Requirements: 13.4
    """
    fake_users_db = _get_fake_users_db()
    user_dict = fake_users_db.get(username)
    
    if not user_dict:
        return None
    
    if not verify_password(password, user_dict["hashed_password"]):
        return None
    
    return User(
        user_id=user_dict["user_id"],
        username=user_dict["username"],
        email=user_dict["email"],
        roles=user_dict["roles"],
        is_active=user_dict["is_active"]
    )
