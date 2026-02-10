"""
Tests for authentication and authorization

Requirements: 13.4
"""
import pytest
from fastapi.testclient import TestClient
from datetime import timedelta

from app.main import app
from app.core.auth import (
    create_access_token,
    decode_access_token,
    verify_password,
    get_password_hash,
    authenticate_user,
    UserRole
)
from app.core.config import settings

client = TestClient(app)


class TestPasswordHashing:
    """Test password hashing functions"""
    
    def test_password_hashing(self):
        """Test password can be hashed and verified"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert verify_password(password, hashed)
    
    def test_wrong_password_fails(self):
        """Test wrong password fails verification"""
        password = "correct_password"
        wrong_password = "wrong_password"
        hashed = get_password_hash(password)
        
        assert not verify_password(wrong_password, hashed)


class TestJWTTokens:
    """Test JWT token creation and validation"""
    
    def test_create_and_decode_token(self):
        """Test token can be created and decoded"""
        data = {
            "sub": "user_123",
            "username": "testuser",
            "roles": [UserRole.CLINICIAN]
        }
        
        token = create_access_token(data)
        assert token is not None
        assert isinstance(token, str)
        
        decoded = decode_access_token(token)
        assert decoded.user_id == "user_123"
        assert decoded.username == "testuser"
        assert UserRole.CLINICIAN in decoded.roles
    
    def test_token_with_expiration(self):
        """Test token with custom expiration"""
        data = {
            "sub": "user_123",
            "username": "testuser",
            "roles": [UserRole.ADMIN]
        }
        
        token = create_access_token(data, expires_delta=timedelta(minutes=30))
        assert token is not None
        
        decoded = decode_access_token(token)
        assert decoded.user_id == "user_123"
    
    def test_invalid_token_raises_error(self):
        """Test invalid token raises HTTPException"""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token("invalid_token")
        
        assert exc_info.value.status_code == 401


class TestUserAuthentication:
    """Test user authentication"""
    
    def test_authenticate_valid_user(self):
        """Test authentication with valid credentials"""
        user = authenticate_user("admin", "admin123")
        
        assert user is not None
        assert user.username == "admin"
        assert UserRole.ADMIN in user.roles
        assert user.is_active
    
    def test_authenticate_invalid_username(self):
        """Test authentication with invalid username"""
        user = authenticate_user("nonexistent", "password")
        assert user is None
    
    def test_authenticate_invalid_password(self):
        """Test authentication with invalid password"""
        user = authenticate_user("admin", "wrong_password")
        assert user is None


class TestLoginEndpoint:
    """Test login endpoint"""
    
    def test_login_success(self):
        """Test successful login"""
        response = client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["username"] == "admin"
        assert UserRole.ADMIN in data["roles"]
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username": "admin", "password": "wrong_password"}
        )
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_login_nonexistent_user(self):
        """Test login with nonexistent user"""
        response = client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username": "nonexistent", "password": "password"}
        )
        
        assert response.status_code == 401


class TestProtectedEndpoints:
    """Test protected endpoints require authentication"""
    
    def test_protected_endpoint_without_token(self):
        """Test accessing protected endpoint without token"""
        response = client.post(
            f"{settings.API_V1_STR}/analyses/",
            json={"patient_id": "test", "image_set_id": "test"}
        )
        
        assert response.status_code == 401  # Unauthorized without auth
    
    def test_protected_endpoint_with_valid_token(self):
        """Test accessing protected endpoint with valid token"""
        # Login first
        login_response = client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username": "clinician", "password": "clinician123"}
        )
        
        token = login_response.json()["access_token"]
        
        # Access protected endpoint
        response = client.post(
            f"{settings.API_V1_STR}/analyses/",
            json={"patient_id": "test", "image_set_id": "test"},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Should not be 403 (may be 404 or other error due to missing data)
        assert response.status_code != 403
    
    def test_protected_endpoint_with_invalid_token(self):
        """Test accessing protected endpoint with invalid token"""
        response = client.post(
            f"{settings.API_V1_STR}/analyses/",
            json={"patient_id": "test", "image_set_id": "test"},
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        assert response.status_code == 401


class TestRoleBasedAccessControl:
    """Test role-based access control"""
    
    def test_viewer_cannot_create_analysis(self):
        """Test viewer role cannot create analysis"""
        # Login as viewer
        login_response = client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username": "viewer", "password": "viewer123"}
        )
        
        token = login_response.json()["access_token"]
        
        # Try to create analysis
        response = client.post(
            f"{settings.API_V1_STR}/analyses/",
            json={"patient_id": "test", "image_set_id": "test"},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 403
        assert "Insufficient permissions" in response.json()["detail"]
    
    def test_clinician_can_create_analysis(self):
        """Test clinician role can create analysis"""
        # Login as clinician
        login_response = client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username": "clinician", "password": "clinician123"}
        )
        
        token = login_response.json()["access_token"]
        
        # Try to create analysis
        response = client.post(
            f"{settings.API_V1_STR}/analyses/",
            json={"patient_id": "test", "image_set_id": "test"},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Should not be 403 (may be other error due to missing data)
        assert response.status_code != 403


class TestGetCurrentUser:
    """Test get current user endpoint"""
    
    def test_get_current_user_info(self):
        """Test getting current user info from token"""
        # Login first
        login_response = client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        token = login_response.json()["access_token"]
        
        # Get current user info
        response = client.get(
            f"{settings.API_V1_STR}/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["username"] == "admin"
        assert UserRole.ADMIN in data["roles"]


class TestLogout:
    """Test logout endpoint"""
    
    def test_logout_success(self):
        """Test successful logout"""
        # Login first
        login_response = client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        token = login_response.json()["access_token"]
        
        # Logout
        response = client.post(
            f"{settings.API_V1_STR}/auth/logout",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]


# Property-based tests
from hypothesis import given, strategies as st, settings as hypothesis_settings


@given(
    username=st.text(min_size=3, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    password=st.text(min_size=8, max_size=50)
)
@hypothesis_settings(max_examples=10, deadline=500)
def test_property_password_hashing_reversible(username, password):
    """
    Property 32: Password hashing is one-way but verifiable
    
    Validates: Requirements 13.4
    """
    hashed = get_password_hash(password)
    
    # Hash should be different from password
    assert hashed != password
    
    # Original password should verify against hash
    assert verify_password(password, hashed)
    
    # Different password should not verify
    if len(password) > 0:
        wrong_password = password + "x"
        assert not verify_password(wrong_password, hashed)
