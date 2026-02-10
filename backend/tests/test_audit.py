"""
Tests for audit logging

Requirements: 13.3
"""
import pytest
from datetime import datetime, timedelta

from app.core.audit import (
    log_audit_event,
    get_audit_logs,
    clear_audit_logs,
    AuditAction,
    AuditLog
)

# Property-based testing
from hypothesis import given, strategies as st, settings as hypothesis_settings


@pytest.fixture(autouse=True)
def clear_logs():
    """Clear audit logs before each test"""
    clear_audit_logs()
    yield
    clear_audit_logs()


class TestAuditLogging:
    """Test audit logging functionality"""
    
    def test_log_audit_event(self):
        """Test logging an audit event"""
        log = log_audit_event(
            action=AuditAction.VIEW_PATIENT,
            user_id="user_123",
            resource_type="patient",
            resource_id="patient_456",
            details={"field": "value"},
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert log is not None
        assert log.action == AuditAction.VIEW_PATIENT
        assert log.user_id == "user_123"
        assert log.resource_type == "patient"
        assert log.resource_id == "patient_456"
        assert log.details["field"] == "value"
        assert log.ip_address == "192.168.1.1"
        assert log.user_agent == "Mozilla/5.0"
        assert isinstance(log.timestamp, datetime)
    
    def test_log_multiple_events(self):
        """Test logging multiple audit events"""
        log1 = log_audit_event(
            action=AuditAction.CREATE_PATIENT,
            user_id="user_1",
            resource_type="patient",
            resource_id="patient_1"
        )
        
        log2 = log_audit_event(
            action=AuditAction.VIEW_ANALYSIS,
            user_id="user_2",
            resource_type="analysis",
            resource_id="analysis_1"
        )
        
        logs = get_audit_logs()
        assert len(logs) == 2
        assert logs[0].action == AuditAction.VIEW_ANALYSIS  # Most recent first
        assert logs[1].action == AuditAction.CREATE_PATIENT
    
    def test_audit_log_to_dict(self):
        """Test converting audit log to dictionary"""
        log = log_audit_event(
            action=AuditAction.LOGIN,
            user_id="user_123",
            resource_type="auth"
        )
        
        log_dict = log.to_dict()
        
        assert isinstance(log_dict, dict)
        assert log_dict["action"] == "login"
        assert log_dict["user_id"] == "user_123"
        assert log_dict["resource_type"] == "auth"
        assert "timestamp" in log_dict
    
    def test_audit_log_to_json(self):
        """Test converting audit log to JSON"""
        log = log_audit_event(
            action=AuditAction.LOGOUT,
            user_id="user_123",
            resource_type="auth"
        )
        
        log_json = log.to_json()
        
        assert isinstance(log_json, str)
        assert "logout" in log_json
        assert "user_123" in log_json


class TestAuditLogQuery:
    """Test audit log querying"""
    
    def setup_method(self):
        """Set up test data"""
        # Create test logs
        log_audit_event(
            action=AuditAction.CREATE_PATIENT,
            user_id="user_1",
            resource_type="patient",
            resource_id="patient_1"
        )
        
        log_audit_event(
            action=AuditAction.VIEW_PATIENT,
            user_id="user_1",
            resource_type="patient",
            resource_id="patient_1"
        )
        
        log_audit_event(
            action=AuditAction.CREATE_ANALYSIS,
            user_id="user_2",
            resource_type="analysis",
            resource_id="analysis_1"
        )
        
        log_audit_event(
            action=AuditAction.VIEW_ANALYSIS,
            user_id="user_2",
            resource_type="analysis",
            resource_id="analysis_1"
        )
    
    def test_get_all_logs(self):
        """Test getting all audit logs"""
        logs = get_audit_logs()
        assert len(logs) == 4
    
    def test_filter_by_user_id(self):
        """Test filtering logs by user ID"""
        logs = get_audit_logs(user_id="user_1")
        assert len(logs) == 2
        assert all(log.user_id == "user_1" for log in logs)
    
    def test_filter_by_resource_type(self):
        """Test filtering logs by resource type"""
        logs = get_audit_logs(resource_type="patient")
        assert len(logs) == 2
        assert all(log.resource_type == "patient" for log in logs)
    
    def test_filter_by_resource_id(self):
        """Test filtering logs by resource ID"""
        logs = get_audit_logs(resource_id="patient_1")
        assert len(logs) == 2
        assert all(log.resource_id == "patient_1" for log in logs)
    
    def test_filter_by_action(self):
        """Test filtering logs by action"""
        logs = get_audit_logs(action=AuditAction.CREATE_PATIENT)
        assert len(logs) == 1
        assert logs[0].action == AuditAction.CREATE_PATIENT
    
    def test_filter_by_time_range(self):
        """Test filtering logs by time range"""
        now = datetime.utcnow()
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)
        
        logs = get_audit_logs(start_time=past, end_time=future)
        assert len(logs) == 4
    
    def test_filter_with_limit(self):
        """Test limiting number of results"""
        logs = get_audit_logs(limit=2)
        assert len(logs) == 2
    
    def test_combined_filters(self):
        """Test combining multiple filters"""
        logs = get_audit_logs(
            user_id="user_1",
            resource_type="patient"
        )
        assert len(logs) == 2
        assert all(log.user_id == "user_1" and log.resource_type == "patient" for log in logs)
    
    def test_logs_sorted_by_timestamp(self):
        """Test logs are sorted by timestamp descending"""
        logs = get_audit_logs()
        
        for i in range(len(logs) - 1):
            assert logs[i].timestamp >= logs[i + 1].timestamp


class TestAuditActions:
    """Test different audit action types"""
    
    def test_data_access_actions(self):
        """Test data access audit actions"""
        actions = [
            AuditAction.VIEW_PATIENT,
            AuditAction.VIEW_ANALYSIS,
            AuditAction.VIEW_IMAGE
        ]
        
        for action in actions:
            log = log_audit_event(
                action=action,
                user_id="user_123",
                resource_type="test"
            )
            assert log.action == action
    
    def test_data_modification_actions(self):
        """Test data modification audit actions"""
        actions = [
            AuditAction.CREATE_PATIENT,
            AuditAction.UPDATE_PATIENT,
            AuditAction.DELETE_PATIENT,
            AuditAction.UPLOAD_IMAGE,
            AuditAction.DELETE_IMAGE,
            AuditAction.CREATE_ANALYSIS,
            AuditAction.DELETE_ANALYSIS
        ]
        
        for action in actions:
            log = log_audit_event(
                action=action,
                user_id="user_123",
                resource_type="test"
            )
            assert log.action == action
    
    def test_authentication_actions(self):
        """Test authentication audit actions"""
        actions = [
            AuditAction.LOGIN,
            AuditAction.LOGOUT,
            AuditAction.LOGIN_FAILED
        ]
        
        for action in actions:
            log = log_audit_event(
                action=action,
                user_id="user_123",
                resource_type="auth"
            )
            assert log.action == action


# Property-based tests
@given(
    user_id=st.text(min_size=1, max_size=50),
    resource_type=st.text(min_size=1, max_size=50),
    resource_id=st.text(min_size=1, max_size=50)
)
@hypothesis_settings(max_examples=5)
def test_property_audit_log_completeness(user_id, resource_type, resource_id):
    """
    Property 31: All audit logs contain required fields
    
    Validates: Requirements 13.3
    """
    clear_audit_logs()
    
    log = log_audit_event(
        action=AuditAction.VIEW_PATIENT,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id
    )
    
    # All required fields must be present
    assert log.timestamp is not None
    assert log.action is not None
    assert log.user_id == user_id
    assert log.resource_type == resource_type
    assert log.resource_id == resource_id
    
    # Log must be retrievable
    logs = get_audit_logs(user_id=user_id)
    assert len(logs) == 1
    assert logs[0].user_id == user_id


@given(
    num_logs=st.integers(min_value=1, max_value=20)
)
@hypothesis_settings(max_examples=5)
def test_property_audit_log_ordering(num_logs):
    """
    Property 31: Audit logs are always ordered by timestamp descending
    
    Validates: Requirements 13.3
    """
    clear_audit_logs()
    
    # Create multiple logs
    for i in range(num_logs):
        log_audit_event(
            action=AuditAction.VIEW_PATIENT,
            user_id=f"user_{i}",
            resource_type="patient",
            resource_id=f"patient_{i}"
        )
    
    # Retrieve logs
    logs = get_audit_logs()
    
    # Verify count
    assert len(logs) == num_logs
    
    # Verify ordering (most recent first)
    for i in range(len(logs) - 1):
        assert logs[i].timestamp >= logs[i + 1].timestamp
