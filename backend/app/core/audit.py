"""
Audit logging for security and compliance

Requirements: 13.3
"""
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Audit action types"""
    # Data access
    VIEW_PATIENT = "view_patient"
    VIEW_ANALYSIS = "view_analysis"
    VIEW_IMAGE = "view_image"
    
    # Data modification
    CREATE_PATIENT = "create_patient"
    UPDATE_PATIENT = "update_patient"
    DELETE_PATIENT = "delete_patient"
    
    UPLOAD_IMAGE = "upload_image"
    DELETE_IMAGE = "delete_image"
    
    CREATE_ANALYSIS = "create_analysis"
    DELETE_ANALYSIS = "delete_analysis"
    
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    
    # Authorization
    ACCESS_DENIED = "access_denied"


class AuditLog:
    """Audit log entry"""
    
    def __init__(
        self,
        action: AuditAction,
        user_id: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        self.timestamp = datetime.utcnow()
        self.action = action
        self.user_id = user_id
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.details = details or {}
        self.ip_address = ip_address
        self.user_agent = user_agent
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


# In-memory audit log storage (replace with database in production)
audit_logs: list[AuditLog] = []


def log_audit_event(
    action: AuditAction,
    user_id: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> AuditLog:
    """
    Log an audit event
    
    Requirements: 13.3
    
    Args:
        action: The action being performed
        user_id: ID of the user performing the action
        resource_type: Type of resource (patient, analysis, image, etc.)
        resource_id: ID of the specific resource
        details: Additional details about the action
        ip_address: IP address of the request
        user_agent: User agent string from the request
    
    Returns:
        The created audit log entry
    """
    audit_log = AuditLog(
        action=action,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    # Store in memory (replace with database write in production)
    audit_logs.append(audit_log)
    
    # Also log to application logger
    logger.info(
        f"AUDIT: {action.value} by user {user_id} on {resource_type} "
        f"{resource_id or 'N/A'} from {ip_address or 'unknown'}"
    )
    
    return audit_log


def get_audit_logs(
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    action: Optional[AuditAction] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100
) -> list[AuditLog]:
    """
    Query audit logs with filters
    
    Requirements: 13.3
    
    Args:
        user_id: Filter by user ID
        resource_type: Filter by resource type
        resource_id: Filter by resource ID
        action: Filter by action type
        start_time: Filter by start timestamp
        end_time: Filter by end timestamp
        limit: Maximum number of results
    
    Returns:
        List of matching audit log entries
    """
    results = audit_logs.copy()
    
    # Apply filters
    if user_id:
        results = [log for log in results if log.user_id == user_id]
    
    if resource_type:
        results = [log for log in results if log.resource_type == resource_type]
    
    if resource_id:
        results = [log for log in results if log.resource_id == resource_id]
    
    if action:
        results = [log for log in results if log.action == action]
    
    if start_time:
        results = [log for log in results if log.timestamp >= start_time]
    
    if end_time:
        results = [log for log in results if log.timestamp <= end_time]
    
    # Sort by timestamp descending (most recent first)
    results.sort(key=lambda x: x.timestamp, reverse=True)
    
    # Apply limit
    return results[:limit]


def clear_audit_logs():
    """Clear all audit logs (for testing only)"""
    global audit_logs
    audit_logs = []
